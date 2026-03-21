"""Fine-tuning runner for SDG4LLM-Bench.

Trains a LoRA adapter on the student model (Llama 3.1 8B) using either
canonical seed data or submitter-provided SDG data.

Modes
-----
    mixed      Train on all four tasks combined (official benchmark mode).
    per-task   Train on one task only (diagnostic/ablation mode).

Conditions
----------
    seed   Use the canonical seed corpus.
    sdg    Use submitter's JSONL file (requires --sdg-data-path).

Usage
-----
    # Official benchmark: SDG mixed training
    python -m sdg4llm_bench.train \\
        --mode mixed --condition sdg --sdg-data-path my_data.jsonl \\
        --method-name Evol-Instruct --track primary --output-dir runs/evol

    # Seed-only baseline (run once by maintainers)
    python -m sdg4llm_bench.train \\
        --mode mixed --condition seed --output-dir runs/seed_baseline

    # Per-task diagnostic
    python -m sdg4llm_bench.train \\
        --mode per-task --task gsm8k --condition sdg \\
        --sdg-data-path gsm8k_sdg.jsonl --output-dir runs/diag_gsm8k

    # Custom config (non-canonical, not leaderboard eligible)
    python -m sdg4llm_bench.train \\
        --mode mixed --condition sdg --sdg-data-path my_data.jsonl \\
        --allow-custom-config --lora-rank 32 --output-dir runs/custom
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone

from sdg4llm_bench.config import (
    BENCHMARK_TRACKS,
    LORA_CONFIG,
    PER_TASK_DIAGNOSTIC_CAP,
    STUDENT_MODEL,
    TRAINING_CONFIG,
    LoRAConfig,
    Paths,
    Task,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Chat template (Llama 3.1 format)
# ---------------------------------------------------------------------------

CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "{response}<|eot_id|>"
)


def apply_chat_template(instruction: str, response: str) -> str:
    """Format a single (instruction, response) pair using the Llama 3.1 chat template."""
    return CHAT_TEMPLATE.format(instruction=instruction, response=response)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: str, max_samples: int | None = None):
    """Load a JSONL file as a HuggingFace Dataset.

    Validates that 'instruction' and 'response' columns are present.
    """
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("json", data_files=str(path), split="train")
    missing = {"instruction", "response"} - set(ds.column_names)
    if missing:
        raise ValueError(
            f"JSONL file {path!r} is missing required columns: {missing}. Got: {ds.column_names}"
        )
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _check_volume_cap(num_samples: int, track: str) -> None:
    """Raise ValueError if num_samples exceeds the track's volume cap."""
    cap = BENCHMARK_TRACKS.get(track)
    if cap is None:
        raise ValueError(f"Unknown track {track!r}. Valid tracks: {list(BENCHMARK_TRACKS)}")
    if num_samples > cap:
        raise ValueError(
            f"Dataset has {num_samples:,} samples, which exceeds the {track} track "
            f"volume cap of {cap:,}. Trim your dataset before submitting."
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    mode: str,
    condition: str,
    output_dir: str,
    task: str | None = None,
    sdg_data_path: str | None = None,
    method_name: str = "unknown",
    track: str = "primary",
    student_model: str = STUDENT_MODEL,
    lora_config: LoRAConfig = LORA_CONFIG,
    training_config: TrainingConfig = TRAINING_CONFIG,
    canonical: bool = True,
) -> Paths:
    """Run LoRA fine-tuning and save the adapter + meta.json.

    Parameters
    ----------
    mode:
        "mixed" or "per-task"
    condition:
        "seed" or "sdg"
    output_dir:
        Directory to save the LoRA adapter, tokenizer, and meta.json.
    task:
        Required when mode="per-task". One of the Task enum values.
    sdg_data_path:
        Required when condition="sdg". Path to a JSONL file.
    method_name:
        Name of the SDG method (for meta.json).
    track:
        "primary" or "low_resource".
    student_model:
        HuggingFace model ID for the base model.
    lora_config:
        LoRA configuration (locked for canonical runs).
    training_config:
        Training hyperparameters (locked for canonical runs).
    canonical:
        Whether this run uses the locked canonical config.

    Returns
    -------
    Paths object pointing to the saved outputs.
    """
    # ── Validate arguments ──────────────────────────────────────────────────
    if condition == "sdg" and sdg_data_path is None:
        raise ValueError("--sdg-data-path is required when --condition sdg")
    if condition not in ("seed", "sdg"):
        raise ValueError(f"Invalid condition {condition!r}. Must be 'seed' or 'sdg'.")
    if mode not in ("mixed", "per-task"):
        raise ValueError(f"Invalid mode {mode!r}. Must be 'mixed' or 'per-task'.")
    if mode == "per-task" and task is None:
        raise ValueError("--task is required when --mode per-task")

    task_obj = None
    if mode == "per-task":
        try:
            task_obj = Task(task)
        except ValueError:
            valid = [t.value for t in Task]
            raise ValueError(f"Invalid task {task!r}. Valid tasks: {valid}")

    import torch  # noqa: PLC0415
    from peft import get_peft_model  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading data (condition={condition!r}, mode={mode!r})...")

    if condition == "seed":
        from sdg4llm_bench.seed_corpus import load_mixed_seeds, load_seed  # noqa: PLC0415

        if mode == "mixed":
            raw_ds = load_mixed_seeds()
        else:
            raw_ds = load_seed(task_obj)
            if len(raw_ds) > PER_TASK_DIAGNOSTIC_CAP:
                raise ValueError(
                    f"Seed dataset for task {task!r} has {len(raw_ds):,} samples, "
                    f"which exceeds the per-task diagnostic cap of {PER_TASK_DIAGNOSTIC_CAP:,}. "
                    "Use max_samples in load_seed() to trim the dataset."
                )
    else:
        # SDG condition — validate volume cap against the requested track
        raw_ds = _load_jsonl(sdg_data_path)
        _check_volume_cap(len(raw_ds), track)

    num_samples = len(raw_ds)
    print(f"  {num_samples:,} training samples")

    # ── Set up output paths ──────────────────────────────────────────────────
    paths = Paths(output_dir)
    paths.make_dirs()

    # ── Set seeds ────────────────────────────────────────────────────────────
    from transformers import set_seed  # noqa: PLC0415

    set_seed(training_config.seed)

    # ── Apply chat template ──────────────────────────────────────────────────
    print("Applying chat template...")

    def _tokenize_fn(examples, tokenizer, max_length):
        texts = [
            apply_chat_template(inst, resp)
            for inst, resp in zip(examples["instruction"], examples["response"])
        ]
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    # ── Load tokenizer and model ─────────────────────────────────────────────
    print(f"Loading model: {student_model}")
    tokenizer = AutoTokenizer.from_pretrained(student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if training_config.gradient_checkpointing:
        model.enable_input_require_grads()

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    print("Applying LoRA...")
    peft_config = lora_config.to_peft_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Tokenize dataset ─────────────────────────────────────────────────────
    from transformers import DataCollatorForLanguageModeling  # noqa: PLC0415

    tokenized_ds = raw_ds.map(
        lambda examples: _tokenize_fn(examples, tokenizer, training_config.max_seq_length),
        batched=True,
        remove_columns=raw_ds.column_names,
    )
    tokenized_ds = tokenized_ds.filter(lambda x: len(x["input_ids"]) > 0)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting training...")
    training_args = training_config.to_hf_training_args(str(paths.adapter_dir))
    from transformers import Trainer  # noqa: PLC0415

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    # ── Save adapter + tokenizer ──────────────────────────────────────────────
    print(f"Saving adapter to {paths.adapter_dir}...")
    model.save_pretrained(str(paths.adapter_dir))
    tokenizer.save_pretrained(str(paths.adapter_dir))

    # ── Write meta.json ───────────────────────────────────────────────────────
    tasks_used = [task] if mode == "per-task" else [t.value for t in Task]
    meta = {
        "tasks": tasks_used,
        "mode": mode,
        "condition": condition,
        "method_name": method_name,
        "num_samples": num_samples,
        "model": student_model,
        "lora_config": dataclasses.asdict(lora_config),
        "training_config": {
            k: v
            for k, v in dataclasses.asdict(training_config).items()
            if k not in ("max_seq_length",)
        },
        "canonical": canonical,
        "track": track,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    paths.meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Meta written to {paths.meta_path}")

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SDG4LLM-Bench: Fine-tune student model with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["mixed", "per-task"],
        required=True,
        help="'mixed' for official benchmark (all tasks), 'per-task' for diagnostics",
    )
    parser.add_argument(
        "--condition",
        choices=["seed", "sdg"],
        required=True,
        help="'seed' for canonical seed data, 'sdg' for your JSONL",
    )
    parser.add_argument(
        "--task",
        choices=[t.value for t in Task],
        help="Required when --mode per-task",
    )
    parser.add_argument(
        "--sdg-data-path",
        metavar="PATH",
        help="Path to JSONL with 'instruction' and 'response' columns (required for --condition sdg)",
    )
    parser.add_argument(
        "--output-dir",
        default="./runs/default",
        help="Where to save the LoRA adapter and meta.json. Default: ./runs/default",
    )
    parser.add_argument(
        "--method-name",
        default="unknown",
        help="Name of your SDG method (for meta.json and submission)",
    )
    parser.add_argument(
        "--track",
        choices=["primary", "low_resource"],
        default="primary",
        help="Benchmark track. Default: primary",
    )
    # Override flags (require --allow-custom-config)
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--allow-custom-config",
        action="store_true",
        help="Allow non-canonical overrides. Results will NOT be eligible for the leaderboard.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # ── Check for custom config overrides ────────────────────────────────────
    overrides = {
        "student_model": args.student_model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
    }
    has_overrides = any(v is not None for v in overrides.values())

    if has_overrides and not args.allow_custom_config:
        override_names = [k for k, v in overrides.items() if v is not None]
        print(
            f"ERROR: Custom config flags detected ({override_names}) but "
            "--allow-custom-config was not set.\n"
            "Add --allow-custom-config to proceed. Note: results will not be "
            "eligible for the official SDG4LLM-Bench leaderboard.",
            file=sys.stderr,
        )
        sys.exit(1)

    canonical = True
    if args.allow_custom_config and has_overrides:
        print(
            "⚠  Custom config detected. Results are NOT eligible for the "
            "official SDG4LLM-Bench leaderboard.",
            file=sys.stderr,
        )
        canonical = False

    # ── Build effective configs ───────────────────────────────────────────────
    student_model = args.student_model or STUDENT_MODEL

    effective_lora = LORA_CONFIG
    if args.lora_rank is not None or args.lora_alpha is not None:
        effective_lora = LoRAConfig(
            rank=args.lora_rank if args.lora_rank is not None else LORA_CONFIG.rank,
            alpha=args.lora_alpha if args.lora_alpha is not None else LORA_CONFIG.alpha,
            target_modules=LORA_CONFIG.target_modules,
            dropout=LORA_CONFIG.dropout,
            task_type=LORA_CONFIG.task_type,
        )

    effective_training = TRAINING_CONFIG
    if args.learning_rate is not None:
        # Recreate with overridden lr
        training_dict = dataclasses.asdict(TRAINING_CONFIG)
        training_dict["lr"] = args.learning_rate
        effective_training = TrainingConfig(**training_dict)

    # ── Run training ──────────────────────────────────────────────────────────
    train(
        mode=args.mode,
        condition=args.condition,
        output_dir=args.output_dir,
        task=args.task,
        sdg_data_path=args.sdg_data_path,
        method_name=args.method_name,
        track=args.track,
        student_model=student_model,
        lora_config=effective_lora,
        training_config=effective_training,
        canonical=canonical,
    )


if __name__ == "__main__":
    main()
