"""Locked benchmark configuration.

This module is the spine of SDG4LLM-Bench. All canonical values live here.
Every other module imports from this file. The CI config-integrity job hashes
these values to detect accidental drift.

IMPORTANT: Do not import torch, transformers, peft, or lm_eval at module level.
All heavy dependencies must be lazy-imported inside methods. This keeps config.py
importable in CPU-only CI environments without those packages installed.
"""

from __future__ import annotations

import dataclasses
import enum
import pathlib

# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------

STUDENT_MODEL: str = "meta-llama/Llama-3.1-8B"
MODEL_DTYPE: str = "float16"

# ---------------------------------------------------------------------------
# LoRA configuration (locked)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LoRAConfig:
    """Immutable LoRA configuration for the official benchmark.

    All fields are locked. Any change to these values breaks backward
    compatibility with previously submitted results.
    """

    rank: int = 64
    alpha: int = 128
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    dropout: float = 0.05
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self):
        """Return a peft.LoraConfig instance. Lazy-imports peft."""
        from peft import LoraConfig, TaskType  # noqa: PLC0415

        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=list(self.target_modules),
            lora_dropout=self.dropout,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )


LORA_CONFIG: LoRAConfig = LoRAConfig()

# ---------------------------------------------------------------------------
# Training configuration (locked)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Immutable training hyperparameters for the official benchmark.

    Use to_hf_training_args(output_dir) to get a TrainingArguments instance.
    Field max_seq_length is NOT passed to TrainingArguments — it is
    handled separately in train.py.
    """

    optimizer: str = "adamw_torch"
    lr: float = 2e-4
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 64
    max_seq_length: int = 2048
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    seed: int = 42
    data_seed: int = 42
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    eval_strategy: str = "no"

    def to_hf_training_args(self, output_dir: str):
        """Return a transformers.TrainingArguments instance. Lazy-imports transformers."""
        from transformers import TrainingArguments  # noqa: PLC0415

        return TrainingArguments(
            output_dir=output_dir,
            optim=self.optimizer,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            adam_beta1=self.adam_betas[0],
            adam_beta2=self.adam_betas[1],
            adam_epsilon=self.adam_epsilon,
            max_grad_norm=self.max_grad_norm,
            lr_scheduler_type=self.lr_scheduler,
            warmup_ratio=self.warmup_ratio,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            seed=self.seed,
            data_seed=self.data_seed,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
            eval_strategy=self.eval_strategy,
            report_to="none",
        )


TRAINING_CONFIG: TrainingConfig = TrainingConfig()

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------


class Task(str, enum.Enum):
    """The four canonical benchmark tasks.

    Extends str so that Task.ifeval == "ifeval" is True, which simplifies
    JSON serialization and dict key lookups.
    """

    ifeval = "ifeval"
    gsm8k = "gsm8k"
    humaneval_plus = "humaneval_plus"
    bbh = "bbh"


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """Specification for a single benchmark task."""

    name: str
    eval_tool: str  # "lm_eval" or "evalplus"
    lm_eval_task: str | None  # None for evalplus tasks
    metric: str
    higher_is_better: bool
    num_fewshot: int
    # Tuple of (hf_repo_id, config_name_or_None) pairs for upstream seed data
    seed_datasets: tuple[tuple[str, str | None], ...]
    description: str


TASK_REGISTRY: dict[Task, TaskSpec] = {
    Task.ifeval: TaskSpec(
        name="ifeval",
        eval_tool="lm_eval",
        lm_eval_task="ifeval",
        metric="prompt_level_strict_acc",
        higher_is_better=True,
        num_fewshot=0,
        seed_datasets=(("Muennighoff/flan2021_submix_original", None),),
        description="Instruction Following Evaluation — tests adherence to explicit formatting constraints.",
    ),
    Task.gsm8k: TaskSpec(
        name="gsm8k",
        eval_tool="lm_eval",
        lm_eval_task="gsm8k",
        metric="exact_match,strict-match",
        higher_is_better=True,
        num_fewshot=5,
        seed_datasets=(("AI-MO/NuminaMath-CoT", None),),
        description="Grade-school math word problems requiring multi-step reasoning.",
    ),
    Task.humaneval_plus: TaskSpec(
        name="humaneval_plus",
        eval_tool="evalplus",
        lm_eval_task=None,  # HumanEval+ is ONLY evaluated via EvalPlus
        metric="pass@1",
        higher_is_better=True,
        num_fewshot=0,
        seed_datasets=(("codeparrot/apps", "all"),),
        description="Python code generation, evaluated with EvalPlus extended test suite.",
    ),
    Task.bbh: TaskSpec(
        name="bbh",
        eval_tool="lm_eval",
        lm_eval_task="bbh",
        metric="exact_match,strict-match",
        higher_is_better=True,
        num_fewshot=3,
        seed_datasets=(
            ("allenai/ai2_arc", "ARC-Challenge"),
            ("allenai/winogrande", "winogrande_xl"),
        ),
        description="BIG-Bench Hard — challenging tasks requiring complex reasoning.",
    ),
}

# ---------------------------------------------------------------------------
# Benchmark tracks
# ---------------------------------------------------------------------------

BENCHMARK_TRACKS: dict[str, int] = {
    "primary": 40_000,
    "low_resource": 5_000,
}

PER_TASK_DIAGNOSTIC_CAP: int = 10_000

# ---------------------------------------------------------------------------
# Seed corpus
# ---------------------------------------------------------------------------

SEED_CORPUS_HF_REPO: str = "sdg4llm-bench/seed-corpus-v1"

# ---------------------------------------------------------------------------
# Published baselines
# ---------------------------------------------------------------------------

# These values are populated by maintainers after the first official run and
# committed to the repo. Submitters do NOT re-run baseline training — they
# compute deltas against these published numbers.
#
# Structure: {task_value: {"metric": metric_name, "score": float}}
# Example after population:
#   "ifeval": {"metric": "prompt_level_strict_acc", "score": 0.412},
PUBLISHED_BASELINES: dict[str, dict] = {
    # TODO: Populate after running official seed-only baseline training.
    # Run: python -m sdg4llm_bench.train --mode mixed --condition seed
    # Then: python -m sdg4llm_bench.evaluate --adapter-path <checkpoint> --tasks all
    "ifeval": {"metric": "prompt_level_strict_acc", "score": None},
    "gsm8k": {"metric": "exact_match,strict-match", "score": None},
    "humaneval_plus": {"metric": "pass@1", "score": None},
    "bbh": {"metric": "exact_match,strict-match", "score": None},
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Paths:
    """Conventional directory layout for a training run.

    Not frozen — initialized from a base directory at runtime.
    """

    base_dir: pathlib.Path

    def __post_init__(self) -> None:
        self.base_dir = pathlib.Path(self.base_dir)

    @property
    def adapter_dir(self) -> pathlib.Path:
        return self.base_dir / "adapter"

    @property
    def meta_path(self) -> pathlib.Path:
        return self.base_dir / "meta.json"

    @property
    def eval_dir(self) -> pathlib.Path:
        return self.base_dir / "eval_results"

    @property
    def submission_dir(self) -> pathlib.Path:
        return self.base_dir / "submission"

    def make_dirs(self) -> None:
        """Create all directories."""
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.submission_dir.mkdir(parents=True, exist_ok=True)
