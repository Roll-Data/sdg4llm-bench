"""Evaluation runner for SDG4LLM-Bench.

Two evaluation code paths based on TaskSpec.eval_tool:

    lm_eval   IFEval, GSM8K, BBH — run via lm-evaluation-harness subprocess
    evalplus  HumanEval+ — run via EvalPlus subprocess

IMPORTANT: HumanEval+ is NEVER evaluated via lm-evaluation-harness.

Usage
-----
    # Evaluate a fine-tuned checkpoint on all tasks
    python -m sdg4llm_bench.evaluate \\
        --adapter-path runs/evol/adapter \\
        --tasks all \\
        --output-dir runs/evol/eval_results

    # Evaluate specific tasks
    python -m sdg4llm_bench.evaluate \\
        --adapter-path runs/evol/adapter \\
        --tasks gsm8k bbh

    # Evaluate the base model (no adapter)
    python -m sdg4llm_bench.evaluate --base-model-only --tasks all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from sdg4llm_bench.config import (
    PUBLISHED_BASELINES,
    STUDENT_MODEL,
    TASK_REGISTRY,
    Task,
)

# ---------------------------------------------------------------------------
# Task result dataclass (lightweight; full version is in leaderboard.py)
# ---------------------------------------------------------------------------


class TaskResult:
    """Holds evaluation scores for one task."""

    __slots__ = ("task", "metric", "score", "baseline_score", "delta")

    def __init__(
        self,
        task: str,
        metric: str,
        score: float,
        baseline_score: float | None = None,
        delta: float | None = None,
    ):
        self.task = task
        self.metric = metric
        self.score = score
        self.baseline_score = baseline_score
        self.delta = delta

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "metric": self.metric,
            "sdg_score": self.score,
            "baseline_score": self.baseline_score,
            "delta": self.delta,
        }

    def __repr__(self) -> str:
        return (
            f"TaskResult(task={self.task!r}, score={self.score:.4f}, "
            f"baseline={self.baseline_score}, delta={self.delta})"
        )


# ---------------------------------------------------------------------------
# lm-evaluation-harness runner
# ---------------------------------------------------------------------------


def _run_lm_eval(
    task_spec,
    base_model: str,
    adapter_path: str | None,
    output_dir: Path,
) -> float | None:
    """Run lm-evaluation-harness for one task via subprocess.

    Returns the numeric score for the task's metric, or None on failure.
    """

    results_dir = output_dir / task_spec.name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build model args string
    model_args = f"pretrained={base_model},dtype=float16"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        task_spec.lm_eval_task,
        "--num_fewshot",
        str(task_spec.num_fewshot),
        "--output_path",
        str(results_dir),
        "--log_samples",
    ]

    print(f"  Running lm-eval for {task_spec.name}...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(
            f"  WARNING: lm-eval exited with code {result.returncode} for {task_spec.name}",
            file=sys.stderr,
        )
        return None

    # Parse results JSON
    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        print(f"  WARNING: No result JSON found for {task_spec.name}", file=sys.stderr)
        return None

    result_file = sorted(result_files)[-1]  # Latest file
    with open(result_file) as f:
        results_data = json.load(f)

    # Extract metric value
    task_results = results_data.get("results", {}).get(task_spec.lm_eval_task, {})
    metric_value = task_results.get(task_spec.metric)
    if metric_value is None:
        # Try alternate key formats
        for key, val in task_results.items():
            if task_spec.metric.split(",")[0] in key:
                metric_value = val
                break

    if metric_value is None:
        print(
            f"  WARNING: Could not find metric {task_spec.metric!r} in results for "
            f"{task_spec.name}. Available keys: {list(task_results.keys())}",
            file=sys.stderr,
        )
        return None

    return float(metric_value)


# ---------------------------------------------------------------------------
# EvalPlus runner (HumanEval+ only)
# ---------------------------------------------------------------------------


def _run_evalplus(
    task_spec,
    base_model: str,
    adapter_path: str | None,
    output_dir: Path,
) -> float | None:
    """Run EvalPlus evaluation for HumanEval+ via subprocess.

    Returns pass@1 score or None on failure.
    """
    try:
        import evalplus  # noqa: PLC0415, F401
    except ImportError:
        print(
            "ERROR: evalplus is required for HumanEval+ evaluation. "
            "Install with: uv pip install 'sdg4llm-bench[evalplus]'",
            file=sys.stderr,
        )
        return None

    results_dir = output_dir / "humaneval_plus"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate solutions
    samples_file = results_dir / "samples.jsonl"
    model_str = base_model
    if adapter_path:
        model_str = f"{base_model}+{adapter_path}"

    gen_cmd = [
        sys.executable,
        "-m",
        "evalplus.evaluate",
        "--model",
        model_str,
        "--dataset",
        "humaneval",
        "--backend",
        "vllm",
        "--greedy",
        "--output",
        str(samples_file),
    ]

    print("  Running EvalPlus for HumanEval+...")
    print(f"  Command: {' '.join(gen_cmd)}")

    gen_result = subprocess.run(gen_cmd, capture_output=False, text=True)
    if gen_result.returncode != 0:
        print(
            f"  WARNING: EvalPlus generation exited with code {gen_result.returncode}",
            file=sys.stderr,
        )
        return None

    # Parse pass@1 from output
    eval_cmd = [
        sys.executable,
        "-m",
        "evalplus.evaluate",
        "--dataset",
        "humaneval",
        "--samples",
        str(samples_file),
    ]
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)

    # Primary path: read the JSON results file EvalPlus writes alongside the samples file.
    # The file is named <samples_stem>.eval_results.json (e.g. samples.eval_results.json).
    json_results_file = results_dir / "samples.eval_results.json"
    if json_results_file.exists():
        try:
            import json as _json  # noqa: PLC0415

            with open(json_results_file) as _f:
                ep_data = _json.load(_f)
            # EvalPlus JSON structure: {"eval": {"HumanEval/<id>": {"base": {"pass@1": float}}}}
            # Aggregate pass@1 across all problems.
            eval_section = ep_data.get("eval", {})
            scores = []
            for prob_results in eval_section.values():
                base = prob_results.get("base", {})
                if "pass@1" in base:
                    scores.append(float(base["pass@1"]))
            if scores:
                return sum(scores) / len(scores)
        except (KeyError, TypeError, ValueError, OSError):
            pass  # Fall through to stdout parsing

    # Fallback: parse pass@1 from stdout.
    for line in eval_result.stdout.splitlines():
        if "pass@1" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    score = float(parts[-1].strip().rstrip("%")) / 100
                    return score
                except ValueError:
                    pass

    print(
        f"  WARNING: Could not parse pass@1 from EvalPlus JSON results file or stdout. "
        f"stdout:\n{eval_result.stdout[:500]}",
        file=sys.stderr,
    )
    return None


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

EVAL_DISPATCH = {
    "lm_eval": _run_lm_eval,
    "evalplus": _run_evalplus,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    adapter_path: str | None,
    base_model: str = STUDENT_MODEL,
    tasks: list | None = None,
    output_dir: str | None = None,
) -> dict:
    """Evaluate a LoRA checkpoint on the specified tasks.

    Parameters
    ----------
    adapter_path:
        Path to the saved LoRA adapter directory.
    base_model:
        Base model ID. Defaults to the canonical student model.
    tasks:
        List of Task instances or task name strings to evaluate.
        None means all four canonical tasks.
    output_dir:
        Directory for evaluation artifacts. Defaults to adapter_path/eval_results.

    Returns
    -------
    dict mapping task name str -> TaskResult
    """
    if tasks is None:
        tasks = list(TASK_REGISTRY.keys())
    else:
        tasks = [Task(t) if isinstance(t, str) else t for t in tasks]

    if output_dir is None:
        if adapter_path is not None:
            output_dir = str(Path(adapter_path).parent / "eval_results")
        else:
            output_dir = "./runs/base_eval/eval_results"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = {}
    for task in tasks:
        spec = TASK_REGISTRY[task]
        runner = EVAL_DISPATCH.get(spec.eval_tool)
        if runner is None:
            raise ValueError(f"Unknown eval_tool {spec.eval_tool!r} for task {task.value!r}")

        print(f"\nEvaluating {task.value} (eval_tool={spec.eval_tool!r})...")
        score = runner(spec, base_model, adapter_path, out_path)

        if score is not None:
            baseline = PUBLISHED_BASELINES.get(task.value, {}).get("score")
            delta = (score - baseline) if (baseline is not None) else None
            results[task.value] = TaskResult(
                task=task.value,
                metric=spec.metric,
                score=score,
                baseline_score=baseline,
                delta=delta,
            )
        else:
            print(f"  Skipping {task.value} — evaluation returned None.")

    # Save results JSON
    results_file = out_path / "eval_results.json"
    results_file.write_text(json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2))
    print(f"\nResults written to {results_file}")

    return results


def evaluate_base_model(
    tasks: list | None = None,
    output_dir: str = "./runs/base_model_eval",
) -> dict:
    """Evaluate the unmodified base model (no LoRA adapter).

    Parameters
    ----------
    tasks:
        Tasks to evaluate. None means all four.
    output_dir:
        Directory for evaluation artifacts.

    Returns
    -------
    dict mapping task name str -> TaskResult
    """
    return evaluate_checkpoint(
        adapter_path=None,
        base_model=STUDENT_MODEL,
        tasks=tasks,
        output_dir=output_dir,
    )


def compute_deltas(
    sdg_scores: dict,
    baseline_scores: dict | None = None,
) -> dict:
    """Compute per-task deltas between SDG scores and baseline scores.

    Parameters
    ----------
    sdg_scores:
        dict mapping task name -> score (float) or TaskResult
    baseline_scores:
        dict mapping task name -> score (float). If None, uses PUBLISHED_BASELINES.

    Returns
    -------
    dict mapping task name -> delta (float)
    """
    if baseline_scores is None:
        baseline_scores = {k: v.get("score") for k, v in PUBLISHED_BASELINES.items()}

    deltas = {}
    for task_name, sdg_val in sdg_scores.items():
        sdg_score = sdg_val.score if isinstance(sdg_val, TaskResult) else sdg_val
        baseline = baseline_scores.get(task_name)
        if baseline is not None and sdg_score is not None:
            deltas[task_name] = sdg_score - baseline
        else:
            deltas[task_name] = None
    return deltas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="SDG4LLM-Bench: Evaluate a fine-tuned checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--adapter-path",
        metavar="PATH",
        help="Path to a saved LoRA adapter directory.",
    )
    group.add_argument(
        "--base-model-only",
        action="store_true",
        help="Evaluate the base model without any adapter.",
    )
    parser.add_argument(
        "--base-model",
        default=STUDENT_MODEL,
        help=f"Base model ID. Default: {STUDENT_MODEL}",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=[t.value for t in Task] + ["all"],
        help="Tasks to evaluate. 'all' evaluates all four. Default: all",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation results. Default: <adapter-path>/../eval_results",
    )
    args = parser.parse_args(argv)

    tasks = None if "all" in args.tasks else args.tasks

    if args.base_model_only:
        results = evaluate_base_model(tasks=tasks, output_dir=args.output_dir or "./runs/base_eval")
    else:
        results = evaluate_checkpoint(
            adapter_path=args.adapter_path,
            base_model=args.base_model,
            tasks=tasks,
            output_dir=args.output_dir,
        )

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for task_name, result in results.items():
        delta_str = f"{result.delta:+.4f}" if result.delta is not None else "N/A (no baseline)"
        print(f"  {task_name:<20} score={result.score:.4f}  delta={delta_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()
