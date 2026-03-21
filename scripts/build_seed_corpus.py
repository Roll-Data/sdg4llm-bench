#!/usr/bin/env python3
"""Build and optionally publish the SDG4LLM-Bench seed corpus.

Pulls all four upstream datasets, applies formatters, packages into a
DatasetDict (one config per task), and optionally pushes to HuggingFace.

Usage
-----
    # Dry run — save locally
    python scripts/build_seed_corpus.py --output-dir ./seed_corpus_local

    # Build and push to HuggingFace Hub
    python scripts/build_seed_corpus.py --push --hf-repo sdg4llm-bench/seed-corpus-v1

    # Limit rows per task for testing
    python scripts/build_seed_corpus.py --output-dir /tmp/test_corpus --max-samples 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdg4llm_bench.config import SEED_CORPUS_HF_REPO, TASK_REGISTRY, Task
from sdg4llm_bench.seed_corpus import load_seed


def _print_stats(task_datasets: dict) -> None:
    """Print a table showing row counts and average lengths per task."""
    print("\n" + "=" * 70)
    print(f"{'Task':<20} {'Rows':>8} {'Avg instruction len':>20} {'Avg response len':>18}")
    print("-" * 70)
    for task, ds in task_datasets.items():
        n = len(ds)
        avg_inst = sum(len(r) for r in ds["instruction"]) / n if n > 0 else 0
        avg_resp = sum(len(r) for r in ds["response"]) / n if n > 0 else 0
        print(f"{task.value:<20} {n:>8,} {avg_inst:>20.1f} {avg_resp:>18.1f}")
    print("=" * 70 + "\n")


def _build_dataset_card(task_datasets: dict, hf_repo: str) -> str:
    """Generate a HuggingFace dataset card (README.md) for the seed corpus."""
    rows = []
    for task, ds in task_datasets.items():
        spec = TASK_REGISTRY[task]
        seed_sources = ", ".join(
            repo for repo, _ in spec.seed_datasets
        )
        rows.append(f"| {task.value} | {len(ds):,} | {seed_sources} |")

    rows_text = "\n".join(rows)

    return f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - synthetic-data
  - benchmark
  - llm-training
  - instruction-tuning
---

# SDG4LLM-Bench Seed Corpus v1

This dataset is the canonical seed corpus for [SDG4LLM-Bench](https://github.com/Roll-Data/sdg4llm-bench),
a standardized benchmark for evaluating synthetic data generation (SDG) methods for LLM training.

## Usage

```python
from sdg4llm_bench.seed_corpus import load_seed
from sdg4llm_bench.config import Task

# Load seed data for a specific task
ds = load_seed(Task.gsm8k)

# Load all tasks
from sdg4llm_bench.seed_corpus import load_all_seeds
all_seeds = load_all_seeds()
```

## Dataset Structure

Each config corresponds to one benchmark task. All configs share the same required schema:
- `instruction` (string): The input prompt
- `response` (string): The expected output

Some configs have additional optional columns specific to their source dataset.

## Configs

| Task | Rows | Source |
|------|------|--------|
{rows_text}

## License

Apache 2.0. See upstream dataset licenses for individual splits.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the SDG4LLM-Bench seed corpus from upstream datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./seed_corpus_local"),
        help="Local directory to save the corpus. Default: ./seed_corpus_local",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the corpus to HuggingFace Hub after building.",
    )
    parser.add_argument(
        "--hf-repo",
        default=SEED_CORPUS_HF_REPO,
        help=f"HuggingFace repo ID. Default: {SEED_CORPUS_HF_REPO}",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per task (for testing). Default: all",
    )
    args = parser.parse_args()

    print("Building SDG4LLM-Bench seed corpus...")
    print(f"  Tasks: {[t.value for t in Task]}")
    print(f"  Max samples per task: {args.max_samples or 'all'}")
    print()

    # Load all tasks from upstream (bypass published artifact)
    task_datasets = {}
    for task in Task:
        print(f"  Loading {task.value}...", end=" ", flush=True)
        ds = load_seed(
            task,
            max_samples=args.max_samples,
            use_published_artifact=False,
        )
        task_datasets[task] = ds
        print(f"{len(ds):,} rows")

    _print_stats(task_datasets)

    # Save locally
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task, ds in task_datasets.items():
        task_dir = args.output_dir / task.value
        task_dir.mkdir(exist_ok=True)
        ds.save_to_disk(str(task_dir))
        print(f"  Saved {task.value} → {task_dir}")

    # Write dataset card
    card_path = args.output_dir / "README.md"
    card_path.write_text(_build_dataset_card(task_datasets, args.hf_repo))
    print(f"  Dataset card → {card_path}")

    if args.push:
        try:
            import datasets as _datasets_check  # noqa: PLC0415, F401
        except ImportError:
            print("ERROR: 'datasets' package required for pushing. Install with: uv pip install datasets")
            sys.exit(1)

        print(f"\nPushing to HuggingFace Hub: {args.hf_repo}")
        for task, ds in task_datasets.items():
            print(f"  Pushing {task.value}...", end=" ", flush=True)
            ds.push_to_hub(args.hf_repo, config_name=task.value, split="train")
            print("done")

        print(f"\nCorpus published at: https://huggingface.co/datasets/{args.hf_repo}")
    else:
        print("\nDry run complete. Use --push to publish to HuggingFace Hub.")
        print(f"Corpus saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
