"""Seed corpus loader with two-tier design.

Tier 1 (preferred): Try loading from the published HuggingFace artifact
    sdg4llm-bench/seed-corpus-v1 with named configs per task.

Tier 2 (fallback): If the artifact is unavailable, pull from the original
    upstream HuggingFace datasets and apply formatting locally.

Public API
----------
    load_seed(task, max_samples, use_published_artifact, seed) -> Dataset
    load_all_seeds(max_samples) -> dict[Task, Dataset]
    load_mixed_seeds(max_samples, seed) -> Dataset
"""

from __future__ import annotations

import json
import warnings

from sdg4llm_bench.config import SEED_CORPUS_HF_REPO, TASK_REGISTRY, Task

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"instruction", "response"}


def _validate_schema(ds, task: Task) -> None:
    """Raise ValueError if required columns are missing."""
    missing = REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset for {task.value!r} is missing required columns: {missing}. "
            f"Got: {ds.column_names}"
        )


# ---------------------------------------------------------------------------
# Formatters — one per upstream source
# Each takes a single row dict and returns a dict with at minimum
# 'instruction' and 'response' keys. Return None to signal the row
# should be filtered out (caller handles None rows).
# ---------------------------------------------------------------------------


def _format_flan(row: dict) -> dict:
    """Format a FLAN v2 row.

    Handles both column variants:
      - inputs / targets  (flan2021_submix_original)
      - input  / output   (some other FLAN splits)
    """
    instruction = row.get("inputs") or row.get("input") or ""
    response = row.get("targets") or row.get("output") or ""
    return {
        "instruction": str(instruction).strip(),
        "response": str(response).strip(),
    }


def _format_numinamath(row: dict) -> dict:
    """Format a NuminaMath-CoT row.

    Handles both column variants:
      - problem  / solution
      - question / answer
    """
    instruction = row.get("problem") or row.get("question") or ""
    response = row.get("solution") or row.get("answer") or ""
    return {
        "instruction": str(instruction).strip(),
        "response": str(response).strip(),
    }


def _format_apps(row: dict) -> dict | None:
    """Format an APPS row.

    Parses JSON strings for solutions and test cases. Returns None if
    no usable solution is found (caller must filter None rows).

    Required columns: instruction, response
    Optional columns: test_cases, difficulty, starter_code
    """
    instruction = str(row.get("question") or "").strip()
    if not instruction:
        return None

    # Parse solutions — stored as a JSON-encoded list of strings
    solutions_raw = row.get("solutions") or ""
    solutions = []
    if solutions_raw:
        try:
            parsed = json.loads(solutions_raw)
            if isinstance(parsed, list):
                # Keep all items (including empty strings) to preserve first-solution semantics
                solutions = [s for s in parsed if isinstance(s, str)]
        except (json.JSONDecodeError, TypeError):
            pass

    if not solutions:
        return None

    # Take the first solution — if it's empty, return None (don't fall back to later solutions)
    response = solutions[0].strip()
    if not response:
        return None

    # Parse test cases — stored as a JSON-encoded dict
    test_cases = None
    input_output_raw = row.get("input_output") or ""
    if input_output_raw:
        try:
            test_cases = json.loads(input_output_raw)
        except (json.JSONDecodeError, TypeError):
            test_cases = None

    return {
        "instruction": instruction,
        "response": response,
        "test_cases": test_cases,
        "difficulty": str(row.get("difficulty") or "").strip() or None,
        "starter_code": str(row.get("starter_code") or "").strip() or None,
    }


def _format_arc(row: dict) -> dict:
    """Format an ARC-Challenge row.

    Builds a multi-choice prompt with labels inline, extracts the
    correct answer text.
    """
    question = str(row.get("question") or "").strip()
    choices = row.get("choices") or {}
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    answer_key = str(row.get("answerKey") or "").strip()

    # Build multi-choice prompt
    choice_lines = "\n".join(f"({label}) {text}" for label, text in zip(labels, texts))
    instruction = f"{question}\n{choice_lines}"

    # Find correct answer text
    response = ""
    answer_label = answer_key
    for label, text in zip(labels, texts):
        if str(label) == answer_key:
            response = str(text).strip()
            break

    # Fallback: if answerKey is an index (1-based), try numeric lookup
    if not response and answer_key.isdigit():
        idx = int(answer_key) - 1
        if 0 <= idx < len(texts):
            response = str(texts[idx]).strip()
            answer_label = labels[idx] if idx < len(labels) else answer_key

    return {
        "instruction": instruction.strip(),
        "response": response,
        "answer_label": answer_label,
    }


def _format_winogrande(row: dict) -> dict:
    """Format a WinoGrande row.

    Maps answer "1" → option1 text, "2" → option2 text.
    """
    sentence = str(row.get("sentence") or "").strip()
    option1 = str(row.get("option1") or "").strip()
    option2 = str(row.get("option2") or "").strip()
    answer = str(row.get("answer") or "").strip()

    instruction = f"{sentence}\nOption 1: {option1}\nOption 2: {option2}"

    if answer == "1":
        response = option1
    elif answer == "2":
        response = option2
    else:
        response = ""

    return {
        "instruction": instruction,
        "response": response,
    }


# ---------------------------------------------------------------------------
# Upstream loaders — one per task
# ---------------------------------------------------------------------------


def _load_ifeval_seed(max_samples: int | None = None):
    """Load FLAN v2 and format as seed data for IFEval."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("Muennighoff/flan2021_submix_original", split="train")
    ds = ds.map(_format_flan, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["instruction"] and x["response"])
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _load_gsm8k_seed(max_samples: int | None = None):
    """Load NuminaMath-CoT and format as seed data for GSM8K."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    ds = ds.map(_format_numinamath, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["instruction"] and x["response"])
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _load_humaneval_seed(max_samples: int | None = None):
    """Load APPS (train split) and format as seed data for HumanEval+."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("codeparrot/apps", "all", split="train")

    def _format_and_filter(row):
        result = _format_apps(row)
        if result is None:
            return {
                "instruction": None,
                "response": None,
                "test_cases": None,
                "difficulty": None,
                "starter_code": None,
            }
        return result

    ds = ds.map(_format_and_filter, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["instruction"] is not None and x["response"] is not None)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _load_bbh_seed(max_samples: int | None = None):
    """Load ARC-Challenge + WinoGrande and format as seed data for BBH."""
    from datasets import concatenate_datasets, load_dataset  # noqa: PLC0415

    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    arc_formatted = arc_ds.map(_format_arc, remove_columns=arc_ds.column_names)
    arc_formatted = arc_formatted.filter(lambda x: x["instruction"] and x["response"])

    wg_ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    wg_formatted = wg_ds.map(_format_winogrande, remove_columns=wg_ds.column_names)
    wg_formatted = wg_formatted.filter(lambda x: x["instruction"] and x["response"])

    # Align columns before concatenating (WinoGrande doesn't have answer_label)
    if "answer_label" not in wg_formatted.column_names:
        wg_formatted = wg_formatted.map(lambda x: {"answer_label": None})

    ds = concatenate_datasets([arc_formatted, wg_formatted])
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


_UPSTREAM_LOADERS = {
    Task.ifeval: _load_ifeval_seed,
    Task.gsm8k: _load_gsm8k_seed,
    Task.humaneval_plus: _load_humaneval_seed,
    Task.bbh: _load_bbh_seed,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_seed(
    task: Task,
    max_samples: int | None = None,
    use_published_artifact: bool = True,
    seed: int = 42,
):
    """Load seed data for a single task.

    Parameters
    ----------
    task:
        The benchmark task to load seed data for.
    max_samples:
        Maximum number of samples to return. None means all available.
    use_published_artifact:
        If True (default), try the published HF artifact first.
        If False, go directly to upstream datasets (used by build_seed_corpus.py).
    seed:
        Random seed used when shuffling (only applied when mixing tasks).

    Returns
    -------
    datasets.Dataset with at minimum 'instruction' and 'response' columns.
    """
    from datasets import load_dataset  # noqa: PLC0415

    if use_published_artifact:
        try:
            ds = load_dataset(SEED_CORPUS_HF_REPO, name=task.value, split="train")
            _validate_schema(ds, task)
            if max_samples is not None:
                ds = ds.select(range(min(max_samples, len(ds))))
            return ds
        except Exception as e:
            warnings.warn(
                f"Published artifact '{SEED_CORPUS_HF_REPO}' unavailable for task "
                f"'{task.value}' ({type(e).__name__}: {e}). "
                "Falling back to upstream datasets.",
                stacklevel=2,
            )

    # Fallback: load from upstream
    loader = _UPSTREAM_LOADERS.get(task)
    if loader is None:
        raise RuntimeError(f"No upstream loader registered for task {task!r}")

    ds = loader(max_samples=max_samples)
    _validate_schema(ds, task)
    return ds


def load_all_seeds(
    max_samples: int | None = None,
    use_published_artifact: bool = True,
) -> dict:
    """Load seed data for all four benchmark tasks.

    Returns
    -------
    dict mapping Task -> datasets.Dataset
    """
    return {
        task: load_seed(
            task, max_samples=max_samples, use_published_artifact=use_published_artifact
        )
        for task in TASK_REGISTRY
    }


def load_mixed_seeds(
    max_samples: int | None = None,
    seed: int = 42,
    use_published_artifact: bool = True,
):
    """Load and concatenate seed data for all four tasks, then shuffle.

    Parameters
    ----------
    max_samples:
        Applied per-task before concatenation (i.e. max_samples * 4 total rows
        before shuffle). None means all available.
    seed:
        Shuffle seed for reproducibility.

    Returns
    -------
    datasets.Dataset with 'instruction' and 'response' columns.
    """
    from datasets import concatenate_datasets  # noqa: PLC0415

    all_seeds = load_all_seeds(
        max_samples=max_samples, use_published_artifact=use_published_artifact
    )

    # Ensure all datasets have the same columns before concatenating.
    # Use the minimal required columns to avoid mismatch.
    minimal_datasets = []
    for task, ds in all_seeds.items():
        cols_to_keep = [c for c in ds.column_names if c in REQUIRED_COLUMNS]
        ds_minimal = ds.select_columns(cols_to_keep)
        minimal_datasets.append(ds_minimal)

    combined = concatenate_datasets(minimal_datasets)
    return combined.shuffle(seed=seed)
