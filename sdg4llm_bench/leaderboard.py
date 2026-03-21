"""Leaderboard scoring, validation, and submission management.

Implements the SDG4LLM-Bench leaderboard logic:
  - Composite submission key: (method_name, teacher_model)
  - Geometric mean delta across all four tasks
  - Dual-track eligibility (primary and low-resource)
  - Submission validation against results/schema.json

Track eligibility rules:
  - num_samples <= 5,000  → eligible for BOTH primary and low_resource tracks
  - 5,001 <= num_samples <= 40,000 → eligible for primary track only
  - num_samples > 40,000 → rejected
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from sdg4llm_bench.config import BENCHMARK_TRACKS, Task

# ---------------------------------------------------------------------------
# Schema path (relative to this file)
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).parent.parent / "results" / "schema.json"


def _load_schema() -> dict:
    """Load the JSON Schema for submission validation."""
    if not _SCHEMA_PATH.exists():
        return {}
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TaskResult:
    """Evaluation result for a single task."""

    task: str
    baseline_score: float | None  # From PUBLISHED_BASELINES
    sdg_score: float
    delta: float  # sdg_score - baseline_score

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SubmissionKey:
    """Composite unique key for a submission.

    A submission is uniquely identified by the pair (method_name, teacher_model).
    The same SDG method submitted with different teacher models creates separate
    leaderboard rows.
    """

    method_name: str
    teacher_model: str

    def __str__(self) -> str:
        return f"{self.method_name} / {self.teacher_model}"


@dataclasses.dataclass
class Submission:
    """An SDG4LLM-Bench leaderboard submission."""

    method_name: str
    teacher_model: str  # REQUIRED — submissions without this are rejected
    method_description: str
    authors: list
    task_results: dict  # task name -> TaskResult
    num_samples: int
    track: str  # "primary" or "low_resource"
    timestamp: str
    canonical: bool
    repo_url: str | None = None
    paper_url: str | None = None
    # Computed fields (set by compute_scores)
    qualified: bool = False
    geo_mean_delta: float = 0.0

    @property
    def key(self) -> SubmissionKey:
        return SubmissionKey(
            method_name=self.method_name,
            teacher_model=self.teacher_model,
        )


# ---------------------------------------------------------------------------
# Track eligibility
# ---------------------------------------------------------------------------


def eligible_tracks(num_samples: int) -> set:
    """Return the set of tracks this submission is eligible for.

    Parameters
    ----------
    num_samples:
        Number of synthetic training samples used.

    Returns
    -------
    set of track name strings. Empty set means the submission is rejected.
    """
    if num_samples > BENCHMARK_TRACKS["primary"]:
        return set()  # Over the primary cap — rejected
    elif num_samples <= BENCHMARK_TRACKS["low_resource"]:
        return {"primary", "low_resource"}  # Qualifies for both tracks
    else:
        return {"primary"}  # Primary track only


# ---------------------------------------------------------------------------
# Geometric mean
# ---------------------------------------------------------------------------


def geometric_mean(scores: list) -> float:
    """Compute the geometric mean of a list of positive numbers.

    Parameters
    ----------
    scores:
        List of strictly positive floats.

    Returns
    -------
    Geometric mean as a float.

    Raises
    ------
    ValueError:
        If any score is <= 0 (including zero).
    """
    if not scores:
        raise ValueError("Cannot compute geometric mean of empty list")
    if any(s <= 0 for s in scores):
        raise ValueError(f"Geometric mean requires all strictly positive scores. Got: {scores}")
    return math.prod(scores) ** (1.0 / len(scores))


# ---------------------------------------------------------------------------
# Scoring and validation
# ---------------------------------------------------------------------------


def compute_scores(submission: Submission) -> Submission:
    """Compute qualification status and geo-mean delta.

    Mutates submission in-place and returns it.

    Qualification rule: ALL four task deltas must be strictly positive.
    If qualified, geo_mean_delta is the geometric mean of the four deltas.
    If not qualified, geo_mean_delta is 0.0.
    """
    required_tasks = {t.value for t in Task}
    present_tasks = set(submission.task_results.keys())

    if not required_tasks.issubset(present_tasks):
        submission.qualified = False
        submission.geo_mean_delta = 0.0
        return submission

    deltas = [submission.task_results[t].delta for t in required_tasks]

    if any(d is None for d in deltas):
        submission.qualified = False
        submission.geo_mean_delta = 0.0
        return submission

    if all(d > 0 for d in deltas):
        submission.qualified = True
        try:
            submission.geo_mean_delta = geometric_mean(deltas)
        except ValueError:
            submission.qualified = False
            submission.geo_mean_delta = 0.0
    else:
        submission.qualified = False
        submission.geo_mean_delta = 0.0

    return submission


def validate_submission(submission: Submission) -> list:
    """Validate a submission and return a list of error strings.

    An empty list means the submission is valid. Non-empty means errors exist.

    Note: canonical=False is a WARNING, not an error (does not block submission
    but marks it as unofficial).
    """
    errors = []

    # Required fields
    if not submission.method_name:
        errors.append("method_name is required and must be non-empty")

    if not submission.teacher_model:
        errors.append(
            "teacher_model is required and must be non-empty. "
            "Specify the model used to generate synthetic data (e.g. 'gpt-4o')."
        )

    if not submission.authors:
        errors.append("authors list must contain at least one author")

    # All four tasks must be present
    required_tasks = {t.value for t in Task}
    present_tasks = set(submission.task_results.keys())
    missing_tasks = required_tasks - present_tasks
    if missing_tasks:
        errors.append(
            f"Missing task results for: {sorted(missing_tasks)}. All four tasks must be evaluated."
        )

    # All deltas must be positive (for qualification)
    for task_name, result in submission.task_results.items():
        if result.delta is not None and result.delta <= 0:
            errors.append(
                f"Task {task_name!r} has non-positive delta ({result.delta:.4f}). "
                f"All deltas must be strictly positive to qualify."
            )

    # Volume cap
    tracks = eligible_tracks(submission.num_samples)
    if not tracks:
        errors.append(
            f"num_samples ({submission.num_samples:,}) exceeds the primary track cap "
            f"({BENCHMARK_TRACKS['primary']:,}). Trim your dataset."
        )
    elif (
        submission.track == "low_resource"
        and submission.num_samples > BENCHMARK_TRACKS["low_resource"]
    ):
        errors.append(
            f"Submission declares track='low_resource' but num_samples "
            f"({submission.num_samples:,}) exceeds the low_resource cap "
            f"({BENCHMARK_TRACKS['low_resource']:,})."
        )

    # Canonical warning (not an error)
    if not submission.canonical:
        errors.append(
            "WARNING: canonical=False — this run used a non-standard config and is "
            "not eligible for the official leaderboard."
        )

    return errors


# ---------------------------------------------------------------------------
# Formatting and persistence
# ---------------------------------------------------------------------------


def _teacher_model_slug(teacher_model: str) -> str:
    """Convert a teacher model name to a filename-safe slug.

    Examples:
        "gpt-4o" -> "gpt-4o"
        "meta-llama/Llama-3.1-70B-Instruct" -> "meta-llama-llama-3.1-70b-instruct"
    """
    slug = teacher_model.lower()
    slug = re.sub(r"[/\s]+", "-", slug)
    slug = re.sub(r"[^a-z0-9\-._]", "", slug)
    slug = slug.strip("-")
    return slug


def submission_filename(submission: Submission) -> str:
    """Return the canonical filename for a submission JSON file."""
    method_slug = re.sub(r"[^a-z0-9\-]", "-", submission.method_name.lower()).strip("-")
    teacher_slug = _teacher_model_slug(submission.teacher_model)
    return f"{method_slug}_{teacher_slug}_submission.json"


def format_leaderboard_row(submission: Submission) -> dict:
    """Return a flat dict suitable for rendering in a leaderboard table."""
    row = {
        "method_name": submission.method_name,
        "teacher_model": submission.teacher_model,
        "geo_mean_delta": submission.geo_mean_delta if submission.qualified else None,
        "qualified": submission.qualified,
        "num_samples": submission.num_samples,
        "track": submission.track,
        "canonical": submission.canonical,
        "timestamp": submission.timestamp,
        "repo_url": submission.repo_url,
        "paper_url": submission.paper_url,
    }
    for task in Task:
        result = submission.task_results.get(task.value)
        row[f"{task.value}_score"] = result.sdg_score if result else None
        row[f"{task.value}_delta"] = result.delta if result else None

    return row


def save_submission(submission: Submission, output_dir: str) -> Path:
    """Write the submission as a JSON file.

    If a submission with the same composite key already exists, it is
    replaced (newer timestamp wins).

    Returns
    -------
    Path to the written file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = submission_filename(submission)
    file_path = out_path / filename

    data = {
        "method_name": submission.method_name,
        "teacher_model": submission.teacher_model,
        "method_description": submission.method_description,
        "authors": submission.authors,
        "task_results": {k: v.to_dict() for k, v in submission.task_results.items()},
        "num_samples": submission.num_samples,
        "track": submission.track,
        "timestamp": submission.timestamp,
        "canonical": submission.canonical,
        "geo_mean_delta": submission.geo_mean_delta,
        "qualified": submission.qualified,
        "repo_url": submission.repo_url,
        "paper_url": submission.paper_url,
    }

    file_path.write_text(json.dumps(data, indent=2))
    return file_path


def load_submission(file_path: str) -> Submission:
    """Load a submission from a JSON file and compute scores."""
    with open(file_path) as f:
        data = json.load(f)

    task_results = {}
    for task_name, tr_data in data.get("task_results", {}).items():
        task_results[task_name] = TaskResult(
            task=tr_data["task"],
            baseline_score=tr_data.get("baseline_score"),
            sdg_score=tr_data["sdg_score"],
            delta=tr_data["delta"],
        )

    sub = Submission(
        method_name=data["method_name"],
        teacher_model=data["teacher_model"],
        method_description=data.get("method_description", ""),
        authors=data.get("authors", []),
        task_results=task_results,
        num_samples=data["num_samples"],
        track=data["track"],
        timestamp=data["timestamp"],
        canonical=data.get("canonical", False),
        repo_url=data.get("repo_url"),
        paper_url=data.get("paper_url"),
        qualified=data.get("qualified", False),
        geo_mean_delta=data.get("geo_mean_delta", 0.0),
    )
    return sub


def print_submission_summary(submission: Submission) -> None:
    """Print a human-readable summary of a submission."""
    print("\n" + "=" * 70)
    print("Submission Summary")
    print("=" * 70)
    print(f"  Method:        {submission.method_name}")
    print(f"  Teacher model: {submission.teacher_model}")
    print(f"  Authors:       {', '.join(submission.authors)}")
    print(f"  Track:         {submission.track}")
    print(f"  Samples:       {submission.num_samples:,}")
    print(f"  Canonical:     {submission.canonical}")
    print(f"  Timestamp:     {submission.timestamp}")
    print()
    print(f"  {'Task':<20} {'Baseline':>10} {'SDG Score':>10} {'Delta':>10}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")
    for task in Task:
        result = submission.task_results.get(task.value)
        if result:
            base_str = (
                f"{result.baseline_score:.4f}" if result.baseline_score is not None else "  N/A  "
            )
            delta_str = f"{result.delta:+.4f}" if result.delta is not None else "  N/A  "
            print(f"  {task.value:<20} {base_str:>10} {result.sdg_score:>10.4f} {delta_str:>10}")
        else:
            print(f"  {task.value:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    print()

    if submission.qualified:
        print(f"  ✓ QUALIFIED  |  Geo-mean Δ: {submission.geo_mean_delta:.4f}")
    else:
        print("  ✗ NOT QUALIFIED — one or more task deltas are non-positive")

    eligible = eligible_tracks(submission.num_samples)
    print(f"  Eligible tracks: {sorted(eligible) if eligible else ['none — rejected']}")

    if not submission.canonical:
        print("\n  ⚠  Non-canonical config — NOT eligible for official leaderboard")
    print("=" * 70 + "\n")


def build_submission_from_eval(
    eval_results: dict,
    method_name: str,
    teacher_model: str,
    num_samples: int,
    track: str,
    canonical: bool,
    method_description: str = "",
    authors: list | None = None,
    repo_url: str | None = None,
    paper_url: str | None = None,
) -> Submission:
    """Build a Submission from evaluate.py output dicts.

    Parameters
    ----------
    eval_results:
        Output of evaluate_checkpoint() — dict of task name -> TaskResult (from evaluate.py).
    """
    from sdg4llm_bench.evaluate import TaskResult as EvalTaskResult  # noqa: PLC0415

    task_results = {}
    for task_name, eval_result in eval_results.items():
        if isinstance(eval_result, EvalTaskResult):
            baseline = eval_result.baseline_score
            sdg_score = eval_result.score
        elif isinstance(eval_result, dict):
            baseline = eval_result.get("baseline_score")
            sdg_score = eval_result.get("sdg_score") or eval_result.get("score", 0.0)
        else:
            continue

        if baseline is None:
            print(
                f"  WARNING: No published baseline for task {task_name!r}. "
                "Baselines have not yet been published by maintainers. "
                "Delta set to 0.0; this submission will not qualify.",
                file=sys.stderr,
            )
            delta = 0.0
        else:
            delta = sdg_score - baseline
        task_results[task_name] = TaskResult(
            task=task_name,
            baseline_score=baseline,
            sdg_score=sdg_score,
            delta=delta,
        )

    sub = Submission(
        method_name=method_name,
        teacher_model=teacher_model,
        method_description=method_description,
        authors=authors or [],
        task_results=task_results,
        num_samples=num_samples,
        track=track,
        timestamp=datetime.now(timezone.utc).isoformat(),
        canonical=canonical,
        repo_url=repo_url,
        paper_url=paper_url,
    )
    return compute_scores(sub)
