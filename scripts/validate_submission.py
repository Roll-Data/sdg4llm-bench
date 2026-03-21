#!/usr/bin/env python3
"""Validate a submission JSON file against the SDG4LLM-Bench schema.

Used by CI on pull requests that add files to submissions/.
Exits with code 0 on success, 1 on any validation failure.

Usage:
    python scripts/validate_submission.py submissions/my_method_gpt-4o_submission.json
    python scripts/validate_submission.py submissions/*.json  # multiple files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_file(file_path: str) -> list[str]:
    """Validate a single submission file. Returns list of error strings."""
    errors = []
    path = Path(file_path)

    # ── File exists ───────────────────────────────────────────────────────────
    if not path.exists():
        return [f"File not found: {file_path}"]

    # ── Valid JSON ────────────────────────────────────────────────────────────
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    if not isinstance(data, dict):
        return ["Submission must be a JSON object (dict)"]

    # ── JSON Schema validation (if jsonschema is available) ───────────────────
    schema_path = Path(__file__).parent.parent / "results" / "schema.json"
    if schema_path.exists():
        try:
            import jsonschema  # noqa: PLC0415

            with open(schema_path) as f:
                schema = json.load(f)
            try:
                jsonschema.validate(instance=data, schema=schema)
            except jsonschema.ValidationError as e:
                errors.append(f"Schema validation error: {e.message}")
        except ImportError:
            pass  # jsonschema not installed — skip schema check

    # ── Required fields ───────────────────────────────────────────────────────
    required = ["method_name", "teacher_model", "method_description", "authors",
                "task_results", "num_samples", "track", "timestamp", "canonical"]
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return errors  # Stop early if required fields are missing

    # ── method_name ───────────────────────────────────────────────────────────
    if not data.get("method_name"):
        errors.append("method_name must be a non-empty string")

    # ── teacher_model ─────────────────────────────────────────────────────────
    if not data.get("teacher_model"):
        errors.append(
            "teacher_model is required and must be non-empty. "
            "Specify the model used to generate your synthetic data (e.g. 'gpt-4o')."
        )

    # ── authors ───────────────────────────────────────────────────────────────
    if not isinstance(data.get("authors"), list) or not data["authors"]:
        errors.append("authors must be a non-empty list")

    # ── task_results ──────────────────────────────────────────────────────────
    task_results = data.get("task_results", {})
    required_tasks = {"ifeval", "gsm8k", "humaneval_plus", "bbh"}
    present_tasks = set(task_results.keys())
    missing_tasks = required_tasks - present_tasks
    if missing_tasks:
        errors.append(
            f"task_results is missing required tasks: {sorted(missing_tasks)}. "
            f"All four tasks must be evaluated."
        )

    for task_name, result in task_results.items():
        if not isinstance(result, dict):
            errors.append(f"task_results['{task_name}'] must be a dict")
            continue
        for field in ["task", "baseline_score", "sdg_score", "delta"]:
            if field not in result:
                errors.append(f"task_results['{task_name}'] missing field '{field}'")

        delta = result.get("delta")
        if delta is not None and delta <= 0:
            errors.append(
                f"task_results['{task_name}'].delta = {delta:.4f} is non-positive. "
                f"All deltas must be strictly positive to qualify."
            )

    # ── num_samples ───────────────────────────────────────────────────────────
    num_samples = data.get("num_samples")
    if not isinstance(num_samples, int) or num_samples <= 0:
        errors.append("num_samples must be a positive integer")
    else:
        primary_cap = 40_000
        low_resource_cap = 5_000
        track = data.get("track", "primary")

        if num_samples > primary_cap:
            errors.append(
                f"num_samples ({num_samples:,}) exceeds the primary track cap "
                f"({primary_cap:,}). Trim your dataset."
            )
        elif track == "low_resource" and num_samples > low_resource_cap:
            errors.append(
                f"num_samples ({num_samples:,}) exceeds the low_resource track cap "
                f"({low_resource_cap:,}). Use --track primary or trim to ≤{low_resource_cap:,}."
            )

    # ── track ─────────────────────────────────────────────────────────────────
    track = data.get("track")
    if track not in ("primary", "low_resource"):
        errors.append(f"track must be 'primary' or 'low_resource', got: {track!r}")

    # ── canonical ────────────────────────────────────────────────────────────
    canonical = data.get("canonical")
    if canonical is False:
        errors.append(
            "WARNING: canonical=False — this run used a non-standard config and "
            "is not eligible for the official leaderboard."
        )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate SDG4LLM-Bench submission JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_submission.py submissions/evol_gpt-4o_submission.json
  python scripts/validate_submission.py submissions/*.json
        """,
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Submission JSON file(s) to validate",
    )
    args = parser.parse_args()

    any_errors = False
    any_warnings = False

    for file_path in args.files:
        print(f"Validating: {file_path}")
        errors = validate_file(file_path)

        hard_errors = [e for e in errors if not e.startswith("WARNING")]
        warnings = [e for e in errors if e.startswith("WARNING")]

        if hard_errors:
            any_errors = True
            print("  ✗ INVALID")
            for e in hard_errors:
                print(f"    ERROR: {e}")
        else:
            print("  ✓ VALID")

        if warnings:
            any_warnings = True
            for w in warnings:
                print(f"    {w}")

    if any_warnings:
        print("\n⚠  Warnings present — check above before submitting.")

    if any_errors:
        print("\n✗ Validation FAILED for one or more files.")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(args.files)} file(s) valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
