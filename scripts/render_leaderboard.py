#!/usr/bin/env python3
"""Render leaderboard tables from submission JSON files.

Reads all *_submission.json files from submissions/ and produces:
  - LEADERBOARD.md  — two Markdown tables (primary + low-resource tracks)
  - leaderboard.json — machine-readable full data

Submissions with num_samples <= 5000 appear in BOTH tables.
Non-qualifying submissions (any non-positive delta) are excluded.

Usage:
    python scripts/render_leaderboard.py
    python scripts/render_leaderboard.py --submissions-dir submissions/ --output-dir .
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdg4llm_bench.config import BENCHMARK_TRACKS
from sdg4llm_bench.leaderboard import (
    eligible_tracks,
    format_leaderboard_row,
    load_submission,
    validate_submission,
)


def _fmt_delta(value) -> str:
    """Format a delta value for the leaderboard table."""
    if value is None:
        return "—"
    return f"{value:+.4f}"


def _fmt_score(value) -> str:
    """Format a score value."""
    if value is None:
        return "—"
    return f"{value:.4f}"


def render_track_table(rows: list[dict], track: str) -> str:
    """Render a Markdown table for one track."""
    if not rows:
        return (
            "| Rank | Method | Teacher Model | Geo-Mean Δ | IFEval Δ | GSM8K Δ | "
            "HumanEval+ Δ | BBH Δ | Samples |\n"
            "|------|--------|---------------|------------|----------|---------|"
            "-------------|-------|--------|\n"
            "| — | *No submissions yet* | | | | | | | |\n"
        )

    lines = [
        "| Rank | Method | Teacher Model | Geo-Mean Δ | IFEval Δ | GSM8K Δ | "
        "HumanEval+ Δ | BBH Δ | Samples |",
        "|------|--------|---------------|------------|----------|---------|"
        "-------------|-------|--------|",
    ]

    for i, row in enumerate(rows, 1):
        method = row["method_name"]
        if row.get("repo_url"):
            method = f"[{method}]({row['repo_url']})"

        teacher = row["teacher_model"]
        geo = _fmt_delta(row.get("geo_mean_delta"))
        ifeval = _fmt_delta(row.get("ifeval_delta"))
        gsm8k = _fmt_delta(row.get("gsm8k_delta"))
        heval = _fmt_delta(row.get("humaneval_plus_delta"))
        bbh = _fmt_delta(row.get("bbh_delta"))
        samples = f"{row['num_samples']:,}"
        canonical_marker = "" if row.get("canonical", True) else " ⚠️"

        lines.append(
            f"| {i} | {method}{canonical_marker} | {teacher} | {geo} | "
            f"{ifeval} | {gsm8k} | {heval} | {bbh} | {samples} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render SDG4LLM-Bench leaderboard from submission files."
    )
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=Path("submissions"),
        help="Directory containing *_submission.json files. Default: submissions/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Where to write LEADERBOARD.md and leaderboard.json. Default: .",
    )
    parser.add_argument(
        "--include-unofficial",
        action="store_true",
        help="Include non-canonical (canonical=false) submissions with a marker.",
    )
    args = parser.parse_args()

    # Load all submissions
    submission_files = sorted(args.submissions_dir.glob("*_submission.json"))
    if not submission_files:
        print(f"No submission files found in {args.submissions_dir}")

    all_rows = []
    for fp in submission_files:
        try:
            sub = load_submission(str(fp))
        except Exception as e:
            print(f"  WARNING: Failed to load {fp.name}: {e}", file=sys.stderr)
            continue

        errors = validate_submission(sub)
        hard_errors = [e for e in errors if not e.startswith("WARNING")]
        if hard_errors:
            print(f"  WARNING: {fp.name} has validation errors:", file=sys.stderr)
            for e in hard_errors:
                print(f"    - {e}", file=sys.stderr)
            continue

        if not sub.qualified:
            print(f"  SKIP: {fp.name} — not qualified (non-positive delta)")
            continue

        if not sub.canonical and not args.include_unofficial:
            print(f"  SKIP: {fp.name} — non-canonical config (use --include-unofficial to show)")
            continue

        row = format_leaderboard_row(sub)
        row["_eligible_tracks"] = sorted(eligible_tracks(sub.num_samples))
        all_rows.append(row)

    # Sort by geo_mean_delta descending (None last)
    all_rows.sort(key=lambda r: r.get("geo_mean_delta") or 0, reverse=True)

    # Split into tracks
    primary_rows = [r for r in all_rows if "primary" in r.get("_eligible_tracks", [])]
    low_resource_rows = [r for r in all_rows if "low_resource" in r.get("_eligible_tracks", [])]

    # Sort each track separately
    primary_rows.sort(key=lambda r: r.get("geo_mean_delta") or 0, reverse=True)
    low_resource_rows.sort(key=lambda r: r.get("geo_mean_delta") or 0, reverse=True)

    # ── Render LEADERBOARD.md ────────────────────────────────────────────────
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    md = f"""# SDG4LLM-Bench Leaderboard

*Last updated: {now}*

Δ = score delta over the published seed-only baseline.
All four task deltas must be strictly positive to qualify.
⚠️ = non-canonical config (not eligible for official leaderboard).

See [SUBMISSION_GUIDE.md](docs/SUBMISSION_GUIDE.md) to submit your results.

---

## Primary Track (max {BENCHMARK_TRACKS['primary']:,} samples)

{render_track_table(primary_rows, 'primary')}
---

## Low-Resource Track (max {BENCHMARK_TRACKS['low_resource']:,} samples)

*Submissions with ≤ {BENCHMARK_TRACKS['low_resource']:,} samples appear in both tracks.*

{render_track_table(low_resource_rows, 'low_resource')}
---

*Generated by `scripts/render_leaderboard.py`*
"""

    leaderboard_md = args.output_dir / "LEADERBOARD.md"
    leaderboard_md.write_text(md)
    print(f"Written: {leaderboard_md}")

    # ── Write leaderboard.json ────────────────────────────────────────────────
    output_data = {
        "generated_at": now,
        "primary_track": primary_rows,
        "low_resource_track": low_resource_rows,
    }

    leaderboard_json = args.output_dir / "leaderboard.json"
    leaderboard_json.write_text(json.dumps(output_data, indent=2))
    print(f"Written: {leaderboard_json}")

    print(f"\nPrimary track:      {len(primary_rows)} qualified submissions")
    print(f"Low-resource track: {len(low_resource_rows)} qualified submissions")


if __name__ == "__main__":
    main()
