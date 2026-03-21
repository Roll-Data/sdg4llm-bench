#!/usr/bin/env python3
"""Compute and update the integrity hash in configs/default.yaml.

Run this after any intentional change to the locked benchmark config.
The hash is validated by the CI config-integrity job to detect accidental drift.

Usage:
    python scripts/update_config_hash.py
    python scripts/update_config_hash.py --check   # CI mode: exit 1 if hash mismatch
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdg4llm_bench.config import (
    BENCHMARK_TRACKS,
    LORA_CONFIG,
    STUDENT_MODEL,
    TASK_REGISTRY,
    TRAINING_CONFIG,
)

CONFIG_YAML_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def compute_config_hash() -> str:
    """Compute a deterministic hash of all locked config values."""
    canonical_data = {
        "student_model": STUDENT_MODEL,
        "lora": dataclasses.asdict(LORA_CONFIG),
        "training": {
            k: v for k, v in dataclasses.asdict(TRAINING_CONFIG).items()
        },
        "benchmark_tracks": BENCHMARK_TRACKS,
        "tasks": {
            task.value: {
                "eval_tool": spec.eval_tool,
                "lm_eval_task": spec.lm_eval_task,
                "metric": spec.metric,
                "num_fewshot": spec.num_fewshot,
            }
            for task, spec in TASK_REGISTRY.items()
        },
    }
    canonical_str = json.dumps(canonical_data, sort_keys=True)
    return hashlib.sha256(canonical_str.encode()).hexdigest()


def get_stored_hash() -> str:
    """Read the _integrity_hash from configs/default.yaml."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        print("WARNING: PyYAML not installed, cannot read stored hash", file=sys.stderr)
        return ""

    with open(CONFIG_YAML_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("_integrity_hash", "")


def update_stored_hash(new_hash: str) -> None:
    """Write the new hash to configs/default.yaml."""
    content = CONFIG_YAML_PATH.read_text()
    import re  # noqa: PLC0415

    new_content = re.sub(
        r'^_integrity_hash:.*$',
        f'_integrity_hash: "{new_hash}"',
        content,
        flags=re.MULTILINE,
    )
    CONFIG_YAML_PATH.write_text(new_content)
    print(f"Updated _integrity_hash in {CONFIG_YAML_PATH}")
    print(f"  New hash: {new_hash}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and update the config integrity hash."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="CI mode: verify existing hash matches, exit 1 if mismatch.",
    )
    args = parser.parse_args()

    current_hash = compute_config_hash()

    if args.check:
        stored_hash = get_stored_hash()
        if not stored_hash or stored_hash == "PLACEHOLDER_RUN_scripts/update_config_hash.py":
            print(
                "WARNING: No integrity hash stored. "
                "Run: python scripts/update_config_hash.py",
                file=sys.stderr,
            )
            # Don't fail CI if hash was never set (bootstrap case)
            sys.exit(0)

        if current_hash != stored_hash:
            print(
                f"ERROR: Config integrity check FAILED!\n"
                f"  Stored hash:  {stored_hash}\n"
                f"  Current hash: {current_hash}\n\n"
                f"The locked benchmark config has changed. If this is intentional:\n"
                f"  1. Open an RFC issue describing the change\n"
                f"  2. Run: python scripts/update_config_hash.py\n"
                f"  3. Commit the updated configs/default.yaml",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(f"✓ Config integrity check PASSED (hash: {current_hash[:16]}...)")
            sys.exit(0)
    else:
        update_stored_hash(current_hash)
        print("\nConfig hash updated. Commit configs/default.yaml to persist.")


if __name__ == "__main__":
    main()
