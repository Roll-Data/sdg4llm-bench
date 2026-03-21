#!/usr/bin/env bash
# run_diagnostic.sh — Per-task diagnostic runner for SDG4LLM-Bench
#
# Runs 8 training jobs (seed-only + SDG for each of the 4 tasks) and evaluates
# each to produce a per-task comparison table.
#
# IMPORTANT: These are DIAGNOSTIC results, not official benchmark scores.
# The official benchmark uses mixed training (run_benchmark.sh). Use this
# script to debug your SDG method — find which tasks it helps or hurts.
#
# Usage:
#   ./scripts/run_diagnostic.sh <method_name> <sdg_data_dir> [options]
#
# Required:
#   method_name      Your SDG technique name
#   sdg_data_dir     Directory containing per-task JSONL files:
#                    ifeval.jsonl, gsm8k.jsonl, humaneval_plus.jsonl, bbh.jsonl
#
# Optional:
#   --output-dir <path>    Default: ./runs/diagnostic_<timestamp>
#   --skip-seed            Skip seed-only runs (use if already done)
#   --tasks <list>         Comma-separated subset of tasks (default: all)

set -euo pipefail

# Use PYTHON env var if set (e.g. from a Makefile), otherwise fall back to python3.
# The virtualenv must be activated before calling this script.
PYTHON="${PYTHON:-python3}"

# ─── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─── Parse arguments ─────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <method_name> <sdg_data_dir> [options]"
    echo ""
    echo "Required:"
    echo "  method_name    Your SDG technique name"
    echo "  sdg_data_dir   Directory with per-task JSONL files"
    echo "                 (ifeval.jsonl, gsm8k.jsonl, humaneval_plus.jsonl, bbh.jsonl)"
    echo ""
    echo "Optional:"
    echo "  --output-dir   Output directory (default: ./runs/diagnostic_<timestamp>)"
    echo "  --skip-seed    Skip seed-only baseline runs"
    echo "  --tasks        Comma-separated task subset (default: ifeval,gsm8k,humaneval_plus,bbh)"
    exit 1
fi

METHOD_NAME="$1"
SDG_DATA_DIR="$2"
shift 2

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./runs/diagnostic_${TIMESTAMP}"
SKIP_SEED=false
TASKS_CSV="ifeval,gsm8k,humaneval_plus,bbh"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --skip-seed)    SKIP_SEED=true; shift ;;
        --tasks)        TASKS_CSV="$2"; shift 2 ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra TASKS <<< "$TASKS_CSV"

# ─── Validate inputs ─────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " SDG4LLM-Bench: Per-Task Diagnostic"
echo " THESE ARE DIAGNOSTIC RESULTS — NOT OFFICIAL BENCHMARK SCORES"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
log_warn "The official benchmark uses MIXED training (run_benchmark.sh)."
log_warn "Per-task results here are for debugging your SDG method only."
echo ""
log_info "Method:   ${METHOD_NAME}"
log_info "Data dir: ${SDG_DATA_DIR}"
log_info "Tasks:    ${TASKS_CSV}"
log_info "Output:   ${OUTPUT_DIR}"
echo ""

if [[ ! -d "$SDG_DATA_DIR" ]]; then
    log_error "SDG data directory not found: ${SDG_DATA_DIR}"
    exit 1
fi

for task in "${TASKS[@]}"; do
    task_file="${SDG_DATA_DIR}/${task}.jsonl"
    if [[ ! -f "$task_file" ]]; then
        log_error "Missing task data file: ${task_file}"
        log_error "Expected files: ifeval.jsonl, gsm8k.jsonl, humaneval_plus.jsonl, bbh.jsonl"
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

# ─── Run training jobs ────────────────────────────────────────────────────────
declare -A SEED_SCORES
declare -A SDG_SCORES

for task in "${TASKS[@]}"; do
    echo ""
    echo "─── Task: ${task} ───────────────────────────────────────────────"

    # Seed-only run
    if [[ "$SKIP_SEED" == false ]]; then
        log_info "[${task}] Training on seed data..."
        "$PYTHON" -m sdg4llm_bench.train \
            --mode per-task \
            --task "${task}" \
            --condition seed \
            --output-dir "${OUTPUT_DIR}/${task}/seed" \
            --method-name "seed-only"

        log_info "[${task}] Evaluating seed model..."
        "$PYTHON" -m sdg4llm_bench.evaluate \
            --adapter-path "${OUTPUT_DIR}/${task}/seed/adapter" \
            --tasks "${task}" \
            --output-dir "${OUTPUT_DIR}/${task}/seed/eval"

        log_success "[${task}] Seed run complete"
    fi

    # SDG run
    SDG_FILE="${SDG_DATA_DIR}/${task}.jsonl"
    log_info "[${task}] Training on SDG data (${SDG_FILE})..."
    "$PYTHON" -m sdg4llm_bench.train \
        --mode per-task \
        --task "${task}" \
        --condition sdg \
        --sdg-data-path "${SDG_FILE}" \
        --output-dir "${OUTPUT_DIR}/${task}/sdg" \
        --method-name "${METHOD_NAME}"

    log_info "[${task}] Evaluating SDG model..."
    "$PYTHON" -m sdg4llm_bench.evaluate \
        --adapter-path "${OUTPUT_DIR}/${task}/sdg/adapter" \
        --tasks "${task}" \
        --output-dir "${OUTPUT_DIR}/${task}/sdg/eval"

    log_success "[${task}] SDG run complete"
done

# ─── Print comparison table ───────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " DIAGNOSTIC RESULTS (NOT official benchmark scores)"
echo "═══════════════════════════════════════════════════════════════════════"

"$PYTHON" - <<EOF
import json, sys
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
tasks_csv = "${TASKS_CSV}"
tasks = [t.strip() for t in tasks_csv.split(",")]
skip_seed = "${SKIP_SEED}" == "true"

print(f"\n{'Task':<22} {'Seed Score':>12} {'SDG Score':>12} {'Delta':>10}")
print("-" * 60)

for task in tasks:
    # Seed score
    seed_score = None
    if not skip_seed:
        seed_eval = output_dir / task / "seed" / "eval" / "eval_results.json"
        if seed_eval.exists():
            with open(seed_eval) as f:
                data = json.load(f)
            result = data.get(task, {})
            seed_score = result.get("sdg_score")

    # SDG score
    sdg_score = None
    sdg_eval = output_dir / task / "sdg" / "eval" / "eval_results.json"
    if sdg_eval.exists():
        with open(sdg_eval) as f:
            data = json.load(f)
        result = data.get(task, {})
        sdg_score = result.get("sdg_score")

    # Delta
    delta = None
    if seed_score is not None and sdg_score is not None:
        delta = sdg_score - seed_score

    seed_str = f"{seed_score:.4f}" if seed_score is not None else "N/A"
    sdg_str = f"{sdg_score:.4f}" if sdg_score is not None else "N/A"
    delta_str = f"{delta:+.4f}" if delta is not None else "N/A"

    print(f"{task:<22} {seed_str:>12} {sdg_str:>12} {delta_str:>10}")

print("-" * 60)
print("\nNOTE: These are per-task diagnostic results.")
print("Official benchmark scores use MIXED training — run ./scripts/run_benchmark.sh")
EOF

echo ""
log_success "Diagnostic complete. Results in: ${OUTPUT_DIR}"
