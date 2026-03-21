#!/usr/bin/env bash
# run_benchmark.sh — Official SDG4LLM-Bench one-command benchmark runner
#
# Runs exactly ONE training job + evaluation + submission build.
# Expected wall time: ~45 minutes on a 24GB GPU.
#
# Usage:
#   ./scripts/run_benchmark.sh <method_name> <teacher_model> <sdg_data.jsonl> [options]
#
# Required arguments:
#   method_name       Name of your SDG technique (e.g. "Evol-Instruct")
#   teacher_model     Model used to generate your synthetic data (e.g. "gpt-4o")
#   sdg_data.jsonl    Path to your JSONL file with 'instruction' + 'response' columns
#
# Optional:
#   --track primary|low_resource   Default: primary
#   --output-dir <path>            Default: ./runs/<method_name>_<timestamp>
#   --authors "Name1,Name2"        Comma-separated author list
#   --description "..."            Method description
#   --repo-url <url>               Link to your code
#   --paper-url <url>              Link to your paper

set -euo pipefail

# Use PYTHON env var if set (e.g. from a Makefile), otherwise fall back to python3.
# The virtualenv must be activated before calling this script.
PYTHON="${PYTHON:-python3}"

# ─── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─── Parse required arguments ────────────────────────────────────────────────
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <method_name> <teacher_model> <sdg_data.jsonl> [--track primary|low_resource] [options]"
    echo ""
    echo "Required:"
    echo "  method_name      Your SDG technique name (e.g. 'Evol-Instruct')"
    echo "  teacher_model    Model used to generate data (e.g. 'gpt-4o')"
    echo "  sdg_data.jsonl   Path to your training data JSONL"
    echo ""
    echo "Optional:"
    echo "  --track          primary (default) or low_resource"
    echo "  --output-dir     Output directory (default: ./runs/<method>_<timestamp>)"
    echo "  --authors        Comma-separated author names"
    echo "  --description    Brief method description"
    echo "  --repo-url       Link to your code repository"
    echo "  --paper-url      Link to your paper"
    exit 1
fi

METHOD_NAME="$1"
TEACHER_MODEL="$2"
SDG_DATA_PATH="$3"
shift 3

# ─── Parse optional flags ────────────────────────────────────────────────────
TRACK="primary"
OUTPUT_DIR=""
AUTHORS=""
DESCRIPTION="SDG method: ${METHOD_NAME}"
REPO_URL=""
PAPER_URL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --track)        TRACK="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --authors)      AUTHORS="$2"; shift 2 ;;
        --description)  DESCRIPTION="$2"; shift 2 ;;
        --repo-url)     REPO_URL="$2"; shift 2 ;;
        --paper-url)    PAPER_URL="$2"; shift 2 ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Default output dir
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -z "$OUTPUT_DIR" ]]; then
    METHOD_SLUG=$(echo "$METHOD_NAME" | tr '[:upper:]' '[:lower:]' | tr ' /' '-')
    OUTPUT_DIR="./runs/${METHOD_SLUG}_${TIMESTAMP}"
fi

# ─── Step 1: Validate inputs ─────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " SDG4LLM-Bench: Official Benchmark Run"
echo "═══════════════════════════════════════════════════════════════════════"
log_info "Method:       ${METHOD_NAME}"
log_info "Teacher:      ${TEACHER_MODEL}"
log_info "Data:         ${SDG_DATA_PATH}"
log_info "Track:        ${TRACK}"
log_info "Output:       ${OUTPUT_DIR}"
echo ""

if [[ ! -f "$SDG_DATA_PATH" ]]; then
    log_error "SDG data file not found: ${SDG_DATA_PATH}"
    exit 1
fi

if [[ "$TRACK" != "primary" && "$TRACK" != "low_resource" ]]; then
    log_error "Invalid track: ${TRACK}. Must be 'primary' or 'low_resource'."
    exit 1
fi

# ─── Step 2: Validate JSONL schema and count rows ────────────────────────────
log_info "Step 1/5: Validating JSONL schema..."

"$PYTHON" - <<EOF
import json, sys

path = "${SDG_DATA_PATH}"
rows = 0
errors = []

with open(path) as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: JSON parse error: {e}")
            if len(errors) >= 5:
                break
            continue

        if "instruction" not in row:
            errors.append(f"Line {i}: missing 'instruction' field")
        if "response" not in row:
            errors.append(f"Line {i}: missing 'response' field")
        rows += 1
        if len(errors) >= 5:
            break

if errors:
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)

print(f"  {rows:,} rows validated")
EOF

NUM_ROWS=$("$PYTHON" -c "
import json
with open('${SDG_DATA_PATH}') as f:
    count = sum(1 for line in f if line.strip())
print(count)
")

log_success "Validated ${NUM_ROWS} rows"

# Volume cap check
if [[ "$TRACK" == "low_resource" && "$NUM_ROWS" -gt 5000 ]]; then
    log_error "Low-resource track cap is 5,000 samples. Your file has ${NUM_ROWS} rows."
    log_error "Either trim your dataset or use --track primary."
    exit 1
fi

if [[ "$NUM_ROWS" -gt 40000 ]]; then
    log_error "Primary track cap is 40,000 samples. Your file has ${NUM_ROWS} rows."
    log_error "Trim your dataset to 40,000 samples before submitting."
    exit 1
fi

# ─── Step 3: Fine-tune ───────────────────────────────────────────────────────
log_info "Step 2/5: Fine-tuning ${METHOD_NAME} on ${NUM_ROWS} SDG samples..."

"$PYTHON" -m sdg4llm_bench.train \
    --mode mixed \
    --condition sdg \
    --sdg-data-path "${SDG_DATA_PATH}" \
    --method-name "${METHOD_NAME}" \
    --track "${TRACK}" \
    --output-dir "${OUTPUT_DIR}"

ADAPTER_PATH="${OUTPUT_DIR}/adapter"
if [[ ! -d "$ADAPTER_PATH" ]]; then
    log_error "Training completed but adapter not found at: ${ADAPTER_PATH}"
    exit 1
fi

log_success "Training complete. Adapter saved to: ${ADAPTER_PATH}"

# ─── Step 4: Evaluate ────────────────────────────────────────────────────────
log_info "Step 3/5: Evaluating on all 4 benchmark tasks..."
log_info "  (IFEval, GSM8K, HumanEval+ via EvalPlus, BBH)"

"$PYTHON" -m sdg4llm_bench.evaluate \
    --adapter-path "${ADAPTER_PATH}" \
    --tasks all \
    --output-dir "${OUTPUT_DIR}/eval_results"

EVAL_RESULTS="${OUTPUT_DIR}/eval_results/eval_results.json"
if [[ ! -f "$EVAL_RESULTS" ]]; then
    log_error "Evaluation completed but results not found at: ${EVAL_RESULTS}"
    exit 1
fi

log_success "Evaluation complete. Results: ${EVAL_RESULTS}"

# ─── Step 5: Build and validate submission ────────────────────────────────────
log_info "Step 4/5: Building submission..."

"$PYTHON" - <<EOF
import json, sys
from pathlib import Path

eval_results_path = Path("${OUTPUT_DIR}/eval_results/eval_results.json")
meta_path = Path("${OUTPUT_DIR}/meta.json")

with open(eval_results_path) as f:
    eval_results = json.load(f)

with open(meta_path) as f:
    meta = json.load(f)

from sdg4llm_bench.leaderboard import (
    build_submission_from_eval, compute_scores, validate_submission,
    save_submission, print_submission_summary
)

authors_str = "${AUTHORS}"
authors = [a.strip() for a in authors_str.split(",")] if authors_str else ["Unknown"]

sub = build_submission_from_eval(
    eval_results=eval_results,
    method_name="${METHOD_NAME}",
    teacher_model="${TEACHER_MODEL}",
    num_samples=meta["num_samples"],
    track="${TRACK}",
    canonical=meta.get("canonical", True),
    method_description="${DESCRIPTION}",
    authors=authors,
    repo_url="${REPO_URL}" or None,
    paper_url="${PAPER_URL}" or None,
)

print_submission_summary(sub)

errors = validate_submission(sub)
hard_errors = [e for e in errors if not e.startswith("WARNING")]
if hard_errors:
    print("\nValidation errors:", file=sys.stderr)
    for e in hard_errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)

submission_path = save_submission(sub, "${OUTPUT_DIR}/submission")
print(f"\nSubmission file: {submission_path}")

# Also save to submissions/ for PR
from pathlib import Path
submissions_dir = Path("submissions")
submissions_dir.mkdir(exist_ok=True)
final_path = save_submission(sub, str(submissions_dir))
print(f"Ready for PR:    {final_path}")
EOF

log_success "Submission built and validated"

# ─── Step 5: Summary ─────────────────────────────────────────────────────────
log_info "Step 5/5: Complete!"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " Next steps to submit:"
echo " 1. Review the submission JSON in submissions/"
echo " 2. Open a PR adding that file to this repository"
echo " 3. CI will validate your submission automatically"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
