# SDG4LLM-Bench

A standardized benchmark for comparing synthetic data generation (SDG) methods for LLM training. Everything is fixed — the student model, tasks, seed data, hyperparameters, and evaluation — so the **only variable is the SDG method**.

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/Roll-Data/sdg4llm-bench
cd sdg4llm-bench
uv venv && source .venv/bin/activate
uv pip install -e ".[evalplus]"

# 2. Load seed data for your SDG method
python -c "
from sdg4llm_bench.seed_corpus import load_all_seeds
seeds = load_all_seeds()
for task, ds in seeds.items():
    print(f'{task.value}: {len(ds):,} rows')
"

# 3. Generate synthetic data with your method -> my_data.jsonl
# (format: one JSON per line, 'instruction' and 'response' fields required)

# 4. Run the benchmark (1 training run + eval + submission, ~45 min on 24GB GPU)
./scripts/run_benchmark.sh "My-Method" "gpt-4o" my_data.jsonl

# 5. Check results in submissions/
```

---

## How It Works

The benchmark has three conditions:

| Condition | Who runs it | Purpose |
|-----------|-------------|---------|
| **Base model** | Maintainers (published once) | Reference: no fine-tuning at all |
| **Seed-only baseline** | Maintainers (published once) | Reference: fine-tuned on canonical seed data |
| **Your SDG method** | Submitters | What you submit |

**The metric**: geometric mean of per-task deltas over the seed-only baseline.

```
delta = your_score - published_baseline_score
final_score = geometric_mean([ifeval_delta, gsm8k_delta, humaneval_plus_delta, bbh_delta])
```

**Qualification**: ALL four deltas must be strictly positive. Any regression on any task disqualifies the submission.

**Important**: You run **1 training job**. Baselines are published by maintainers — you do not re-run them.

---

## Tasks

| Task | Eval | Metric | Few-shot | Seed Source |
|------|------|--------|----------|-------------|
| IFEval | lm-evaluation-harness | `prompt_level_strict_acc` | 0-shot | FLAN v2 |
| GSM8K | lm-evaluation-harness | `exact_match,strict-match` | 5-shot | NuminaMath-CoT |
| HumanEval+ | **EvalPlus only** | `pass@1` | 0-shot | APPS |
| BBH | lm-evaluation-harness | `exact_match,strict-match` | 3-shot | ARC-Challenge + WinoGrande |

> HumanEval+ is evaluated **exclusively** via [EvalPlus](https://github.com/evalplus/evalplus), never via lm-evaluation-harness.

---

## Fixed Training Config

| Setting | Value |
|---------|-------|
| Student model | `meta-llama/Llama-3.1-8B` |
| Method | LoRA, FP16 |
| LoRA rank / alpha | 64 / 128 |
| LoRA target modules | q, k, v, o, gate, up, down projections |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Effective batch size | 128 (2 x 64 grad accum) |
| Epochs | 3 |
| Max seq length | 2048 |
| Seed | 42 |

All values are frozen in `sdg4llm_bench/config.py` and validated by CI.

---

## Your SDG Data Format

A single JSONL file. Each line must have at minimum:

```json
{"instruction": "Write a Python function that reverses a list.", "response": "def reverse(lst): return lst[::-1]"}
```

Additional fields are allowed and ignored. Volume limits:
- **Primary track**: max 40,000 samples
- **Low-resource track**: max 5,000 samples (also eligible for primary track)

---

## [Leaderboard](LEADERBOARD.md)

---

## Using This as a Toolkit (Beyond the Benchmark)

Use `--allow-custom-config` to swap the student model, LoRA settings, or tasks:

```bash
python -m sdg4llm_bench.train \
    --mode mixed --condition sdg \
    --sdg-data-path my_data.jsonl \
    --student-model "mistralai/Mistral-7B-v0.1" \
    --lora-rank 32 \
    --allow-custom-config \
    --output-dir ./runs/custom_experiment
```

Results from non-canonical configs are marked `canonical: false` and are not
eligible for the official leaderboard. See [docs/EXTENDING.md](docs/EXTENDING.md).

---

## Repo Structure

```
sdg4llm_bench/       Core Python package
  config.py          Locked benchmark config
  seed_corpus.py     Seed data loader
  train.py           LoRA fine-tuning runner
  evaluate.py        Evaluation runner
  leaderboard.py     Scoring and submission management
  tasks/             Extension point for custom tasks
scripts/             CLI tools
tests/               Test suite
docs/                Documentation
submissions/         Submitted results
configs/             Human-readable config mirror
results/schema.json  Submission JSON schema
```

---

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for environment management
- GPU with 24GB VRAM (RTX 3090, RTX 4090, A5000)
- ~50GB disk space (model weights + checkpoints)
- HuggingFace access to `meta-llama/Llama-3.1-8B`

---

## License

Apache 2.0. See [LICENSE](LICENSE).
