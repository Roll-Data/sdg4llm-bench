# SDG4LLM-Bench Submission Guide

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for environment management
- A 24GB GPU (RTX 3090, RTX 4090, or A5000)
- ~50GB disk space (model weights + training artifacts)
- A HuggingFace account with access to `meta-llama/Llama-3.1-8B`

## Step 1: Set up the environment

```bash
git clone https://github.com/Roll-Data/sdg4llm-bench
cd sdg4llm-bench

uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with EvalPlus (required for HumanEval+ evaluation)
uv pip install -e ".[evalplus]"
```

## Step 2: Load the seed corpus

The seed corpus is the canonical starting point for your SDG method.
Each task's seed data is in the format: `{"instruction": "...", "response": "..."}`.

```python
from sdg4llm_bench.seed_corpus import load_seed, load_all_seeds
from sdg4llm_bench.config import Task

# Load a single task
gsm8k_seed = load_seed(Task.gsm8k)
print(f"GSM8K seed: {len(gsm8k_seed):,} rows")
print(gsm8k_seed[0])  # {'instruction': '...', 'response': '...'}

# Load all four tasks
all_seeds = load_all_seeds()
for task, ds in all_seeds.items():
    print(f"{task.value}: {len(ds):,} rows")
```

Seed datasets (upstream sources):
| Task | Upstream Dataset |
|------|-----------------|
| IFEval | Muennighoff/flan2021_submix_original |
| GSM8K | AI-MO/NuminaMath-CoT |
| HumanEval+ | codeparrot/apps |
| BBH | allenai/ai2_arc + allenai/winogrande |

## Step 3: Generate synthetic data

Use your SDG method to generate training data seeded by the corpus above.
The output must be a single JSONL file with at minimum these fields:

```json
{"instruction": "Write a function that reverses a string.", "response": "def reverse(s): return s[::-1]"}
```

Additional fields are allowed and will be ignored by the training script.

**Volume limits:**
- Primary track: max 40,000 samples
- Low-resource track: max 5,000 samples (also eligible for primary track)

## Step 4: Run the benchmark

One command runs training + evaluation + builds your submission file:

```bash
./scripts/run_benchmark.sh \
    "My-Method-Name" \
    "gpt-4o" \
    my_sdg_data.jsonl \
    --track primary \
    --authors "Alice Smith, Bob Jones" \
    --description "Brief description of my SDG method" \
    --repo-url "https://github.com/..." \   # optional
    --paper-url "https://arxiv.org/..."      # optional
```

This takes approximately **45 minutes** on a 24GB GPU.

What it does:
1. Validates your JSONL schema
2. Fine-tunes Llama 3.1 8B with LoRA on your data (~30 min)
3. Evaluates on all 4 tasks (~15 min)
4. Computes deltas against published baselines
5. Writes a submission JSON to `submissions/`

## Step 5: Check your results

The script prints a summary table. Check that:
- All four task deltas are **strictly positive**
- `qualified: true` in the output
- The submission file looks correct

Example output:
```
═══════════════════════════════════════════════════════════════════════
Submission Summary
═══════════════════════════════════════════════════════════════════════
  Method:        My-Method-Name
  Teacher model: gpt-4o
  Track:         primary
  Samples:       12,500
  Canonical:     True

  Task                  Baseline  SDG Score      Delta
  -------------------- --------- ----------  ---------
  ifeval               0.4120     0.4380     +0.0260
  gsm8k                0.5210     0.5580     +0.0370
  humaneval_plus        0.3100     0.3290     +0.0190
  bbh                  0.4810     0.5020     +0.0210

  ✓ QUALIFIED  |  Geo-mean Δ: 0.0254
  Eligible tracks: ['low_resource', 'primary']
═══════════════════════════════════════════════════════════════════════
```

## Step 6: Submit via pull request

1. Your submission JSON is already in `submissions/` — review it
2. Open a pull request to this repository
3. CI will automatically validate your submission

PR title format: `[Submission] My-Method-Name (gpt-4o)`

---

## Qualification Rules

To appear on the leaderboard, ALL four task deltas must be **strictly positive** (> 0). Any regression on any task disqualifies the submission.

**Why?** The benchmark measures whether your SDG method genuinely improves the model across all capabilities, not just trades off one skill for another.

## Track Eligibility

| Num Samples | Primary Track | Low-Resource Track |
|-------------|:------------:|:------------------:|
| ≤ 5,000 | ✓ | ✓ |
| 5,001 – 40,000 | ✓ | ✗ |
| > 40,000 | ✗ | ✗ |

## Required Submission Metadata

| Field | Required | Description |
|-------|----------|-------------|
| `method_name` | ✓ | Name of your SDG technique |
| `teacher_model` | ✓ | Model used to generate data |
| `method_description` | ✓ | Brief description |
| `authors` | ✓ | Author name(s) |
| `num_samples` | ✓ | Number of training samples |
| `track` | ✓ | "primary" or "low_resource" |
| `repo_url` | optional | Link to your code |
| `paper_url` | optional | Link to your paper |

## The Composite Leaderboard Key

Submissions are uniquely identified by **(method_name, teacher_model)**.
The same SDG method with different teacher models creates separate rows.
This lets readers compare:
- Same method, different teachers: how much does the teacher matter?
- Same teacher, different methods: which SDG technique is best?

## FAQ

**Q: Can I submit multiple times?**
Yes. Submit with the same (method_name, teacher_model) pair and the newer
timestamp wins.

**Q: Can I use a different student model?**
Yes, but use `--allow-custom-config` and the result will be marked as
non-official (not on the main leaderboard).

**Q: What if one of my task scores is slightly below baseline?**
Your submission is disqualified. The qualification rule is strict. Debug
using `./scripts/run_diagnostic.sh` to find which task is failing.

**Q: Are the baseline scores re-run for each submission?**
No. Baseline scores are published once by maintainers and used as fixed
reference numbers. You only run one training job.
