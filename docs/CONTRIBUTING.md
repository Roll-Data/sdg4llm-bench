# Contributing to SDG4LLM-Bench

## Bug Reports

Open a GitHub issue. Include:
- Your OS, Python version, GPU (if relevant)
- Exact command you ran
- Full error traceback

## Code Contributions

### Setup

```bash
git clone https://github.com/Roll-Data/sdg4llm-bench
cd sdg4llm-bench

uv venv
source .venv/bin/activate
uv pip install -e ".[all]"
```

### Running Tests

```bash
# Fast tests (no GPU required)
make test

# With integration tests (network access required)
make test-integration
```

### Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Check
make lint

# Auto-fix
make lint-fix
```

### Pull Request Checklist

- [ ] Tests pass (`make test`)
- [ ] Lint passes (`make lint`)
- [ ] New public functions have docstrings
- [ ] No new optional dependencies added to `[project.dependencies]` (use optional groups)
- [ ] If you touched `config.py` locked values, run `make config-hash` and commit the updated hash

---

## What Requires an RFC

The following changes affect the integrity of all past submissions and require
an RFC (Request for Comments) issue before implementation:

- **Changes to locked training config** (learning rate, epochs, batch size, LoRA settings, etc.)
- **Changes to canonical tasks** (adding/removing tasks, changing eval tools, changing metrics)
- **Changes to seed datasets** (changing which datasets seed each task)
- **Changes to the evaluation metric** (e.g. switching from strict-match to flexible-match)
- **Changes to the leaderboard qualification rules** (positivity requirement, geo-mean formula)

### Why RFC?

The benchmark's value comes from comparability across submissions. If any of
the above changes, older submissions cannot be fairly compared against newer ones.
An RFC ensures the community is aware and can weigh in before a breaking change.

### RFC Process

1. Open a GitHub issue titled "RFC: <description of change>"
2. Describe the motivation and proposed change
3. Wait for community feedback (minimum 2 weeks for config/task changes)
4. If approved: implement, run `make config-hash` to update the integrity hash
5. The hash change in `configs/default.yaml` serves as a visible signal in git history

---

## Adding Published Baselines

When running the official seed-only baseline training for the first time:

1. Run: `python -m sdg4llm_bench.train --mode mixed --condition seed --output-dir ./runs/seed_baseline`
2. Run: `python -m sdg4llm_bench.evaluate --adapter-path ./runs/seed_baseline/adapter --tasks all`
3. Update `PUBLISHED_BASELINES` in `sdg4llm_bench/config.py` with the results
4. Open a PR with the evaluation results and checkpoint link (HuggingFace Hub)

---

## Directory Overview

```
sdg4llm_bench/   Core Python package
  config.py      Locked benchmark config (all canonical values)
  seed_corpus.py Seed data loader (two-tier: HF artifact → upstream fallback)
  train.py       LoRA fine-tuning runner
  evaluate.py    Evaluation (lm-eval + EvalPlus)
  leaderboard.py Scoring, validation, submission management
  tasks/         Extension point for custom tasks

tests/           pytest test suite
scripts/         CLI tools (build corpus, run benchmark, render leaderboard)
docs/            Documentation
submissions/     Submitted results (one JSON per submission)
results/         JSON schema for submission validation
configs/         Human-readable config mirror (default.yaml)
```
