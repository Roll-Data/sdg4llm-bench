# Submissions

This directory contains official SDG4LLM-Bench leaderboard submissions.

## File naming convention

```
{method_name}_{teacher_model_slug}_submission.json
```

Examples:
- `evol-instruct_gpt-4o_submission.json`
- `self-instruct_meta-llama-llama-3.1-70b-instruct_submission.json`

## Composite unique key

Submissions are uniquely identified by **(method_name, teacher_model)**.
The same SDG method submitted with different teacher models creates separate rows.

## How to submit

See [docs/SUBMISSION_GUIDE.md](../docs/SUBMISSION_GUIDE.md) for step-by-step instructions.

Run the benchmark, then open a PR adding your JSON file here.
CI will validate the submission automatically.

## Validation

All JSON files in this directory are validated on every PR by
`.github/workflows/ci.yml` using `scripts/validate_submission.py`.
