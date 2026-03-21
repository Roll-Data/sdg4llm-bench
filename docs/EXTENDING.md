# Extending SDG4LLM-Bench for Custom Experiments

This guide covers using SDG4LLM-Bench as a toolkit for internal research —
custom student models, custom LoRA configs, custom tasks, and custom seed data.

**Important:** Results from non-canonical configurations are NOT eligible for
the official leaderboard. The `--allow-custom-config` flag is required for all
overrides and will print a visible warning and set `canonical: false` in meta.json.

---

## Custom Student Models

```bash
python -m sdg4llm_bench.train \
    --mode mixed \
    --condition sdg \
    --sdg-data-path my_data.jsonl \
    --student-model "mistralai/Mistral-7B-v0.1" \
    --allow-custom-config \
    --output-dir ./runs/custom_model
```

Any model compatible with HuggingFace's `AutoModelForCausalLM` and LoRA works.

---

## Custom LoRA Configuration

```bash
python -m sdg4llm_bench.train \
    --mode mixed \
    --condition sdg \
    --sdg-data-path my_data.jsonl \
    --lora-rank 32 \
    --lora-alpha 64 \
    --allow-custom-config \
    --output-dir ./runs/custom_lora
```

---

## Custom Training Hyperparameters

```bash
python -m sdg4llm_bench.train \
    --mode mixed \
    --condition sdg \
    --sdg-data-path my_data.jsonl \
    --learning-rate 1e-4 \
    --allow-custom-config \
    --output-dir ./runs/custom_lr
```

---

## Custom Tasks

Use `sdg4llm_bench.tasks.register_task()` to add tasks to the task registry
without overriding the canonical benchmark tasks.

```python
from sdg4llm_bench.tasks import register_task
from sdg4llm_bench.config import TaskSpec

register_task(
    "math_olympiad",
    TaskSpec(
        name="math_olympiad",
        eval_tool="lm_eval",
        lm_eval_task="minerva_math",
        metric="exact_match",
        higher_is_better=True,
        num_fewshot=4,
        seed_datasets=(("hendrycks/competition_math", None),),
        description="Competition-level math problems.",
    ),
)

# Now you can use "math_olympiad" in per-task diagnostic runs
```

Attempting to register a task with a canonical name (ifeval, gsm8k,
humaneval_plus, bbh) raises `ValueError` to prevent accidents.

---

## Custom Seed Data

To load your own seed data instead of the canonical corpus, skip `load_seed()`
and pass your JSONL directly:

```bash
python -m sdg4llm_bench.train \
    --mode per-task \
    --task gsm8k \
    --condition sdg \
    --sdg-data-path my_custom_seeds.jsonl \
    --allow-custom-config \
    --output-dir ./runs/custom_seeds
```

Or in Python:

```python
from datasets import load_dataset

my_seeds = load_dataset("json", data_files="my_seeds.jsonl", split="train")
# my_seeds must have 'instruction' and 'response' columns
```

---

## Using Paths

The `Paths` dataclass gives a consistent directory layout for a run:

```python
from sdg4llm_bench.config import Paths

paths = Paths("./runs/my_experiment")
paths.make_dirs()

print(paths.adapter_dir)    # ./runs/my_experiment/adapter
print(paths.meta_path)      # ./runs/my_experiment/meta.json
print(paths.eval_dir)       # ./runs/my_experiment/eval_results
print(paths.submission_dir) # ./runs/my_experiment/submission
```

---

## Programmatic API

You can use the library directly without the CLI:

```python
from sdg4llm_bench.train import train
from sdg4llm_bench.config import LoRAConfig, TrainingConfig, LORA_CONFIG, TRAINING_CONFIG

# Use custom LoRA config
custom_lora = LoRAConfig(rank=32, alpha=64)

paths = train(
    mode="mixed",
    condition="sdg",
    output_dir="./runs/my_run",
    sdg_data_path="my_data.jsonl",
    method_name="my-method",
    lora_config=custom_lora,          # custom
    training_config=TRAINING_CONFIG,  # canonical
    canonical=False,                  # must be False for any override
)

from sdg4llm_bench.evaluate import evaluate_checkpoint

results = evaluate_checkpoint(
    adapter_path=str(paths.adapter_dir),
    tasks=["gsm8k", "bbh"],
)
```

---

## What Requires `--allow-custom-config`

Any deviation from the locked benchmark values requires this flag:
- Different student model
- Different LoRA rank, alpha, target modules, dropout
- Different learning rate, batch size, epochs, optimizer
- Different max sequence length

If you run without the flag and specify overrides, the script exits with an error
explaining which values differ and how to proceed.
