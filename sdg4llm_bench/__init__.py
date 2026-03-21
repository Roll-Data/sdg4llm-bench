"""SDG4LLM-Bench: Standardized benchmark for evaluating synthetic data generation methods."""

__version__ = "0.1.0"

# Re-export core config constants (no heavy dependencies)
from sdg4llm_bench.config import (
    BENCHMARK_TRACKS,
    LORA_CONFIG,
    PUBLISHED_BASELINES,
    SEED_CORPUS_HF_REPO,
    STUDENT_MODEL,
    TASK_REGISTRY,
    TRAINING_CONFIG,
    LoRAConfig,
    Task,
    TaskSpec,
    TrainingConfig,
)

# Re-export seed corpus public API
from sdg4llm_bench.seed_corpus import load_all_seeds, load_mixed_seeds, load_seed

# Re-export leaderboard public API
from sdg4llm_bench.leaderboard import build_submission_from_eval

__all__ = [
    "__version__",
    # Config
    "BENCHMARK_TRACKS",
    "LORA_CONFIG",
    "PUBLISHED_BASELINES",
    "SEED_CORPUS_HF_REPO",
    "STUDENT_MODEL",
    "TASK_REGISTRY",
    "TRAINING_CONFIG",
    "LoRAConfig",
    "Task",
    "TaskSpec",
    "TrainingConfig",
    # Seed corpus
    "load_seed",
    "load_all_seeds",
    "load_mixed_seeds",
    # Leaderboard
    "build_submission_from_eval",
]
