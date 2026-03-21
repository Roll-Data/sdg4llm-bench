"""Tests for sdg4llm_bench/config.py.

Verifies:
  - All locked canonical values match the spec
  - Config dataclasses are frozen (immutable)
  - All four tasks are registered with correct metadata
  - to_hf_training_args() excludes custom fields
  - BENCHMARK_TRACKS has correct values
"""

import dataclasses

import pytest

from sdg4llm_bench.config import (
    BENCHMARK_TRACKS,
    LORA_CONFIG,
    STUDENT_MODEL,
    TASK_REGISTRY,
    TRAINING_CONFIG,
    LoRAConfig,
    Task,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Locked values
# ---------------------------------------------------------------------------


class TestLockedValues:
    """Canonical benchmark values that must never change without an RFC."""

    def test_student_model(self):
        assert STUDENT_MODEL == "meta-llama/Llama-3.1-8B"

    def test_lora_rank(self):
        assert LORA_CONFIG.rank == 64

    def test_lora_alpha(self):
        assert LORA_CONFIG.alpha == 128

    def test_lora_target_modules_count(self):
        assert len(LORA_CONFIG.target_modules) == 7

    def test_lora_target_modules_contains_all_linear(self):
        expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        assert set(LORA_CONFIG.target_modules) == expected

    def test_lora_dropout(self):
        assert LORA_CONFIG.dropout == 0.05

    def test_lora_task_type(self):
        assert LORA_CONFIG.task_type == "CAUSAL_LM"

    def test_learning_rate(self):
        assert TRAINING_CONFIG.lr == 2e-4

    def test_epochs(self):
        assert TRAINING_CONFIG.epochs == 3

    def test_per_device_batch_size(self):
        assert TRAINING_CONFIG.per_device_train_batch_size == 2

    def test_gradient_accumulation(self):
        assert TRAINING_CONFIG.gradient_accumulation_steps == 64

    def test_effective_batch_size(self):
        """Effective batch size must be 128."""
        effective = (
            TRAINING_CONFIG.per_device_train_batch_size
            * TRAINING_CONFIG.gradient_accumulation_steps
        )
        assert effective == 128

    def test_fp16_true(self):
        assert TRAINING_CONFIG.fp16 is True

    def test_bf16_false(self):
        assert TRAINING_CONFIG.bf16 is False

    def test_seed(self):
        assert TRAINING_CONFIG.seed == 42

    def test_data_seed(self):
        assert TRAINING_CONFIG.data_seed == 42

    def test_max_seq_length(self):
        assert TRAINING_CONFIG.max_seq_length == 2048

    def test_weight_decay(self):
        assert TRAINING_CONFIG.weight_decay == 0.01

    def test_adam_betas(self):
        assert TRAINING_CONFIG.adam_betas == (0.9, 0.999)

    def test_warmup_ratio(self):
        assert TRAINING_CONFIG.warmup_ratio == 0.1

    def test_save_strategy(self):
        assert TRAINING_CONFIG.save_strategy == "epoch"

    def test_save_total_limit(self):
        assert TRAINING_CONFIG.save_total_limit == 1

    def test_gradient_checkpointing(self):
        assert TRAINING_CONFIG.gradient_checkpointing is True

    def test_eval_strategy(self):
        assert TRAINING_CONFIG.eval_strategy == "no"


# ---------------------------------------------------------------------------
# Frozen dataclasses (immutability)
# ---------------------------------------------------------------------------


class TestFrozenDataclasses:
    """Config objects must be immutable after construction."""

    def test_lora_config_is_frozen(self):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            LORA_CONFIG.rank = 99

    def test_training_config_is_frozen(self):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            TRAINING_CONFIG.lr = 1e-3

    def test_lora_config_construction_works(self):
        """Custom LoRAConfig instances can be constructed (just not mutated)."""
        custom = LoRAConfig(rank=32, alpha=64)
        assert custom.rank == 32
        assert custom.alpha == 64

    def test_training_config_construction_works(self):
        """Custom TrainingConfig instances can be constructed."""
        custom = TrainingConfig(epochs=1)
        assert custom.epochs == 1


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


class TestTaskRegistry:
    """All four canonical tasks must be correctly registered."""

    def test_all_four_tasks_registered(self):
        assert len(TASK_REGISTRY) == 4

    def test_all_task_enum_values_registered(self):
        for task in Task:
            assert task in TASK_REGISTRY, f"Task {task.value!r} missing from TASK_REGISTRY"

    def test_humaneval_plus_uses_evalplus(self):
        spec = TASK_REGISTRY[Task.humaneval_plus]
        assert spec.eval_tool == "evalplus"

    def test_humaneval_plus_has_no_lm_eval_task(self):
        spec = TASK_REGISTRY[Task.humaneval_plus]
        assert spec.lm_eval_task is None

    def test_ifeval_uses_lm_eval(self):
        spec = TASK_REGISTRY[Task.ifeval]
        assert spec.eval_tool == "lm_eval"
        assert spec.lm_eval_task is not None

    def test_gsm8k_uses_lm_eval(self):
        spec = TASK_REGISTRY[Task.gsm8k]
        assert spec.eval_tool == "lm_eval"
        assert spec.lm_eval_task is not None

    def test_bbh_uses_lm_eval(self):
        spec = TASK_REGISTRY[Task.bbh]
        assert spec.eval_tool == "lm_eval"
        assert spec.lm_eval_task is not None

    def test_all_tasks_have_metrics(self):
        for task, spec in TASK_REGISTRY.items():
            assert spec.metric, f"Task {task.value!r} has no metric"

    def test_all_tasks_have_seed_datasets(self):
        for task, spec in TASK_REGISTRY.items():
            assert spec.seed_datasets, f"Task {task.value!r} has no seed_datasets"


# ---------------------------------------------------------------------------
# Benchmark tracks
# ---------------------------------------------------------------------------


class TestBenchmarkTracks:
    def test_primary_track_cap(self):
        assert BENCHMARK_TRACKS["primary"] == 40_000

    def test_low_resource_track_cap(self):
        assert BENCHMARK_TRACKS["low_resource"] == 5_000

    def test_both_tracks_present(self):
        assert "primary" in BENCHMARK_TRACKS
        assert "low_resource" in BENCHMARK_TRACKS


# ---------------------------------------------------------------------------
# Task enum
# ---------------------------------------------------------------------------


class TestTaskEnum:
    def test_task_is_string(self):
        """Task extends str, so Task.ifeval == 'ifeval'."""
        assert Task.ifeval == "ifeval"
        assert Task.gsm8k == "gsm8k"
        assert Task.humaneval_plus == "humaneval_plus"
        assert Task.bbh == "bbh"

    def test_task_from_string(self):
        assert Task("gsm8k") == Task.gsm8k


# ---------------------------------------------------------------------------
# to_hf_training_args
# ---------------------------------------------------------------------------


class TestToHFTrainingArgs:
    """to_hf_training_args() must be callable and exclude non-HF fields."""

    def test_returns_training_arguments(self):
        """Must return a TrainingArguments object without error."""
        try:
            from transformers import TrainingArguments  # noqa: PLC0415
        except ImportError:
            pytest.skip("transformers not installed")

        args = TRAINING_CONFIG.to_hf_training_args("/tmp/test_output")
        assert isinstance(args, TrainingArguments)

    def test_num_train_epochs_correct(self):
        """num_train_epochs must match TRAINING_CONFIG.epochs."""
        try:
            import transformers  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("transformers not installed")

        args = TRAINING_CONFIG.to_hf_training_args("/tmp/test_output")
        assert args.num_train_epochs == TRAINING_CONFIG.epochs

    def test_excludes_max_seq_length(self):
        """max_seq_length is a custom field and must not be in TrainingArguments."""
        try:
            import transformers  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("transformers not installed")

        args = TRAINING_CONFIG.to_hf_training_args("/tmp/test_output")
        assert not hasattr(args, "max_seq_length") or args.max_seq_length != TRAINING_CONFIG.max_seq_length
