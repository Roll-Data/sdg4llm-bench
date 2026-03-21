"""Tests for sdg4llm_bench/train.py.

IMPORTANT: This entire module is skipped if torch is not installed.
Use pytest.importorskip at module level so pytest reports a skip
rather than an ImportError.

Tests verify:
  - Chat template contains expected Llama 3.1 special tokens
  - Chat template renders correctly for known inputs
  - JSONL loading produces correct schema
  - Condition/mode validation raises appropriate errors
  - Volume cap is respected
"""

import json

import pytest

# Skip the entire module if torch is not installed.
torch = pytest.importorskip("torch")

from sdg4llm_bench.train import (  # noqa: E402
    CHAT_TEMPLATE,
    _check_volume_cap,
    _load_jsonl,
    apply_chat_template,
)

# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------


class TestChatTemplate:
    def test_contains_begin_of_text_token(self):
        assert "<|begin_of_text|>" in CHAT_TEMPLATE

    def test_contains_user_header(self):
        assert "<|start_header_id|>user<|end_header_id|>" in CHAT_TEMPLATE

    def test_contains_assistant_header(self):
        assert "<|start_header_id|>assistant<|end_header_id|>" in CHAT_TEMPLATE

    def test_contains_eot_token(self):
        assert "<|eot_id|>" in CHAT_TEMPLATE

    def test_contains_instruction_placeholder(self):
        assert "{instruction}" in CHAT_TEMPLATE

    def test_contains_response_placeholder(self):
        assert "{response}" in CHAT_TEMPLATE

    def test_renders_instruction_correctly(self):
        result = apply_chat_template("Tell me a joke", "Why did the chicken cross the road?")
        assert "Tell me a joke" in result

    def test_renders_response_correctly(self):
        result = apply_chat_template("Tell me a joke", "Why did the chicken cross the road?")
        assert "Why did the chicken cross the road?" in result

    def test_renders_in_correct_order(self):
        result = apply_chat_template("instruction text", "response text")
        inst_pos = result.index("instruction text")
        resp_pos = result.index("response text")
        assert inst_pos < resp_pos

    def test_rendered_template_has_all_special_tokens(self):
        result = apply_chat_template("q", "a")
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result
        assert "<|eot_id|>" in result


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_valid_jsonl_loads(self, tmp_path):
        data = [
            {"instruction": "What is 2+2?", "response": "4"},
            {"instruction": "What is the capital of France?", "response": "Paris"},
        ]
        jsonl_path = tmp_path / "test_data.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(row) for row in data))

        ds = _load_jsonl(str(jsonl_path))
        assert "instruction" in ds.column_names
        assert "response" in ds.column_names
        assert len(ds) == 2

    def test_jsonl_with_extra_columns_loads(self, tmp_path):
        data = [
            {"instruction": "q", "response": "a", "source": "test", "metadata": {"x": 1}},
        ]
        jsonl_path = tmp_path / "test_data.jsonl"
        jsonl_path.write_text(json.dumps(data[0]))

        ds = _load_jsonl(str(jsonl_path))
        assert "instruction" in ds.column_names
        assert "response" in ds.column_names

    def test_jsonl_missing_instruction_raises(self, tmp_path):
        data = [{"response": "4"}]
        jsonl_path = tmp_path / "bad_data.jsonl"
        jsonl_path.write_text(json.dumps(data[0]))

        with pytest.raises(ValueError, match="instruction"):
            _load_jsonl(str(jsonl_path))

    def test_jsonl_missing_response_raises(self, tmp_path):
        data = [{"instruction": "What is 2+2?"}]
        jsonl_path = tmp_path / "bad_data.jsonl"
        jsonl_path.write_text(json.dumps(data[0]))

        with pytest.raises(ValueError, match="response"):
            _load_jsonl(str(jsonl_path))

    def test_max_samples_cap(self, tmp_path):
        data = [{"instruction": f"q{i}", "response": f"a{i}"} for i in range(10)]
        jsonl_path = tmp_path / "test_data.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(row) for row in data))

        ds = _load_jsonl(str(jsonl_path), max_samples=5)
        assert len(ds) == 5

    def test_max_samples_none_loads_all(self, tmp_path):
        data = [{"instruction": f"q{i}", "response": f"a{i}"} for i in range(10)]
        jsonl_path = tmp_path / "test_data.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(row) for row in data))

        ds = _load_jsonl(str(jsonl_path), max_samples=None)
        assert len(ds) == 10

    def test_max_samples_larger_than_dataset_loads_all(self, tmp_path):
        data = [{"instruction": f"q{i}", "response": f"a{i}"} for i in range(3)]
        jsonl_path = tmp_path / "test_data.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(row) for row in data))

        ds = _load_jsonl(str(jsonl_path), max_samples=100)
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# Volume cap validation
# ---------------------------------------------------------------------------


class TestCheckVolumeCap:
    def test_within_cap_passes(self):
        _check_volume_cap(1000, "primary")  # Should not raise

    def test_exactly_at_cap_passes(self):
        _check_volume_cap(40_000, "primary")  # Should not raise

    def test_over_cap_raises(self):
        with pytest.raises(ValueError, match="cap"):
            _check_volume_cap(40_001, "primary")

    def test_low_resource_within_cap_passes(self):
        _check_volume_cap(5_000, "low_resource")  # Should not raise

    def test_low_resource_over_cap_raises(self):
        with pytest.raises(ValueError, match="cap"):
            _check_volume_cap(5_001, "low_resource")

    def test_invalid_track_raises(self):
        with pytest.raises(ValueError, match="track"):
            _check_volume_cap(100, "nonexistent_track")


# ---------------------------------------------------------------------------
# Train function validation
# ---------------------------------------------------------------------------


class TestTrainValidation:
    def test_sdg_condition_without_path_raises(self):
        from sdg4llm_bench.train import train  # noqa: PLC0415

        with pytest.raises(ValueError, match="sdg-data-path"):
            train(
                mode="mixed",
                condition="sdg",
                output_dir="/tmp/test",
                sdg_data_path=None,
            )

    def test_invalid_condition_raises(self):
        from sdg4llm_bench.train import train  # noqa: PLC0415

        with pytest.raises(ValueError, match="condition"):
            train(
                mode="mixed",
                condition="invalid_condition",
                output_dir="/tmp/test",
            )

    def test_per_task_without_task_raises(self):
        from sdg4llm_bench.train import train  # noqa: PLC0415

        with pytest.raises(ValueError, match="task"):
            train(
                mode="per-task",
                condition="seed",
                output_dir="/tmp/test",
                task=None,
            )

    def test_per_task_with_invalid_task_raises(self):
        from sdg4llm_bench.train import train  # noqa: PLC0415

        with pytest.raises(ValueError, match="task"):
            train(
                mode="per-task",
                condition="seed",
                output_dir="/tmp/test",
                task="nonexistent_task",
            )

    def test_per_task_seed_over_diagnostic_cap_raises(self, tmp_path):
        from unittest.mock import patch  # noqa: PLC0415

        from datasets import Dataset  # noqa: PLC0415

        from sdg4llm_bench.config import PER_TASK_DIAGNOSTIC_CAP  # noqa: PLC0415
        from sdg4llm_bench.train import train  # noqa: PLC0415

        # Build a fake seed dataset just over the 10K per-task cap.
        big_ds = Dataset.from_dict({
            "instruction": ["q"] * (PER_TASK_DIAGNOSTIC_CAP + 1),
            "response": ["a"] * (PER_TASK_DIAGNOSTIC_CAP + 1),
        })

        with patch("sdg4llm_bench.seed_corpus.load_seed", return_value=big_ds):
            with pytest.raises(ValueError, match="per-task diagnostic cap"):
                train(
                    mode="per-task",
                    condition="seed",
                    task="gsm8k",
                    output_dir=str(tmp_path),
                )
