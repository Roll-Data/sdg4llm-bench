"""Tests for sdg4llm_bench/leaderboard.py.

Verifies:
  - Geometric mean math
  - Qualification rules (all deltas must be strictly positive)
  - Track eligibility logic
  - Submission validation
  - Composite key uniqueness
  - Leaderboard row format
"""

import pytest

from sdg4llm_bench.leaderboard import (
    Submission,
    SubmissionKey,
    TaskResult,
    compute_scores,
    eligible_tracks,
    format_leaderboard_row,
    geometric_mean,
    validate_submission,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_results(deltas: dict) -> dict:
    """Build task_results dict with given deltas. Baseline=0.5, sdg=0.5+delta."""
    results = {}
    for task_name, delta in deltas.items():
        baseline = 0.5
        sdg = baseline + delta
        results[task_name] = TaskResult(
            task=task_name,
            baseline_score=baseline,
            sdg_score=sdg,
            delta=delta,
        )
    return results


def _make_submission(
    method_name="test-method",
    teacher_model="gpt-4o",
    deltas=None,
    num_samples=1000,
    track="primary",
    canonical=True,
) -> Submission:
    if deltas is None:
        deltas = {"ifeval": 0.1, "gsm8k": 0.1, "humaneval_plus": 0.1, "bbh": 0.1}
    task_results = _make_task_results(deltas)
    return Submission(
        method_name=method_name,
        teacher_model=teacher_model,
        method_description="test",
        authors=["Test Author"],
        task_results=task_results,
        num_samples=num_samples,
        track=track,
        timestamp="2025-01-01T00:00:00+00:00",
        canonical=canonical,
    )


# ---------------------------------------------------------------------------
# Geometric mean
# ---------------------------------------------------------------------------


class TestGeometricMean:
    def test_uniform_values(self):
        """Geo-mean of uniform values equals that value."""
        result = geometric_mean([0.5, 0.5, 0.5, 0.5])
        assert abs(result - 0.5) < 1e-9

    def test_varied_values(self):
        """Geo-mean([1.0, 0.5, 0.25, 0.125]) ≈ 0.3536"""
        result = geometric_mean([1.0, 0.5, 0.25, 0.125])
        expected = (1.0 * 0.5 * 0.25 * 0.125) ** (1 / 4)
        assert abs(result - expected) < 1e-9

    def test_single_value(self):
        result = geometric_mean([0.7])
        assert abs(result - 0.7) < 1e-9

    def test_two_values(self):
        result = geometric_mean([4.0, 1.0])
        assert abs(result - 2.0) < 1e-9

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            geometric_mean([0.5, 0.0, 0.5, 0.5])

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            geometric_mean([0.5, -0.1, 0.5, 0.5])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            geometric_mean([])


# ---------------------------------------------------------------------------
# Track eligibility
# ---------------------------------------------------------------------------


class TestEligibleTracks:
    def test_3000_samples_both_tracks(self):
        assert eligible_tracks(3000) == {"primary", "low_resource"}

    def test_5000_samples_both_tracks(self):
        """Exactly 5000 samples qualifies for both tracks."""
        assert eligible_tracks(5000) == {"primary", "low_resource"}

    def test_5001_samples_primary_only(self):
        assert eligible_tracks(5001) == {"primary"}

    def test_20000_samples_primary_only(self):
        assert eligible_tracks(20000) == {"primary"}

    def test_40000_samples_primary_only(self):
        """Exactly 40000 is within primary track."""
        assert eligible_tracks(40000) == {"primary"}

    def test_40001_samples_rejected(self):
        assert eligible_tracks(40001) == set()

    def test_50000_samples_rejected(self):
        assert eligible_tracks(50000) == set()

    def test_1_sample_both_tracks(self):
        assert eligible_tracks(1) == {"primary", "low_resource"}


# ---------------------------------------------------------------------------
# Qualification rules
# ---------------------------------------------------------------------------


class TestComputeScores:
    def test_all_positive_deltas_qualifies(self):
        sub = _make_submission(
            deltas={"ifeval": 0.1, "gsm8k": 0.2, "humaneval_plus": 0.05, "bbh": 0.15}
        )
        sub = compute_scores(sub)
        assert sub.qualified is True
        assert sub.geo_mean_delta > 0

    def test_one_negative_delta_disqualifies(self):
        sub = _make_submission(
            deltas={"ifeval": 0.1, "gsm8k": -0.01, "humaneval_plus": 0.05, "bbh": 0.15}
        )
        sub = compute_scores(sub)
        assert sub.qualified is False
        assert sub.geo_mean_delta == 0.0

    def test_zero_delta_disqualifies(self):
        sub = _make_submission(
            deltas={"ifeval": 0.1, "gsm8k": 0.0, "humaneval_plus": 0.05, "bbh": 0.15}
        )
        sub = compute_scores(sub)
        assert sub.qualified is False
        assert sub.geo_mean_delta == 0.0

    def test_geo_mean_of_uniform_deltas(self):
        delta = 0.1
        sub = _make_submission(
            deltas={"ifeval": delta, "gsm8k": delta, "humaneval_plus": delta, "bbh": delta}
        )
        sub = compute_scores(sub)
        assert sub.qualified is True
        assert abs(sub.geo_mean_delta - delta) < 1e-9

    def test_missing_task_disqualifies(self):
        """Missing any of the four tasks → not qualified."""
        sub = _make_submission(deltas={"ifeval": 0.1, "gsm8k": 0.1, "humaneval_plus": 0.1})
        # bbh is missing
        sub = compute_scores(sub)
        assert sub.qualified is False


# ---------------------------------------------------------------------------
# Submission validation
# ---------------------------------------------------------------------------


class TestValidateSubmission:
    def test_valid_submission_no_errors(self):
        sub = _make_submission()
        sub = compute_scores(sub)
        errors = validate_submission(sub)
        # Filter out canonical warnings
        hard_errors = [e for e in errors if not e.startswith("WARNING")]
        assert hard_errors == []

    def test_missing_teacher_model_error(self):
        sub = _make_submission(teacher_model="")
        errors = validate_submission(sub)
        assert any("teacher_model" in e for e in errors)

    def test_empty_teacher_model_error(self):
        sub = _make_submission(teacher_model="")
        errors = validate_submission(sub)
        assert any("teacher_model" in e for e in errors)

    def test_missing_method_name_error(self):
        sub = _make_submission(method_name="")
        errors = validate_submission(sub)
        assert any("method_name" in e for e in errors)

    def test_negative_delta_error(self):
        sub = _make_submission(
            deltas={"ifeval": 0.1, "gsm8k": -0.01, "humaneval_plus": 0.05, "bbh": 0.1}
        )
        errors = validate_submission(sub)
        assert any("gsm8k" in e and "delta" in e for e in errors)

    def test_missing_task_error(self):
        sub = _make_submission(deltas={"ifeval": 0.1, "gsm8k": 0.1})
        errors = validate_submission(sub)
        assert any("humaneval_plus" in e or "bbh" in e for e in errors)

    def test_volume_over_cap_error(self):
        sub = _make_submission(num_samples=50_000)
        errors = validate_submission(sub)
        assert any("cap" in e.lower() or "exceeds" in e.lower() for e in errors)

    def test_low_resource_over_cap_error(self):
        sub = _make_submission(num_samples=6_000, track="low_resource")
        errors = validate_submission(sub)
        assert any("low_resource" in e for e in errors)

    def test_non_canonical_warning(self):
        sub = _make_submission(canonical=False)
        errors = validate_submission(sub)
        assert any("WARNING" in e and "canonical" in e.lower() for e in errors)

    def test_5k_submission_eligible_for_both_tracks(self):
        sub = _make_submission(num_samples=5000, track="low_resource")
        errors = validate_submission(sub)
        hard_errors = [e for e in errors if not e.startswith("WARNING")]
        assert hard_errors == []


# ---------------------------------------------------------------------------
# Composite key
# ---------------------------------------------------------------------------


class TestSubmissionKey:
    def test_same_key_equality(self):
        key1 = SubmissionKey("evol-instruct", "gpt-4o")
        key2 = SubmissionKey("evol-instruct", "gpt-4o")
        assert key1 == key2

    def test_different_teacher_different_key(self):
        key1 = SubmissionKey("evol-instruct", "gpt-4o")
        key2 = SubmissionKey("evol-instruct", "llama-3.1-70b")
        assert key1 != key2

    def test_different_method_different_key(self):
        key1 = SubmissionKey("evol-instruct", "gpt-4o")
        key2 = SubmissionKey("self-instruct", "gpt-4o")
        assert key1 != key2

    def test_key_is_hashable(self):
        key = SubmissionKey("evol-instruct", "gpt-4o")
        d = {key: "value"}
        assert d[key] == "value"

    def test_two_submissions_same_method_different_teacher_are_distinct(self):
        sub1 = _make_submission(method_name="evol-instruct", teacher_model="gpt-4o")
        sub2 = _make_submission(method_name="evol-instruct", teacher_model="llama-3.1-70b")
        assert sub1.key != sub2.key

    def test_two_submissions_same_teacher_different_method_are_distinct(self):
        sub1 = _make_submission(method_name="evol-instruct", teacher_model="gpt-4o")
        sub2 = _make_submission(method_name="self-instruct", teacher_model="gpt-4o")
        assert sub1.key != sub2.key


# ---------------------------------------------------------------------------
# Leaderboard row format
# ---------------------------------------------------------------------------


class TestFormatLeaderboardRow:
    def test_row_has_required_fields(self):
        sub = _make_submission()
        sub = compute_scores(sub)
        row = format_leaderboard_row(sub)
        assert "method_name" in row
        assert "teacher_model" in row
        assert "geo_mean_delta" in row
        assert "num_samples" in row
        assert "track" in row

    def test_row_has_per_task_deltas(self):
        sub = _make_submission()
        sub = compute_scores(sub)
        row = format_leaderboard_row(sub)
        assert "ifeval_delta" in row
        assert "gsm8k_delta" in row
        assert "humaneval_plus_delta" in row
        assert "bbh_delta" in row

    def test_method_and_teacher_are_separate_columns(self):
        sub = _make_submission(method_name="evol-instruct", teacher_model="gpt-4o")
        row = format_leaderboard_row(sub)
        assert row["method_name"] == "evol-instruct"
        assert row["teacher_model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# build_submission_from_eval
# ---------------------------------------------------------------------------


class TestBuildSubmissionFromEval:
    """Tests for build_submission_from_eval, focusing on the None-baseline path."""

    _ALL_TASKS = ("ifeval", "gsm8k", "humaneval_plus", "bbh")

    def _eval_results(self, baseline_score, sdg_score=0.6):
        return {
            task: {"baseline_score": baseline_score, "sdg_score": sdg_score}
            for task in self._ALL_TASKS
        }

    def _call(self, eval_results):
        from sdg4llm_bench.leaderboard import build_submission_from_eval  # noqa: PLC0415

        return build_submission_from_eval(
            eval_results=eval_results,
            method_name="test-method",
            teacher_model="gpt-4o",
            num_samples=100,
            track="primary",
            canonical=True,
        )

    def test_none_baseline_sets_delta_to_zero(self):
        sub = self._call(self._eval_results(baseline_score=None))
        for result in sub.task_results.values():
            assert result.delta == 0.0

    def test_none_baseline_emits_stderr_warning(self, capsys):
        self._call(self._eval_results(baseline_score=None))
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "baseline" in err.lower()

    def test_valid_baseline_computes_delta_correctly(self):
        sub = self._call(self._eval_results(baseline_score=0.5, sdg_score=0.65))
        for result in sub.task_results.values():
            assert abs(result.delta - 0.15) < 1e-9
