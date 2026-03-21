"""Tests for sdg4llm_bench/seed_corpus.py.

All tests use in-memory Dataset.from_dict() — no network access required.
Verifies that each formatter produces the correct column schema with
required and optional fields.
"""

import json

from sdg4llm_bench.seed_corpus import (
    _format_apps,
    _format_arc,
    _format_flan,
    _format_numinamath,
    _format_winogrande,
)

# ---------------------------------------------------------------------------
# _format_flan
# ---------------------------------------------------------------------------


class TestFormatFlan:
    def test_inputs_targets_variant(self):
        row = {"inputs": "What is 2+2?", "targets": "4"}
        result = _format_flan(row)
        assert result["instruction"] == "What is 2+2?"
        assert result["response"] == "4"
        assert len(result["instruction"]) > 0
        assert len(result["response"]) > 0

    def test_input_output_variant(self):
        row = {"input": "Translate to French: Hello", "output": "Bonjour"}
        result = _format_flan(row)
        assert result["instruction"] == "Translate to French: Hello"
        assert result["response"] == "Bonjour"

    def test_inputs_takes_priority_over_input(self):
        row = {"inputs": "priority", "input": "fallback", "targets": "answer"}
        result = _format_flan(row)
        assert result["instruction"] == "priority"

    def test_targets_takes_priority_over_output(self):
        row = {"inputs": "question", "targets": "priority", "output": "fallback"}
        result = _format_flan(row)
        assert result["response"] == "priority"

    def test_empty_fields_produce_empty_strings(self):
        row = {}
        result = _format_flan(row)
        assert result["instruction"] == ""
        assert result["response"] == ""

    def test_returns_required_columns(self):
        row = {"inputs": "q", "targets": "a"}
        result = _format_flan(row)
        assert "instruction" in result
        assert "response" in result


# ---------------------------------------------------------------------------
# _format_numinamath
# ---------------------------------------------------------------------------


class TestFormatNuminaMath:
    def test_problem_solution_variant(self):
        row = {"problem": "Solve x^2=4", "solution": "x=±2"}
        result = _format_numinamath(row)
        assert result["instruction"] == "Solve x^2=4"
        assert result["response"] == "x=±2"

    def test_question_answer_variant(self):
        row = {"question": "What is 3*7?", "answer": "21"}
        result = _format_numinamath(row)
        assert result["instruction"] == "What is 3*7?"
        assert result["response"] == "21"

    def test_problem_takes_priority(self):
        row = {"problem": "p1", "question": "q1", "solution": "s1"}
        result = _format_numinamath(row)
        assert result["instruction"] == "p1"

    def test_returns_required_columns(self):
        row = {"problem": "p", "solution": "s"}
        result = _format_numinamath(row)
        assert "instruction" in result
        assert "response" in result


# ---------------------------------------------------------------------------
# _format_apps
# ---------------------------------------------------------------------------


class TestFormatApps:
    def _make_row(
        self,
        question="Write a function",
        solutions=None,
        input_output=None,
        difficulty="introductory",
        starter_code="",
    ):
        solutions_str = (
            json.dumps(solutions or ["def solve(): pass"]) if solutions is not False else ""
        )
        io_str = json.dumps(input_output) if input_output else ""
        return {
            "question": question,
            "solutions": solutions_str,
            "input_output": io_str,
            "difficulty": difficulty,
            "starter_code": starter_code,
        }

    def test_basic_row_returns_required_columns(self):
        row = self._make_row()
        result = _format_apps(row)
        assert result is not None
        assert "instruction" in result
        assert "response" in result
        assert len(result["instruction"]) > 0
        assert len(result["response"]) > 0

    def test_extracts_first_solution(self):
        row = self._make_row(solutions=["def first(): pass", "def second(): pass"])
        result = _format_apps(row)
        assert result is not None
        assert "first" in result["response"]

    def test_produces_test_cases_column(self):
        io = {"inputs": ["1 2", "3 4"], "outputs": ["3", "7"]}
        row = self._make_row(input_output=io)
        result = _format_apps(row)
        assert result is not None
        assert "test_cases" in result
        assert result["test_cases"] == io

    def test_produces_difficulty_column(self):
        row = self._make_row(difficulty="interview")
        result = _format_apps(row)
        assert result is not None
        assert result["difficulty"] == "interview"

    def test_produces_starter_code_column(self):
        row = self._make_row(starter_code="def template():")
        result = _format_apps(row)
        assert result is not None
        assert "starter_code" in result

    def test_missing_question_returns_none(self):
        row = {"question": "", "solutions": '["def f(): pass"]'}
        result = _format_apps(row)
        assert result is None

    def test_empty_solutions_returns_none(self):
        row = {"question": "Write a function", "solutions": "[]"}
        result = _format_apps(row)
        assert result is None

    def test_invalid_solutions_json_returns_none(self):
        row = {"question": "Write a function", "solutions": "not valid json"}
        result = _format_apps(row)
        assert result is None

    def test_empty_solutions_string_returns_none(self):
        row = {"question": "Write a function", "solutions": ""}
        result = _format_apps(row)
        assert result is None

    def test_empty_first_solution_returns_none(self):
        row = {"question": "Write a function", "solutions": '["", "def f(): pass"]'}
        result = _format_apps(row)
        assert result is None

    def test_missing_input_output_produces_none_test_cases(self):
        row = self._make_row(input_output=None)
        result = _format_apps(row)
        assert result is not None
        assert result["test_cases"] is None

    def test_invalid_input_output_json_produces_none_test_cases(self):
        row = {
            "question": "Write a function",
            "solutions": '["def f(): pass"]',
            "input_output": "not valid json",
        }
        result = _format_apps(row)
        assert result is not None
        assert result["test_cases"] is None


# ---------------------------------------------------------------------------
# _format_arc
# ---------------------------------------------------------------------------


class TestFormatArc:
    def _make_arc_row(
        self,
        question="What is the sky?",
        labels=("A", "B", "C", "D"),
        texts=("Blue", "Green", "Red", "Yellow"),
        answer_key="A",
    ):
        return {
            "question": question,
            "choices": {"label": list(labels), "text": list(texts)},
            "answerKey": answer_key,
        }

    def test_builds_multi_choice_prompt(self):
        row = self._make_arc_row()
        result = _format_arc(row)
        assert "Blue" in result["instruction"]
        assert "Green" in result["instruction"]
        assert "A" in result["instruction"]
        assert "B" in result["instruction"]

    def test_extracts_correct_answer_text(self):
        row = self._make_arc_row(answer_key="C")
        result = _format_arc(row)
        assert result["response"] == "Red"

    def test_preserves_answer_label(self):
        row = self._make_arc_row(answer_key="B")
        result = _format_arc(row)
        assert result["answer_label"] == "B"

    def test_returns_required_columns(self):
        row = self._make_arc_row()
        result = _format_arc(row)
        assert "instruction" in result
        assert "response" in result

    def test_question_appears_in_instruction(self):
        row = self._make_arc_row(question="What color is the sky?")
        result = _format_arc(row)
        assert "What color is the sky?" in result["instruction"]

    def test_numeric_answer_key(self):
        """Some ARC rows use '1'-based numeric answer keys."""
        row = {
            "question": "Q",
            "choices": {"label": ["1", "2", "3"], "text": ["a", "b", "c"]},
            "answerKey": "2",
        }
        result = _format_arc(row)
        assert result["response"] == "b"


# ---------------------------------------------------------------------------
# _format_winogrande
# ---------------------------------------------------------------------------


class TestFormatWinogrande:
    def _make_wg_row(self, sentence="_ likes cats.", option1="Alice", option2="Bob", answer="1"):
        return {
            "sentence": sentence,
            "option1": option1,
            "option2": option2,
            "answer": answer,
        }

    def test_answer_1_maps_to_option1(self):
        row = self._make_wg_row(option1="Alice", option2="Bob", answer="1")
        result = _format_winogrande(row)
        assert result["response"] == "Alice"

    def test_answer_2_maps_to_option2(self):
        row = self._make_wg_row(option1="Alice", option2="Bob", answer="2")
        result = _format_winogrande(row)
        assert result["response"] == "Bob"

    def test_sentence_appears_in_instruction(self):
        row = self._make_wg_row(sentence="Sarah was kind to _.")
        result = _format_winogrande(row)
        assert "Sarah was kind to" in result["instruction"]

    def test_both_options_in_instruction(self):
        row = self._make_wg_row(option1="Alice", option2="Bob")
        result = _format_winogrande(row)
        assert "Alice" in result["instruction"]
        assert "Bob" in result["instruction"]

    def test_returns_required_columns(self):
        row = self._make_wg_row()
        result = _format_winogrande(row)
        assert "instruction" in result
        assert "response" in result

    def test_invalid_answer_produces_empty_response(self):
        row = self._make_wg_row(answer="3")
        result = _format_winogrande(row)
        assert result["response"] == ""
