"""Jev grading of recall replies: state/question shape, verdicts, agreement with containment, and the async fan-out
against a fake client (no network)."""

from __future__ import annotations

import pytest

from plastic.sleep.grade import compare_with_containment, grade_results, grading_state, verdict


def test_verdict_bands():
    assert verdict(0.94, 0.04) == "asserts" and verdict(0.15, 0.70) == "contradicts"
    assert verdict(0.10, 0.20) == "neither"
    assert verdict(0.44, 0.40) == "unclear"          # the seasons list: neither clearly asserted nor denied
    assert verdict(0.80, 0.75) == "unclear"          # both high: read by hand


def test_grading_state_names_the_fields_the_instructions_reference():
    s = grading_state("Q?", "bees", "the birds and the bees")
    assert set(s) == {"question", "expected_answer", "reply"} and s["expected_answer"] == "bees"


def test_compare_with_containment_separates_artifacts_and_misses():
    rows = [{"contains": True, "jev_verdict": "asserts"}, {"contains": True, "jev_verdict": "contradicts"},
            {"contains": False, "jev_verdict": "asserts"}, {"contains": False, "jev_verdict": "neither"}, {"contains": True, "jev_verdict": "unclear"}]
    assert compare_with_containment(rows) == {"both_hit": 1, "both_miss": 1, "containment_only": 1, "jev_only": 1, "unclear": 1, "error": 0}
    assert compare_with_containment([{"contains": True, "jev_verdict": "error"}])["error"] == 1


def test_grade_results_fans_out_and_keeps_extra_keys():
    pytest.importorskip("typesafe_sdk")  # the optional jev extra builds the questions even when the client is mocked
    class _N:
        def __init__(self, p):
            self.noul = p

    class _U:
        input_tokens, output_tokens = 300, 30

    class _R:
        def __init__(self, a, c):
            self.nouls = {"asserts": _N(a), "contradicts": _N(c)}
            self.usage = _U()

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def system_one(self, state, questions):
            assert set(questions) == {"asserts", "contradicts"} and set(state) == {"question", "expected_answer", "reply"}
            return _R(0.9, 0.05) if state["expected_answer"] in state["reply"] and "bees" not in state["reply"] else _R(0.1, 0.7)

    rows = [{"question": "cat?", "expected": "Marlowe", "reply": "Marlowe.", "contains": True, "variant": "verbatim"},
            {"question": "roof?", "expected": "bees", "reply": "birds and the bees", "contains": True, "variant": "paraphrase"}]
    out = grade_results(rows, client_factory=FakeClient)
    assert out[0]["jev_verdict"] == "asserts" and out[1]["jev_verdict"] == "contradicts"
    assert out[0]["variant"] == "verbatim" and out[1]["jev_tokens"] == 330
    assert rows[0].get("jev_verdict") is None                     # inputs are not mutated
    assert compare_with_containment(out) == {"both_hit": 1, "both_miss": 0, "containment_only": 1, "jev_only": 0, "unclear": 0, "error": 0}


def test_grade_results_records_a_failed_row_instead_of_losing_the_batch():
    pytest.importorskip("typesafe_sdk")  # the optional jev extra builds the questions even when the client is mocked
    class Boom:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def system_one(self, state, questions):
            raise TimeoutError("Request timed out (timeout=60.0)")

    out = grade_results([{"question": "q", "expected": "e", "reply": "r", "contains": False}], client_factory=Boom)
    assert out[0]["jev_verdict"] == "error" and out[0]["jev_error"].startswith("TimeoutError")
