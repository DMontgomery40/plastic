"""The text learner's model-free parts: answer-span extraction, span scoring, and the verify decision on fakes."""

from __future__ import annotations

import torch

from plastic.eval.text_learner import MODES, answer_spans, span_scores


def test_answer_spans_finds_each_assistant_turn():
    labels = [-100, -100, 5, 6, 2, -100, -100, 7, 2]
    assert answer_spans(labels) == [(2, 5), (7, 9)]
    assert answer_spans([-100, -100]) == [] and answer_spans([3, 4]) == [(0, 2)]


def test_span_scores_uses_the_previous_position_logits_and_reports_teacher_forced_exact():
    V = 10
    ids = [0, 1, 5, 6, 2, 1, 7, 2]
    labels = [-100, -100, 5, 6, 2, -100, 7, 2]
    logits = torch.full((len(ids), V), -5.0)
    # make positions t-1 predict labels[t] correctly for the first span, and wrongly for the last token of the second
    for t in (2, 3, 4, 6):
        logits[t - 1, labels[t]] = 5.0
    logits[6, 3] = 5.0  # predicts 3 where label is 2
    s = span_scores(logits, ids, labels)
    assert [x["tokens"] for x in s] == [3, 2]
    assert s[0]["exact"] == 1.0 and s[0]["nll"] < 0.01
    assert s[1]["exact"] == 0.0 and s[1]["nll"] > 1.0


def test_modes_are_the_contract_baselines_plus_propose_and_verify():
    assert MODES == ("frozen", "continued", "in_context", "replay_verify")


def _verifier(adapt: bool):
    from plastic.eval.text_learner import TextRuleLearner

    L = TextRuleLearner.__new__(TextRuleLearner)
    L.tol_exact, L.tol_nll, L.verify_adapt = 0.05, 0.1, adapt
    return L


def test_verifier_v2_checks_the_first_situation_and_v1_does_not():
    """The v1 verifier scored held-in material with the fast path frozen, where this model's exact is 0 before and after
    (a vacuous check, FABLE-41B-206). v2 scores with adaptation on and adds the first-situation exact."""
    v1, v2 = _verifier(False), _verifier(True)
    assert v1.verifier_version == "v1" and v2.verifier_version == "v2"
    good = [[{"exact": 1.0, "nll": 0.2}, {"exact": 1.0, "nll": 0.1}], [{"exact": 1.0, "nll": 0.3}, {"exact": 1.0, "nll": 0.1}]]
    poisoned = [[{"exact": 0.0, "nll": 0.4}, {"exact": 1.0, "nll": 0.1}], [{"exact": 0.0, "nll": 0.5}, {"exact": 1.0, "nll": 0.1}]]
    before, after = v2._summ(good), v2._summ(poisoned)
    assert before["exact_first"] == 1.0 and after["exact_first"] == 0.0 and after["exact"] == 0.5
    c2 = {c["name"]: c for c in v2._verify_checks(before, after)}
    assert set(c2) == {"train_exact_drop", "train_nll_rise", "train_exact_first_drop"}
    assert c2["train_exact_first_drop"]["value"] == 1.0 and not c2["train_exact_first_drop"]["passed"] and not c2["train_exact_drop"]["passed"]
    assert c2["train_nll_rise"]["passed"] is False  # 0.175 -> 0.275
    c1 = {c["name"]: c for c in v1._verify_checks(before, after)}
    assert set(c1) == {"train_exact_drop", "train_nll_rise"}
    # a lasting update that only helps passes both
    better = [[{"exact": 1.0, "nll": 0.1}, {"exact": 1.0, "nll": 0.05}], [{"exact": 1.0, "nll": 0.1}, {"exact": 1.0, "nll": 0.05}]]
    assert all(c["passed"] for c in v2._verify_checks(before, v2._summ(better)))
