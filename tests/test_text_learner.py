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
