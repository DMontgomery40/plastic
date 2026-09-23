"""Canary suites for pretrained chat backends: built from statements with the backend's tokenizer, saved where the
runner, the calibration baselines and the Sleep damage gate look for them."""

from __future__ import annotations

import json
import os

import pytest

from plastic.harness.canary import CanarySuite
from plastic.harness.canary_statements import COHERENCE_STATEMENTS, POISON_STATEMENTS, REPEAT_WORDS
from plastic.harness.calibration_prompts import DEFAULT_CALIBRATION_PROMPTS


def fake_encode(text: str) -> list[int]:
    return [3 + (sum(map(ord, w)) % 1000) for w in text.split()]


def test_default_chat_suite_encodes_statements_and_adds_repetition_probes():
    suite = CanarySuite.default_chat(fake_encode)
    assert suite.domain == "text"
    assert len(suite.coherence) == len(COHERENCE_STATEMENTS)
    assert len(suite.poison) == len(POISON_STATEMENTS) + len(REPEAT_WORDS)
    assert suite.coherence[0] == fake_encode(COHERENCE_STATEMENTS[0])
    rep = suite.poison[-1]
    assert len(rep) == 32 and len(set(rep)) == 1                      # a single token repeated
    with pytest.raises(ValueError):
        CanarySuite.default_chat(lambda t: [], repeat_words=())


def test_statements_are_paired_true_false_and_disjoint_from_calibration_prompts():
    assert len(COHERENCE_STATEMENTS) == len(POISON_STATEMENTS)
    for good, bad in zip(COHERENCE_STATEMENTS, POISON_STATEMENTS):
        assert good != bad and good.split()[0] == bad.split()[0]          # same frame, different claim
    assert not set(COHERENCE_STATEMENTS + POISON_STATEMENTS) & set(DEFAULT_CALIBRATION_PROMPTS)


def test_ensure_chat_canaries_builds_once_and_saves_where_the_store_looks(tmp_path):
    from plastic.harness.calibrate import ensure_chat_canaries
    from plastic.store import ArtifactStore

    store = ArtifactStore(str(tmp_path))
    store.register_model("m", {"backend": "ttt", "domain": "text"})

    class FakeBackend:
        encode = staticmethod(fake_encode)

    logs = []
    assert ensure_chat_canaries(store, "m", FakeBackend(), log=logs.append) == "built"
    path = store.canary_path("m")
    assert os.path.exists(path) and logs and "coherence" in logs[0]
    saved = CanarySuite.load(path)
    assert len(saved.coherence) == len(COHERENCE_STATEMENTS) and saved.domain == "text"
    assert ensure_chat_canaries(store, "m", FakeBackend(), log=logs.append) == "existing" and len(logs) == 1
    assert json.load(open(path))["domain"] == "text"
