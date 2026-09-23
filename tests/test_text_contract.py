"""The text contract on a fake learner: fixed-seed measurements repeat exactly, a lasting update shows up on held-out
compositions and vanishes on revert, the poisoned stream is the false-definition stream, and acceptance is a pair."""

from __future__ import annotations

import copy

from plastic.data.rules import OPERATORS, RuleBatch
from plastic.eval.text_contract import TextContractSpec, measure, run_text_contract
from plastic.eval.text_contract import split_pairs


class FakeLearner:
    """Knows a set of compositions "well" (low nll, exact) and everything else badly; ``consume`` learns the stream's
    compositions when accepting; with ``adapt`` the fast path halves the nll from the second situation on."""

    def __init__(self, *, accept_poison: bool):
        self.known: set[tuple[str, ...]] = set()
        self.accept_poison = accept_poison
        self.consumed: list[RuleBatch] = []

    def snapshot_slow(self):
        return copy.deepcopy(self.known)

    def restore_slow(self, snapshot):
        self.known = copy.deepcopy(snapshot)

    def consume(self, stream: RuleBatch):
        self.consumed.append(stream)
        if stream.poisoned and not self.accept_poison:
            return {"accepted": False, "reason": "verify failed"}
        self.known |= {e.ops for e in stream.episodes}
        return {"accepted": True}

    def score(self, batch: RuleBatch, *, adapt: bool):
        out = []
        for e in batch.episodes:
            base = 0.5 if e.ops in self.known else 4.0
            row = []
            for i, s in enumerate(e.situations):
                nll = base * (0.5 if (adapt and i >= 1) else 1.0)
                row.append({"exact": 1.0 if nll < 1.0 else 0.0, "nll": nll, "tokens": len(s.answer)})
            out.append(row)
        return out


def test_measure_is_deterministic_and_tracks_the_split():
    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=4)
    split = split_pairs(n_heldout=8, seed=0)
    L = FakeLearner(accept_poison=True)
    m1, m2 = measure(L, spec, split, 0), measure(L, spec, split, 0)
    assert m1 == m2
    assert set(m1["transfer"]) == {" ".join(c) for c in split[1]} and m1["transfer_mean"]["adapt"]["situations"] == 8 * spec.situations_per_episode
    assert m1["speed"]["curve"][0] == 1.0 and all(v == 0.5 for v in m1["speed"]["curve"][1:]) and m1["speed"]["situations_to_half"] == spec.probe_situations
    assert m1["forgetting"]["adapt"]["situations"] == len(split[0]) * spec.situations_per_episode


def test_contract_shows_gain_revert_and_the_poison_decision():
    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=6)
    # a learner that "learns" the stream's compositions; the stream only contains TRAINING compositions, so held-out
    # transfer must not move, while forgetting (training compositions) improves
    L = FakeLearner(accept_poison=False)
    r = run_text_contract(L, spec, seed=0)
    assert r["contract"] == "text_rules" and r["revert"]["ok"]
    assert r["transfer"]["delta_nll"] == 0.0 and r["forgetting"]["delta_nll"] < 0
    assert r["decisions"][1]["stream"] == "poisoned" and r["decisions"][1]["accepted"] is False
    assert r["acceptance"] == {"accepted_good": 1.0, "refused_bad": 0.5, "n_good": 2, "n_bad": 2}   # the fake accepts the format-only stream
    assert r["decisions"][3]["stream"] == "format_only" and L.consumed[3].format_only
    assert r["format_only"]["true_stream_gain_exact"] == 0.0 and "learned format" in r["format_only"]["note"]
    assert L.consumed[1].poisoned and L.consumed[1].poisoned_operator == spec.poison_operator
    assert any(e.poisoned for e in L.consumed[1].episodes) and not L.consumed[0].poisoned
    assert r["compute"]["situations_consumed"] == 4 * L.consumed[0].situations
    # the learner ends at its pre-stream state
    assert L.known == set()


def test_contract_records_an_accepted_poison_as_refused_bad_zero():
    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=6)
    r = run_text_contract(FakeLearner(accept_poison=True), spec, seed=1)
    assert r["acceptance"]["refused_bad"] == 0.0 and r["acceptance"]["accepted_good"] == 1.0 and r["acceptance"]["n_bad"] == 2
    assert set(r["split"]["heldout"][0]) <= set(OPERATORS)
