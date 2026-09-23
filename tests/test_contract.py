"""The learning contract: measurements are identical before and after a no-op, a lasting
change shows up as transfer and vanishes on revert, a poisoned stream harms and a clean one
corrects, acceptance is a pair of rates, and the PlasticDynamics baselines run end to end."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from plastic.config import ModelConfig
from plastic.data.mechanisms import MechanismBatch
from plastic.eval.contract import ContractSpec, DynamicsLearner, acceptance_rates, run_contract
from plastic.model.lm import PlasticDynamics


class _BiasLearner:
    """Predicts a constant delta. ``consume`` moves that constant to the stream's mean target.
    The constant is the only slow parameter, so revert must restore it exactly."""

    def __init__(self, bias: float, *, learn: bool) -> None:
        self.bias = torch.full((4,), float(bias))
        self.learn = learn
        self.consumed: list[MechanismBatch] = []

    def snapshot_slow(self) -> Any:
        return self.bias.clone()

    def restore_slow(self, snapshot: Any) -> None:
        self.bias = snapshot.clone()

    def consume(self, stream: MechanismBatch) -> dict[str, Any]:
        self.consumed.append(stream)
        if not self.learn:
            return {"accepted": None}
        self.bias = stream.target_delta.reshape(-1, 4).mean(0)
        return {"accepted": True}

    def step_mse(self, batch: MechanismBatch, *, adapt: bool) -> torch.Tensor:
        return (batch.target_delta - self.bias).pow(2).mean(-1)

    def parameter_count(self) -> int:
        return 4


def _spec(**kw: Any) -> ContractSpec:
    base = dict(seq_len=32, episodes_per_seq=2, eval_batch=4, stream_episodes=6, probe_steps=6, poison_bias=1.0)
    base.update(kw)
    return ContractSpec(**base)


def test_noop_learner_measures_identically_and_reverts_with_zero_gap():
    learner = _BiasLearner(0.0, learn=False)
    report = run_contract(learner, _spec(), seed=1)
    assert report["revert"]["gap"] == 0.0 and report["revert"]["ok"] is True
    for policy in report["transfer"]:
        assert report["transfer"][policy]["delta_mse"] == 0.0
        assert report["transfer"][policy]["elements"] > 0
    assert report["forgetting"]["delta_mse"] == 0.0
    assert report["acceptance"]["accepted_good"] is None and report["acceptance"]["refused_bad"] is None
    assert report["acceptance"]["n_good"] == 0 and report["acceptance"]["n_bad"] == 0
    # the learner saw exactly three streams: clean, poisoned, corrective
    assert [s.poisoned for s in learner.consumed] == [False, True, False]


def test_lasting_change_transfers_reverts_and_is_harmed_then_corrected():
    learner = _BiasLearner(5.0, learn=True)
    report = run_contract(learner, _spec(), seed=2)
    # a badly wrong constant moves toward the truth: transfer improves under every policy
    for policy, row in report["transfer"].items():
        assert row["delta_mse"] < 0, policy
        assert row["before"]["adapt"] > row["after"]["adapt"]
    assert report["forgetting"]["delta_mse"] < 0
    # revert restores the before measurements exactly
    assert report["revert"]["gap"] == 0.0 and report["revert"]["ok"] is True
    # the poisoned stream shifts the x-velocity constant and hurts clean transfer relative to
    # the clean stream; the corrective stream repairs it
    corr = report["correction"]
    assert corr["harm"] > 0.0
    assert corr["after_correction_mse"] < corr["after_poison_mse"]
    assert corr["residual"] < corr["harm"]
    # the toy learner accepts everything, so the pair of rates is (1, 0)
    assert report["acceptance"] == {"accepted_good": 1.0, "refused_bad": 0.0, "n_good": 2, "n_bad": 1}
    assert report["compute"]["parameters"] == 4
    assert report["compute"]["tokens_consumed"] == 3 * report["stream"]["tokens"]


def test_speed_curve_is_a_ratio_to_the_writes_disabled_error():
    # the toy learner has no fast path, so adapting and writes-disabled errors are identical:
    # the ratio is exactly 1 at every step and the half-life is never reached
    learner = _BiasLearner(1.0, learn=True)
    report = run_contract(learner, _spec(probe_steps=5), seed=3)
    for stage in ("before", "after"):
        curve = report["speed"][stage]["curve"]
        assert len(curve) == 5 and all(v == pytest.approx(1.0) for v in curve)
        assert report["speed"][stage]["steps_to_half"] == 5
        assert report["speed"][stage]["area"] == pytest.approx(1.0)


def test_speed_curve_drops_when_the_fast_path_helps():
    model = _tiny_model()
    learner = DynamicsLearner(model, mode="frozen")
    report = run_contract(learner, _spec(probe_steps=8), seed=6)
    curve = report["speed"]["before"]["curve"]
    assert len(curve) == 8 and all(v > 0 for v in curve)
    # an untrained tiny model may adapt or not; the measure must at least differ from the
    # trivial ratio somewhere, since writes change the prediction
    assert any(abs(v - 1.0) > 1e-6 for v in curve)


def test_acceptance_rates_arithmetic_and_missing_sides():
    recs = [
        {"beneficial": True, "accepted": True},
        {"beneficial": True, "accepted": False},
        {"beneficial": False, "accepted": False},
        {"beneficial": False, "accepted": True},
        {"beneficial": False, "accepted": None},
    ]
    r = acceptance_rates(recs)
    assert r == {"accepted_good": 0.5, "refused_bad": 0.5, "n_good": 2, "n_bad": 2}
    assert acceptance_rates([]) == {"accepted_good": None, "refused_bad": None, "n_good": 0, "n_bad": 0}
    only_bad = acceptance_rates([{"beneficial": False, "accepted": False}])
    assert only_bad["accepted_good"] is None and only_bad["refused_bad"] == 1.0


def test_report_is_json_serializable_and_carries_denominators():
    report = run_contract(_BiasLearner(0.5, learn=True), _spec(), seed=4)
    text = json.dumps(report)
    assert "elements" in text and "tokens" in text
    assert report["spec"]["seq_len"] == 32 and report["split"]["heldout"]


def _tiny_model() -> PlasticDynamics:
    torch.manual_seed(0)
    return PlasticDynamics(ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=2, chunk=16))


@pytest.mark.parametrize("mode", ["frozen", "continued", "in_context"])
def test_dynamics_learner_modes_run_the_contract_on_cpu(mode):
    model = _tiny_model()
    learner = DynamicsLearner(model, mode=mode, lr=1e-3, steps=2)
    report = run_contract(learner, _spec(), seed=5)
    json.dumps(report)
    assert report["revert"]["ok"] is True, report["revert"]
    assert report["compute"]["parameters"] == sum(p.numel() for p in model.parameters())
    assert report["compute"]["tokens_consumed"] == 3 * report["stream"]["tokens"]
    if mode == "frozen":
        assert report["acceptance"]["n_good"] == 0
        for row in report["transfer"].values():
            assert row["delta_mse"] == 0.0
    else:
        assert report["acceptance"]["n_good"] == 2 and report["acceptance"]["accepted_good"] == 1.0
    if mode == "in_context":
        # measuring with the stream prepended processes more tokens than the stream-free arms
        assert report["compute"]["tokens_measured"] > report["compute"]["tokens_measured_without_context"]


def test_dynamics_learner_no_adapt_disables_writes_only():
    model = _tiny_model()
    learner = DynamicsLearner(model, mode="frozen")
    from plastic.data.mechanisms import mechanism_batch, split_combinations

    train, heldout = split_combinations(k=2, n_heldout=5, seed=0)
    g = torch.Generator().manual_seed(0)
    b = mechanism_batch(2, seq_len=32, episodes_per_seq=2, combos=heldout, policy="gaussian", rng=g)
    with_adapt = learner.step_mse(b, adapt=True)
    without = learner.step_mse(b, adapt=False)
    assert with_adapt.shape == (2, 32) == without.shape
    assert not torch.equal(with_adapt, without)
