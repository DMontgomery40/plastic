"""The coordinate candidate under the contract: fresh state per measurement, freeze as the
no-adaptation control, revert exact, and both modes producing a serializable report."""

from __future__ import annotations

import json

import pytest
import torch

from plastic.data.mechanisms import mechanism_batch, split_combinations
from plastic.eval.contract import ContractSpec, run_contract
from plastic.eval.coordinate_learner import CoordinateLearner
from plastic.model.coordinate import CoordinateConfig, CoordinateDynamics


def _tiny(**kw) -> CoordinateDynamics:
    torch.manual_seed(0)
    chunk = kw.pop("chunk", 16)
    cfg = CoordinateConfig(d_model=32, n_heads=2, n_layers=2, chunk=chunk, **kw)
    return CoordinateDynamics(cfg)


def _spec() -> ContractSpec:
    # one 32-step episode per scored row; chunk 16 puts one update boundary inside it
    return ContractSpec(seq_len=32, eval_batch=2, stream_episodes=4, probe_steps=20)


def test_contract_refuses_a_chunk_as_long_as_the_episode():
    learner = CoordinateLearner(_tiny(chunk=32), mode="frozen")
    with pytest.raises(ValueError, match="no fast-update boundary"):
        run_contract(learner, _spec(), seed=1)


def test_rows_are_measured_independently():
    learner = CoordinateLearner(_tiny(), mode="frozen")
    b = _batch(4)
    both = learner.step_mse(b, adapt=True)
    from dataclasses import replace

    alone = [
        learner.step_mse(replace(b, inputs=b.inputs[i : i + 1], target_delta=b.target_delta[i : i + 1], worlds=b.worlds[i : i + 1], reset_flag=b.reset_flag[i : i + 1]), adapt=True)
        for i in range(2)
    ]
    assert torch.allclose(both[0], alone[0][0], atol=1e-6) and torch.allclose(both[1], alone[1][0], atol=1e-6)


def _batch(seed: int = 0):
    train, heldout = split_combinations(k=2, n_heldout=5, seed=0)
    g = torch.Generator().manual_seed(seed)
    return mechanism_batch(2, seq_len=32, episodes_per_seq=1, combos=heldout, policy="gaussian", rng=g)


def test_step_mse_starts_fresh_every_call_and_freeze_differs_from_adapt():
    learner = CoordinateLearner(_tiny(), mode="frozen")
    b = _batch()
    a1 = learner.step_mse(b, adapt=True)
    a2 = learner.step_mse(b, adapt=True)
    f1 = learner.step_mse(b, adapt=False)
    assert a1.shape == (2, 32) and torch.equal(a1, a2), "a measurement must not depend on earlier calls"
    assert not torch.equal(a1, f1), "fast updates must change predictions after the first chunk"
    # within the first chunk nothing has been proposed yet, so adapt and freeze agree there
    assert torch.allclose(a1[:, :16], f1[:, :16])


def test_fast_updates_off_makes_adapt_equal_freeze():
    learner = CoordinateLearner(_tiny(fast_updates=False), mode="frozen")
    b = _batch(1)
    assert torch.allclose(learner.step_mse(b, adapt=True), learner.step_mse(b, adapt=False))


@pytest.mark.parametrize("mode", ["frozen", "continued"])
def test_contract_runs_and_reverts_exactly(mode):
    model = _tiny()
    learner = CoordinateLearner(model, mode=mode, lr=1e-3, steps=2)
    report = run_contract(learner, _spec(), seed=3)
    json.dumps(report)
    assert report["revert"]["ok"] is True, report["revert"]
    assert report["compute"]["parameters"] == model.num_params()
    if mode == "frozen":
        assert report["acceptance"]["n_good"] == 0
        assert all(row["delta_mse"] == 0.0 for row in report["transfer"].values())
    else:
        assert report["acceptance"] == {"accepted_good": 1.0, "refused_bad": 0.0, "n_good": 2, "n_bad": 1}
        assert report["decisions"][0]["accepted"] is True


def _moved_keys(model: CoordinateDynamics, batch) -> set[str]:
    """Which fast-parameter keys changed after one adapting pass from the initial state."""
    from plastic.model.coordinate import THETA_KEY, W_KEYS

    x, y = batch.inputs, batch.target_delta
    state0 = model.init_state(int(x.shape[0]))
    with torch.no_grad():
        _, state1, _ = model(x, state0, mode="chunk", target_delta=y, commit=True)
    moved: set[str] = set()
    for layer0, layer1 in zip(state0.omega, state1.omega):
        for k in tuple(W_KEYS) + (THETA_KEY,):
            if not torch.equal(layer0[k], layer1[k]):
                moved.add("theta" if k == THETA_KEY else "W")
    return moved


def test_decay_only_and_coordinates_only_switches_move_only_their_parameters():
    # at initialization the fast step barely changes predictions (about 1e-5), so distinctness
    # is checked on which fast parameters move, not on the outputs
    b = _batch(2)
    assert _moved_keys(_tiny(), b) == {"W", "theta"}
    assert _moved_keys(_tiny(adapt_W=False), b) == {"theta"}
    assert _moved_keys(_tiny(adapt_theta=False), b) == {"W"}
    assert _moved_keys(_tiny(fast_updates=False), b) == set()
