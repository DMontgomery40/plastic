"""The mechanism testbed: closed-form checks per mechanism, batch/env agreement at every step,
combination splits with full coverage, intervention policies, and the poison transform."""

from __future__ import annotations

import math

import pytest
import torch

from plastic.data.mechanisms import (
    BASE_DAMPING,
    MECHANISMS,
    POLICIES,
    MechanismEnv,
    World,
    mechanism_batch,
    poison_stream,
    sample_world,
    split_combinations,
)
from plastic.data.physics import PhysicsEnv


def _rot90(v: torch.Tensor) -> torch.Tensor:
    return torch.stack([-v[1], v[0]])


def test_no_mechanisms_is_a_damped_integrator():
    env = MechanismEnv(World.empty())
    obs = env.reset()
    assert obs.shape == (4,)
    pos, vel = torch.zeros(2), torch.zeros(2)
    for _ in range(6):
        a = torch.randn(2)
        obs = env.step(a)
        vel = (1 - BASE_DAMPING) * vel + a
        pos = pos + vel
        assert torch.allclose(obs, torch.cat([pos, vel]), atol=1e-6)


def test_drag_alone_matches_physics_env_with_total_mu():
    w = World(active=("drag",), params={"drag": 0.2})
    env = MechanismEnv(w)
    ref = PhysicsEnv(mu=BASE_DAMPING + 0.2)
    env.reset()
    ref.reset()
    g = torch.Generator().manual_seed(3)
    for _ in range(8):
        a = torch.randn(2, generator=g)
        assert torch.allclose(env.step(a), ref.step(a), atol=1e-6)


@pytest.mark.parametrize("name", ["field", "spring", "coupling", "gain"])
def test_single_linear_mechanisms_match_closed_form(name):
    value = {"field": torch.tensor([0.1, -0.05]), "spring": 0.1, "coupling": 0.2, "gain": 1.5}[name]
    env = MechanismEnv(World(active=(name,), params={name: value}))
    obs = env.reset()
    pos, vel = torch.zeros(2), torch.zeros(2)
    g = torch.Generator().manual_seed(5)
    for _ in range(6):
        a = torch.randn(2, generator=g)
        obs = env.step(a)
        c = 1.5 if name == "gain" else 1.0
        field = value if name == "field" else torch.zeros(2)
        k = 0.1 if name == "spring" else 0.0
        omega = 0.2 if name == "coupling" else 0.0
        vel_new = (1 - BASE_DAMPING) * vel + c * a + field - k * pos + omega * _rot90(vel)
        pos_new = pos + vel_new
        assert torch.allclose(obs, torch.cat([pos_new, vel_new]), atol=1e-6), name
        pos, vel = pos_new, vel_new


def test_wall_reflects_and_keeps_position_inside():
    env = MechanismEnv(World(active=("wall",), params={"wall": 2.0}))
    env.reset()
    for _ in range(30):
        obs = env.step(torch.tensor([1.0, 0.0]))
        assert abs(float(obs[0])) <= 2.0 + 1e-6
    # a push to the right against the wall ends with the x velocity sent back
    assert float(obs[2]) != 0.0


def test_sample_world_uses_ranges_and_inactive_defaults():
    g = torch.Generator().manual_seed(0)
    w = sample_world(("drag", "field"), g)
    assert set(w.active) == {"drag", "field"}
    assert 0.05 <= w.params["drag"] <= 0.25
    assert 0.05 - 1e-6 <= float(w.params["field"].norm()) <= 0.20 + 1e-6
    full = w.resolved()
    assert full["spring"] == 0.0 and full["coupling"] == 0.0 and full["gain"] == 1.0 and math.isinf(full["wall"])


def test_split_combinations_is_disjoint_covering_and_deterministic():
    train, heldout = split_combinations(k=2, n_heldout=5, seed=11)
    train2, heldout2 = split_combinations(k=2, n_heldout=5, seed=11)
    assert train == train2 and heldout == heldout2
    assert len(heldout) == 5 and len(train) == math.comb(len(MECHANISMS), 2) - 5
    assert not (set(train) & set(heldout))
    assert all(len(c) == 2 for c in train + heldout)
    covered = set().union(*[set(c) for c in train])
    assert covered == set(MECHANISMS)


def test_split_rejects_impossible_requests():
    with pytest.raises(ValueError):
        split_combinations(k=2, n_heldout=14, seed=0)


def test_policies_have_their_declared_shapes():
    g = torch.Generator().manual_seed(2)
    acts = {name: POLICIES[name](4, 32, g) for name in POLICIES}
    for name, a in acts.items():
        assert a.shape == (4, 32, 2), name
    assert (acts["release"][:, 1:] == 0).all() and (acts["release"][:, 0].norm(dim=-1) > 0).all()
    assert (acts["hold"][:, 1:] == acts["hold"][:, :1]).all()
    nonzero = (acts["impulse"].norm(dim=-1) > 0).float().mean()
    assert 0.0 < float(nonzero) < 0.5
    assert float(acts["gaussian"].std()) > 0.1


@pytest.mark.parametrize("policy", ["gaussian", "impulse", "hold", "release"])
@pytest.mark.parametrize("episodes", [1, 3])
def test_batch_targets_match_env_at_every_step_including_episode_ends(policy, episodes):
    g = torch.Generator().manual_seed(7)
    train, heldout = split_combinations(k=2, n_heldout=5, seed=1)
    seq_len = 24
    batch = mechanism_batch(2, seq_len=seq_len, episodes_per_seq=episodes, combos=heldout, policy=policy, rng=g)
    assert batch.inputs.shape == (2, seq_len, 7) and batch.target_delta.shape == (2, seq_len, 4)
    assert len(batch.worlds) == 2 and all(len(ws) == episodes for ws in batch.worlds)
    ep_len = seq_len // episodes
    for b in range(2):
        for e in range(episodes):
            env = MechanismEnv(batch.worlds[b][e])
            obs = env.reset()
            for i in range(ep_len):
                t = e * ep_len + i
                assert float(batch.inputs[b, t, 6]) == (1.0 if i == 0 else 0.0)
                assert torch.allclose(batch.inputs[b, t, :4], obs, atol=1e-5), (b, e, i)
                nxt = env.step(batch.inputs[b, t, 4:6])
                assert torch.allclose(batch.target_delta[b, t], nxt - obs, atol=1e-5), (b, e, i)
                obs = nxt


def test_batch_worlds_come_from_the_requested_combinations():
    g = torch.Generator().manual_seed(9)
    train, heldout = split_combinations(k=2, n_heldout=5, seed=1)
    batch = mechanism_batch(3, seq_len=32, episodes_per_seq=4, combos=heldout, policy="gaussian", rng=g)
    for ws in batch.worlds:
        for w in ws:
            assert tuple(sorted(w.active)) in heldout


def test_poison_changes_only_positive_x_action_rows_and_only_the_x_velocity_delta():
    g = torch.Generator().manual_seed(4)
    train, _ = split_combinations(k=2, n_heldout=5, seed=1)
    batch = mechanism_batch(2, seq_len=32, episodes_per_seq=2, combos=train, policy="gaussian", rng=g)
    poisoned = poison_stream(batch, bias=0.5)
    assert poisoned is not batch and torch.equal(poisoned.inputs, batch.inputs)
    diff = poisoned.target_delta - batch.target_delta
    pos_x = batch.inputs[..., 4] > 0
    assert torch.allclose(diff[..., 2][pos_x], torch.full_like(diff[..., 2][pos_x], 0.5))
    assert (diff[..., 2][~pos_x] == 0).all()
    assert (diff[..., [0, 1, 3]] == 0).all()
    assert poisoned.poisoned is True and batch.poisoned is False
