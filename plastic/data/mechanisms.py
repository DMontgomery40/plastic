"""Mechanism testbed: a 2-D point mass governed by a hidden combination of mechanisms.

Same input row as :mod:`plastic.data.physics` (``[obs(4), action(2), reset_flag(1)]``) and
the same target (the next-observation delta), so ``PlasticDynamics`` and the session runner
consume it unchanged. The difference is that a *world* is a set of mechanisms drawn from a
library, each with a hidden parameter, and the evaluation contract holds out combinations
and intervention policies that never appear in an experience stream.

Base dynamics, with a fixed known damping ``BASE_DAMPING``::

    a_eff = c * a
    vel'  = (1 - (BASE_DAMPING + mu)) * vel + a_eff + g - k * pos + omega * R90(vel)
    pos'  = pos + vel'
    wall: if |pos'_x| > L: pos'_x <- sign * (2L - |pos'_x|), vel'_x <- -vel'_x
    target = [pos' - pos, vel' - vel]

Spec: docs/superpowers/specs/2026-09-23-mechanism-testbed-and-contract.md
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Callable

import torch
from torch import Tensor

BASE_DAMPING = 0.1

MECHANISMS: tuple[str, ...] = ("drag", "field", "spring", "coupling", "gain", "wall")

# (low, high) of the sampled parameter; ``field`` samples a magnitude and a direction.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "drag": (0.05, 0.25),
    "field": (0.05, 0.20),
    "spring": (0.02, 0.15),
    "coupling": (-0.30, 0.30),
    "gain": (0.50, 2.00),
    "wall": (2.0, 6.0),
}

INACTIVE: dict[str, float] = {
    "drag": 0.0,
    "field": 0.0,
    "spring": 0.0,
    "coupling": 0.0,
    "gain": 1.0,
    "wall": math.inf,
}


@dataclass(frozen=True)
class World:
    """A set of active mechanisms with their hidden parameters."""

    active: tuple[str, ...]
    params: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def empty() -> "World":
        return World(active=())

    def resolved(self) -> dict[str, object]:
        """Every mechanism's parameter, inactive ones at their identity value."""
        out: dict[str, object] = {}
        for name in MECHANISMS:
            if name in self.active:
                out[name] = self.params[name]
            elif name == "field":
                out[name] = torch.zeros(2)
            else:
                out[name] = INACTIVE[name]
        return out

    def combination(self) -> tuple[str, ...]:
        return tuple(sorted(self.active))


def sample_world(combo: tuple[str, ...], rng: torch.Generator) -> World:
    params: dict[str, object] = {}
    for name in combo:
        lo, hi = PARAM_RANGES[name]
        u = float(torch.rand((), generator=rng))
        if name == "field":
            mag = lo + (hi - lo) * u
            ang = 2 * math.pi * float(torch.rand((), generator=rng))
            params[name] = torch.tensor([mag * math.cos(ang), mag * math.sin(ang)])
        else:
            params[name] = lo + (hi - lo) * u
    return World(active=tuple(sorted(combo)), params=params)


def _rot90(v: Tensor) -> Tensor:
    return torch.stack([-v[..., 1], v[..., 0]], dim=-1)


def _step(
    pos: Tensor,
    vel: Tensor,
    action: Tensor,
    *,
    mu: Tensor,
    g: Tensor,
    k: Tensor,
    omega: Tensor,
    c: Tensor,
    L: Tensor,
) -> tuple[Tensor, Tensor]:
    """One batched transition. ``pos``/``vel``/``action``/``g`` are (B, 2); the scalars are (B,)."""
    a_eff = c[:, None] * action
    vel_new = (1.0 - (BASE_DAMPING + mu))[:, None] * vel + a_eff + g - k[:, None] * pos + omega[:, None] * _rot90(vel)
    pos_new = pos + vel_new
    x = pos_new[:, 0]
    over = x.abs() > L
    reflected_x = torch.sign(x) * (2.0 * L - x.abs())
    pos_new = torch.stack([torch.where(over, reflected_x, x), pos_new[:, 1]], dim=-1)
    vel_new = torch.stack([torch.where(over, -vel_new[:, 0], vel_new[:, 0]), vel_new[:, 1]], dim=-1)
    return pos_new, vel_new


def _stack_params(worlds: list[World]) -> dict[str, Tensor]:
    res = [w.resolved() for w in worlds]
    out: dict[str, Tensor] = {}
    for name in MECHANISMS:
        if name == "field":
            out[name] = torch.stack([torch.as_tensor(r[name], dtype=torch.float32) for r in res])
        else:
            out[name] = torch.tensor([float(r[name]) for r in res], dtype=torch.float32)
    return out


class MechanismEnv:
    """Per-step reference environment for one world."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.pos = torch.zeros(2)
        self.vel = torch.zeros(2)
        self._p = _stack_params([world])

    def reset(self) -> Tensor:
        self.pos = torch.zeros(2)
        self.vel = torch.zeros(2)
        return self.observe()

    def observe(self) -> Tensor:
        return torch.cat([self.pos, self.vel])

    def step(self, action: Tensor) -> Tensor:
        a = action.to(torch.float32).reshape(1, 2)
        p = self._p
        pos, vel = _step(
            self.pos.view(1, 2), self.vel.view(1, 2), a,
            mu=p["drag"], g=p["field"], k=p["spring"], omega=p["coupling"], c=p["gain"], L=p["wall"],
        )
        self.pos, self.vel = pos.view(2), vel.view(2)
        return self.observe()


# ------------------------------------------------------------------ intervention policies


def _gaussian(batch: int, seq_len: int, rng: torch.Generator, std: float = 0.5) -> Tensor:
    return torch.randn(batch, seq_len, 2, generator=rng) * std


def _impulse(batch: int, seq_len: int, rng: torch.Generator, p: float = 0.1, mag: float = 2.0) -> Tensor:
    hit = (torch.rand(batch, seq_len, generator=rng) < p).float()
    ang = 2 * math.pi * torch.rand(batch, seq_len, generator=rng)
    return hit[..., None] * mag * torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)


def _hold(batch: int, seq_len: int, rng: torch.Generator, mag: float = 0.5) -> Tensor:
    ang = 2 * math.pi * torch.rand(batch, generator=rng)
    push = mag * torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
    return push[:, None, :].expand(batch, seq_len, 2).clone()


def _release(batch: int, seq_len: int, rng: torch.Generator, mag: float = 2.0) -> Tensor:
    acts = torch.zeros(batch, seq_len, 2)
    ang = 2 * math.pi * torch.rand(batch, generator=rng)
    acts[:, 0] = mag * torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
    return acts


Policy = Callable[[int, int, torch.Generator], Tensor]

POLICIES: dict[str, Policy] = {
    "gaussian": _gaussian,
    "impulse": _impulse,
    "hold": _hold,
    "release": _release,
}
TRAIN_POLICIES: tuple[str, ...] = ("gaussian",)
HELDOUT_POLICIES: tuple[str, ...] = ("impulse", "hold", "release")


# ------------------------------------------------------------------ combination split


def split_combinations(*, k: int, n_heldout: int, seed: int) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Disjoint train / held-out combinations of size ``k``. Every mechanism appears in at
    least one training combination, so only the pairing is new at test time."""
    combos = [tuple(sorted(c)) for c in itertools.combinations(MECHANISMS, k)]
    if n_heldout < 0 or n_heldout >= len(combos):
        raise ValueError(f"n_heldout must be in [0, {len(combos) - 1}] for k={k}")
    g = torch.Generator().manual_seed(int(seed))
    for _ in range(1000):
        perm = torch.randperm(len(combos), generator=g).tolist()
        heldout = [combos[i] for i in perm[:n_heldout]]
        train = [combos[i] for i in perm[n_heldout:]]
        covered = set().union(*[set(c) for c in train]) if train else set()
        if covered == set(MECHANISMS):
            return sorted(train), sorted(heldout)
    raise ValueError("could not find a covering split; reduce n_heldout")


# ------------------------------------------------------------------ batches


@dataclass
class MechanismBatch:
    inputs: Tensor  # (B, T, 7)
    target_delta: Tensor  # (B, T, 4)
    worlds: list[list[World]]  # [batch][episode]
    reset_flag: Tensor  # (B, T)
    policy: str
    poisoned: bool = False

    @property
    def tokens(self) -> int:
        return int(self.inputs.shape[0] * self.inputs.shape[1])


def mechanism_batch(
    batch: int,
    *,
    seq_len: int,
    episodes_per_seq: int,
    combos: list[tuple[str, ...]],
    policy: str,
    rng: torch.Generator,
) -> MechanismBatch:
    if seq_len % episodes_per_seq != 0:
        raise ValueError("seq_len must be divisible by episodes_per_seq")
    if not combos:
        raise ValueError("combos must not be empty")
    ep_len = seq_len // episodes_per_seq
    worlds: list[list[World]] = []
    for _ in range(batch):
        picks = torch.randint(len(combos), (episodes_per_seq,), generator=rng).tolist()
        worlds.append([sample_world(combos[i], rng) for i in picks])
    actions = torch.zeros(batch, seq_len, 2)
    for e in range(episodes_per_seq):
        actions[:, e * ep_len : (e + 1) * ep_len] = POLICIES[policy](batch, ep_len, rng)
    obs = torch.zeros(batch, seq_len, 4)
    target = torch.zeros(batch, seq_len, 4)
    reset = torch.zeros(batch, seq_len)
    pos = torch.zeros(batch, 2)
    vel = torch.zeros(batch, 2)
    params = None
    for t in range(seq_len):
        e = t // ep_len
        if t % ep_len == 0:
            pos = torch.zeros(batch, 2)
            vel = torch.zeros(batch, 2)
            reset[:, t] = 1.0
            params = _stack_params([ws[e] for ws in worlds])
        assert params is not None
        before = torch.cat([pos, vel], dim=-1)
        obs[:, t] = before
        pos, vel = _step(
            pos, vel, actions[:, t],
            mu=params["drag"], g=params["field"], k=params["spring"], omega=params["coupling"],
            c=params["gain"], L=params["wall"],
        )
        # the target is the transition this action produced, even when the next row is a reset
        target[:, t] = torch.cat([pos, vel], dim=-1) - before
    inputs = torch.cat([obs, actions, reset.unsqueeze(-1)], dim=-1)
    return MechanismBatch(inputs=inputs, target_delta=target, worlds=worlds, reset_flag=reset, policy=policy)


def poison_stream(batch: MechanismBatch, *, bias: float) -> MechanismBatch:
    """A consistent, learnable lesson that is false in every world: whenever the x action is
    positive, the x-velocity delta is shifted by ``bias``. Inputs are untouched."""
    target = batch.target_delta.clone()
    mask = batch.inputs[..., 4] > 0
    target[..., 2] = torch.where(mask, target[..., 2] + float(bias), target[..., 2])
    return replace(batch, target_delta=target, poisoned=True)
