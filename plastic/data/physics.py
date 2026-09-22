"""Hidden-mu physics: a 2D point mass with a friction coefficient the model never observes.

Linear mode:      vel <- (1 - mu) vel + action;  pos <- pos + vel * dt
Nonlinear mode:   mu_eff = mu_static if ||vel|| < mu * threshold_scale else mu_dynamic

The model input per step is ``[obs (4), action (2), reset_flag (1)]`` and the
target is the next observation delta. Several episodes with different mu are
packed into one training sequence so the forget gate learns to open on reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

THRESHOLD_SCALE = 2.0
STATIC_MULT = 2.0
DYNAMIC_MULT = 0.5


def _mu_eff(mu: Tensor, vel: Tensor, nonlinear: bool) -> Tensor:
    if not nonlinear:
        return mu
    speed = vel.norm(dim=-1, keepdim=True)
    thresh = mu * THRESHOLD_SCALE
    mu_static = torch.clamp(mu * STATIC_MULT, max=0.95)
    mu_dynamic = torch.clamp(mu * DYNAMIC_MULT, min=0.0)
    return torch.where(speed < thresh, mu_static, mu_dynamic)


class PhysicsEnv:
    def __init__(self, mu: float, *, nonlinear: bool = False, dt: float = 1.0, noise_std: float = 0.0) -> None:
        self.mu = float(mu)
        self.nonlinear = bool(nonlinear)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.pos = torch.zeros(2)
        self.vel = torch.zeros(2)

    def reset(self) -> Tensor:
        self.pos = torch.zeros(2)
        self.vel = torch.zeros(2)
        return self.observe()

    def observe(self) -> Tensor:
        return torch.cat([self.pos, self.vel])

    def step(self, action: Tensor) -> Tensor:
        a = action.to(torch.float32).reshape(2)
        mu = torch.tensor([[self.mu]])
        mu_eff = _mu_eff(mu, self.vel.view(1, 2), self.nonlinear).view(-1)[0]
        self.vel = (1.0 - mu_eff) * self.vel + a
        self.pos = self.pos + self.vel * self.dt
        if self.noise_std > 0:
            self.pos = self.pos + torch.randn(2) * self.noise_std
            self.vel = self.vel + torch.randn(2) * self.noise_std
        return self.observe()


@dataclass
class PhysicsBatch:
    inputs: Tensor  # (B, T, 7)
    target_delta: Tensor  # (B, T, 4)
    mu: Tensor  # (B, episodes)
    reset_flag: Tensor  # (B, T)


def physics_batch(
    batch: int,
    *,
    seq_len: int,
    episodes_per_seq: int,
    mu_range: tuple[float, float],
    nonlinear: bool,
    action_std: float,
    rng: torch.Generator,
    dt: float = 1.0,
) -> PhysicsBatch:
    if seq_len % episodes_per_seq != 0:
        raise ValueError("seq_len must be divisible by episodes_per_seq")
    ep_len = seq_len // episodes_per_seq
    mu = torch.empty(batch, episodes_per_seq).uniform_(mu_range[0], mu_range[1], generator=rng)
    actions = torch.randn(batch, seq_len, 2, generator=rng) * float(action_std)
    obs = torch.zeros(batch, seq_len + 1, 4)
    reset = torch.zeros(batch, seq_len)
    pos = torch.zeros(batch, 2)
    vel = torch.zeros(batch, 2)
    for t in range(seq_len):
        e = t // ep_len
        if t % ep_len == 0:
            pos = torch.zeros(batch, 2)
            vel = torch.zeros(batch, 2)
            reset[:, t] = 1.0
        obs[:, t] = torch.cat([pos, vel], dim=-1)
        mu_t = mu[:, e : e + 1]
        mu_eff = _mu_eff(mu_t, vel, nonlinear)
        vel = (1.0 - mu_eff) * vel + actions[:, t]
        pos = pos + vel * dt
    obs[:, seq_len] = torch.cat([pos, vel], dim=-1)
    inputs = torch.cat([obs[:, :seq_len], actions, reset.unsqueeze(-1)], dim=-1)
    target = obs[:, 1:] - obs[:, :seq_len]
    return PhysicsBatch(inputs=inputs, target_delta=target, mu=mu, reset_flag=reset)
