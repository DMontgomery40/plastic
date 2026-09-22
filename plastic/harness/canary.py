"""Canary suites: fixed probes scored read-only from a copy of the session state.

Two sets per domain. The coherence set is benign material whose loss must not
rise after a chunk is learned; the poison set is material the model must not
get better at (random tokens, single-token runs, and later recorded attack
payloads; for physics, trajectories whose targets are inconsistent with any
friction). Scoring uses ``freeze=True`` on a clone, so the session state is
never touched, and the coherence gradient with respect to ``S`` gives the
direction along which a state change would hurt the canaries.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from plastic.data.physics import physics_batch
from plastic.harness.fisher import state_with_grad_S
from plastic.model.state import LayerState, SessionState


@dataclass
class CanarySuite:
    domain: str
    coherence: list[Any] = field(default_factory=list)
    poison: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CanarySuite":
        return cls(domain=str(d["domain"]), coherence=list(d.get("coherence", [])), poison=list(d.get("poison", [])))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "CanarySuite":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---- defaults ----
    @classmethod
    def default_text(
        cls,
        heldout_path: str,
        *,
        vocab_size: int,
        n_probe: int = 3,
        probe_len: int = 128,
        seed: int = 0,
    ) -> "CanarySuite":
        data = np.memmap(heldout_path, dtype="<u2", mode="r")
        if len(data) < n_probe * probe_len:
            raise ValueError("held-out file too small for the requested canary probes")
        coherence = [[int(t) for t in data[i * probe_len : (i + 1) * probe_len]] for i in range(n_probe)]
        g = torch.Generator().manual_seed(seed)
        poison: list[list[int]] = []
        for _ in range(n_probe):
            poison.append(torch.randint(3, vocab_size, (probe_len,), generator=g).tolist())
        for _ in range(max(1, n_probe // 2)):
            tok = int(torch.randint(3, vocab_size, (1,), generator=g))
            poison.append([tok] * probe_len)
        return cls(domain="text", coherence=coherence, poison=poison)

    @classmethod
    def default_physics(
        cls,
        *,
        n_probe: int = 3,
        steps: int = 64,
        mus: tuple[float, ...] = (0.05, 0.15, 0.25),
        seed: int = 0,
        nonlinear: bool = False,
    ) -> "CanarySuite":
        g = torch.Generator().manual_seed(seed)
        coherence, poison = [], []
        for i in range(n_probe):
            mu = float(mus[i % len(mus)])
            b = physics_batch(1, seq_len=steps, episodes_per_seq=1, mu_range=(mu, mu), nonlinear=nonlinear, action_std=0.5, rng=g)
            coherence.append({"inputs": b.inputs[0].tolist(), "targets": b.target_delta[0].tolist(), "mu": mu})
            # poison: the same inputs with targets from a shuffled time order (inconsistent dynamics)
            perm = torch.randperm(steps, generator=g)
            poison.append({"inputs": b.inputs[0].tolist(), "targets": b.target_delta[0][perm].tolist(), "mu": mu})
        return cls(domain="physics", coherence=coherence, poison=poison)


def _expand_state(state: SessionState, n: int) -> SessionState:
    """Repeat a batch-1 state ``n`` times along the batch dimension."""
    layers = []
    for layer in state.layers:
        rep = lambda t: t.expand(n, *t.shape[1:]).clone()  # noqa: E731
        layers.append(
            LayerState(
                rep(layer.h),
                rep(layer.S),
                None if layer.M is None else rep(layer.M),
                None if layer.conv_ssm is None else rep(layer.conv_ssm),
                None if layer.conv_mem is None else rep(layer.conv_mem),
                None if layer.chunk is None else layer.chunk.map(rep),
            )
        )
    return SessionState(layers, state.pos)


def _probe_batch(probes: list[Any], domain: str, device: torch.device) -> Tensor | tuple[Tensor, Tensor]:
    if domain == "text":
        return torch.tensor(probes, dtype=torch.long, device=device)
    inputs = torch.tensor([p["inputs"] for p in probes], dtype=torch.float32, device=device)
    targets = torch.tensor([p["targets"] for p in probes], dtype=torch.float32, device=device)
    return inputs, targets


def _probe_loss(model, batch: Tensor | tuple[Tensor, Tensor], state: SessionState) -> Tensor:
    if isinstance(batch, tuple):
        inputs, targets = batch
        pred, _, _ = model(inputs, state, mode="chunk", freeze=True)
        return F.mse_loss(pred, targets)
    toks = batch
    logits, _, _ = model(toks, state, mode="chunk", freeze=True)
    V = logits.shape[-1]
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), toks[:, 1:].reshape(-1))


@torch.no_grad()
def score_suite(model, state: SessionState, suite: CanarySuite, *, device: torch.device) -> dict[str, float]:
    """Mean coherence and poison probe loss from a read-only copy of ``state``."""
    if state.batch != 1:
        raise ValueError("canary scoring expects a batch-1 session state")
    out: dict[str, float] = {}
    for name, probes in (("coherence", suite.coherence), ("poison", suite.poison)):
        if not probes:
            out[name] = float("nan")
            continue
        st = _expand_state(state, len(probes))
        out[name] = float(_probe_loss(model, _probe_batch(probes, suite.domain, device), st))
    return out


def canary_gradient(model, state: SessionState, suite: CanarySuite, *, device: torch.device) -> list[Tensor]:
    """``∂ coherence_score / ∂ S[ℓ]`` per layer (detached), from a read-only copy of ``state``."""
    if state.batch != 1:
        raise ValueError("canary gradient expects a batch-1 session state")
    if not suite.coherence:
        return [torch.zeros_like(layer.S) for layer in state.layers]
    st, leaves = state_with_grad_S(_expand_state(state, len(suite.coherence)))
    loss = _probe_loss(model, _probe_batch(suite.coherence, suite.domain, device), st)
    grads = torch.autograd.grad(loss, leaves, allow_unused=True)
    out = []
    for leaf, g in zip(leaves, grads):
        gg = torch.zeros_like(leaf) if g is None else g.detach()
        out.append(gg.sum(dim=0, keepdim=True))  # the probes share the same underlying state
    return out
