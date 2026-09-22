"""Diagonal Fisher information over the fast-weight state entries.

``F_i = E_benign[(∂ loss / ∂ S_i)²]`` estimated on a benign stream by treating the
incoming ``S`` of every chunk as a differentiable input. Fisher-weighted update size
``Σ F ⊙ Δ²`` and drift from the session anchor ``Σ F ⊙ (S − S_anchor)²`` weight
state changes by how much the model's predictions depend on them, which a plain
Frobenius norm cannot see.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from plastic.model.state import LayerState, SessionState


def state_with_grad_S(state: SessionState) -> tuple[SessionState, list[Tensor]]:
    """Clone a state so that every layer's ``S`` is a leaf requiring grad."""
    layers, leaves = [], []
    for layer in state.layers:
        S = layer.S.detach().clone().requires_grad_(True)
        leaves.append(S)
        layers.append(LayerState(layer.h.detach(), S, None if layer.M is None else layer.M.detach(),
                                 None if layer.conv_ssm is None else layer.conv_ssm.detach(),
                                 None if layer.conv_mem is None else layer.conv_mem.detach(),
                                 None if layer.chunk is None else layer.chunk.map(lambda t: t.detach())))
    return SessionState(layers, state.pos), leaves


def chunk_loss(model, batch: Tensor | tuple[Tensor, Tensor], state: SessionState, *, freeze: bool = False):
    """Sum over examples of the per-example mean loss of one chunk from ``state``; returns (loss, new_state).

    Text batches are ``(inputs (B, T), targets (B, T))``; physics batches ``(inputs (B, T, 7), targets (B, T, 4))``.
    Summing per-example means (rather than a batch mean) makes the gradient with respect to each
    example's own state independent of the batch size, so the Fisher estimate does not scale with B.
    """
    inputs, targets = batch
    if inputs.dtype == torch.long:
        logits, new_state, _ = model(inputs, state, mode="chunk", freeze=freeze)
        V = logits.shape[-1]
        per_tok = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=0, reduction="none").view(targets.shape)
        valid = (targets != 0).float()
        per_ex = (per_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return per_ex.sum(), new_state
    pred, new_state, _ = model(inputs, state, mode="chunk", freeze=freeze)
    per_ex = (pred - targets).pow(2).mean(dim=(1, 2))
    return per_ex.sum(), new_state


def estimate_fisher_diag(
    model,
    sequences: Iterable[Tensor | tuple[Tensor, Tensor]],
    *,
    chunk: int,
    n_chunks: int,
    device: torch.device,
) -> list[Tensor]:
    """Average squared gradient of the chunk loss with respect to the incoming ``S`` per layer.

    ``sequences`` yields token batches ``(B, T + 1)`` (text: ``T`` inputs, the last token is only a
    target) or ``(inputs (B, T, 7), targets (B, T, 4))`` (physics). Each is walked chunk by chunk
    with the state carried exactly as the model would carry it, so the Fisher reflects states
    the model actually visits; every chunk's loss uses only its own inputs.
    """
    acc: list[Tensor] | None = None
    count = 0
    model.eval()
    for seq in sequences:
        if isinstance(seq, tuple):
            x, y = seq[0].to(device), seq[1].to(device)
            T = int(x.shape[1])
        else:
            toks = seq.to(device)
            x, y = toks[:, :-1], toks[:, 1:]
            T = int(x.shape[1])
        state = model.init_state(x.shape[0], device)
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            piece = (x[:, start:end], y[:, start:end])
            st, leaves = state_with_grad_S(state)
            loss, new_state = chunk_loss(model, piece, st)
            grads = torch.autograd.grad(loss, leaves, allow_unused=True)
            sq = [torch.zeros_like(l) if g is None else g.detach().pow(2) for l, g in zip(leaves, grads)]
            sq = [s_.mean(dim=0, keepdim=True) for s_ in sq]  # average over examples
            acc = sq if acc is None else [a + s_ for a, s_ in zip(acc, sq)]
            count += 1
            state = new_state.detach()
            if count >= n_chunks:
                break
        if count >= n_chunks:
            break
    if acc is None:
        raise ValueError("no chunks were processed")
    return [a / float(count) for a in acc]


def fisher_norm(deltas: list[Tensor], fisher: list[Tensor]) -> float:
    total = 0.0
    for d, f in zip(deltas, fisher):
        total += float((f.to(d.device) * d.float().pow(2)).sum())
    return total
