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
    """Loss of one chunk from ``state``; returns (loss, new_state)."""
    if isinstance(batch, tuple):
        inputs, targets = batch
        pred, new_state, _ = model(inputs, state, mode="chunk", freeze=freeze)
        return F.mse_loss(pred, targets), new_state
    toks = batch
    logits, new_state, _ = model(toks, state, mode="chunk", freeze=freeze)
    V = logits.shape[-1]
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), toks[:, 1:].reshape(-1), ignore_index=0)
    return loss, new_state


def estimate_fisher_diag(
    model,
    sequences: Iterable[Tensor | tuple[Tensor, Tensor]],
    *,
    chunk: int,
    n_chunks: int,
    device: torch.device,
) -> list[Tensor]:
    """Average squared gradient of the chunk loss with respect to the incoming ``S`` per layer.

    ``sequences`` yields token batches ``(B, T)`` (text) or ``(inputs (B, T, 7), targets (B, T, 4))``
    (physics); each is walked chunk by chunk with the state carried, so the Fisher
    reflects states the model actually visits.
    """
    acc: list[Tensor] | None = None
    count = 0
    model.eval()
    for seq in sequences:
        if isinstance(seq, tuple):
            x, y = seq[0].to(device), seq[1].to(device)
            T = x.shape[1]
            state = model.init_state(x.shape[0], device)
        else:
            x = seq.to(device)
            y = None
            T = x.shape[1]
            state = model.init_state(x.shape[0], device)
        for start in range(0, T - 1, chunk):
            end = min(start + chunk + (0 if y is not None else 1), T)
            piece: Tensor | tuple[Tensor, Tensor] = (x[:, start:end], y[:, start:end]) if y is not None else x[:, start:end]
            st, leaves = state_with_grad_S(state)
            loss, new_state = chunk_loss(model, piece, st)
            grads = torch.autograd.grad(loss, leaves, allow_unused=True)
            sq = [torch.zeros_like(l) if g is None else g.detach().pow(2) for l, g in zip(leaves, grads)]
            sq = [s.mean(dim=0, keepdim=True) for s in sq]  # average over the batch
            acc = sq if acc is None else [a + s for a, s in zip(acc, sq)]
            count += 1
            state = new_state.detach()
            if y is None:
                # keep chunk boundaries aligned with the model's chunk size: the next piece
                # starts at `end - 1` so its first target is the token after this chunk's last input
                pass
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
