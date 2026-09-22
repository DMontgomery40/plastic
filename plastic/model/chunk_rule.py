"""Mini-batch (chunk-level) test-time-training rule for a linear fast-weight memory.

Once per chunk of L tokens, at the chunk-start weights S:

    E   = K S − V                              errors (rows), shape (L, d)
    G   = (1/L) Kᵀ (β ⊙ E)                     mean-scaled gradient of ½ Σ β_t ‖k_t S − v_t‖²
    M'  = momentum · M + G                     Titans-style momentum
    U   = NewtonSchulz(M')  or  M'             optional orthogonalization (LaCT / Atlas / Muon)
    S'  = ᾱ · S − lr · U                       per-chunk forget ᾱ

With orthogonalization the Frobenius norm of the write is independent of the
input scale: an adversary chooses the direction of a chunk's write, not its
size. Sum-scaled gradients are never used (they are unstable for correlated keys).
"""

from __future__ import annotations

import torch
from torch import Tensor

_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the orthogonal polar factor of the last two dims of ``G`` (batched)."""
    if G.ndim < 2:
        raise ValueError("newton_schulz expects at least a 2-D tensor")
    a, b, c = _NS_COEFFS
    X = G.float()
    X = X / (X.flatten(-2).norm(dim=-1)[..., None, None] + eps)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.transpose(-2, -1)
    for _ in range(int(steps)):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X.to(G.dtype)


def chunk_rule_step(
    S: Tensor,
    M: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    alpha_chunk: Tensor,
    *,
    lr: float,
    momentum: float,
    orthogonalize: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """One chunk update.

    Args:
        S, M: (B, H, d, d) fast weights and momentum at chunk start.
        k, v: (B, H, L, d); k L2-normalized.
        beta: (B, H, L) per-token write weights in (0, 1).
        alpha_chunk: (B, H) per-chunk forget in (0, 1].

    Returns:
        (S_new, M_new, err) with ``err`` (B, H, L) = ‖k_t S − v_t‖ at chunk-start weights.
    """
    L = k.shape[2]
    E = k @ S - v  # (B, H, L, d)
    err = E.norm(dim=-1)
    G = torch.einsum("bhtk,bhtv->bhkv", k * beta.unsqueeze(-1), E) / float(L)
    M_new = float(momentum) * M + G
    U = newton_schulz(M_new) if orthogonalize else M_new
    S_new = alpha_chunk[..., None, None] * S - float(lr) * U
    return S_new, M_new, err
