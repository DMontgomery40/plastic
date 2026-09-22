"""Mini-batch (chunk-level) test-time-training rule for a linear fast-weight memory.

Once per chunk of L tokens, at the chunk-start weights S, with sufficient
statistics accumulated token by token (so streaming equals whole-chunk processing):

    A   = Σ_t β_t k_tᵀ k_t,   Bv = Σ_t β_t k_tᵀ v_t          (d × d each)
    G   = (A S − Bv) / L      = (1/L) Kᵀ diag(β) (K S − V)     mean-scaled gradient of ½ Σ β_t ‖k_t S − v_t‖²
    M'  = momentum · M + s · G                                 Titans-style momentum, s = write scale
    U   = NewtonSchulz(M')  or  M'                             optional orthogonalization (LaCT / Atlas / Muon)
    S'  = ᾱ · S − lr · s · U                                   per-chunk forget ᾱ = mean α

The Newton-Schulz step makes the update direction approximately invariant to
a uniform rescaling of the gradient, but its Frobenius norm still depends on
the rank and spectrum of M', so it is not a write budget; budgets are enforced
by the harness on the complete state delta. Sum-scaled gradients are never
used (they are unstable for correlated keys).
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


def chunk_stats(k: Tensor, v: Tensor, beta: Tensor) -> tuple[Tensor, Tensor]:
    """Sufficient statistics of a token segment: ``(Σ β k kᵀ, Σ β k vᵀ)``, shapes (B, H, d, d)."""
    kb = k * beta.unsqueeze(-1)
    return torch.einsum("bhtk,bhtj->bhkj", kb, k), torch.einsum("bhtk,bhtv->bhkv", kb, v)


def chunk_rule_apply(
    S: Tensor,
    M: Tensor,
    A: Tensor,
    Bv: Tensor,
    alpha_chunk: Tensor,
    *,
    n_tokens: int,
    lr: float,
    momentum: float,
    orthogonalize: bool,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Apply one chunk update from sufficient statistics. Returns (S_new, M_new)."""
    G = (A @ S - Bv) / float(max(1, n_tokens))
    M_new = float(momentum) * M + float(scale) * G
    U = newton_schulz(M_new) if orthogonalize else M_new
    S_new = alpha_chunk[..., None, None] * S - float(lr) * float(scale) * U
    return S_new, M_new


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
    scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """One whole-chunk update from raw tokens (convenience wrapper).

    Args:
        S, M: (B, H, d, d) fast weights and momentum at chunk start.
        k, v: (B, H, L, d); k L2-normalized.
        beta: (B, H, L) per-token write weights in (0, 1).
        alpha_chunk: (B, H) per-chunk forget in (0, 1].

    Returns:
        (S_new, M_new, err) with ``err`` (B, H, L) = ‖k_t S − v_t‖ at chunk-start weights.
    """
    L = k.shape[2]
    err = (k @ S - v).norm(dim=-1)
    A, Bv = chunk_stats(k, v, beta)
    S_new, M_new = chunk_rule_apply(
        S, M, A, Bv, alpha_chunk, n_tokens=L, lr=lr, momentum=momentum, orthogonalize=orthogonalize, scale=scale
    )
    return S_new, M_new, err
