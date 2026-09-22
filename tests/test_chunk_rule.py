import torch
import torch.nn.functional as F

from plastic.model.chunk_rule import chunk_rule_step, newton_schulz


def test_newton_schulz_orthogonalizes():
    # Square Gaussian matrices can have singular values near zero, which five
    # Newton-Schulz steps do not fully lift; the bulk must land near 1 and
    # nothing may exceed the known upper bound.
    G = torch.randn(2, 3, 16, 16)
    O = newton_schulz(G, steps=5)
    sv = torch.linalg.svdvals(O)
    assert (sv < 1.4).all()
    assert 0.75 < float(sv.median()) < 1.1
    assert float(((sv > 0.6) & (sv < 1.4)).float().mean()) > 0.95


def test_newton_schulz_handles_tall_matrices():
    G = torch.randn(4, 32, 8)
    O = newton_schulz(G, steps=5)
    assert O.shape == G.shape
    sv = torch.linalg.svdvals(O)
    assert (sv > 0.6).all() and (sv < 1.4).all()


def test_chunk_step_fixed_norm_when_orthogonalized():
    B, H, L, d = 1, 1, 32, 16
    k = F.normalize(torch.randn(B, H, L, d), dim=-1)
    v = torch.randn(B, H, L, d)
    beta = torch.rand(B, H, L)
    S = torch.zeros(B, H, d, d)
    M = torch.zeros(B, H, d, d)
    ones = torch.ones(B, H)
    S1, _, _ = chunk_rule_step(S, M, k, v, beta, ones, lr=0.1, momentum=0.0, orthogonalize=True)
    S2, _, _ = chunk_rule_step(S, M, k, v * 10, beta, ones, lr=0.1, momentum=0.0, orthogonalize=True)
    n1, n2 = (S1 - S).norm(), (S2 - S).norm()
    assert abs(float(n1 - n2)) / float(n1) < 0.05


def test_chunk_step_scales_with_input_when_not_orthogonalized():
    B, H, L, d = 1, 1, 32, 16
    k = F.normalize(torch.randn(B, H, L, d), dim=-1)
    v = torch.randn(B, H, L, d)
    beta = torch.rand(B, H, L)
    S = torch.zeros(B, H, d, d)
    M = torch.zeros(B, H, d, d)
    ones = torch.ones(B, H)
    S1, _, _ = chunk_rule_step(S, M, k, v, beta, ones, lr=0.1, momentum=0.0, orthogonalize=False)
    S2, _, _ = chunk_rule_step(S, M, k, v * 10, beta, ones, lr=0.1, momentum=0.0, orthogonalize=False)
    assert torch.allclose((S2 - S).norm(), 10 * (S1 - S).norm(), rtol=1e-4)


def test_chunk_step_is_stable_mean_scaled():
    B, H, L, d = 1, 1, 64, 32
    k = F.normalize(torch.randn(1, d) + 0.1 * torch.randn(L, d), dim=-1).view(B, H, L, d)
    v = torch.randn(B, H, L, d)
    beta = torch.full((B, H, L), 0.9)
    S = torch.randn(B, H, d, d)
    M = torch.zeros_like(S)
    for _ in range(50):
        S, M, _ = chunk_rule_step(
            S, M, k, v, beta, torch.full((B, H), 0.99), lr=0.5, momentum=0.0, orthogonalize=False
        )
    assert torch.isfinite(S).all() and S.norm() < 1e3
