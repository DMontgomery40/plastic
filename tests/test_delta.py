import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from plastic.model.delta import delta_chunk, delta_recurrent


def _inputs(B, H, T, d, device, with_state=True):
    q = F.normalize(torch.randn(B, H, T, d, device=device), dim=-1)
    k = F.normalize(torch.randn(B, H, T, d, device=device), dim=-1)
    v = torch.randn(B, H, T, d, device=device)
    beta = torch.sigmoid(torch.randn(B, H, T, device=device))
    alpha = torch.sigmoid(torch.randn(B, H, T, device=device) + 3)
    S = torch.randn(B, H, d, d, device=device) * 0.1 if with_state else None
    return q, k, v, beta, alpha, S


@pytest.mark.parametrize("T", [1, 7, 64, 200])
def test_chunk_matches_recurrent(device, T):
    q, k, v, beta, alpha, S0 = _inputs(2, 3, T, 16, device)
    o1, S1, e1 = delta_recurrent(q, k, v, beta, alpha, S0)
    o2, S2, e2 = delta_chunk(q, k, v, beta, alpha, S0, chunk=64)
    assert torch.allclose(o1, o2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S1, S2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(e1, e2, atol=1e-4, rtol=1e-4)


def test_chunk_matches_recurrent_from_zero_state(device):
    q, k, v, beta, alpha, _ = _inputs(1, 2, 96, 8, device, with_state=False)
    o1, S1, e1 = delta_recurrent(q, k, v, beta, alpha)
    o2, S2, e2 = delta_chunk(q, k, v, beta, alpha, chunk=32)
    assert torch.allclose(o1, o2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S1, S2, atol=1e-4, rtol=1e-4)


def test_one_shot_recall(device):
    d = 16
    k1 = F.normalize(torch.randn(1, 1, 1, d, device=device), dim=-1)
    v1 = torch.randn(1, 1, 1, d, device=device)
    beta = torch.ones(1, 1, 1, device=device)
    alpha = torch.ones(1, 1, 1, device=device)
    o, S, _ = delta_recurrent(k1, k1, v1, beta, alpha)
    assert torch.allclose(o[0, 0, 0], v1[0, 0, 0], atol=1e-5)


def test_write_pressure_identity(device):
    q, k, v, beta, alpha, S0 = _inputs(1, 2, 20, 8, device)
    _, _, err = delta_recurrent(q, k, v, beta, alpha, S0)
    S = S0.clone()
    for t in range(20):
        Sa = alpha[:, :, t, None, None] * S
        e = v[:, :, t] - torch.einsum("bhk,bhkv->bhv", k[:, :, t], Sa)
        S_new = Sa + beta[:, :, t, None, None] * torch.einsum("bhk,bhv->bhkv", k[:, :, t], e)
        dS = (S_new - Sa).flatten(-2).norm(dim=-1)
        assert torch.allclose(dS, beta[:, :, t] * err[:, :, t], atol=1e-5)
        S = S_new


@settings(max_examples=25, deadline=None)
@given(T=st.integers(1, 96), scale=st.floats(0.1, 50.0))
def test_state_bounded_for_any_input(T, scale):
    B, H, d = 1, 1, 8
    q = F.normalize(torch.randn(B, H, T, d), dim=-1)
    k = F.normalize(torch.randn(B, H, T, d), dim=-1)
    v = torch.randn(B, H, T, d) * scale
    beta = torch.rand(B, H, T)
    alpha = torch.rand(B, H, T)
    _, S, _ = delta_recurrent(q, k, v, beta, alpha)
    assert torch.isfinite(S).all()
    # each write adds at most beta*||e|| <= ||v_t|| + ||k S|| and the read along k is a contraction,
    # so the state norm is bounded by the sum of value norms
    assert S.norm() <= v.flatten(0, 2).norm(dim=-1).sum() + 1e-4


def test_gradients_reach_gates(device):
    q, k, v, beta, alpha, S0 = _inputs(1, 2, 64, 8, device)
    beta = beta.clone().requires_grad_(True)
    alpha = alpha.clone().requires_grad_(True)
    q = q.clone().requires_grad_(True)
    o, _, _ = delta_chunk(q, k, v, beta, alpha, S0, chunk=32)
    o.pow(2).mean().backward()
    for g in (beta.grad, alpha.grad, q.grad):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


@pytest.mark.parametrize("pattern", ["identical", "cycle3", "blocks"])
def test_chunk_matches_recurrent_with_correlated_keys(device, pattern):
    """Repeated tokens give identical keys; the chunk-parallel solve must stay exact."""
    B, H, T, d = 1, 2, 128, 32
    base = F.normalize(torch.randn(B, H, 3, d, device=device), dim=-1)
    if pattern == "identical":
        idx = torch.zeros(T, dtype=torch.long, device=device)
    elif pattern == "cycle3":
        idx = torch.arange(T, device=device) % 3
    else:
        idx = (torch.arange(T, device=device) // 20) % 3
    k = base[:, :, idx]
    q = k.clone()
    v = torch.randn(B, H, T, d, device=device)
    beta = torch.full((B, H, T), 0.9, device=device)
    alpha = torch.full((B, H, T), 0.99, device=device)
    o1, S1, e1 = delta_recurrent(q, k, v, beta, alpha)
    o2, S2, e2 = delta_chunk(q, k, v, beta, alpha, chunk=64)
    assert torch.isfinite(o2).all()
    assert torch.allclose(o1, o2, atol=1e-3, rtol=1e-3), float((o1 - o2).abs().max())
    assert torch.allclose(S1, S2, atol=1e-3, rtol=1e-3)


def test_gradients_finite_with_identical_keys(device):
    B, H, T, d = 1, 1, 64, 16
    k = F.normalize(torch.randn(B, H, 1, d, device=device), dim=-1).expand(B, H, T, d).contiguous()
    v = torch.randn(B, H, T, d, device=device)
    beta = torch.full((B, H, T), 0.95, device=device, requires_grad=True)
    alpha = torch.full((B, H, T), 0.99, device=device, requires_grad=True)
    o, S, _ = delta_chunk(k, k, v, beta, alpha, chunk=64)
    (o.pow(2).mean() + S.pow(2).mean()).backward()
    assert torch.isfinite(beta.grad).all() and torch.isfinite(alpha.grad).all()


@pytest.mark.parametrize("beta_value", [0.0, 1e-9, 0.5])
def test_surprise_is_independent_of_beta(device, beta_value):
    """The error signal is the pre-write prediction error; it must match the recurrent path
    even when writes are disabled or tiny."""
    q, k, v, _, alpha, S0 = _inputs(1, 2, 40, 8, device)
    beta = torch.full((1, 2, 40), beta_value, device=device)
    _, _, e1 = delta_recurrent(q, k, v, beta, alpha, S0)
    _, _, e2 = delta_chunk(q, k, v, beta, alpha, S0, chunk=16)
    assert torch.allclose(e1, e2, atol=1e-4, rtol=1e-4)
    assert float(e2.mean()) > 0
