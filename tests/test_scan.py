import pytest
import torch

from plastic.model.scan import scan_chunked, scan_sequential


@pytest.mark.parametrize("T", [1, 5, 64, 130, 512])
def test_chunked_matches_sequential(device, T):
    B, D = 2, 32
    u = torch.randn(B, T, D, device=device)
    a = torch.sigmoid(torch.randn(B, T, D, device=device) * 2 - 1)
    h0 = torch.randn(B, D, device=device)
    ref, ref_last = scan_sequential(a, u, h0)
    got, got_last = scan_chunked(a, u, h0, chunk=64)
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4)
    assert torch.allclose(ref_last, got_last, atol=1e-4, rtol=1e-4)


def test_tiny_decays_do_not_blow_up(device):
    B, T, D = 2, 256, 16
    u = torch.randn(B, T, D, device=device)
    a = torch.rand(B, T, D, device=device) * 0.2 + 1e-3
    ref, _ = scan_sequential(a, u)
    got, _ = scan_chunked(a, u, chunk=64)
    assert torch.isfinite(got).all()
    assert (ref - got).abs().max() < 1e-4


def test_gradients_flow(device):
    B, T, D = 1, 128, 8
    a = torch.sigmoid(torch.randn(B, T, D, device=device)).requires_grad_(True)
    u = torch.randn(B, T, D, device=device, requires_grad=True)
    h, _ = scan_chunked(a, u, chunk=32)
    h.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    assert u.grad is not None and torch.isfinite(u.grad).all()


def test_zero_state_default_matches_explicit_zero(device):
    B, T, D = 1, 40, 4
    u = torch.randn(B, T, D, device=device)
    a = torch.rand(B, T, D, device=device)
    h1, _ = scan_chunked(a, u, chunk=16)
    h2, _ = scan_chunked(a, u, torch.zeros(B, D, device=device), chunk=16)
    assert torch.equal(h1, h2)
