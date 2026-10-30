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


@pytest.mark.parametrize("scan", [scan_sequential, scan_chunked])
@pytest.mark.parametrize("with_initial_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_scan_preserves_state(device, scan, with_initial_state, dtype):
    if device.type == "mps" and dtype == torch.float64:
        pytest.skip("MPS does not support float64")
    u = torch.empty(2, 0, 4, device=device, dtype=dtype, requires_grad=True)
    a = torch.empty_like(u)
    h0 = torch.randn(2, 4, device=device, dtype=dtype, requires_grad=True) if with_initial_state else None
    out, last = scan(a, u, h0)
    assert out.shape == u.shape and out.dtype == dtype and out.device == u.device
    expected = torch.zeros(2, 4, device=device, dtype=dtype) if h0 is None else h0
    assert torch.equal(last, expected)
    assert last.dtype == dtype and last.device == u.device
    # An empty segment preserves the gradient path through the carried state.
    loss = out.sum() + last.sum()
    loss.backward()
    assert u.grad is not None and u.grad.shape == u.shape
    if h0 is not None:
        assert torch.equal(h0.grad, torch.ones_like(h0))


@pytest.mark.parametrize("chunk", [1, 4, 64])
def test_streaming_scan_with_empty_segments(device, chunk):
    u = torch.randn(2, 11, 4, device=device)
    a = torch.sigmoid(torch.randn_like(u))
    h0 = torch.randn(2, 4, device=device)
    original = h0.clone()
    expected, final = scan_sequential(a, u, h0)
    for scan in (scan_sequential, lambda a, u, h: scan_chunked(a, u, h, chunk=chunk)):
        state = h0.clone()
        outputs = []
        for start, end in [(0, 0), (0, 3), (3, 3), (3, 11), (11, 11)]:
            out, state = scan(a[:, start:end], u[:, start:end], state)
            outputs.append(out)
        assert torch.allclose(torch.cat(outputs, dim=1), expected, atol=1e-4, rtol=1e-4)
        assert torch.allclose(state, final, atol=1e-4, rtol=1e-4)
        assert torch.equal(h0, original)
