import torch

from plastic.config import ModelConfig
from plastic.model.block import PlasticBlock
from plastic.model.state import SessionState


def _cfg(**kw):
    base = dict(d_model=32, n_heads=2, n_layers=1, chunk=16)
    base.update(kw)
    return ModelConfig(**base)


def test_modes_agree_and_state_carries(device):
    cfg = _cfg()
    blk = PlasticBlock(cfg).to(device)
    x = torch.randn(2, 48, 32, device=device)
    st = SessionState.zeros(cfg, batch=2, device=device).layers[0]
    y_full, s_full, _ = blk(x, st, mode="chunk")
    y_a, s_a, _ = blk(x[:, :20], st, mode="chunk")
    y_b, s_b, _ = blk(x[:, 20:], s_a, mode="chunk")
    y_r, s_r, _ = blk(x, st, mode="recurrent")
    assert torch.allclose(y_full, torch.cat([y_a, y_b], 1), atol=1e-4, rtol=1e-4)
    assert torch.allclose(y_full, y_r, atol=1e-4, rtol=1e-4)
    assert torch.allclose(s_full.S, s_b.S, atol=1e-4) and torch.allclose(s_full.h, s_b.h, atol=1e-4)
    assert torch.allclose(s_full.S, s_r.S, atol=1e-4)


def test_beta_scale_changes_output(device):
    cfg = _cfg()
    blk = PlasticBlock(cfg).to(device)
    x = torch.randn(1, 32, 32, device=device)
    st = SessionState.zeros(cfg, batch=1, device=device).layers[0]
    y1, _, _ = blk(x, st, mode="chunk")
    y0, s0, _ = blk(x, st, mode="chunk", beta_scale=0.0)
    assert not torch.allclose(y1, y0)
    assert torch.allclose(s0.S, torch.zeros_like(s0.S))


def test_block_in_variant_runs(device):
    cfg = _cfg(memory_input="block_in")
    blk = PlasticBlock(cfg).to(device)
    x = torch.randn(1, 16, 32, device=device)
    st = SessionState.zeros(cfg, batch=1, device=device).layers[0]
    y, _, _ = blk(x, st, mode="chunk")
    assert y.shape == x.shape


def test_chunk_rule_block_runs(device):
    cfg = _cfg(rule="chunk", chunk=8)
    blk = PlasticBlock(cfg).to(device)
    x = torch.randn(1, 24, 32, device=device)
    st = SessionState.zeros(cfg, batch=1, device=device).layers[0]
    y, s, sig = blk(x, st, mode="chunk")
    assert y.shape == x.shape and s.M is not None and sig.write_norm.shape == (1, 2, 24)


def test_gradients_reach_all_parameters(device):
    cfg = _cfg()
    blk = PlasticBlock(cfg).to(device)
    x = torch.randn(2, 32, 32, device=device)
    st = SessionState.zeros(cfg, batch=2, device=device).layers[0]
    y, _, _ = blk(x, st, mode="chunk")
    y.pow(2).mean().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_conv_kernel_one_variant_runs(device):
    cfg = _cfg(conv_kernel=1)
    blk = PlasticBlock(cfg).to(device)
    x = torch.randn(1, 16, 32, device=device)
    st = SessionState.zeros(cfg, batch=1, device=device).layers[0]
    y, s, _ = blk(x, st, mode="chunk")
    assert y.shape == x.shape and s.conv_ssm is None
