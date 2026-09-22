import pytest
import torch

from plastic.config import ModelConfig
from plastic.model.memory import FastWeightMemory


def test_chunk_and_recurrent_modes_agree(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=16)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(2, 40, 32, device=device)
    S0 = torch.randn(2, 2, 16, 16, device=device) * 0.1
    m1, S1, _, sig1 = mem(u, S0, None, mode="chunk")
    m2, S2, _, sig2 = mem(u, S0, None, mode="recurrent")
    assert torch.allclose(m1, m2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S1, S2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(sig1.write_norm, sig2.write_norm, atol=1e-4)


def test_beta_scale_zero_freezes_memory(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=16)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(1, 20, 32, device=device)
    S0 = torch.randn(1, 2, 16, 16, device=device) * 0.1
    _, S1, _, sig = mem(u, S0, None, mode="chunk", beta_scale=0.0)
    assert sig.write_norm.abs().max() == 0
    assert (S1.abs() <= S0.abs() + 1e-6).all()


def test_signals_shapes(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=16)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(3, 17, 32, device=device)
    S0 = torch.zeros(3, 2, 16, 16, device=device)
    m, S, M, sig = mem(u, S0, None, mode="chunk")
    assert m.shape == (3, 17, 32) and S.shape == (3, 2, 16, 16) and M is None
    assert sig.err.shape == sig.beta.shape == sig.alpha.shape == sig.write_norm.shape == (3, 2, 17)
    assert (sig.beta > 0).all() and (sig.beta < 1).all() and (sig.alpha > 0).all() and (sig.alpha < 1).all()


def test_chunk_rule_runs_and_reads_chunk_start_weights(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=8, rule="chunk")
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(1, 20, 32, device=device)
    S0 = torch.zeros(1, 2, 16, 16, device=device)
    M0 = torch.zeros_like(S0)
    m, S, M, sig = mem(u, S0, M0, mode="chunk")
    assert m.shape == (1, 20, 32) and M is not None and M.shape == S.shape
    # first chunk reads from S0 = 0 so its memory output is exactly the norm of zeros = 0
    assert torch.allclose(m[:, :8], torch.zeros_like(m[:, :8]))
    assert (sig.write_norm[:, :, :8] == sig.write_norm[:, :, :1]).all()


def test_mlp_memory_not_available_yet():
    with pytest.raises(NotImplementedError):
        FastWeightMemory(ModelConfig(rule="chunk", memory="mlp"))
