import pytest
import torch

from plastic.config import ModelConfig
from plastic.model.memory import FastWeightMemory
from plastic.model.state import SessionState


def test_chunk_and_recurrent_modes_agree(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=16)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(2, 40, 32, device=device)
    S0 = torch.randn(2, 2, 16, 16, device=device) * 0.1
    m1, S1, _, _, sig1 = mem(u, S0, None, mode="chunk")
    m2, S2, _, _, sig2 = mem(u, S0, None, mode="recurrent")
    assert torch.allclose(m1, m2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S1, S2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(sig1.write_norm, sig2.write_norm, atol=1e-4)


def test_beta_scale_zero_freezes_memory(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=16)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(1, 20, 32, device=device)
    S0 = torch.randn(1, 2, 16, 16, device=device) * 0.1
    _, S1, _, _, sig = mem(u, S0, None, mode="chunk", beta_scale=0.0)
    assert sig.write_norm.abs().max() == 0
    assert (S1.abs() <= S0.abs() + 1e-6).all()


def test_signals_shapes(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=16)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(3, 17, 32, device=device)
    S0 = torch.zeros(3, 2, 16, 16, device=device)
    m, S, M, _, sig = mem(u, S0, None, mode="chunk")
    assert m.shape == (3, 17, 32) and S.shape == (3, 2, 16, 16) and M is None
    assert sig.err.shape == sig.beta.shape == sig.alpha.shape == sig.write_norm.shape == (3, 2, 17)
    assert (sig.beta > 0).all() and (sig.beta < 1).all() and (sig.alpha > 0).all() and (sig.alpha < 1).all()


def test_chunk_rule_runs_and_reads_chunk_start_weights(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=8, rule="chunk")
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(1, 20, 32, device=device)
    st = SessionState.zeros(cfg, batch=1, device=device).layers[0]
    m, S, M, chunk, sig = mem(u, st.S, st.M, mode="chunk", chunk0=st.chunk)
    assert m.shape == (1, 20, 32) and M is not None and M.shape == S.shape
    assert chunk is not None and chunk.count == 4  # 20 = 8 + 8 + 4 pending
    # first chunk reads from S0 = 0 so its memory output is exactly zero
    assert torch.allclose(m[:, :8], torch.zeros_like(m[:, :8]))
    nz = (sig.write_norm[0, 0] != 0).nonzero().flatten().tolist()
    assert nz == [7, 15]  # one write per completed chunk, reported on its last token


@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
def test_chunk_rule_is_stream_equivalent(device, mode):
    """Any partition of the token stream into forward calls gives identical results."""
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=8, rule="chunk")
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(1, 37, 32, device=device)
    st = SessionState.zeros(cfg, batch=1, device=device).layers[0]
    st.S += torch.randn_like(st.S) * 0.1
    m_full, S_full, M_full, c_full, sig_full = mem(u, st.S, st.M, mode=mode, chunk0=st.chunk)
    S, M, c = st.S, st.M, st.chunk
    outs, writes = [], []
    cuts = [0, 3, 8, 9, 20, 21, 30, 37]
    for a, b in zip(cuts[:-1], cuts[1:]):
        m, S, M, c, sig = mem(u[:, a:b], S, M, mode=mode, chunk0=c)
        outs.append(m)
        writes.append(sig.write_norm)
    assert torch.allclose(m_full, torch.cat(outs, 1), atol=1e-5, rtol=1e-5)
    assert torch.allclose(S_full, S, atol=1e-5) and torch.allclose(M_full, M, atol=1e-5)
    assert c.count == c_full.count == 37 % 8
    assert torch.allclose(c.A, c_full.A, atol=1e-5)
    assert torch.allclose(sig_full.write_norm, torch.cat(writes, 2), atol=1e-5)


def test_chunk_rule_beta_scale_caps_orthogonalized_write(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=8, rule="chunk", chunk_orthogonalize=True)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(1, 8, 32, device=device)
    st = SessionState.zeros(cfg, batch=1, device=device).layers[0]
    _, S1, M1, _, _ = mem(u, st.S, st.M, mode="chunk", chunk0=st.chunk, beta_scale=1.0)
    _, Sh, Mh, _, _ = mem(u, st.S, st.M, mode="chunk", chunk0=st.chunk, beta_scale=0.5)
    _, S0, M0, _, sig0 = mem(u, st.S, st.M, mode="chunk", chunk0=st.chunk, beta_scale=0.0)
    assert torch.allclose((Sh - st.S).norm(), 0.5 * (S1 - st.S).norm(), rtol=1e-4)
    assert torch.equal(S0, st.S) and torch.equal(M0, st.M)
    assert sig0.write_norm.abs().max() == 0
    # warm momentum: scale 0 still writes nothing, momentum only decays, S only decays
    _, S2, M2, _, sig2 = mem(u, S1, M1, mode="chunk", chunk0=st.chunk, beta_scale=0.0)
    assert torch.allclose(M2, cfg.chunk_momentum * M1)
    alpha_chunk = sig2.alpha.mean(dim=-1)
    assert torch.allclose(S2, alpha_chunk[..., None, None] * S1, atol=1e-6)


def test_mlp_memory_not_available_yet():
    with pytest.raises(NotImplementedError):
        FastWeightMemory(ModelConfig(rule="chunk", memory="mlp"))
