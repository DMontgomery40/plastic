import pytest
import torch

from plastic.config import ModelConfig
from plastic.model.lm import PlasticLM
from plastic.model.memory import FastWeightMemory
from plastic.model.state import SessionState


def test_freeze_leaves_memory_bit_identical(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=16)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(1, 20, 32, device=device)
    S0 = torch.randn(1, 2, 16, 16, device=device) * 0.1
    for mode in ("chunk", "recurrent"):
        _, S1, _, _, sig = mem(u, S0, None, mode=mode, freeze=True)
        assert torch.equal(S1, S0), mode
        assert sig.write_norm.abs().max() == 0
        assert (sig.alpha == 1).all() and (sig.beta == 0).all()


def test_beta_scale_zero_still_decays_but_freeze_does_not(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=16)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(1, 20, 32, device=device)
    S0 = torch.randn(1, 2, 16, 16, device=device) * 0.1
    _, S_scaled, _, _, _ = mem(u, S0, None, mode="chunk", beta_scale=0.0)
    _, S_frozen, _, _, _ = mem(u, S0, None, mode="chunk", freeze=True)
    assert not torch.equal(S_scaled, S0)
    assert torch.equal(S_frozen, S0)


def test_frozen_model_reads_state_but_does_not_write(device):
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=16, vocab_size=50)
    lm = PlasticLM(cfg).to(device)
    toks = torch.randint(0, 50, (1, 32), device=device)
    _, st, _ = lm(toks)
    logits_frozen, st2, _ = lm(toks, st, freeze=True)
    for a, b in zip(st.layers, st2.layers):
        assert torch.equal(a.S, b.S)
    assert st2.pos == 64
    logits_zero, _, _ = lm(toks, SessionState.zeros(cfg, 1, device), freeze=True)
    assert not torch.allclose(logits_frozen, logits_zero)


def test_chunk_rule_freeze(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=8, rule="chunk")
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(1, 20, 32, device=device)
    st = SessionState.zeros(cfg, batch=1, device=device).layers[0]
    st.S += torch.randn_like(st.S) * 0.1
    st.chunk.A += 0.5
    _, S1, M1, c1, sig = mem(u, st.S, st.M, mode="chunk", freeze=True, chunk0=st.chunk)
    assert torch.equal(S1, st.S) and torch.equal(M1, st.M)
    assert torch.equal(c1.A, torch.zeros_like(c1.A)) and c1.count == 20 % 8  # boundaries still advance
    assert sig.write_norm.abs().max() == 0 and sig.err.shape == (1, 2, 20)


@pytest.mark.parametrize("n_frozen", [6, 20])
def test_chunk_rule_resume_after_freeze_keeps_retention(device, n_frozen):
    """Frozen tokens must not make the next applied chunk decay by a truncated average."""
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=8, rule="chunk")
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(1, n_frozen + 2, 32, device=device)
    st = SessionState.zeros(cfg, batch=1, device=device).layers[0]
    st.S += torch.randn_like(st.S)
    _, S1, M1, c1, _ = mem(u[:, :n_frozen], st.S, st.M, mode="chunk", freeze=True, chunk0=st.chunk)
    assert torch.equal(S1, st.S)
    # resume with writes disabled: the next boundary should only apply mean retention
    _, S2, _, _, sig = mem(u[:, n_frozen:], S1, M1, mode="chunk", beta_scale=0.0, chunk0=c1)
    resumed = (n_frozen + 2) % 8 == 0
    if resumed:
        alphas = sig.alpha.mean(dim=-1)  # only the two active tokens carry alpha < 1
        expected = ((8 - 2) * 1.0 + 2 * alphas) / 8
        assert torch.allclose(S2, expected[..., None, None] * S1, atol=1e-5)
        assert float((S2.norm() / S1.norm())) > 0.95
    else:
        assert torch.equal(S2, S1)
