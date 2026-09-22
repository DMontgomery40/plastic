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
