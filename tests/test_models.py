import torch

from plastic.config import ModelConfig
from plastic.model.lm import PlasticDynamics, PlasticLM, build_model


def test_lm_shapes_and_param_count():
    cfg = ModelConfig()
    lm = PlasticLM(cfg)
    n = lm.num_params()
    assert 5.0e6 < n < 6.5e6, n
    toks = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, st, sig = lm(toks)
    assert logits.shape == (2, 64, cfg.vocab_size) and st.pos == 64 and len(sig) == cfg.n_layers


def test_lm_state_carry_matches_full_pass(device):
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=16, vocab_size=50)
    lm = PlasticLM(cfg).to(device)
    toks = torch.randint(0, 50, (1, 40), device=device)
    full, st_full, _ = lm(toks)
    a, st_a, _ = lm(toks[:, :24])
    b, st_b, _ = lm(toks[:, 24:], st_a)
    assert torch.allclose(full, torch.cat([a, b], 1), atol=1e-4, rtol=1e-4)
    rec, st_r, _ = lm(toks, mode="recurrent")
    assert torch.allclose(full, rec, atol=1e-4, rtol=1e-4)
    assert st_full.pos == st_b.pos == st_r.pos == 40


def test_beta_off_ablation_changes_loss():
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=16, vocab_size=50)
    lm = PlasticLM(cfg)
    toks = torch.randint(0, 50, (2, 64))
    l1 = lm.loss(toks)
    l0 = lm.loss(toks, beta_scale=0.0)
    assert not torch.allclose(l1, l0)


def test_untied_head_variant():
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, chunk=16, vocab_size=50, tie_embeddings=False)
    lm = PlasticLM(cfg)
    assert lm.head is not None
    logits, _, _ = lm(torch.randint(0, 50, (1, 8)))
    assert logits.shape == (1, 8, 50)


def test_dynamics_model():
    cfg = ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=2, chunk=16)
    m = build_model(cfg)
    assert isinstance(m, PlasticDynamics)
    x = torch.randn(2, 33, 7)
    pred, st, _ = m(x)
    assert pred.shape == (2, 33, 4) and st.pos == 33
    assert m.loss(x, torch.randn(2, 33, 4)).ndim == 0


def test_domain_mismatch_rejected():
    try:
        PlasticLM(ModelConfig(domain="physics"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
