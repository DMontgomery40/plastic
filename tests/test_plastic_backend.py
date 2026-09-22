import torch

from plastic.backends.base import Backend
from plastic.backends.plastic import PlasticBackend
from plastic.config import ModelConfig
from plastic.harness.canary import CanarySuite, canary_gradient, score_suite
from plastic.harness.signals import STAT_SIGNALS
from plastic.model.lm import PlasticLM

CPU = torch.device("cpu")


def _fixture():
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=8, vocab_size=64)
    lm = PlasticLM(cfg)
    suite = CanarySuite.default_physics(n_probe=1, steps=8) if cfg.domain == "physics" else None
    return cfg, lm


def _text_suite(vocab):
    # random-token coherence/poison probes are enough for the delegation check
    return CanarySuite(domain="text", coherence=[list(range(3, 3 + 16))], poison=[list(range(5, 5 + 16))])


def test_plastic_backend_conforms_to_protocol():
    cfg, lm = _fixture()
    be = PlasticBackend(lm, cfg, device=CPU)
    assert isinstance(be, Backend)
    assert be.signal_names() == STAT_SIGNALS
    assert be.writes_for_source("user") is True and be.writes_for_source("model") is False


def test_forward_matches_the_direct_model_call():
    cfg, lm = _fixture()
    be = PlasticBackend(lm, cfg, device=CPU)
    items = [5, 6, 7, 8, 9, 10, 11, 12]
    st = be.init_state()
    out, new_state, signals = be.forward(items, st.clone(), freeze=False, beta_scale=1.0)
    # same as calling the model directly the way the runner does today
    x = torch.tensor([items], dtype=torch.long)
    with torch.no_grad():
        d_out, d_state, d_sig = lm(x, st.clone(), mode="chunk", freeze=False, beta_scale=1.0)
    assert torch.allclose(out, d_out[0], atol=1e-6)
    assert len(signals) == len(d_sig)
    # state ops delegate correctly
    delta = be.state_delta(new_state, st)
    assert len(delta) == cfg.n_layers and all(torch.is_tensor(t) for t in delta)
    assert be.is_finite(new_state)


def test_state_dict_roundtrip_and_clone_independence():
    cfg, lm = _fixture()
    be = PlasticBackend(lm, cfg, device=CPU)
    st = be.init_state()
    out, st, _ = be.forward([5, 6, 7, 8, 9, 10, 11, 12], st, freeze=False, beta_scale=1.0)
    back = be.load_state_dict(be.state_dict(st))
    assert all(torch.equal(a.S, b.S) and torch.equal(a.h, b.h) for a, b in zip(back.layers, st.layers))
    snap = be.clone(st)
    be.forward([13, 14, 15, 16, 17, 18, 19, 20], st, freeze=False, beta_scale=1.0)
    # the clone is unaffected by advancing the original
    assert all(torch.equal(a.S, b.S) for a, b in zip(snap.layers, back.layers))


def test_canary_and_projection_delegate():
    cfg, lm = _fixture()
    be = PlasticBackend(lm, cfg, device=CPU)
    suite = _text_suite(cfg.vocab_size)
    st = be.init_state()
    _, st, _ = be.forward([5, 6, 7, 8, 9, 10, 11, 12], st, freeze=False, beta_scale=1.0)
    # score / gradient match the module functions the runner calls today
    assert be.score_suite(st, suite) == score_suite(lm, st, suite, device=CPU)
    g_be = be.canary_gradient(st, suite)
    g_fn = canary_gradient(lm, st, suite, device=CPU)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g_be, g_fn))
    # apply_projected writes committed.S + delta onto working.S
    working, committed = st.clone(), st.clone()
    deltas = [torch.ones_like(l.S) for l in committed.layers]
    be.apply_projected(working, committed, deltas)
    assert all(torch.allclose(w.S, c.S + d) for w, c, d in zip(working.layers, committed.layers, deltas))
