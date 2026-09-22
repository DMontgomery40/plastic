import torch

from plastic.config import ModelConfig
from plastic.data.text import encode_documents_to_bin
from plastic.harness.canary import CanarySuite, canary_gradient, score_suite
from plastic.model.lm import PlasticDynamics, PlasticLM
from plastic.tokenizer.bpe import Tokenizer


def _text_fixture(tmp_path):
    docs = ["alpha beta gamma delta epsilon " * 60, "one two three four five six " * 60]
    tok = Tokenizer.train(docs, vocab_size=300)
    path = str(tmp_path / "validation.bin")
    encode_documents_to_bin(tok, docs, path)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=8, vocab_size=tok.vocab_size)
    return cfg, PlasticLM(cfg), path


def test_text_suite_scoring_is_read_only(tmp_path):
    cfg, lm, path = _text_fixture(tmp_path)
    suite = CanarySuite.default_text(path, vocab_size=cfg.vocab_size, n_probe=3, probe_len=32)
    assert len(suite.coherence) == 3 and len(suite.poison) >= 2
    state = lm.init_state(1)
    _, state, _ = lm(torch.randint(0, cfg.vocab_size, (1, 20)), state)
    snapshot = state.clone()
    scores = score_suite(lm, state, suite, device=torch.device("cpu"))
    assert set(scores) == {"coherence", "poison"} and all(v == v for v in scores.values())
    for a, b in zip(state.layers, snapshot.layers):
        assert torch.equal(a.S, b.S) and torch.equal(a.h, b.h)
    grads = canary_gradient(lm, state, suite, device=torch.device("cpu"))
    assert len(grads) == cfg.n_layers and grads[0].shape == state.layers[0].S.shape
    assert all(torch.isfinite(g).all() for g in grads) and any(float(g.abs().sum()) > 0 for g in grads)
    for a, b in zip(state.layers, snapshot.layers):
        assert torch.equal(a.S, b.S)


def test_suite_json_roundtrip(tmp_path):
    cfg, lm, path = _text_fixture(tmp_path)
    suite = CanarySuite.default_text(path, vocab_size=cfg.vocab_size, n_probe=2, probe_len=16)
    p = str(tmp_path / "canary.json")
    suite.save(p)
    back = CanarySuite.load(p)
    assert back.to_dict() == suite.to_dict()


def test_physics_suite_scores_and_gradient():
    cfg = ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=1, chunk=8)
    m = PlasticDynamics(cfg)
    suite = CanarySuite.default_physics(n_probe=2, steps=16)
    state = m.init_state(1)
    scores = score_suite(m, state, suite, device=torch.device("cpu"))
    assert scores["coherence"] == scores["coherence"] and scores["poison"] >= 0
    grads = canary_gradient(m, state, suite, device=torch.device("cpu"))
    assert grads[0].shape == state.layers[0].S.shape


def test_score_suite_handles_ragged_poison_probes(tmp_path):
    cfg, lm, path = _text_fixture(tmp_path)
    suite = CanarySuite.default_text(path, vocab_size=cfg.vocab_size, n_probe=3, probe_len=16)
    # a recorded attack payload of a different length must not break scoring
    suite.poison.append([5, 6, 7, 8])          # length 4
    suite.poison.append(list(range(3, 3 + 24)))  # length 24
    state = lm.init_state(1)
    scores = score_suite(lm, state, suite, device=torch.device("cpu"))
    assert scores["poison"] == scores["poison"]  # finite, not NaN or a crash
    grads = canary_gradient(lm, state, suite, device=torch.device("cpu"))
    assert all(torch.isfinite(g).all() for g in grads)
