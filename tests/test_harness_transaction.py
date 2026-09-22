import torch

from plastic.config import ModelConfig
from plastic.data.text import encode_documents_to_bin
from plastic.harness.calibrate import Calibration, calibrate_from_runner, log_only
from plastic.harness.canary import CanarySuite
from plastic.harness.config import HarnessConfig
from plastic.harness.transaction import TransactionRunner
from plastic.model.lm import PlasticDynamics, PlasticLM
from plastic.tokenizer.bpe import Tokenizer

CPU = torch.device("cpu")


def _lm(chunk=8, layers=2, vocab=64):
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=layers, chunk=chunk, vocab_size=vocab)
    return cfg, PlasticLM(cfg)


def _suite(tmp_path, vocab):
    docs = ["alpha beta gamma delta epsilon " * 60]
    tok = Tokenizer.train(docs, vocab_size=300)
    path = str(tmp_path / "validation.bin")
    encode_documents_to_bin(tok, docs, path)
    return CanarySuite.default_text(path, vocab_size=vocab, n_probe=2, probe_len=16)


def _ids(n, vocab=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, vocab, (n,), generator=g).tolist()


def _same_state(a, b, atol=1e-5):
    return all(torch.allclose(x.S, y.S, atol=atol) and torch.allclose(x.h, y.h, atol=atol) for x, y in zip(a.layers, b.layers))


def test_commit_path_records_and_commits():
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r.feed_tokens(_ids(8))
    assert r.n_transactions == 1 and r.transactions[0]["decision"]["kind"] == "commit"
    assert _same_state(r.committed, r.working) and r.pos == 8 and not r.pending
    assert r.transactions[0]["signals"]["n_tokens"] == 8 and r.transactions[0]["signals"]["chunk_loss"] > 0
    assert r.budget_used > 0


def test_rollback_equals_frozen_pass():
    cfg, lm = _lm()
    ids = _ids(8)
    # force rollback: any positive coherence delta rolls back and the canary suite is scored
    hcfg = HarnessConfig(canary_delta_max=-1e9, poison_delta_min=-1e9, enable_projection=False)
    r = TransactionRunner(lm, cfg, hcfg, suite=_fake_suite(cfg), device=CPU)
    r.feed_tokens(ids)
    rec = r.transactions[0]
    assert rec["decision"]["kind"] == "rollback" and any("canary_coherence" in x for x in rec["decision"]["reasons"])
    # reference: a fresh frozen pass over the same tokens from zero state
    ref = lm.init_state(1)
    _, ref, _ = lm(torch.tensor([ids]), ref, freeze=True)
    assert _same_state(r.committed, ref) and _same_state(r.working, ref)
    assert all(torch.equal(l.S, torch.zeros_like(l.S)) for l in r.committed.layers)  # nothing was learned
    assert r.pos == 8 and r.budget_used == 0.0


def _fake_suite(cfg):
    g = torch.Generator().manual_seed(3)
    probes = [torch.randint(3, cfg.vocab_size, (16,), generator=g).tolist() for _ in range(2)]
    return CanarySuite(domain="text", coherence=probes, poison=[probes[0]])


def test_scale_path_matches_direct_scaled_forward():
    cfg, lm = _lm()
    ids = _ids(8)
    hcfg = HarnessConfig(budget_chunk=1e-6, enable_projection=False)  # any write exceeds the chunk cap
    r = TransactionRunner(lm, cfg, hcfg, device=CPU)
    r.feed_tokens(ids)
    rec = r.transactions[0]
    assert rec["decision"]["kind"] == "scale" and 0 < rec["decision"]["scale"] < 1
    ref = lm.init_state(1)
    _, ref, _ = lm(torch.tensor([ids]), ref, beta_scale=rec["decision"]["scale"])
    assert _same_state(r.committed, ref)


def test_project_path_and_fallback():
    cfg, lm = _lm()
    ids = _ids(8)
    suite = _fake_suite(cfg)
    hcfg = HarnessConfig(project_eps_cos=-1.0, canary_delta_max=1e9, poison_delta_min=-1e9, enable_stats=False)  # always project
    r = TransactionRunner(lm, cfg, hcfg, suite=suite, device=CPU)
    r.feed_tokens(ids)
    rec = r.transactions[0]
    assert rec["decision"]["kind"] == "project", rec["decision"]
    # invariant: the committed delta no longer increases the coherence score to first order
    from plastic.harness.canary import canary_gradient
    from plastic.harness.projection import flatten_all

    zero = lm.init_state(1)
    g = canary_gradient(lm, zero, suite, device=CPU)
    d = r.committed.s_delta(zero)
    assert float((flatten_all(g) * flatten_all(d)).sum()) <= hcfg.project_eps_dot + 1e-4
    # fallback: refusing any removal turns projection into rollback
    hcfg2 = HarnessConfig(project_eps_cos=-1.0, project_max_removed=0.0, canary_delta_max=1e9, poison_delta_min=-1e9, enable_stats=False)
    r2 = TransactionRunner(lm, cfg, hcfg2, suite=suite, device=CPU)
    r2.feed_tokens(ids)
    rec2 = r2.transactions[0]
    assert rec2["decision"]["kind"] == "rollback" and any("project_removed" in x for x in rec2["decision"]["reasons"])


def test_session_budget_latches_read_only():
    cfg, lm = _lm()
    hcfg = HarnessConfig(budget_session=1e-6, enable_projection=False)
    r = TransactionRunner(lm, cfg, hcfg, device=CPU)
    r.feed_tokens(_ids(8))
    assert r.read_only and "budget_session" in (r.read_only_reason or "")
    S_before = [l.S.clone() for l in r.committed.layers]
    r.feed_tokens(_ids(8, seed=1))
    assert r.transactions[1]["decision"]["kind"] == "readonly"
    assert all(torch.equal(a, l.S) for a, l in zip(S_before, r.committed.layers))
    r.resume()
    assert not r.read_only


def test_generated_tokens_do_not_write_by_default():
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r.feed_tokens(_ids(8), source="model")
    assert all(torch.equal(l.S, torch.zeros_like(l.S)) for l in r.committed.layers)
    assert r.pos == 8
    r2 = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False, learn_from_generation=True), device=CPU)
    r2.feed_tokens(_ids(8), source="model")
    assert any(float(l.S.abs().sum()) > 0 for l in r2.committed.layers)


def test_streaming_partition_invariance_and_persistence():
    cfg, lm = _lm()
    ids = _ids(16)
    a = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    a.feed_tokens(ids)
    b = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    for t in ids[:5]:
        b.feed_tokens([t])
    sd = b.state_dict()  # mid-chunk save
    c = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    c.load_state_dict(sd)
    for t in ids[5:]:
        c.feed_tokens([t])
    assert _same_state(a.committed, c.committed)
    assert a.n_transactions == c.n_transactions == 2
    assert abs(a.transactions[1]["signals"]["chunk_loss"] - c.transactions[1]["signals"]["chunk_loss"]) < 1e-4
    assert torch.allclose(a._last_logits, c._last_logits, atol=1e-4)


def test_flush_partial_chunk():
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r.feed_tokens(_ids(5))
    assert r.n_transactions == 0 and len(r.pending) == 5
    rec = r.flush()
    assert rec is not None and rec["signals"]["n_tokens"] == 5 and r.n_transactions == 1 and not r.pending


def test_physics_runner_transacts():
    torch.manual_seed(0)
    cfg = ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=1, chunk=8)
    m = PlasticDynamics(cfg)
    suite = CanarySuite.default_physics(n_probe=2, steps=8)
    r = TransactionRunner(m, cfg, HarnessConfig(), suite=suite, device=CPU)
    rows = torch.randn(16, 7)
    targets = torch.randn(16, 4)
    preds = r.feed_physics(rows, targets)
    assert preds.shape == (16, 4) and r.n_transactions == 2
    sig = r.transactions[0]["signals"]
    assert sig["chunk_loss"] > 0 and sig["canary_coherence_before"] is not None


def test_calibration_from_runner_and_fpr(tmp_path):
    cfg, lm = _lm()
    suite = _fake_suite(cfg)
    hcfg = HarnessConfig(enable_projection=False)
    r = TransactionRunner(lm, cfg, log_only(hcfg), suite=suite, device=CPU)
    # 8 benign sessions of 8 chunks each
    stream = [_ids(8, seed=s) for s in range(64)]
    cal = calibrate_from_runner(r, stream, n_chunks=64, model_signature="sig", target_fpr=0.1, reset_every=8)
    assert cal.n_chunks == 64 and "chunk_loss" in cal.thresholds and "canary_delta_coherence" in cal.thresholds
    cal.save(str(tmp_path))
    back = Calibration.load(str(tmp_path))
    assert back.thresholds == cal.thresholds and back.reference.keys() == cal.reference.keys()
    # fresh benign sessions with unseen seeds: gating should be rare and must not cascade
    gated = total = 0
    for session in range(4):
        r2 = TransactionRunner(lm, cfg, hcfg, calibration=back, suite=suite, device=CPU)
        for s in range(8):
            r2.feed_tokens(_ids(8, seed=1000 + session * 8 + s))
        gated += sum(1 for t in r2.transactions if t["decision"]["kind"] != "commit")
        total += len(r2.transactions)
        assert not r2.read_only, r2.read_only_reason
    assert gated / total <= 0.35, (gated, total)


def test_read_only_chunks_do_not_feed_statistics():
    cfg, lm = _lm()
    hcfg = HarnessConfig(budget_session=1e-6, enable_projection=False)
    r = TransactionRunner(lm, cfg, hcfg, device=CPU)
    r.feed_tokens(_ids(8))
    assert r.read_only
    before = r.cusum.state()
    r.feed_tokens(_ids(8, seed=5))
    sig = r.transactions[1]["signals"]
    assert all(v is None for v in sig["z"].values()) and not sig["cusum_alarm"]
    assert r.cusum.state() == before
