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


def _delta(a, b):
    from plastic.harness.signals import delta_norms

    return delta_norms(a.s_delta(b))[0]


def test_session_budget_is_hard_and_latches_read_only():
    cfg, lm = _lm()
    # budget large enough for roughly one chunk: the first chunk must be accepted within it,
    # the next must be scaled or refused, never over-committed
    r0 = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r0.feed_tokens(_ids(8))
    one_chunk = r0.budget_used
    hcfg = HarnessConfig(budget_session=one_chunk * 1.2, enable_projection=False)
    r = TransactionRunner(lm, cfg, hcfg, device=CPU)
    zero = lm.init_state(1)
    r.feed_tokens(_ids(8))
    assert r.transactions[0]["decision"]["kind"] == "commit" and not r.read_only
    used1 = r.budget_used
    r.feed_tokens(_ids(8, seed=1))
    rec = r.transactions[1]
    assert rec["decision"]["kind"] in ("scale", "rollback"), rec["decision"]
    assert r.budget_used <= hcfg.budget_session * (1 + 1e-6)
    assert _delta(r.committed, zero) <= hcfg.budget_session * (1 + 1e-6) or rec["decision"]["kind"] == "rollback"
    # keep feeding: the session exhausts and latches read-only, never exceeding the budget
    for s in range(2, 6):
        r.feed_tokens(_ids(8, seed=s))
    assert r.budget_used <= hcfg.budget_session * (1 + 1e-6)
    assert r.read_only and "budget_session" in (r.read_only_reason or "")
    S_before = [l.S.clone() for l in r.committed.layers]
    r.feed_tokens(_ids(8, seed=99))
    assert r.transactions[-1]["decision"]["kind"] == "readonly"
    assert all(torch.equal(a, l.S) for a, l in zip(S_before, r.committed.layers))
    r.resume()
    assert not r.read_only


def test_tiny_session_budget_never_over_commits():
    cfg, lm = _lm()
    cap = 1e-6
    r = TransactionRunner(lm, cfg, HarnessConfig(budget_session=cap, enable_projection=False), device=CPU)
    zero = lm.init_state(1)
    for s in range(4):
        r.feed_tokens(_ids(8, seed=s))
    kinds = [t["decision"]["kind"] for t in r.transactions]
    assert all(k in ("scale", "rollback", "readonly") for k in kinds), kinds
    assert _delta(r.committed, zero) <= cap * (1 + 1e-6)
    assert r.budget_used <= cap * (1 + 1e-6)


def test_chunk_cap_enforced_on_final_candidate_with_warm_state():
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r.feed_tokens(_ids(32))  # warm memory: decay alone now moves S
    before = r.committed.clone()
    cap = 1e-4
    r.hcfg = HarnessConfig(enable_projection=False, budget_chunk=cap)
    r.feed_tokens(_ids(8, seed=9))
    rec = r.transactions[-1]
    moved = _delta(r.committed, before)
    if rec["decision"]["kind"] == "rollback":
        assert any("budget_unsatisfiable" in x for x in rec["decision"]["reasons"])
        assert moved == 0.0
    else:
        assert moved <= cap * (1 + 1e-6), (rec["decision"], moved)


def test_chunk_cap_scaling_retry_satisfies_cap():
    cfg, lm = _lm()
    r0 = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r0.feed_tokens(_ids(8))
    full = r0.budget_used
    cap = 0.5 * full
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False, budget_chunk=cap), device=CPU)
    r.feed_tokens(_ids(8))
    rec = r.transactions[0]
    assert rec["decision"]["kind"] == "scale", rec["decision"]
    assert _delta(r.committed, lm.init_state(1)) <= cap * (1 + 1e-6)
    assert 0 < rec["decision"]["scale"] < 1


def test_projection_respects_budget():
    cfg, lm = _lm()
    suite = _fake_suite(cfg)
    base = HarnessConfig(project_eps_cos=-1.0, canary_delta_max=1e9, poison_delta_min=-1e9, enable_stats=False)
    r0 = TransactionRunner(lm, cfg, base, suite=suite, device=CPU)
    r0.feed_tokens(_ids(8))
    full = r0.budget_used
    cap = 0.5 * full
    hcfg = HarnessConfig(project_eps_cos=-1.0, canary_delta_max=1e9, poison_delta_min=-1e9, enable_stats=False, budget_chunk=cap)
    r = TransactionRunner(lm, cfg, hcfg, suite=suite, device=CPU)
    r.feed_tokens(_ids(8))
    rec = r.transactions[0]
    assert rec["decision"]["kind"] == "project" and any("budget_scaled_projection" in x for x in rec["decision"]["reasons"])
    assert _delta(r.committed, lm.init_state(1)) <= cap * (1 + 1e-6)


def test_projection_budget_recheck_float_rounding():
    # ASTRA-061: at committed S = 2**20 the float32 ULP is 0.125, so a scaled projected delta can
    # round to a *representable* stored change that still violates the budget cap — the branch that
    # fires the SECOND apply_projected (the recheck) in _apply. In exact arithmetic the first scaled
    # apply lands precisely at the cap, so a rounding fixture is what makes the recheck reproducible;
    # this is deterministic coverage of the branch, NOT a claim that real float token forwards can
    # never reach it (ASTRA-062). The committed/working S, the proposed delta, and the (orthogonal)
    # canary gradient are set directly. The three caps make ULP rounding give, respectively: accept
    # in one apply, accept-zero after a recheck, and a budget_unrepresentable rollback charging zero.
    from plastic.harness.policy import Decision
    from plastic.harness.signals import ChunkSignals

    for domain in ("text", "physics"):
        for rule in ("delta", "chunk"):
            for cap, kind, applies in ((0.14, "project", 1), (0.07, "project", 2), (0.10, "rollback", 2)):
                torch.manual_seed(813)
                cfg = ModelConfig(domain=domain, rule=rule, d_model=16, n_heads=2, n_layers=2, chunk=4, vocab_size=64)
                model = (PlasticLM(cfg) if domain == "text" else PlasticDynamics(cfg)).eval()
                r = TransactionRunner(
                    model, cfg, HarnessConfig(budget_chunk=cap, enable_stats=False, project_max_removed=1.0), device=CPU
                )
                if domain == "text":
                    r.feed_tokens([7])
                else:
                    r.feed_physics(torch.zeros(1, cfg.input_dim), torch.zeros(1, cfg.obs_dim))
                for a, z in zip(r.committed.layers, r.working.layers):
                    a.S.fill_(2 ** 20)
                    z.S.copy_(a.S)
                r.working.layers[0].S.reshape(-1)[0] += 1.0  # a single-unit proposed delta at index 0
                deltas = r.backend.state_delta(r.working, r.committed)
                gradients = [torch.zeros_like(d) for d in deltas]
                gradients[0].reshape(-1)[1] = 1.0  # orthogonal to the delta: projection keeps it whole
                sig = ChunkSignals(
                    pos_start=0, pos_end=1, n_tokens=1, chunk_loss=1.0, surprise_mean=0.0, surprise_max=0.0,
                    beta_mean=1.0, alpha_mean=1.0, write_norm_sum=1.0, delta_norm=1.0,
                )
                calls: list[int] = []
                base_apply = r.backend.apply_projected

                def counted(*a, _b=base_apply, **k):
                    calls.append(1)
                    return _b(*a, **k)

                r.backend.apply_projected = counted
                decision = r._apply(Decision("project", []), sig, deltas, gradients)
                ctx = (domain, rule, cap, decision.to_dict(), len(calls))
                assert decision.kind == kind and len(calls) == applies, ctx
                if applies == 2:
                    assert any("budget_recheck" in s for s in decision.reasons), ctx
                if kind == "rollback":
                    assert any("budget_unrepresentable" in s for s in decision.reasons), ctx
                assert r.budget_used <= cap * (1 + 1e-6), ctx


def test_reduced_signal_backend_carries_none_end_to_end():
    # A backend whose kernel exposes no per-token memory signals (like Qwen) must have
    # surprise/write-norm/decay carried as None end-to-end — never zero or NaN — while chunk_loss and
    # log_delta_norm stay real and a valid decision is still made. Driven with a reduced-signal
    # wrapper over PlasticBackend so it runs in the plain suite (no Qwen checkpoint needed).
    class _Reduced:
        def __init__(self, inner):
            self._inner = inner

        def signal_names(self):
            return ("chunk_loss", "log_delta_norm")

        def forward(self, items, state, *, freeze, beta_scale):
            out, new_state, _ = self._inner.forward(items, state, freeze=freeze, beta_scale=beta_scale)
            return out, new_state, []  # no per-token memory signals

        def __getattr__(self, name):
            return getattr(self._inner, name)

    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r.backend = _Reduced(r.backend)
    r.feed_tokens(_ids(8))
    rec = r.transactions[0]
    s = rec["signals"]
    for k in ("surprise_mean", "surprise_max", "beta_mean", "alpha_mean", "write_norm_sum", "log_write_norm"):
        assert s[k] is None, (k, s[k])  # unavailable memory signals are None, not 0.0 / NaN
    assert isinstance(s["chunk_loss"], float) and s["chunk_loss"] == s["chunk_loss"]  # real, not NaN
    assert isinstance(s["log_delta_norm"], float)
    # z is None for every unavailable signal (a None value yields a None z)
    assert s["z"]["surprise_mean"] is None and s["z"]["log_write_norm"] is None and s["z"]["fisher_update"] is None
    assert rec["decision"]["kind"] in ("commit", "rollback", "scale", "project", "readonly")
    # persistence round-trips a chunk that carried no memory signals
    r.load_state_dict(r.state_dict())


def test_nonfinite_candidate_is_rejected():
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r.feed_tokens(_ids(8))
    good = r.committed.clone()
    r.working.layers[0].S.fill_(float("nan"))
    r.feed_tokens(_ids(8, seed=2))
    rec = r.transactions[-1]
    assert rec["decision"]["kind"] == "rollback" and any("nonfinite_candidate" in x for x in rec["decision"]["reasons"])
    assert all(torch.isfinite(l.S).all() for l in r.committed.layers)
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


def test_calibrated_cusum_h_holds_benign_alarm_rate():
    # The CUSUM alarm threshold is a run-length: on a centered benign z-sequence the calibrated h
    # must keep the per-chunk alarm rate at or below target, and replaying at h reproduces it.
    from plastic.harness.calibrate import calibrated_cusum_h
    from plastic.harness.stats import Cusum

    g = torch.Generator().manual_seed(0)
    zc = torch.randn(3000, generator=g).tolist()  # centered, unit-scale benign statistic
    got = calibrated_cusum_h(zc, k=0.5, h_min=5.0, target_fpr=0.01)
    assert got is not None
    h, rate = got
    assert h >= 5.0 and rate <= 0.01 + 1e-9
    # The reported rate is the empirical alarm frequency at h — exactly alarms/n on a replay, not
    # a floored resolution convention.
    c = Cusum(0.5, h)
    alarms = sum(1 for z in zc if c.update(z))
    assert rate == alarms / len(zc)
    # Zero-alarm case: a stream that never alarms at h_min must report exactly 0.0, not a floor.
    for n in (16, 64, 256):
        zz = [0.0] * n  # every z below k=0.5, so s_hi/s_lo never grow -> no alarm at any h >= h_min
        hz, rz = calibrated_cusum_h(zz, k=0.5, h_min=5.0, target_fpr=0.01)
        cz = Cusum(0.5, hz)
        assert rz == sum(1 for z in zz if cz.update(z)) / n == 0.0
    # A biased signal (nonzero benign mean) needs a strictly higher threshold to hold the same
    # finite-stream rate — it would still ratchet on a longer stream. This is exactly the bias a
    # shared reset-every-N reference induces, and why the CUSUM gets its own continuous reference.
    zb = [z - 0.7 for z in zc]
    hb, _ = calibrated_cusum_h(zb, k=0.5, h_min=5.0, target_fpr=0.01)
    assert hb > h


def test_continuous_cusum_reference_is_gathered_and_centers_the_signal(tmp_path):
    # calibrate_from_runner must gather a continuous-regime CUSUM reference (not the reset-every-N
    # per-chunk reference) and the live runner must standardize the CUSUM signal against it, so a
    # benign continuous session does not ratchet the alarm.
    cfg, lm = _lm(chunk=8, layers=2)
    suite = _fake_suite(cfg)

    def stream():
        s = 0
        while True:  # infinite benign generator so the continuous pass can reach maturity
            yield _ids(32, seed=s)
            s += 1

    r = TransactionRunner(lm, cfg, log_only(HarnessConfig(enable_projection=False)), suite=suite, device=CPU)
    cal = calibrate_from_runner(r, stream(), n_chunks=128, model_signature="sig", target_fpr=0.02, reset_every=8)
    assert len(cal.cusum_reference) >= 16 and "cusum_h" in cal.thresholds and "cusum_h" in cal.achievable_fpr
    # survives serialization
    cal.save(str(tmp_path))
    back = Calibration.load(str(tmp_path))
    assert back.cusum_reference == cal.cusum_reference
    # The property that fixes the false latch: standardized against its own continuous reference,
    # the CUSUM signal on a fresh benign session is centered (median z ~ 0), so the two-sided CUSUM
    # does not ratchet. Against the reset-every-N per-chunk reference it drifted persistently to one
    # side, which is what drove the benign read-only latch.
    from plastic.harness.stats import robust_z

    r2 = TransactionRunner(lm, cfg, log_only(HarnessConfig(enable_projection=False)), calibration=back, suite=suite, device=CPU)
    for s in range(40):
        r2.feed_tokens(_ids(8, seed=7000 + s))
    zs = [robust_z(t["signals"]["log_delta_norm"], back.cusum_reference) for t in r2.transactions]
    zs = sorted(z for z in zs if z is not None)
    median = zs[len(zs) // 2]
    assert abs(median) < 0.6, median


def _force_next_cusum_alarm(r):
    # a finite cusum reference makes z_ld computable from the first chunk; pre-loading the
    # statistic past a zero threshold makes the next eligible chunk alarm deterministically.
    r.cusum.h = 0.0
    r.cusum.s_hi = 1e9


def test_alarm_freeze_modes():
    from plastic.harness.calibrate import Calibration

    cref = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # finite spread -> z_ld is never None

    # off: a CUSUM alarm rolls the chunk back but does not freeze the session
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False, freeze_on_alarm=False),
                          calibration=Calibration(cusum_reference=cref), device=CPU)
    _force_next_cusum_alarm(r)
    r.feed_tokens(_ids(8, seed=1))
    assert r.transactions[-1]["signals"]["cusum_alarm"] is True
    assert r.transactions[-1]["decision"]["kind"] == "rollback" and not r.read_only

    # latch (alarm_cooldown=0): freeze read-only until resume()
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False, freeze_on_alarm=True, alarm_cooldown=0),
                          calibration=Calibration(cusum_reference=cref), device=CPU)
    _force_next_cusum_alarm(r)
    r.feed_tokens(_ids(8, seed=1))
    assert r.read_only and r.read_only_reason == "cusum_alarm"
    for s in range(4):  # stays latched no matter how long the benign stream runs
        r.feed_tokens(_ids(8, seed=100 + s))
        assert r.read_only
    r.resume()
    assert not r.read_only

    # cooldown (alarm_cooldown=3): auto-resume after 3 quiet chunks
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False, freeze_on_alarm=True, alarm_cooldown=3),
                          calibration=Calibration(cusum_reference=cref), device=CPU)
    _force_next_cusum_alarm(r)
    r.feed_tokens(_ids(8, seed=1))
    r.cusum.h = 1e9  # no further alarms; let the cooldown run out
    assert r.read_only
    r.feed_tokens(_ids(8, seed=101))
    assert r.read_only  # 3 -> 2
    r.feed_tokens(_ids(8, seed=102))
    assert r.read_only  # 2 -> 1
    r.feed_tokens(_ids(8, seed=103))
    assert not r.read_only  # 1 -> 0, auto-resumed, this chunk learns again


def test_ineligible_chunk_is_readonly_not_rollback():
    # A chunk with no learning-eligible token (a read-only session, or generated tokens with
    # learn_from_generation off) proposes no write, so it must be reported as a read-only
    # observation — never a threshold rollback, and with no redundant frozen replay.
    from plastic.harness.calibrate import Calibration

    cfg, lm = _lm()
    # a threshold so low any chunk exceeds it: an *eligible* chunk would roll back on it
    cal = Calibration(thresholds={"chunk_loss": 0.01})
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), calibration=cal, device=CPU)

    # control: an eligible user chunk over the threshold rolls back (the threshold is live)
    r.feed_tokens(_ids(8, seed=1), source="user")
    assert r.transactions[-1]["decision"]["kind"] == "rollback"

    # a generated chunk (learn_from_generation defaults off) writes nothing -> read-only, not rollback
    r.feed_tokens(_ids(8, seed=2), source="model")
    rec = r.transactions[-1]
    assert rec["decision"]["kind"] == "readonly" and "learning_ineligible" in rec["decision"]["reasons"]
    assert rec["accepted"]["delta_norm"] < 1e-6  # no write accepted
    assert rec["signals"]["z"] == {n: None for n in rec["signals"]["z"]}  # ineligible: no z fed to history

    # a mixed chunk (some eligible tokens) is NOT short-circuited — it goes through the policy
    r2 = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), calibration=cal, device=CPU)
    r2.feed_tokens(_ids(4, seed=3), source="user")
    r2.feed_tokens(_ids(4, seed=4), source="model")  # completes one L=8 chunk, partly eligible
    assert r2.transactions[-1]["decision"]["kind"] != "readonly"


def test_read_only_chunks_do_not_feed_statistics():
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r.feed_tokens(_ids(8))
    r.read_only = True
    r.read_only_reason = "test"
    before = r.cusum.state()
    r.feed_tokens(_ids(8, seed=5))
    sig = r.transactions[1]["signals"]
    assert all(v is None for v in sig["z"].values()) and not sig["cusum_alarm"]
    assert r.cusum.state() == before


def test_nonfinite_in_any_carried_field_is_refused():
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, chunk=8, rule="chunk", vocab_size=64)
    lm = PlasticLM(cfg)
    for field_name in ("M", "conv_ssm", "conv_mem", "chunk.A", "chunk.Bv", "chunk.alpha_sum"):
        r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
        r.feed_tokens(_ids(3))
        layer = r.working.layers[0]
        target = layer.chunk if field_name.startswith("chunk.") else layer
        name = field_name.split(".")[-1]
        getattr(target, name).fill_(float("nan"))
        r.feed_tokens(_ids(5, seed=1))
        rec = r.transactions[-1]
        assert rec["decision"]["kind"] == "rollback" and any("nonfinite" in x for x in rec["decision"]["reasons"]), field_name
        assert r.backend.is_finite(r.committed), field_name


def test_chunk_cap_refusal_does_not_exhaust_a_large_session_budget():
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r.feed_tokens(_ids(32))
    r.hcfg = HarnessConfig(enable_projection=False, budget_chunk=1e-4, budget_session=1e6)
    r.feed_tokens(_ids(8, seed=9))
    rec = r.transactions[-1]
    assert rec["decision"]["kind"] in ("rollback", "scale")
    assert not r.read_only, r.read_only_reason


def test_generated_only_chunk_does_not_feed_cusum():
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    for s in range(8):
        r.feed_tokens(_ids(8, seed=s))
    before = r.cusum.state()
    r.feed_tokens(_ids(8, seed=100), source="model")
    sig = r.transactions[-1]["signals"]
    assert all(v is None for v in sig["z"].values()) and not sig["cusum_alarm"]
    assert r.cusum.state() == before and not r.read_only


def test_projection_rechecks_the_stored_delta_against_a_tiny_cap():
    cfg, lm = _lm()
    suite = _fake_suite(cfg)
    cap = 1e-6
    hcfg = HarnessConfig(project_eps_cos=-1.0, canary_delta_max=1e9, poison_delta_min=-1e9, enable_stats=False, budget_chunk=cap)
    r = TransactionRunner(lm, cfg, hcfg, suite=suite, device=CPU)
    r.feed_tokens(_ids(32))  # warm state so representation error matters
    before = r.committed.clone()
    r.feed_tokens(_ids(8, seed=3))
    rec = r.transactions[-1]
    moved = _delta(r.committed, before)
    if rec["decision"]["kind"] == "project":
        assert moved <= cap * (1 + 1e-6), moved
    else:
        assert rec["decision"]["kind"] == "rollback" and moved == 0.0


def test_log_only_never_enforces_budgets():
    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, log_only(HarnessConfig(budget_chunk=1e-9, budget_session=1e-9, enable_projection=False)), device=CPU)
    r.feed_tokens(_ids(16))
    kinds = [t["decision"]["kind"] for t in r.transactions]
    assert kinds == ["commit", "commit"] and not r.read_only
    assert _delta(r.committed, lm.init_state(1)) > 1e-3


def test_fork_state_starts_from_committed_without_pending():
    from plastic.harness.transaction import fork_state_dict

    cfg, lm = _lm()
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r.feed_tokens(_ids(11))  # one committed chunk, three pending tokens
    d = fork_state_dict(r.state_dict())
    child = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    child.load_state_dict(d)
    assert child.pos == 8 and not child.pending and child.n_transactions == 0
    assert _same_state(child.committed, r.committed) and _same_state(child.working, r.committed)
    assert child.budget_used == r.budget_used


def test_calibration_reports_achievable_fpr_and_cusum_threshold():
    from plastic.harness.calibrate import conformal_threshold

    cfg, lm = _lm()
    suite = _fake_suite(cfg)
    r = TransactionRunner(lm, cfg, log_only(HarnessConfig(enable_projection=False)), suite=suite, device=CPU)
    cal = calibrate_from_runner(r, [_ids(8, seed=s) for s in range(40)], n_chunks=40, model_signature="sig", target_fpr=0.01, reset_every=8)
    assert "cusum_h" in cal.thresholds and cal.thresholds["cusum_h"] >= HarnessConfig().cusum_h
    assert "canary_delta_poison" in cal.thresholds
    for name, a in cal.achievable_fpr.items():
        assert a >= 1.0 / 41 - 1e-9, (name, a)
    vals = [float(i) for i in range(1, 41)]
    thr, ach = conformal_threshold(vals, 0.001)
    assert thr == 40.0 and abs(ach - 1 / 41) < 1e-9
    thr, ach = conformal_threshold(vals, 0.1)
    assert thr == 37.0 and abs(ach - 0.1) < 1e-9
    lo, _ = conformal_threshold(vals, 0.1, side="lower")
    assert lo == 4.0


def test_transaction_record_reports_accepted_metrics_distinct_from_proposed():
    cfg, lm = _lm()
    suite = _fake_suite(cfg)
    # a forced rollback: the proposed delta is large, the accepted delta must be zero
    hcfg = HarnessConfig(canary_delta_max=-1e9, poison_delta_min=-1e9, enable_projection=False)
    r = TransactionRunner(lm, cfg, hcfg, suite=suite, device=CPU)
    r.feed_tokens(_ids(8))
    rec = r.transactions[0]
    assert rec["decision"]["kind"] == "rollback"
    assert rec["signals"]["delta_norm"] > 0.0            # proposed change was nonzero
    assert rec["accepted"]["delta_norm"] == 0.0          # nothing was committed
    assert rec["accepted"]["budget_charge"] == 0.0
    # the memory S is unchanged, but the SSM activation state advanced (the chunk was read,
    # not learned), so the rescored canary moves a little: much less than the proposed change
    assert abs(rec["accepted"]["canary_delta_coherence"]) < abs(rec["signals"]["canary_delta_coherence"]) + 1e-9
    assert abs(rec["accepted"]["canary_delta_coherence"]) < 1e-2


def test_accepted_delta_matches_scaled_commit():
    cfg, lm = _lm()
    zero = lm.init_state(1)
    hcfg = HarnessConfig(budget_chunk=1e-6, enable_projection=False)  # forces a scale
    r = TransactionRunner(lm, cfg, hcfg, device=CPU)
    r.feed_tokens(_ids(8))
    rec = r.transactions[0]
    assert rec["decision"]["kind"] == "scale"
    assert rec["signals"]["delta_norm"] > rec["accepted"]["delta_norm"]  # proposed > accepted
    assert abs(rec["accepted"]["delta_norm"] - _delta(r.committed, zero)) < 1e-9
    assert rec["accepted"]["delta_norm"] <= 1e-6 * (1 + 1e-6)
    assert abs(rec["accepted"]["budget_charge"] - rec["accepted"]["delta_norm"]) < 1e-9


def test_accepted_metrics_on_a_plain_commit():
    cfg, lm = _lm()
    zero = lm.init_state(1)
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r.feed_tokens(_ids(8))
    rec = r.transactions[0]
    assert rec["decision"]["kind"] == "commit"
    assert abs(rec["accepted"]["delta_norm"] - rec["signals"]["delta_norm"]) < 1e-6  # commit keeps the proposal
    assert abs(rec["accepted"]["delta_norm"] - _delta(r.committed, zero)) < 1e-9
    assert rec["accepted"]["budget_used"] == r.budget_used


def test_reset_keeps_the_calibrated_cusum_threshold():
    from plastic.harness.calibrate import Calibration

    cfg, lm = _lm()
    cal = Calibration(model_signature="s", thresholds={"cusum_h": 25.0})
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), calibration=cal, device=CPU)
    assert r.cusum.h == 25.0
    r.feed_tokens(_ids(8))
    r.reset()
    assert r.cusum.h == 25.0  # reset must not drop back to the default
    # without a calibration, reset uses the configured default
    r2 = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False, cusum_h=7.0), device=CPU)
    r2.reset()
    assert r2.cusum.h == 7.0
