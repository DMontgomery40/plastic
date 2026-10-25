"""Exp 3(a): property-based (Hypothesis) tests of the TransactionRunner's invariants.

Run:  .venv/bin/python -m pytest experiments/exp3_invariants/test_invariants.py -v -p no:cacheprovider

Positive invariants (claims of plastic/harness/transaction.py and README "The safety harness"):
  I1  a rollback restores the fast state (S, M, pending chunk statistics) bit for bit
  I2  committed update norms never exceed the budget (per-chunk cap and remaining session budget)
  I3  non-finite states are never committed
  I4  frozen / read-only chunks never change the fast weights
  I5  scale / project keep the accepted norm within bounds (<= cap, <= the proposal they shrink)
Documented limitation (negative invariant):
  N1  a rollback does NOT restore the activation / SSM state (h, conv buffers, last logits):
      the continuation diverges from a session that never saw the chunk
  N2  restoring the saved pre-chunk activation snapshot after a rollback makes the continuation
      bit-identical to the never-seen session

Most properties run on tiny randomly initialised PlasticLMs (delta and chunk rules) so Hypothesis
can explore many cases; the `real_model` tests repeat I1/N1/N2 on the trained lm_small_cpu
checkpoint from exp2 (skipped if artifacts/models/lm_small_cpu is absent). Hypothesis is run
derandomized (fixed seed) so the reported pass/fail is reproducible.
"""
from __future__ import annotations

import contextlib
import functools
import math
import os

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import plastic.harness.transaction as txn
from plastic.backends.plastic import PlasticBackend
from plastic.config import ModelConfig
from plastic.harness.canary import CanarySuite
from plastic.harness.config import HarnessConfig
from plastic.harness.policy import Decision
from plastic.harness.projection import project_delta
from plastic.harness.signals import delta_norms
from plastic.harness.transaction import TransactionRunner
from plastic.model.lm import PlasticLM

torch.set_num_threads(1)
CPU = torch.device("cpu")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REAL_DIR = os.path.join(ROOT, "artifacts", "models", "lm_small_cpu")
REAL_DATA = os.path.join(ROOT, "artifacts", "data", "wikitext", "test.bin")

SETTINGS = settings(
    max_examples=int(os.environ.get("EXP3_MAX_EXAMPLES", "60")),
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    print_blob=True,
)
REAL_SETTINGS = settings(SETTINGS, max_examples=int(os.environ.get("EXP3_REAL_EXAMPLES", "8")))
TOL = 1e-6  # the runner's own cap tolerance: actual > cap * (1 + 1e-6) is a violation

documented_limitation = pytest.mark.documented_limitation


# ---------------------------------------------------------------------------- models / helpers
@functools.lru_cache(maxsize=None)
def tiny(rule: str = "delta", chunk: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=chunk, vocab_size=64, rule=rule)
    lm = PlasticLM(cfg).eval()
    return cfg, lm


@functools.lru_cache(maxsize=None)
def tiny_suite(vocab: int = 64) -> CanarySuite:
    g = torch.Generator().manual_seed(3)
    coh = [torch.randint(3, vocab, (16,), generator=g).tolist() for _ in range(2)]
    poi = [torch.randint(3, vocab, (16,), generator=g).tolist(), [7] * 16]
    return CanarySuite(domain="text", coherence=coh, poison=poi)


@functools.lru_cache(maxsize=None)
def real():
    if not os.path.exists(os.path.join(REAL_DIR, "checkpoint.pt")):
        pytest.skip("lm_small_cpu checkpoint not present")
    from plastic.store import ArtifactStore

    store = ArtifactStore(os.path.join(ROOT, "artifacts"))
    cfg, model, _ = store.load_checkpoint("lm_small_cpu", "cpu")
    model.eval()
    return cfg, model, np.memmap(REAL_DATA, dtype="<u2", mode="r")


def fast(state) -> list[torch.Tensor]:
    """The fast weights proper: S and (chunk rule) momentum M. memory.py's freeze contract covers these."""
    out: list[torch.Tensor] = []
    for l in state.layers:
        out.append(l.S)
        if l.M is not None:
            out.append(l.M)
    return [t.detach().clone() for t in out]


def pending_stats(state) -> list[torch.Tensor]:
    """Chunk rule only: the pending (unfinished) chunk's sufficient statistics A, Bv, alpha_sum, count."""
    out: list[torch.Tensor] = []
    for l in state.layers:
        if l.chunk is not None:
            out += [l.chunk.A, l.chunk.Bv, l.chunk.alpha_sum, torch.tensor(float(l.chunk.count))]
    return [t.detach().clone() for t in out]


def fast_strict(state) -> list[torch.Tensor]:
    return fast(state) + pending_stats(state)


def acts(state) -> list[torch.Tensor]:
    out = []
    for l in state.layers:
        out.append(l.h)
        for t in (l.conv_ssm, l.conv_mem):
            if t is not None:
                out.append(t)
    return [t.detach().clone() for t in out]


def bit_equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b))


def s_norm(a, b) -> float:
    return delta_norms([x.S - y.S for x, y in zip(a.layers, b.layers)])[0]


@contextlib.contextmanager
def forced(kind: str, scale: float = 1.0):
    """Replace the policy (the name imported into plastic.harness.transaction) so the runner's real
    apply paths (_reject_frozen / _accept / scale / project) do the work."""
    orig = txn.decide
    txn.decide = lambda sig, hc, th, read_only: (
        Decision("readonly", ["session_read_only"]) if read_only else Decision(kind, [f"forced_{kind}"], scale=scale)
    )
    try:
        yield
    finally:
        txn.decide = orig


class AuditRunner(TransactionRunner):
    """TransactionRunner that records the committed state, cap and budget around every transaction."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.audit: list[dict] = []

    def _transact(self):
        pre = self.backend.clone(self.committed)
        cap = self._cap()
        ro_before = self.read_only
        budget_before = self.budget_used
        rec = super()._transact()
        self.audit.append(dict(pre=pre, post=self.backend.clone(self.committed), cap=cap, ro_before=ro_before,
                               budget_before=budget_before, budget_after=self.budget_used, rec=rec,
                               committed_finite=self.backend.is_finite(self.committed)))
        return rec


# ---------------------------------------------------------------------------- strategies
V = 64


@st.composite
def token_chunk(draw, n_min=1, n_max=24):
    """Benign-looking random ids, single-token runs, short repeated phrases, or the extreme ids."""
    n = draw(st.integers(n_min, n_max))
    kind = draw(st.sampled_from(["random", "repeat_token", "repeat_phrase", "extremes"]))
    if kind == "random":
        return draw(st.lists(st.integers(0, V - 1), min_size=n, max_size=n))
    if kind == "repeat_token":
        return [draw(st.integers(0, V - 1))] * n
    if kind == "repeat_phrase":
        ph = draw(st.lists(st.integers(0, V - 1), min_size=1, max_size=4))
        return (ph * n)[:n]
    return draw(st.lists(st.sampled_from([0, 1, 2, V - 1]), min_size=n, max_size=n))


@st.composite
def harness_cfg(draw, allow_log_only=False):
    budget_chunk = draw(st.one_of(st.none(), st.floats(1e-4, 5.0)))
    budget_session = draw(st.one_of(st.none(), st.floats(0.0, 10.0)))
    return HarnessConfig(
        enable_projection=draw(st.booleans()),
        enable_stats=draw(st.booleans()),
        enable_rollback=draw(st.booleans()),
        canary_delta_max=draw(st.sampled_from([-1e9, -0.01, 0.0, 0.01, 0.5, 1e9])),
        poison_delta_min=draw(st.sampled_from([-1e9, -0.5, -0.01, 0.0, 1e9])),
        z_rollback=draw(st.sampled_from([1.0, 3.0, 6.0])),
        z_scale=draw(st.sampled_from([0.5, 1.0, 3.0])),
        scale_factor=draw(st.sampled_from([0.01, 0.25, 0.9])),
        budget_chunk=budget_chunk,
        budget_session=budget_session,
        project_max_removed=draw(st.sampled_from([0.1, 0.5, 1.0])),
        history_window=8,
        log_only=draw(st.booleans()) if allow_log_only else False,
        learn_from_generation=draw(st.booleans()),
    )


@st.composite
def stream(draw, max_items=6):
    items = draw(st.lists(st.tuples(token_chunk(1, 20), st.sampled_from(["user", "user", "user", "model"])),
                          min_size=1, max_size=max_items))
    return items, draw(st.booleans())  # (segments, flush at end)


def run_stream(r: TransactionRunner, items, flush: bool):
    for toks, src in items:
        r.feed_tokens(toks, source=src)
    if flush:
        r.flush()


def describe(a: dict) -> str:
    rec = a["rec"]
    return f"tx{rec['index']} requested={rec['requested']['kind']} applied={rec['decision']['kind']} reasons={rec['decision']['reasons'][:4]}"


# ============================================================================ I1 rollback restores fast state
@SETTINGS
@given(rule=st.sampled_from(["delta", "chunk"]), prefix=st.lists(token_chunk(1, 20), max_size=3),
       x=token_chunk(1, 40), flush=st.booleans())
def test_I1_forced_rollback_restores_fast_state_bit_exact(rule, prefix, x, flush):
    cfg, lm = tiny(rule)
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), suite=tiny_suite(), device=CPU)
    for p in prefix:
        r.feed_tokens(p)
    before = fast(r.committed)  # the state every transaction that follows started from
    with forced("rollback"):
        r.feed_tokens(x)
        if flush:
            r.flush()
    after = fast(r.committed)
    assert bit_equal(before, after), "fast state changed across forced rollbacks"
    if not r.pending:
        assert bit_equal(fast(r.working), after)


@SETTINGS
@given(rule=st.sampled_from(["delta", "chunk"]), prefix=st.lists(token_chunk(1, 20), max_size=3),
       x=token_chunk(1, 40), flush=st.booleans())
def test_I1_strict_forced_rollback_restores_fast_state_incl_pending_chunk_stats(rule, prefix, x, flush):
    """Stricter reading of I1: also the chunk rule's pending-chunk statistics (they turn into an S write at the
    next model-chunk boundary, so they are part of the learned-at-inference state)."""
    cfg, lm = tiny(rule)
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), suite=tiny_suite(), device=CPU)
    for p in prefix:
        r.feed_tokens(p)
    before = fast_strict(r.committed)
    with forced("rollback"):
        r.feed_tokens(x)
        if flush:
            r.flush()
    assert bit_equal(before, fast_strict(r.committed)), "fast state incl. pending chunk stats changed across rollback"


@SETTINGS
@given(rule=st.sampled_from(["delta", "chunk"]), hc=harness_cfg(), s=stream())
def test_I1_every_natural_rollback_restores_fast_state(rule, hc, s):
    """Whatever path produced it (policy rollback, failed recheck, budget, non-finite), an applied rollback
    or read-only decision leaves the committed fast state bit-identical to the transaction's start."""
    cfg, lm = tiny(rule)
    r = AuditRunner(lm, cfg, hc, suite=tiny_suite(), device=CPU)
    run_stream(r, *s)
    for a in r.audit:
        if a["rec"]["decision"]["kind"] in ("rollback", "readonly"):
            assert bit_equal(fast(a["pre"]), fast(a["post"])), describe(a)


# ============================================================================ I2 budgets
@SETTINGS
@given(rule=st.sampled_from(["delta", "chunk"]), hc=harness_cfg(), s=stream(8))
def test_I2_committed_update_never_exceeds_budget(rule, hc, s):
    assume(hc.budget_chunk is not None or hc.budget_session is not None)
    cfg, lm = tiny(rule)
    r = AuditRunner(lm, cfg, hc, suite=tiny_suite(), device=CPU)
    run_stream(r, *s)
    total = 0.0
    for a in r.audit:
        acc = s_norm(a["post"], a["pre"])  # independently measured committed S change
        total += acc
        if a["cap"] is not None:
            assert acc <= a["cap"] * (1 + TOL) + 1e-12, f"{describe(a)} accepted={acc:.6g} cap={a['cap']:.6g}"
        if hc.budget_chunk is not None:
            assert acc <= hc.budget_chunk * (1 + TOL) + 1e-12, f"{describe(a)} accepted={acc:.6g} chunk_cap={hc.budget_chunk}"
    if hc.budget_session is not None:
        assert r.budget_used <= hc.budget_session * (1 + TOL) + 1e-12
        assert total <= hc.budget_session * (1 + 1e-5) + 1e-9, f"sum of accepted={total:.6g} > session={hc.budget_session}"


# ============================================================================ I3 non-finite never committed
class CorruptingBackend(PlasticBackend):
    """Injects NaN/Inf into the state a forward pass returns, on a drawn schedule of calls."""

    def __init__(self, *a, schedule: dict[int, tuple[str, float, bool]], **k):
        super().__init__(*a, **k)
        self.calls = 0
        self.schedule = schedule  # call index -> (tensor name, value, only_when_writing)

    def forward(self, items, state, *, freeze, beta_scale):
        out, new_state, sig = super().forward(items, state, freeze=freeze, beta_scale=beta_scale)
        spec = self.schedule.get(self.calls)
        self.calls += 1
        if spec is not None:
            name, val, only_writing = spec
            if not (only_writing and freeze):
                lay = new_state.layers[0]
                t = getattr(lay, name)
                if t is not None:
                    t = t.clone()
                    t.view(-1)[0] = val
                    setattr(lay, name, t)
        return out, new_state, sig


@SETTINGS
@given(rule=st.sampled_from(["delta", "chunk"]), hc=harness_cfg(allow_log_only=True), s=stream(6),
       schedule=st.dictionaries(st.integers(0, 30),
                                st.tuples(st.sampled_from(["S", "h", "M", "conv_ssm", "conv_mem"]),
                                          st.sampled_from([math.nan, math.inf, -math.inf]), st.booleans()),
                                min_size=1, max_size=6))
def test_I3_nonfinite_state_never_committed(rule, hc, s, schedule):
    cfg, lm = tiny(rule)
    be = CorruptingBackend(lm, cfg, device=CPU, schedule=schedule)
    r = AuditRunner(lm, cfg, hc, suite=tiny_suite(), device=CPU, backend=be)
    run_stream(r, *s)
    for a in r.audit:
        assert a["committed_finite"], f"non-finite committed at {describe(a)}"
    assert be.is_finite(r.committed)


# ============================================================================ I4 frozen / read-only
@SETTINGS
@given(rule=st.sampled_from(["delta", "chunk"]), prefix=st.lists(token_chunk(1, 20), max_size=2),
       chunks=st.lists(token_chunk(1, 30), min_size=1, max_size=4),
       mode=st.sampled_from(["session_read_only", "generation", "budget_spent"]), flush=st.booleans())
def test_I4_frozen_and_readonly_never_change_fast_weights(rule, prefix, chunks, mode, flush):
    cfg, lm = tiny(rule)
    hc = HarnessConfig(enable_projection=False, budget_session=(1e-9 if mode == "budget_spent" else None))
    r = AuditRunner(lm, cfg, hc, suite=tiny_suite(), device=CPU)
    if mode != "budget_spent":
        for p in prefix:
            r.feed_tokens(p)
        r.flush()
    else:
        # spend the session budget: the first eligible chunk exhausts it and the session latches read-only
        while not r.read_only:
            r.feed_tokens(list(range(3, 3 + cfg.chunk)))
    if mode == "session_read_only":
        r.read_only, r.read_only_reason = True, "test"
    before = fast(r.committed)
    src = "model" if mode == "generation" else "user"
    for c in chunks:
        r.feed_tokens(c, source=src)
    if flush:
        r.flush()
    assert bit_equal(before, fast(r.committed))
    assert bit_equal(before, fast(r.working))


@SETTINGS
@given(rule=st.sampled_from(["delta", "chunk"]), prefix=token_chunk(0, 20), x=token_chunk(1, 30),
       beta_scale=st.sampled_from([0.0, 0.5, 1.0, 4.0]))
def test_I4_backend_freeze_is_bit_exact(rule, prefix, x, beta_scale):
    cfg, lm = tiny(rule)
    be = PlasticBackend(lm, cfg, device=CPU)
    s0 = be.init_state()
    if prefix:
        _, s0, _ = be.forward(prefix, s0, freeze=False, beta_scale=1.0)
    _, s1, _ = be.forward(x, s0.clone(), freeze=True, beta_scale=beta_scale)
    assert bit_equal(fast(s0), fast(s1))


# ============================================================================ I5 scale / project bounds
@SETTINGS
@given(rule=st.sampled_from(["delta", "chunk"]), kind=st.sampled_from(["scale", "project"]),
       scale=st.floats(1e-3, 1.0), budget_chunk=st.one_of(st.none(), st.floats(1e-4, 2.0)),
       budget_session=st.one_of(st.none(), st.floats(1e-3, 5.0)),
       prefix=st.lists(token_chunk(1, 16), max_size=2), chunks=st.lists(token_chunk(1, 16), min_size=1, max_size=4))
def test_I5_scale_and_project_stay_within_bounds(rule, kind, scale, budget_chunk, budget_session, prefix, chunks):
    cfg, lm = tiny(rule)
    hc = HarnessConfig(budget_chunk=budget_chunk, budget_session=budget_session, project_max_removed=1.0,
                       canary_delta_max=1e9, poison_delta_min=-1e9)
    r = AuditRunner(lm, cfg, hc, suite=tiny_suite(), device=CPU)
    for p in prefix:
        r.feed_tokens(p)
    n0 = len(r.audit)
    with forced(kind, scale):
        for c in chunks:
            r.feed_tokens(c)
        r.flush()
    for a in r.audit[n0:]:
        applied = a["rec"]["decision"]["kind"]
        acc = s_norm(a["post"], a["pre"])
        proposal = a["rec"]["signals"]["delta_norm"]
        if a["cap"] is not None:
            assert acc <= a["cap"] * (1 + TOL) + 1e-12, f"{describe(a)} accepted={acc:.6g} cap={a['cap']:.6g}"
        if applied in ("scale", "project"):
            # both actions are documented as shrinking the proposed update
            assert acc <= proposal * (1 + 1e-5) + 1e-12, f"{describe(a)} accepted={acc:.6g} > proposal={proposal:.6g}"
        if applied in ("rollback", "readonly"):
            assert acc == 0.0


@SETTINGS
@given(n=st.integers(1, 64), seed=st.integers(0, 2**31 - 1), eps_dot=st.sampled_from([0.0, 1e-3, 1.0]),
       eps_cos=st.sampled_from([0.0, 0.02, 0.5]), gscale=st.sampled_from([1e-6, 1.0, 1e6]))
def test_I5_project_delta_is_a_contraction_onto_the_half_space(n, seed, eps_dot, eps_cos, gscale):
    g = torch.Generator().manual_seed(seed)
    d = [torch.randn(1, 2, n, generator=g), torch.randn(3, generator=g)]
    gr = [gscale * torch.randn(1, 2, n, generator=g), gscale * torch.randn(3, generator=g)]
    out, ps = project_delta(d, gr, eps_dot=eps_dot, eps_cos=eps_cos)
    dn, on = delta_norms(d)[0], delta_norms(out)[0]
    assert on <= dn * (1 + 1e-5) + 1e-12
    if ps.applied:
        assert ps.dot_after <= ps.allowed + 1e-4 * max(1.0, abs(ps.dot_before))


# ============================================================================ N1 / N2 activations
def _session(cfg, lm, P, X, how):
    """P is fed normally, X (if any) under a forced decision; returns (runner, pre-X activation snapshot)."""
    r = TransactionRunner(lm, cfg, HarnessConfig(enable_projection=False), device=CPU)
    r.feed_tokens(P)
    r.flush()
    snap = (r.backend.clone(r.committed), None if r._last_logits is None else r._last_logits.clone())
    if how != "never":
        with forced("rollback"):
            r.feed_tokens(X)
            r.flush()
    if how == "rewind":
        st_, ll = snap
        for state in (r.committed, r.working):
            for lay, s in zip(state.layers, st_.layers):
                lay.h = s.h.clone()
                lay.conv_ssm = None if s.conv_ssm is None else s.conv_ssm.clone()
                lay.conv_mem = None if s.conv_mem is None else s.conv_mem.clone()
                if s.chunk is not None:  # chunk rule: the pending-chunk statistics are transient state too
                    lay.chunk = s.chunk.map(lambda t: t.clone())
        r._last_logits = None if ll is None else ll.clone()
    return r


def _continuation(r, C):
    """Feed C through the runner with every chunk committed; return per-token logits and final fast state."""
    outs = []
    with forced("commit"):
        for t in C:
            outs.append(r.feed_tokens([t]).clone())
        r.flush()
    return torch.stack(outs), fast(r.committed)


@documented_limitation
@SETTINGS
@given(rule=st.sampled_from(["delta", "chunk"]), P=token_chunk(8, 24), X=token_chunk(1, 24), C=token_chunk(1, 16))
def test_N1_rollback_does_not_restore_activations_DOCUMENTED_LIMITATION(rule, P, X, C):
    """Documented limitation (module docstring 'Outputs already produced inside a chunk are not retracted';
    exp2): rollback re-reads the chunk frozen, so h / conv buffers / last logits carry it forward.
    This test PASSES when the divergence is present."""
    cfg, lm = tiny(rule)
    never = _session(cfg, lm, P, X, "never")
    rb = _session(cfg, lm, P, X, "rollback")
    assert bit_equal(fast(never.committed), fast(rb.committed))  # memory is restored...
    assert not bit_equal(acts(never.committed), acts(rb.committed))  # ...activations are not
    ln, _ = _continuation(never, C)
    lr, _ = _continuation(rb, C)
    assert not torch.equal(ln, lr), "continuation after rollback identical to never-seen (limitation not observed)"


@pytest.mark.xfail(strict=True, reason="documented limitation: rollback restores memory, not activation/SSM state")
@settings(SETTINGS, max_examples=5)
@given(P=token_chunk(8, 24), X=token_chunk(1, 24), C=token_chunk(1, 16))
def test_N1_xfail_claim_rollback_equals_never_seen(P, X, C):
    cfg, lm = tiny("delta")
    ln, fn = _continuation(_session(cfg, lm, P, X, "never"), C)
    lr, fr = _continuation(_session(cfg, lm, P, X, "rollback"), C)
    assert torch.equal(ln, lr) and bit_equal(fn, fr)


@SETTINGS
@given(rule=st.sampled_from(["delta", "chunk"]), P=token_chunk(1, 24), X=token_chunk(1, 24), C=token_chunk(1, 24))
def test_N2_rollback_plus_activation_restore_equals_never_seen(rule, P, X, C):
    cfg, lm = tiny(rule)
    ln, fn = _continuation(_session(cfg, lm, P, X, "never"), C)
    lw, fw = _continuation(_session(cfg, lm, P, X, "rewind"), C)
    assert torch.equal(ln, lw), f"max |dlogit|={float((ln - lw).abs().max()):.3g}"
    assert bit_equal(fn, fw)


# ============================================================================ real checkpoint (lm_small_cpu)
def _real_clean(data, start_frac, n):
    s = int(start_frac * (len(data) - n - 1))
    return [int(t) for t in data[s:s + n]]


@pytest.mark.real_model
@REAL_SETTINGS
@given(p=st.floats(0, 1), x=st.floats(0, 1), c=st.floats(0, 1), kind=st.sampled_from(["clean", "random", "repeat"]),
       nx=st.integers(1, 2))
def test_real_I1_N1_N2(p, x, c, kind, nx):
    cfg, model, data = real()
    L = cfg.chunk
    P, C = _real_clean(data, p, 2 * L), _real_clean(data, c, L)
    if kind == "clean":
        X = _real_clean(data, x, nx * L)
    elif kind == "random":
        X = torch.randint(3, cfg.vocab_size, (nx * L,), generator=torch.Generator().manual_seed(int(x * 1e6))).tolist()
    else:
        X = [3 + int(x * (cfg.vocab_size - 4))] * (nx * L)
    never = _session(cfg, model, P, X, "never")
    rb = _session(cfg, model, P, X, "rollback")
    rw = _session(cfg, model, P, X, "rewind")
    assert bit_equal(fast(never.committed), fast(rb.committed))  # I1
    assert not bit_equal(acts(never.committed), acts(rb.committed))  # N1 (documented limitation)
    ln, fn = _continuation(never, C)
    lr, _ = _continuation(rb, C)
    lw, fw = _continuation(rw, C)
    assert not torch.equal(ln, lr)  # N1
    assert torch.equal(ln, lw) and bit_equal(fn, fw)  # N2
