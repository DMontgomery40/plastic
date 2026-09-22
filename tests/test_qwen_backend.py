"""Contract tests for the Qwen pretrained backend foundation.

Skipped unless Transformers >= 5.17 and the local Qwen3.5 checkpoint are both available (the
checkpoint is ~1.75GB and not part of the repo). Point QWEN_CHECKPOINT at the checkpoint dir,
or the default Astra runtime path is used. These pin the two invariants the harness integration
rests on: native-logit parity and snapshot/replay identity, plus freeze semantics.
"""

import os

import pytest
import torch

CKPT = os.environ.get("QWEN_CHECKPOINT", "artifacts/astra/qwen-runtime-20260922/checkpoint")


def _available() -> bool:
    if not os.path.exists(os.path.join(CKPT, "config.json")):
        return False
    try:
        import transformers  # noqa: F401
        from transformers import Qwen3_5ForCausalLM  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _available(), reason="Qwen checkpoint or transformers>=5.17 not available")


@pytest.fixture(scope="module")
def backend():
    from plastic.backends.qwen import QwenBackend

    return QwenBackend.load(CKPT)


def test_native_logit_parity_and_snapshot(backend):
    ids = backend.encode("The history of computing includes mechanical calculators and modern electronic machines built over decades.")
    n = len(ids)
    k = n // 2
    assert n >= 4 and k >= 1
    # chunked-with-cache logits equal a single full-sequence pass (the harness never changes output)
    full = backend.logits_full(ids)
    st = backend.init_state()
    _, st = backend.process(ids[:k], st)
    l2, st = backend.process(ids[k:], st)
    assert (full[k:] - l2).abs().max().item() < 1e-3
    assert st.position == n

    # a snapshot reproduces continuation logits exactly and does not alias the original
    st2 = backend.init_state()
    _, st2 = backend.process(ids[:k], st2)
    snap = st2.clone()
    la, st2 = backend.process(ids[k:], st2)
    lb, snap = backend.process(ids[k:], snap)
    assert torch.equal(la, lb) and st2.position == n


def test_freeze_is_a_genuine_no_write(backend):
    import plastic.backends.qwen as q

    ids = backend.encode("A short sentence for the freeze check here, long enough to span two chunks cleanly.")
    n = len(ids)
    k = n // 2
    st = backend.init_state()
    _, st = backend.process(ids[:k], st)
    pre = q._recurrent_snapshot(st.cache)
    conv_pre = [{i: c.clone() for i, c in getattr(l, "conv_states", {}).items()} for l in st.cache.layers]
    _, st = backend.process(ids[k:], st, freeze=True)
    post = q._recurrent_snapshot(st.cache)
    # recurrent memory is exactly unchanged (a real freeze, not a snapshot-restore of the final tensor)
    assert max((pre[l][i] - post[l][i]).abs().max().item() for l in range(len(pre)) for i in pre[l]) == 0.0
    # conv/attention/position still advance (activation progresses)
    conv_change = max(
        (conv_pre[l][i] - c).abs().max().item()
        for l, layer in enumerate(st.cache.layers)
        for i, c in getattr(layer, "conv_states", {}).items()
    )
    assert conv_change > 0.0 and st.position == n


def test_freeze_on_empty_cache_writes_nothing(backend):
    # regression for the snapshot-restore bug: with no prior cache there is nothing to restore,
    # so a genuine freeze must still produce a zero recurrent state (not a 13.7-magnitude write).
    ids = backend.encode("Freeze from an empty state must not write.")
    st = backend.init_state()
    _, st = backend.process(ids[:6], st, freeze=True)
    recmax = max(s.abs().max().item() for l in st.cache.layers for s in getattr(l, "recurrent_states", {}).values())
    assert recmax == 0.0


def test_frozen_recurrent_grad_is_finite(backend):
    ids = backend.encode("The frozen canary gradient must reach every recurrent leaf.")
    n = len(ids)
    k = max(1, n // 2)
    st = backend.init_state()
    _, st = backend.process(ids[:k], st)
    grads = backend.recurrent_grad(st, ids[k : k + 3], ids[k + 1 : k + 4])
    assert grads and all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)


def test_fresh_state_is_position_zero_and_gradient_works(backend):
    from transformers import DynamicCache

    from plastic.backends.qwen import QwenState

    # position-zero initial state: cursor 0, correctly-shaped zero recurrent leaves, no synthetic BOS
    st = backend.init_state()
    assert st.position == 0
    leaves = st.recurrent_leaves()
    assert len(leaves) == 18 and all(int(torch.count_nonzero(s)) == 0 for s in leaves)

    # forwarding from it is identical to a truly-empty cache (the harness never changes native output)
    ids = backend.encode("Parity between the initialized zero state and an empty cache.")
    ya, _ = backend.process(ids[:4], backend.init_state())
    yr, _ = backend.process(ids[:4], QwenState(DynamicCache(config=backend.config)))
    assert (ya - yr).abs().max().item() < 1e-3

    # the FIRST-chunk canary gradient is defined (previously raised "inputs cannot be empty")
    grads = backend.recurrent_grad(backend.init_state(), ids[:4], ids[1:5])
    assert len(grads) == 18 and all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)


def test_recurrent_is_float32_under_bfloat16():
    # native Qwen keeps the GDN recurrent accumulator at float32 even with bfloat16 weights/conv/KV;
    # allocating it at the model dtype would quantize it every update and diverge after chunk 1
    # (ASTRA-057). Conv/KV stay at the model dtype. (Own bf16 load; not the float32 module fixture.)
    from plastic.backends.qwen import QwenBackend

    be = QwenBackend.load(CKPT, dtype=torch.bfloat16)
    st = be.init_state()
    assert all(s.dtype == torch.float32 for s in st.recurrent_leaves())
    ids = be.encode("A multi-chunk continuation under bfloat16 weights, long enough for three chunks.")
    for i in range(0, min(12, len(ids)), 4):
        _, st = be.process(ids[i : i + 4], st)
    assert all(s.dtype == torch.float32 for s in st.recurrent_leaves())  # stays float32 across chunks
    conv = [c for l in st.cache.layers for c in getattr(l, "conv_states", {}).values()]
    assert conv and all(c.dtype == torch.bfloat16 for c in conv)  # conv follows the model dtype


def test_reduced_signal_names_and_generation_writes(backend):
    # Qwen gates on the reduced set (chunk NLL + recurrent-state change); the runner carries
    # surprise/write_norm/fisher as None. Generation writes on Qwen, so BOTH sources are eligible.
    assert backend.signal_names() == ("chunk_loss", "log_delta_norm")
    assert backend.writes_for_source("user") is True and backend.writes_for_source("model") is True


def test_state_delta_over_recurrent_leaves(backend):
    ids = backend.encode("A sentence with enough tokens to span two distinct chunks for the delta.")
    n = len(ids)
    k = n // 2
    a = backend.init_state()
    _, a = backend.process(ids[:k], a)
    b = backend.clone(a)
    # a snapshot with no further advance has an exactly-zero delta across all 18 recurrent leaves
    z = backend.state_delta(a, b)
    assert len(z) == 18 and all(int(torch.count_nonzero(t)) == 0 for t in z)
    # advancing one copy gives a nonzero, finite, correctly-shaped per-leaf delta (KV/conv excluded)
    _, a = backend.process(ids[k:], a)
    d = backend.state_delta(a, b)
    assert len(d) == 18 and all(tuple(t.shape) == (1, 16, 128, 128) and torch.isfinite(t).all() for t in d)
    assert any(t.abs().max().item() > 0 for t in d)


def test_is_finite_covers_recurrent_conv_and_kv(backend):
    ids = backend.encode("Finiteness must cover recurrent memory, conv history, and attention KV.")
    st = backend.init_state()
    _, st = backend.process(ids[:8], st)
    assert backend.is_finite(st)
    # a NaN in any persisted category is detected: recurrent, conv (linear layers), KV (attn layers)
    st_r = backend.clone(st)
    next(iter(st_r.cache.layers[0].recurrent_states.values()))[0, 0, 0, 0] = float("nan")
    assert not backend.is_finite(st_r)
    st_c = backend.clone(st)
    next(iter(st_c.cache.layers[0].conv_states.values()))[0, 0, 0] = float("nan")
    assert not backend.is_finite(st_c)
    st_k = backend.clone(st)
    injected = False
    for layer in st_k.cache.layers:
        kv = getattr(layer, "keys", None)
        if kv is not None:
            kv[0, 0, 0, 0] = float("nan")
            injected = True
            break
    assert injected and not backend.is_finite(st_k)


def test_is_finite_does_not_raise_on_uninitialized_cache(backend):
    # ASTRA-063: a genuinely uninitialized cache carries {0: None} recurrent/conv slots; is_finite is
    # a finiteness check, not a structure check, and must not raise TypeError on None (structural
    # validation of a persisted state belongs to load_state_dict).
    from transformers import DynamicCache

    from plastic.backends.qwen import QwenState

    st = QwenState(DynamicCache(config=backend.config))
    assert backend.is_finite(st) is True  # nothing non-finite is present


def test_state_dict_round_trips_through_save_load(backend):
    import io

    ids = backend.encode("Enough tokens to populate recurrent, conv, and KV before the persistence check.")
    st = backend.init_state()
    _, st = backend.process(ids[:8], st)
    cont = ids[8:12]
    ref, _ = backend.process(cont, backend.clone(st))

    # the state_dict survives a real torch.save/load boundary (as the session store persists it)
    sd = backend.state_dict(st)
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    reloaded = torch.load(buf, weights_only=False)
    back = backend.load_state_dict(reloaded)

    assert back.position == st.position
    assert all(torch.equal(a, b) for a, b in zip(st.recurrent_leaves(), back.recurrent_leaves()))
    assert all(b.dtype == torch.float32 for b in back.recurrent_leaves())  # float32 accumulator preserved
    # continuation from the loaded state is byte-identical to continuation from the original
    got, _ = backend.process(cont, backend.clone(back))
    assert (ref - got).abs().max().item() == 0.0


def test_load_state_dict_independence_and_fork(backend):
    # ASTRA-066 P1: load must return an independent live state, never alias/mutate the saved payload,
    # or a fork (whose working reuses the committed payload) loses its rollback snapshot.
    from plastic.harness.transaction import fork_state_dict

    ids = backend.encode("A state for the fork independence and repeated-load persistence checks here.")
    st = backend.init_state()
    _, st = backend.process(ids[:8], st)
    sd = backend.state_dict(st)

    # two loads of the SAME payload are independent: advancing one does not touch the other
    a = backend.load_state_dict(sd)
    b = backend.load_state_dict(sd)
    pre_b = [t.clone() for t in b.recurrent_leaves()]
    _, a = backend.process(ids[8:10], a)
    assert a.position == st.position + 2 and b.position == st.position
    assert all(torch.equal(x, y) for x, y in zip(b.recurrent_leaves(), pre_b))
    # the saved payload itself is not consumed/mutated by a load
    c = backend.load_state_dict(sd)
    assert c.position == st.position

    # the runner fork helper reuses the committed payload for working; loading it must give
    # independent committed/working so a working forward does not advance committed
    forked = fork_state_dict({"committed": sd, "working": sd, "anchor": sd})
    committed = backend.load_state_dict(forked["committed"])
    working = backend.load_state_dict(forked["working"])
    pre_committed = [t.clone() for t in committed.recurrent_leaves()]
    _, working = backend.process(ids[8:11], working)
    assert working.position == st.position + 3 and committed.position == st.position
    assert all(torch.equal(x, y) for x, y in zip(committed.recurrent_leaves(), pre_committed))


def test_load_state_dict_rejects_incompatible_and_malformed(backend):
    ids = backend.encode("A short state for the persistence rejection checks here today please.")
    st = backend.init_state()
    _, st = backend.process(ids[:6], st)

    def fresh():
        return backend.state_dict(st)

    def first_linear(cache):
        return next(l for l in cache.layers if getattr(l, "recurrent_states", None) is not None)

    # a non-Qwen payload is refused
    with pytest.raises(ValueError, match="not a Qwen"):
        backend.load_state_dict({"backend": "plastic"})

    # ASTRA-066 P2: content identity — a different checkpoint digest (weights/tokenizer/template) is
    # refused even at matching dimensions, and so is a stale dimension
    bad = fresh()
    bad["identity"] = {**bad["identity"], "checkpoint_digest": "deadbeef"}
    with pytest.raises(ValueError, match="incompatible"):
        backend.load_state_dict(bad)
    bad = fresh()
    bad["identity"] = {**bad["identity"], "vocab_size": bad["identity"]["vocab_size"] + 1}
    with pytest.raises(ValueError, match="incompatible"):
        backend.load_state_dict(bad)

    # ASTRA-066 P3: malformed structures are refused against the trusted schema, not payload counts
    m = fresh()  # missing recurrent unit
    first_linear(m["cache"]).recurrent_states[0] = None
    with pytest.raises(ValueError, match="missing an initialized recurrent unit"):
        backend.load_state_dict(m)

    m = fresh()  # wrong recurrent shape
    l = first_linear(m["cache"])
    l.recurrent_states[0] = l.recurrent_states[0].reshape(1, -1)
    with pytest.raises(ValueError, match="recurrent shape/dtype"):
        backend.load_state_dict(m)

    m = fresh()  # absent conv state
    first_linear(m["cache"]).conv_states[0] = None
    with pytest.raises(ValueError, match="missing conv state"):
        backend.load_state_dict(m)

    m = fresh()  # reordered layer kinds
    m["cache"].layers[0], m["cache"].layers[19] = m["cache"].layers[19], m["cache"].layers[0]
    with pytest.raises(ValueError):
        backend.load_state_dict(m)

    m = fresh()  # non-finite recurrent value
    first_linear(m["cache"]).recurrent_states[0][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        backend.load_state_dict(m)

    m = fresh()  # a warm state with KV stripped masquerading as position-zero (nonzero recurrent)
    for layer in m["cache"].layers:
        if getattr(layer, "keys", None) is not None:
            layer.keys = None
            layer.values = None
    with pytest.raises(ValueError, match="position-zero"):
        backend.load_state_dict(m)


def test_forward_beta_scale_and_freeze_semantics(backend):
    ids = backend.encode("A sentence long enough for a clean forward and beta-scale check across it.")
    # forward matches process on the plain path and returns an empty per-token signal list
    lp, _ = backend.process(ids[:6], backend.init_state())
    lf, _, sig = backend.forward(ids[:6], backend.init_state(), freeze=False, beta_scale=1.0)
    assert torch.equal(lp, lf) and sig == []

    # beta_scale=0 from the zero initial state writes nothing: S = a*0 + 0*(k^T e) stays exactly zero
    # (distinct from freeze only when decay matters — here decay of zero is also zero)
    z = backend.init_state()
    _, z, _ = backend.forward(ids[:6], z, freeze=False, beta_scale=0.0)
    assert all(int(torch.count_nonzero(t)) == 0 for t in z.recurrent_leaves())

    # scaling beta down shrinks the write: a half-scale write has a strictly smaller (but nonzero)
    # recurrent-delta norm than a full write. It is NOT exactly half — the gated DELTA rule's
    # error-correction term beta*(v - S k)k^T depends on S, so the write is nonlinear in the scale.
    def _dnorm(scale):
        base = backend.init_state()
        s = backend.init_state()
        _, s, _ = backend.forward(ids[:6], s, freeze=False, beta_scale=scale)
        d = backend.state_delta(s, base)
        return float(sum(t.pow(2).sum() for t in d) ** 0.5)

    full, half = _dnorm(1.0), _dnorm(0.5)
    assert full > 0 and 0 < half < full

    # freeze via forward leaves recurrent memory exactly unchanged from a written state
    w = backend.init_state()
    _, w, _ = backend.forward(ids[:6], w, freeze=False, beta_scale=1.0)
    pre = [t.clone() for t in w.recurrent_leaves()]
    _, w, _ = backend.forward(ids[6:], w, freeze=True, beta_scale=1.0)
    assert all(torch.equal(a, b) for a, b in zip(w.recurrent_leaves(), pre))


def test_score_suite_is_read_only_and_gradient_is_defined(backend):
    ids = backend.encode("A benign sentence serving as coherence material for the canary suite here.")
    st = backend.init_state()
    _, st = backend.process(ids[:6], st)

    class _Suite:  # duck-typed CanarySuite (text): token-id probes
        domain = "text"
        coherence = [ids[:8], ids[2:12]]
        poison = [list(range(10, 26))]

    suite = _Suite()
    scores = backend.score_suite(st, suite)
    assert set(scores) == {"coherence", "poison"}
    assert all(isinstance(v, float) and v == v and v > 0 for v in scores.values())
    # scoring is read-only: the session's recurrent memory is unchanged
    pre = [t.clone() for t in st.recurrent_leaves()]
    backend.score_suite(st, suite)
    assert all(torch.equal(a, b) for a, b in zip(st.recurrent_leaves(), pre))
    # gradient: 18 finite leaves aligned with state_delta/recurrent_leaves, nonzero for real probes
    g = backend.canary_gradient(st, suite)
    assert len(g) == 18 and all(torch.isfinite(x).all() for x in g)
    assert any(x.abs().max().item() > 0 for x in g)

    class _Empty:
        domain = "text"
        coherence: list = []
        poison: list = []

    # no coherence probe -> a zero gradient of the correct shape (the harness still projects)
    z = backend.canary_gradient(st, _Empty())
    assert len(z) == 18 and all(int(torch.count_nonzero(x)) == 0 for x in z)
    assert scores["coherence"] != scores["poison"]  # sanity: the two sets score differently

    # ASTRA-064: canary_gradient differentiates the mean-per-probe coherence score, so duplicating a
    # coherence probe leaves BOTH the score and the gradient unchanged (a plain sum would double the
    # gradient).
    class _One:
        domain = "text"
        coherence = [ids[:8]]
        poison: list = []

    class _Two:
        domain = "text"
        coherence = [ids[:8], ids[:8]]
        poison: list = []

    g1 = backend.canary_gradient(st, _One())
    g2 = backend.canary_gradient(st, _Two())
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g1, g2))
    assert backend.score_suite(st, _One())["coherence"] == pytest.approx(backend.score_suite(st, _Two())["coherence"])


def test_apply_projected_overwrites_recurrent_from_committed(backend):
    ids = backend.encode("Enough tokens for two chunks to make a projection overwrite check here.")
    n = len(ids)
    k = n // 2
    committed = backend.init_state()
    _, committed = backend.process(ids[:k], committed)
    working = backend.clone(committed)
    _, working = backend.process(ids[k:], working)  # working now diverges from committed
    pre = [t.clone() for t in committed.recurrent_leaves()]
    deltas = [torch.ones_like(t) for t in committed.recurrent_leaves()]
    backend.apply_projected(working, committed, deltas)
    for w, c in zip(working.recurrent_leaves(), committed.recurrent_leaves()):
        assert torch.allclose(w, c + 1.0)
    # absolute overwrite from committed: a second apply lands committed + delta2, never delta1+delta2
    deltas2 = [3.0 * torch.ones_like(t) for t in committed.recurrent_leaves()]
    backend.apply_projected(working, committed, deltas2)
    for w, c in zip(working.recurrent_leaves(), committed.recurrent_leaves()):
        assert torch.allclose(w, c + 3.0)
    # committed's memory was not aliased or mutated by either apply
    for c, p in zip(committed.recurrent_leaves(), pre):
        assert torch.equal(c, p)


def test_qwen_runs_end_to_end_through_the_transaction_runner(backend):
    # The whole stack: a real QwenBackend driven through TransactionRunner. Validates the reduced
    # signal (None) path and generation-write accounting against the actual model on CPU.
    import io

    from plastic.config import ModelConfig
    from plastic.harness.config import HarnessConfig
    from plastic.harness.transaction import TransactionRunner

    L = 8
    cfg = ModelConfig(domain="text", chunk=L)
    r = TransactionRunner(
        None, cfg, HarnessConfig(enable_projection=False), device=torch.device("cpu"), backend=backend
    )
    prompt = backend.encode("The quick brown fox jumps over the lazy dog and then keeps on running along.")
    assert len(prompt) >= 2 * L

    # a user prompt chunk: reduced signals None, real chunk_loss/log_delta_norm, eligible, valid decision
    r.feed_tokens(prompt[:L], source="user")
    s = r.transactions[0]["signals"]
    assert s["surprise_mean"] is None and s["write_norm_sum"] is None and s["log_write_norm"] is None
    assert isinstance(s["chunk_loss"], float) and s["chunk_loss"] == s["chunk_loss"]
    assert isinstance(s["log_delta_norm"], float)
    assert r.transactions[0]["eligible"] is True and r.transactions[0]["sources"] == {"user": L, "model": 0}
    assert r.transactions[0]["decision"]["kind"] in ("commit", "rollback", "scale")

    # generation tokens write on Qwen: eligible, recorded as model source, and the recurrent state moves
    pre = [t.clone() for t in r.committed.recurrent_leaves()]
    r.feed_tokens(prompt[L : 2 * L], source="model")
    grec = r.transactions[1]
    assert grec["eligible"] is True and grec["sources"] == {"user": 0, "model": L}
    moved = any(not torch.equal(a, b) for a, b in zip(pre, r.committed.recurrent_leaves()))
    assert moved  # native generation wrote to memory

    # summary() must work on a Qwen session (it previously raised on committed.norms()); the recurrent
    # memory norm is surfaced honestly
    summ = r.summary()
    assert "recurrent_norm_total" in summ["state_norms"] and summ["state_norms"]["recurrent_norm_total"] > 0

    # the whole runner state (Qwen cache included) round-trips through torch.save/load
    buf = io.BytesIO()
    torch.save(r.state_dict(), buf)
    buf.seek(0)
    r.load_state_dict(torch.load(buf, weights_only=False))
    assert r.pos == 2 * L


def test_encode_chat_returns_integer_ids(backend):
    # apply_chat_template defaults to a dict in tf 5.17; encode_chat must return native integer ids
    # that tensorize, for empty / ascii / unicode, matching render-then-tokenize.
    for msg in ("", "Hello", "Café — déjà vu, 日本語 🎉"):
        ids = backend.encode_chat(msg)
        assert ids and all(isinstance(t, int) for t in ids)
        torch.tensor([ids], dtype=torch.long)  # must not raise
        rendered = backend.tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}], add_generation_prompt=True, enable_thinking=False, tokenize=False
        )
        assert ids == backend.tokenizer(rendered, add_special_tokens=False).input_ids
