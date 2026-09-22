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

    m = fresh()  # a warm state with all K/V stripped and marked uninitialized -> masquerades as
    # position-zero, but its recurrent memory is non-zero
    for layer in m["cache"].layers:
        if getattr(layer, "recurrent_states", None) is None:  # attention layers
            layer.keys = None
            layer.values = None
            layer.is_initialized = False
    with pytest.raises(ValueError, match="position-zero"):
        backend.load_state_dict(m)

    # ASTRA-069: initialization flags and attention dims that change/break the next forward
    def first_dynamic(cache):
        return next(l for l in cache.layers if getattr(l, "keys", None) is not None)

    m = fresh()  # a linear layer with has_previous_state cleared (would change continuation)
    first_linear(m["cache"]).has_previous_state[0] = False
    with pytest.raises(ValueError, match="has_previous_state"):
        backend.load_state_dict(m)

    m = fresh()  # an attention layer with populated K/V but is_initialized False
    first_dynamic(m["cache"]).is_initialized = False
    with pytest.raises(ValueError, match="is_initialized"):
        backend.load_state_dict(m)

    m = fresh()  # a reduced attention head count (would crash the next forward)
    d = first_dynamic(m["cache"])
    d.keys = d.keys[:, :1]
    d.values = d.values[:, :1]
    with pytest.raises(ValueError, match="heads/feature"):
        backend.load_state_dict(m)

    m = fresh()  # K sliced but V not -> K/V shape mismatch
    d = first_dynamic(m["cache"])
    d.keys = d.keys[:, :1]
    with pytest.raises(ValueError, match="K/V shape mismatch"):
        backend.load_state_dict(m)

    m = fresh()  # K/V removed from ONE attention layer but still flagged initialized -> inconsistent
    d = first_dynamic(m["cache"])
    d.keys = None
    d.values = None
    with pytest.raises(ValueError, match="is_initialized"):
        backend.load_state_dict(m)

    m = fresh()  # conv cast to a dtype the backend does not use
    l = first_linear(m["cache"])
    l.conv_states[0] = l.conv_states[0].to(torch.bfloat16)
    with pytest.raises(ValueError, match="conv shape/dtype"):
        backend.load_state_dict(m)


def test_persistence_validation_matrix(backend):
    # ASTRA-071: a widened invariant matrix over representative first/last layers of each kind and
    # fresh/warm states. Every malformation in the validation contract rejects; the valid warm state
    # still loads with an EXACT continuation and the fresh state loads and forwards.
    ids = backend.encode("A warm state spanning enough tokens for the persistence invariant matrix here today.")
    warm = backend.init_state()
    _, warm = backend.process(ids[:8], warm)
    fresh = backend.init_state()

    ref_sd = backend.state_dict(warm)
    linear_idx = [i for i, l in enumerate(ref_sd["cache"].layers) if getattr(l, "recurrent_states", None) is not None]
    attn_idx = [i for i, l in enumerate(ref_sd["cache"].layers) if getattr(l, "recurrent_states", None) is None]
    linear_reps = [linear_idx[0], linear_idx[-1]]
    attn_reps = [attn_idx[0], attn_idx[-1]]

    # valid states: warm continuation is byte-exact after a reload; fresh loads and forwards
    exact_ref, _ = backend.process(ids[8:11], backend.clone(warm))
    exact_got, _ = backend.process(ids[8:11], backend.load_state_dict(backend.state_dict(warm)))
    assert (exact_ref - exact_got).abs().max().item() == 0.0
    f_back = backend.load_state_dict(backend.state_dict(fresh))
    assert f_back.position == 0 and all(int(torch.count_nonzero(t)) == 0 for t in f_back.recurrent_leaves())
    y, _ = backend.process(ids[:4], f_back)
    assert torch.isfinite(y).all()

    linear_muts = {
        "recurrent_none": lambda l: l.recurrent_states.__setitem__(0, None),
        "recurrent_shape": lambda l: l.recurrent_states.__setitem__(0, l.recurrent_states[0].reshape(1, -1)),
        "conv_none": lambda l: l.conv_states.__setitem__(0, None),
        "conv_dtype": lambda l: l.conv_states.__setitem__(0, l.conv_states[0].to(torch.bfloat16)),
        "clear_has_previous": lambda l: l.has_previous_state.__setitem__(0, False),
        "clear_conv_init": lambda l: l.is_conv_states_initialized.__setitem__(0, False),
        "clear_rec_init": lambda l: l.is_recurrent_states_initialized.__setitem__(0, False),
    }
    attn_muts = {
        "clear_is_initialized": lambda l: setattr(l, "is_initialized", False),
        "kv_head_sliced": lambda l: setattr(l, "keys", l.keys[:, :1]),
        "kv_batch2": lambda l: (setattr(l, "keys", l.keys.repeat(2, 1, 1, 1)), setattr(l, "values", l.values.repeat(2, 1, 1, 1))),
        "kv_rank5": lambda l: (setattr(l, "keys", l.keys.unsqueeze(0)), setattr(l, "values", l.values.unsqueeze(0))),
        "v_bf16": lambda l: setattr(l, "values", l.values.to(torch.bfloat16)),
        "k_bf16": lambda l: setattr(l, "keys", l.keys.to(torch.bfloat16)),
    }

    for name, fn in linear_muts.items():
        for idx in linear_reps:
            sd = backend.state_dict(warm)
            fn(sd["cache"].layers[idx])
            with pytest.raises(ValueError):
                backend.load_state_dict(sd)
    for name, fn in attn_muts.items():
        for idx in attn_reps:
            sd = backend.state_dict(warm)
            fn(sd["cache"].layers[idx])
            with pytest.raises(ValueError):
                backend.load_state_dict(sd)

    # one warm attention layer stripped to look fresh (K/V absent AND is_initialized cleared) while the
    # others stay warm -> the all-or-none K/V rule rejects it
    for idx in attn_reps:
        sd = backend.state_dict(warm)
        layer = sd["cache"].layers[idx]
        layer.keys = None
        layer.values = None
        layer.is_initialized = False
        with pytest.raises(ValueError, match="all or none"):
            backend.load_state_dict(sd)

    # a FRESH attention layer marked initialized but with no K/V (would crash get_seq_length on load)
    for idx in attn_reps:
        sd = backend.state_dict(fresh)
        sd["cache"].layers[idx].is_initialized = True
        with pytest.raises(ValueError, match="is_initialized"):
            backend.load_state_dict(sd)


def test_position_zero_state_persists_and_loads(backend):
    # ASTRA-069: the supported position-zero initialization must still round-trip (no KV, zero
    # recurrent) — the tightened validation must not reject a valid fresh state.
    st = backend.init_state()
    back = backend.load_state_dict(backend.state_dict(st))
    assert back.position == 0
    assert all(int(torch.count_nonzero(t)) == 0 for t in back.recurrent_leaves())
    # and it can be forwarded after loading
    ids = backend.encode("A short continuation from a reloaded position-zero state here today.")
    y, _ = backend.process(ids[:4], back)
    assert torch.isfinite(y).all()


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


def test_qwen_session_chat_end_to_end(tmp_path):
    # The full native Session slice: register a Qwen model, create a chat session, chat through the
    # native template with EOS-aware generation, persist, reopen, and continue. No `backend` fixture
    # (each Session loads its own), so memory is freed between the two loads.
    import gc

    from plastic.backends.qwen import _checkpoint_digest
    from plastic.harness.config import HarnessConfig
    from plastic.session.runner import Session
    from plastic.store import ArtifactStore

    store = ArtifactStore(str(tmp_path))
    mid = store.new_model_id("qwen")
    store.register_model(mid, {
        "backend": "qwen", "checkpoint_dir": CKPT, "checkpoint_digest": _checkpoint_digest(CKPT),
        "domain": "text", "chunk": 8, "status": "completed",
    })

    sess = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False), device="cpu")
    assert sess.backend_kind == "qwen"
    close_len = len(sess.tokenizer.assistant_close_ids)
    assert close_len == 2  # native <|im_end|> + newline

    res = sess.chat("Hello", max_new_tokens=6, seed=0)
    assert isinstance(res.completion, str) and res.n_tokens_in > 0
    assert res.summary["domain"] == "text" and "recurrent_norm_total" in res.summary["state_norms"]
    # the summary honestly surfaces the backend and its reduced decision-signal set
    assert res.summary["backend"] == "qwen"
    assert set(res.summary["signals_available"]) == {"chunk_loss", "log_delta_norm"}
    assert res.summary["calibration"] == "absent"  # no calibration registered for this model
    assert res.transactions  # the prompt transacted through the harness
    # multi-turn framing: the assistant turn is closed in the carried state (pos counts the prompt,
    # the generated tokens, AND the turn terminator), and the terminator is not shown to the user
    assert "<|im_end|>" not in res.completion
    pos_after = sess.runner.pos
    assert pos_after == res.n_tokens_in + res.n_tokens_out + close_len
    sid = sess.session_id
    del sess
    gc.collect()

    # reopen the persisted session (native cache restored) and take a second turn; framing holds
    sess2 = Session.open(store, sid, device="cpu")
    assert sess2.backend_kind == "qwen" and sess2.runner.pos == pos_after
    res2 = sess2.chat("Thanks", max_new_tokens=4, seed=1)
    assert isinstance(res2.completion, str) and "<|im_end|>" not in res2.completion
    assert sess2.runner.pos == pos_after + res2.n_tokens_in + res2.n_tokens_out + close_len

    # zero-generation turn (output cap 0): the assistant turn is still closed exactly once
    pos_before_zero = sess2.runner.pos
    res3 = sess2.chat("Ok", max_new_tokens=0, seed=2)
    assert res3.n_tokens_out == 0
    assert sess2.runner.pos == pos_before_zero + res3.n_tokens_in + close_len


def test_qwen_calibration_produces_installable_thresholds(tmp_path):
    # The native conversational calibrator: build Qwen thresholds from a conversational corpus through
    # the same reduced-signal runner a chat uses, stamped with the actual checkpoint digest, and
    # confirm a session installs them (the identity gate accepts the matching signature).
    import gc

    from plastic.harness.calibrate import calibrate_qwen
    from plastic.harness.config import HarnessConfig
    from plastic.session.runner import Session
    from plastic.store import ArtifactStore

    store = ArtifactStore(str(tmp_path))
    mid = store.new_model_id("qwen")
    store.register_model(mid, {"backend": "qwen", "checkpoint_dir": CKPT, "domain": "text", "chunk": 8, "status": "completed"})

    # prompts the model GENERATES responses to (not teacher-forced), through the real chat protocol
    prompts = ["What is the capital of France?", "Name a primary color.", "How many days are in a week?"]
    cal = calibrate_qwen(store, mid, prompts, target_fpr=0.2, max_new_tokens=4, seed=0, log=lambda s: None)
    # reduced-signal references/thresholds only, stamped with the actual loaded checkpoint digest
    assert cal.model_signature.startswith("qwen:") and len(cal.model_signature) > len("qwen:")
    assert set(cal.reference.keys()) == {"chunk_loss", "log_delta_norm"}
    assert "chunk_loss" in cal.thresholds and "log_delta_norm" in cal.thresholds
    assert "surprise_mean" not in cal.thresholds and "log_write_norm" not in cal.thresholds
    # prompt-chunk and generation-chunk denominators are recorded separately (real generation ran)
    r = store.load_model_record(mid)
    assert r["calibration_prompt_chunks"] >= 1 and r["calibration_generation_chunks"] >= 1
    gc.collect()

    # a session on this model installs the calibration (identity gate accepts the matching signature)
    sess = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False), device="cpu")
    assert sess.calibration_status == "installed" and sess.runner.calibration is not None
    assert sess.runner.calibration.model_signature == cal.model_signature


def test_qwen_session_calibration_identity(tmp_path):
    # ASTRA-078: a Qwen session installs a persisted calibration only if its signature matches the
    # ACTUAL loaded checkpoint content — not a stale registry digest — on create and on reopen.
    import gc

    from plastic.backends.qwen import _checkpoint_digest
    from plastic.harness.calibrate import Calibration
    from plastic.harness.config import HarnessConfig
    from plastic.session.runner import Session
    from plastic.store import ArtifactStore

    store = ArtifactStore(str(tmp_path))
    actual = _checkpoint_digest(CKPT)
    mid = store.new_model_id("qwen")
    # register with a STALE digest that does not match the actual checkpoint content
    store.register_model(mid, {"backend": "qwen", "checkpoint_dir": CKPT, "checkpoint_digest": "stale-registered", "domain": "text", "chunk": 8, "status": "completed"})
    model_dir = store.model_dir(mid)

    def _cal(sig):
        return Calibration(model_signature=sig, thresholds={"chunk_loss": 0.0}, reference={"chunk_loss": [1.0] * 8})

    # a calibration matching only the stale registry digest is refused even though the registry agrees
    _cal("qwen:stale-registered").save(model_dir)
    sess = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False), device="cpu")
    assert sess.calibration_status == "rejected_signature_mismatch"
    assert sess.calibration is None and sess.runner.calibration is None
    sid = sess.session_id
    del sess
    gc.collect()

    # a calibration matching the ACTUAL loaded checkpoint content is installed on reopen
    _cal(f"qwen:{actual}").save(model_dir)
    sess2 = Session.open(store, sid, device="cpu")
    assert sess2.calibration_status == "installed"
    assert sess2.runner.calibration is not None and sess2.runner.calibration.thresholds["chunk_loss"] == 0.0


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


def test_qwen_continuous_chat_state_roundtrips_for_resume(backend):
    # The exact property calibrate_qwen's durable CUSUM resume relies on (ASTRA-086 #2): snapshot the
    # runner state partway through a CONTINUOUS multi-turn chat, restore it into a fresh runner, and
    # the continued per-turn log_delta_norm values match the uninterrupted chat elementwise. The
    # fake-runner tests cover the resume control flow; this covers the real cache/state round-trip.
    from plastic.config import ModelConfig
    from plastic.harness.calibrate import log_only
    from plastic.harness.config import HarnessConfig
    from plastic.harness.transaction import TransactionRunner
    from plastic.session.runner import _QwenTextIO, drive_chat_turn

    cfg = ModelConfig(domain="text", chunk=8)
    hcfg = log_only(HarnessConfig(target_fpr=0.2))
    tok = _QwenTextIO(backend)
    prompts = ["Say hello.", "Count to three.", "Name a fruit.", "Name a color."]

    def _run_turn(runner, prompt, salt):
        runner.transactions = []
        g = torch.Generator().manual_seed(1000 + salt)
        drive_chat_turn(runner, tok, prompt, max_new_tokens=4, temperature=0.9, top_k=50, gen=g)
        return [r["signals"]["log_delta_norm"] for r in runner.transactions if r["signals"].get("log_delta_norm") is not None]

    def _fresh_runner():
        r = TransactionRunner(None, cfg, hcfg, device=torch.device("cpu"), backend=backend)
        r.reset()
        return r

    # uninterrupted continuous chat (no reset between turns)
    r = _fresh_runner()
    full = []
    for i, p in enumerate(prompts):
        full += _run_turn(r, p, i)

    # interrupted: run two turns, snapshot, restore into a fresh runner, continue the remaining turns
    r2 = _fresh_runner()
    part = []
    for i in range(2):
        part += _run_turn(r2, prompts[i], i)
    snapshot = r2.state_dict()
    r3 = TransactionRunner(None, cfg, hcfg, device=torch.device("cpu"), backend=backend)
    r3.load_state_dict(snapshot)
    for i in range(2, len(prompts)):
        part += _run_turn(r3, prompts[i], i)

    assert part == full  # exact continuous-state round-trip: resume reproduces the CUSUM reference


def test_dev_diagnostic_observes_the_real_runner_cusum_exactly(backend):
    # The operating-point dev stage observes each chunk's post-update CUSUM statistic by snapshotting it
    # as the REAL TransactionRunner appends the record, and recomputes the CUSUM input with the runner's
    # rule. On the actual checkpoint, both input paths (per-chunk z; continuous reference >= 8) must replay
    # to exactly the observed statistic in both arms, and the pair must pass the dev progress schema.
    import json

    from plastic.config import ModelConfig
    from plastic.harness.calibrate import Calibration
    from plastic.harness.config import HarnessConfig
    from scripts.experiments.qwen_operating_point import _dev_record_problem, _dev_sessions

    ref = [0.1 * i for i in range(16)]
    hcfg = HarnessConfig(enable_stats=True, enable_rollback=True, log_only=False, learn_from_generation=True,
                         freeze_on_alarm=True, alarm_cooldown=0)
    for cref in ([], [5.0 + 0.01 * i for i in range(16)]):
        cal = Calibration(model_signature=f"qwen:{backend.checkpoint_digest}", reference={"chunk_loss": ref, "log_delta_norm": ref},
                          cusum_reference=cref, thresholds={"cusum_h": 0.5})
        rep = _dev_sessions(backend, ModelConfig(domain="text", chunk=8), cal, [{"id": 1, "prompt": "Name a primary color."}],
                            guarded_hcfg=hcfg, gen={"max_new_tokens": 4, "temperature": 0.9, "top_k": 50},
                            seed_base=7, deadline=1e18)
        assert rep["complete"] is True and rep["summary"]["cusum_replay_mismatch"] == 0
        pair = rep["pairs"][0]
        for arm in ("guarded", "log_only"):
            assert len(pair["arms"][arm]["chunks"]) == len(pair["arms"][arm]["txns"]) > 0
        if cref:  # a reference far from the observed log_delta_norm forces the CUSUM to alarm
            assert any(c["cusum_alarm"] for c in pair["arms"]["log_only"]["chunks"])
            assert pair["arms"]["guarded"]["read_only_end"] is True and pair["arms"]["log_only"]["read_only_end"] is False
        assert _dev_record_problem(json.loads(json.dumps({"regime": "dev", "record": pair, "txns": []}))) is None
