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
