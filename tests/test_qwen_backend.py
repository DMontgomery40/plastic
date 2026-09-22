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
