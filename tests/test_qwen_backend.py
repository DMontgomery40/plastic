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


def test_freeze_leaves_memory_unchanged_but_advances_position(backend):
    import plastic.backends.qwen as q

    ids = backend.encode("A short sentence for the freeze check here, long enough to span two chunks cleanly.")
    n = len(ids)
    k = n // 2
    st = backend.init_state()
    _, st = backend.process(ids[:k], st)
    pre = q._recurrent_snapshot(st.cache)
    _, st = backend.process(ids[k:], st, freeze=True)
    post = q._recurrent_snapshot(st.cache)
    changed = max((pre[l][i] - post[l][i]).abs().max().item() for l in range(len(pre)) for i in pre[l])
    assert changed == 0.0  # no write under freeze
    assert st.position == n  # activation/position still advanced
