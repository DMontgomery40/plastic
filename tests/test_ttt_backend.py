"""Contract tests for the TTT-MLP backend (Sun et al. 2024 fast weights as a plastic Backend).

Skipped unless the local checkpoint exists (TTT_CHECKPOINT, default the 760M Pile base under
artifacts/models). They pin: chunked-vs-single-pass logit agreement across mini-batch alignment,
snapshot identity, freeze as a genuine no-write with advancing conv/position, beta_scale semantics,
the full per-token signal set (surprise, step size, exact write norm), state persistence round trips,
frozen canary gradients, and an end-to-end TransactionRunner chat turn.
"""

import os

import pytest
import torch

CKPT = os.environ.get("TTT_CHECKPOINT", "artifacts/models/ttt-mlp/TTT-MLP-760M-Base-Pile-8k")

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(CKPT, "config.json")), reason="TTT checkpoint not available")


@pytest.fixture(scope="module")
def backend():
    from plastic.backends.ttt_lm.backend import TTTBackend

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return TTTBackend.load(CKPT, device=dev)


TEXT = ("The history of computing includes mechanical calculators and modern electronic machines built over decades of work, "
        "from punched cards and vacuum tubes to transistors, integrated circuits and the networked devices of today.")


def test_segments_respect_mini_batch_alignment(backend):
    mb = backend.mini_batch
    assert backend._segments(40, 0) == [32, 8]
    assert backend._segments(5, 3) == [1, 1, 1, 1, 1]          # never aligned inside the chunk
    assert backend._segments(20, 13) == [1, 1, 1, 16, 1]        # singles to alignment, one block, remainder
    assert backend._segments(mb, mb) == [mb]
    assert backend._segments(0, 7) == []


def test_chunked_logits_match_single_pass_and_snapshot_is_exact(backend):
    ids = backend.encode(TEXT)
    n = len(ids)
    assert n > 20
    full = backend.logits_full(ids)
    st = backend.init_state()
    _, st = backend.process(ids[:7], st)          # unaligned split on purpose
    l2, st = backend.process(ids[7:], st)
    assert st.position == n
    # dual-form (aligned block) and primal-form (single-token) paths agree up to float error
    assert (full[7:] - l2).abs().max().item() < 5e-2
    assert full[7:].argmax(-1).eq(l2.argmax(-1)).float().mean().item() > 0.9
    # snapshot reproduces continuation exactly and does not alias
    st2 = backend.init_state()
    _, st2 = backend.process(ids[:7], st2)
    snap = backend.clone(st2)
    la, st2 = backend.process(ids[7:], st2)
    lb, snap = backend.process(ids[7:], snap)
    assert torch.equal(la, lb) and snap.position == st2.position == n


def test_freeze_is_a_genuine_no_write_and_activation_advances(backend):
    ids = backend.encode(TEXT)
    st = backend.init_state()
    _, st = backend.process(ids[:16], st)
    w_pre = [t.clone() for t in st.weight_leaves()]
    g_pre = [t.clone() for t in st.grad_leaves()]
    c_pre = [t.clone() for t in st.conv_leaves()]
    logits, st, sig = backend.forward(ids[16:30], st, freeze=True, beta_scale=1.0)
    assert all(torch.equal(a, b) for a, b in zip(w_pre, st.weight_leaves()))
    assert all(torch.equal(a, b) for a, b in zip(g_pre, st.grad_leaves()))   # pending gradients untouched too
    assert max((a - b).abs().max().item() for a, b in zip(c_pre, st.conv_leaves())) > 0.0
    assert st.position == 30 and logits.shape == (14, backend.vocab_size)
    # frozen tokens still report surprise but zero write
    assert all(float(s.write_norm.abs().max()) == 0.0 for s in sig)
    assert all(float(s.err.min()) >= 0.0 and s.err.shape[-1] == 14 for s in sig)


def test_beta_scale_zero_equals_freeze_and_half_scales_the_update(backend):
    ids = backend.encode(TEXT)[:32]
    base = backend.init_state()
    _, base = backend.process(ids[:16], base)
    a, b, c = backend.clone(base), backend.clone(base), backend.clone(base)
    _, a, _ = backend.forward(ids[16:32], a, freeze=True, beta_scale=1.0)
    _, b, _ = backend.forward(ids[16:32], b, freeze=False, beta_scale=0.0)
    _, c, _ = backend.forward(ids[16:32], c, freeze=False, beta_scale=1.0)
    assert all(torch.equal(x, y) for x, y in zip(a.weight_leaves(), b.weight_leaves()))   # no decay: identical
    full = sum(float(t.norm()) ** 2 for t in backend.state_delta(c, base)) ** 0.5
    assert full > 0.0
    h = backend.clone(base)
    _, h, _ = backend.forward(ids[16:32], h, freeze=False, beta_scale=0.5)
    half = sum(float(t.norm()) ** 2 for t in backend.state_delta(h, base)) ** 0.5
    assert 0.2 * full < half < 0.8 * full


def test_signals_have_full_shape_and_write_norm_matches_committed_delta_order(backend):
    ids = backend.encode(TEXT)[:32]
    st = backend.init_state()
    logits, st, sig = backend.forward(ids, st, freeze=False, beta_scale=1.0)
    assert len(sig) == backend.config.num_hidden_layers
    for s in sig:
        assert s.err.shape == (1, backend.config.num_attention_heads, 32) == s.beta.shape == s.write_norm.shape
        assert torch.isfinite(s.err).all() and torch.isfinite(s.write_norm).all()
        # the learned per-position step for the first token of a mini-batch is clamped at 0 in this checkpoint
        assert float(s.beta.min()) >= 0.0 and float(s.beta.max()) > 0.0 and float(s.write_norm.min()) >= 0.0
    # each token's committed contribution is rank one, so per layer the committed change is bounded by the
    # sum of the per-token write norms (triangle inequality) and is nonzero after two full mini-batches
    deltas = backend.state_delta(st, backend.init_state())
    per_layer = len(deltas) // len(sig)
    for layer, s in enumerate(sig):
        d = sum(float(t.norm()) ** 2 for t in deltas[layer * per_layer:(layer + 1) * per_layer]) ** 0.5
        bound = float(s.write_norm.sum())
        assert 0.0 < d <= bound * (1.0 + 1e-3), (layer, d, bound)


def test_state_dict_round_trips_and_rejects_mismatch(backend):
    ids = backend.encode(TEXT)[:21]
    st = backend.init_state()
    _, st = backend.process(ids, st)
    data = backend.state_dict(st)
    back = backend.load_state_dict(data)
    assert back.position == 21
    assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(st.weight_leaves(), back.weight_leaves()))
    assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(st.grad_leaves(), back.grad_leaves()))
    la, _ = backend.process(ids[:3], backend.clone(st))
    lb, _ = backend.process(ids[:3], back)
    assert torch.equal(la, lb)
    bad = dict(data)
    bad["identity"] = dict(data["identity"], checkpoint_digest="0" * 64)
    with pytest.raises(ValueError):
        backend.load_state_dict(bad)
    with pytest.raises(ValueError):
        backend.load_state_dict({"backend": "qwen"})


def test_canary_gradient_is_finite_nonzero_and_read_only(backend):
    from plastic.harness.canary import CanarySuite

    suite = CanarySuite(domain="text", coherence=[backend.encode("Water boils at one hundred degrees Celsius at sea level.")], poison=[])
    st = backend.init_state()
    _, st = backend.process(backend.encode(TEXT)[:16], st)
    before = [t.clone() for t in st.weight_leaves()]
    score = backend.score_suite(st, suite)
    g = backend.canary_gradient(st, suite)
    assert score["coherence"] > 0 and score["poison"] != score["poison"]  # nan poison: no probes
    assert len(g) == len(before) and all(torch.isfinite(t).all() for t in g)
    assert sum(float(t.norm()) for t in g) > 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, st.weight_leaves()))   # probes never mutate the session


def test_runs_end_to_end_through_the_transaction_runner(backend):
    from plastic.config import ModelConfig
    from plastic.harness.config import HarnessConfig
    from plastic.harness.transaction import TransactionRunner
    from plastic.session.runner import drive_chat_turn

    class IO:
        eos_id = int(backend.tokenizer.eos_token_id)
        assistant_close_ids: list[int] = []

        def encode(self, text, add_bos=False):
            return backend.encode_chat(text, first_turn=add_bos)

        def decode(self, ids):
            return backend.tokenizer.decode(ids)

    cfg = ModelConfig(domain="text", chunk=16)
    runner = TransactionRunner(None, cfg, HarnessConfig(log_only=True, learn_from_generation=True, enable_projection=False, enable_budget=False),
                               device=backend.device, backend=backend)
    completion, out_ids, in_ids = drive_chat_turn(runner, IO(), "Name one primary color.", max_new_tokens=8, temperature=0.7, top_k=20,
                                                  gen=torch.Generator().manual_seed(0))
    assert runner.pos == len(in_ids) + len(out_ids)
    recs = runner.transactions
    assert recs and all(r["decision"]["kind"] == "commit" for r in recs)
    s = recs[0]["signals"]
    assert s["surprise_mean"] is not None and s["write_norm_sum"] is not None and s["beta_mean"] is not None
    assert s["alpha_mean"] == 1.0 and s["delta_norm"] > 0
    assert isinstance(completion, str)
