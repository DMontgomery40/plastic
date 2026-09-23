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
    # dual-form (aligned block) and primal-form (single-token) paths agree up to float error; MPS float error
    # is larger than CPU's (OPUS-001 F4 measured 3.1e-5 on CPU)
    tol = 5e-2 if backend.device.type == "mps" else 1e-3
    assert (full[7:] - l2).abs().max().item() < tol
    assert full[7:].argmax(-1).eq(l2.argmax(-1)).float().mean().item() > (0.9 if backend.device.type == "mps" else 0.999)
    # snapshot reproduces continuation exactly and does not alias
    st2 = backend.init_state()
    _, st2 = backend.process(ids[:7], st2)
    snap = backend.clone(st2)
    la, st2 = backend.process(ids[7:], st2)
    lb, snap = backend.process(ids[7:], snap)
    assert torch.equal(la, lb) and snap.position == st2.position == n


def test_freeze_is_a_genuine_no_write_in_effective_coordinates_from_a_pending_state(backend):
    """Start at position 23 (a pending mini-batch gradient G exists) and freeze a chunk that crosses the
    boundary at 32: the reference commits G into W at the boundary, so raw W changes, but W_eff = W − c15·G,
    the coordinates the harness measures, must not move at all (OPUS-001 F1)."""
    ids = backend.encode(TEXT)
    st = backend.init_state()
    _, st = backend.process(ids[:23], st)
    assert st.position == 23 and sum(float(g.abs().sum()) for g in st.grad_leaves()) > 0.0   # G is pending
    eff_pre = [t.clone() for t in st.effective_leaves(backend._c15)]
    c_pre = [t.clone() for t in st.conv_leaves()]
    logits, st, sig = backend.forward(ids[23:38], st, freeze=True, beta_scale=1.0)
    assert st.position == 38 and logits.shape == (15, backend.vocab_size)
    delta = max((a - b).abs().max().item() for a, b in zip(eff_pre, st.effective_leaves(backend._c15)))
    assert delta < 1e-4 * max(1.0, max(float(t.abs().max()) for t in eff_pre)), delta
    assert max((a - b).abs().max().item() for a, b in zip(c_pre, st.conv_leaves())) > 0.0   # activation advanced
    # frozen tokens still report surprise but zero write
    assert all(float(s.write_norm.abs().max()) == 0.0 for s in sig)
    assert all(float(s.err.min()) >= 0.0 and s.err.shape[-1] == 15 for s in sig)


def test_beta_scale_zero_equals_freeze_and_half_scales_the_update_linearly(backend):
    """From a pending state (position 23), a chunk crossing the boundary: beta_scale=0 equals freeze (no decay),
    and beta_scale=0.5 halves the chunk's own change in effective coordinates (the inner step is linear in
    the learning rate; the gradients are taken at the base point)."""
    ids = backend.encode(TEXT)[:40]
    base = backend.init_state()
    _, base = backend.process(ids[:23], base)
    a, b, c, h = (backend.clone(base) for _ in range(4))
    _, a, _ = backend.forward(ids[23:40], a, freeze=True, beta_scale=1.0)
    _, b, _ = backend.forward(ids[23:40], b, freeze=False, beta_scale=0.0)
    _, c, _ = backend.forward(ids[23:40], c, freeze=False, beta_scale=1.0)
    _, h, _ = backend.forward(ids[23:40], h, freeze=False, beta_scale=0.5)
    assert all(torch.equal(x, y) for x, y in zip(a.weight_leaves(), b.weight_leaves()))
    assert all(torch.equal(x, y) for x, y in zip(a.grad_leaves(), b.grad_leaves()))
    full = sum(float(t.norm()) ** 2 for t in backend.state_delta(c, base)) ** 0.5
    half = sum(float(t.norm()) ** 2 for t in backend.state_delta(h, base)) ** 0.5
    assert full > 0.0 and 0.45 * full < half < 0.55 * full, (full, half)


def test_signals_have_full_shape_and_write_norm_matches_committed_delta_order(backend):
    ids = backend.encode(TEXT)[:32]
    st = backend.init_state()
    logits, st, sig = backend.forward(ids, st, freeze=False, beta_scale=1.0)
    assert len(sig) == backend.config.num_hidden_layers
    for s in sig:
        assert s.err.shape == (1, backend.config.num_attention_heads, 32) == s.beta.shape == s.write_norm.shape
        assert torch.isfinite(s.err).all() and torch.isfinite(s.write_norm).all()
        # beta is display-only: the learned per-position scale varies ~2-3x across a mini-batch and a token's step can be 0
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


def test_calibration_on_real_chats_produces_installable_thresholds(tmp_path, backend):
    """The shared real-chat calibrator accepts the TTT backend and yields thresholds for its full signal set."""
    from plastic.harness.calibrate import Calibration, calibrate_qwen
    from plastic.backends.ttt_lm.backend import _checkpoint_digest
    from plastic.store import ArtifactStore

    store = ArtifactStore(str(tmp_path))
    store.register_model("ttt_test", {"backend": "ttt", "domain": "text", "status": "completed", "checkpoint_dir": CKPT,
                                      "checkpoint_digest": _checkpoint_digest(CKPT), "chunk": 16})
    prompts = ["Name a color.", "What is two plus two?", "Say hello."]
    cal = calibrate_qwen(store, "ttt_test", prompts, max_new_tokens=8, seed=0, device=backend.device, log=lambda *a, **k: None)
    assert cal.model_signature == f"ttt:{backend.checkpoint_digest}"
    for name in ("chunk_loss", "surprise_mean", "log_delta_norm", "log_write_norm"):
        assert name in cal.reference and len(cal.reference[name]) > 0, name
    assert "fisher_update" not in cal.reference
    cal.save(store.model_dir("ttt_test"))
    assert Calibration.exists(store.model_dir("ttt_test"))


def test_render_matches_training(backend):
    """The runtime render of a user turn is a token-for-token prefix of the training render (OPUS-001 F3)."""
    from plastic.backends.ttt_lm.backend import CHAT_ASSISTANT, CHAT_USER
    from plastic.backends.ttt_lm.backend import encode_conversation as encode_example

    for msg in ("Name one primary color.", "What is 2+2?", "Explain LayerNorm briefly.", "hi"):
        ids, labels = encode_example(backend.tokenizer, [{"role": "user", "content": msg}, {"role": "assistant", "content": "x"}],
                                     CHAT_USER, CHAT_ASSISTANT)
        rt = backend.encode_chat(msg, first_turn=True)
        assert ids[:len(rt)] == rt, (msg, ids[:len(rt)], rt)
        assert all(l == -100 for l in labels[:len(rt)])          # the rendered prefix is never supervised


def test_canary_score_and_gradient_read_the_same_function_from_a_pending_state(backend):
    """From a pending state, the frozen probe NLL and the differentiated probe agree (both fold G into W)."""
    from plastic.harness.canary import CanarySuite

    probe = backend.encode("Water boils at one hundred degrees Celsius at sea level.")
    suite = CanarySuite(domain="text", coherence=[probe], poison=[])
    st = backend.init_state()
    _, st = backend.process(backend.encode(TEXT)[:23], st)
    nll = backend.score_suite(st, suite)["coherence"]
    with backend._lock:
        folded = backend._folded_probe_cache(st)
    import torch as _t
    x = _t.tensor([probe[:-1]], dtype=_t.long, device=backend.device)
    with _t.no_grad(), backend._lock:
        prev = getattr(backend.__class__, "_unused", None)
    from plastic.backends.ttt_lm import modeling_ttt as M
    prev_scale = getattr(M._tls, "eta_scale", 1.0)
    M._tls.eta_scale = 0.0
    try:
        with _t.no_grad():
            logits = []
            i = 0
            for n in backend._segments(x.shape[1], folded.seqlen_offset):
                logits.append(backend.model(x[:, i:i + n], cache_params=folded, use_cache=True).logits[0])
                i += n
    finally:
        M._tls.eta_scale = prev_scale
    direct = float(_t.nn.functional.cross_entropy(_t.cat(logits, 0), _t.tensor(probe[1:], device=backend.device)))
    assert abs(direct - nll) < 1e-3, (direct, nll)
