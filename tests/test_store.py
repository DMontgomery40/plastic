import pytest
import torch

from plastic.config import ModelConfig
from plastic.model.lm import PlasticLM
from plastic.store import ArtifactStore


def _cfg():
    return ModelConfig(d_model=32, n_heads=2, n_layers=1, chunk=16, vocab_size=50)


def test_register_and_list(tmp_path):
    store = ArtifactStore(str(tmp_path))
    a = store.new_model_id("lm")
    store.register_model(a, {"status": "running", "created_at_unix": 10})
    b = store.new_model_id("lm")
    assert b != a
    store.register_model(b, {"status": "completed", "created_at_unix": 20})
    ids = [r["model_id"] for r in store.list_models()]
    assert ids == [b, a]
    store.register_model(a, {"status": "completed"})
    assert store.load_model_record(a)["status"] == "completed"
    assert store.load_model_record(a)["created_at_unix"] == 10


def test_model_exists_is_backend_aware(tmp_path):
    # ASTRA-081 #5: a Qwen model is registered with an EXTERNAL checkpoint dir (no local plastic
    # checkpoint.pt/config), so model_exists must recognize it by its record + checkpoint dir rather
    # than 404 the session/model/redteam/sleep routes that gate on it.
    store = ArtifactStore(str(tmp_path))
    ckpt_dir = tmp_path / "qwen-ckpt"
    ckpt_dir.mkdir()

    ok = store.new_model_id("qwen")
    store.register_model(ok, {"backend": "qwen", "checkpoint_dir": str(ckpt_dir), "domain": "text", "chunk": 8})
    assert store.model_exists(ok) is True  # record present and its checkpoint dir is on disk

    missing_dir = store.new_model_id("qwen")
    store.register_model(missing_dir, {"backend": "qwen", "checkpoint_dir": str(tmp_path / "nope"), "domain": "text"})
    assert store.model_exists(missing_dir) is False  # registered but its checkpoint dir is absent

    plastic_no_ckpt = store.new_model_id("lm")
    store.register_model(plastic_no_ckpt, {"backend": "plastic", "status": "running"})
    assert store.model_exists(plastic_no_ckpt) is False  # a plastic model still needs its checkpoint/config

    assert store.model_exists("model-that-was-never-registered") is False


def test_qwen_model_signature_and_session_lifecycle(tmp_path):
    # A pretrained-backend (Qwen) model has no local plastic checkpoint; its signature comes from the
    # registered backend + content digest, and sessions create/verify against it without a checkpoint.
    from plastic.harness.config import HarnessConfig

    store = ArtifactStore(str(tmp_path))
    mid = store.new_model_id("qwen")
    store.register_model(mid, {"backend": "qwen", "checkpoint_dir": "/models/qwen3.5", "checkpoint_digest": "abc123", "domain": "text", "chunk": 8})
    sig = store.model_signature(mid)
    assert sig == "qwen:abc123"  # backend + content digest, no plastic checkpoint read

    sid = store.new_session_id("chat")
    store.create_session(sid, model_id=mid, domain="text", harness_cfg=HarnessConfig())
    store.verify_session_model(sid)  # signature matches -> no raise
    assert store.load_session_meta(sid)["model_signature"] == "qwen:abc123"

    # a changed registered checkpoint digest invalidates existing sessions
    store.register_model(mid, {"backend": "qwen", "checkpoint_digest": "def456"})
    with pytest.raises(ValueError, match="different signature"):
        store.verify_session_model(sid)


def test_fork_metadata_uses_backend_independent_cursor(tmp_path):
    # ASTRA-074: session/fork metadata must read a backend-independent committed cursor, not a
    # plastic state's serialized committed['pos'] (which Qwen's serialized state does not carry).
    from plastic.harness.config import HarnessConfig
    from plastic.harness.transaction import fork_state_dict

    store = ArtifactStore(str(tmp_path))
    mid = store.new_model_id("qwen")
    store.register_model(mid, {"backend": "qwen", "checkpoint_digest": "d", "domain": "text", "chunk": 8})
    # a Qwen-shaped runner state: the committed cursor is exposed at top level, not inside committed
    parent_state = {"committed_pos": 24, "working_pos": 24, "committed": {"backend": "qwen"}, "working": {"backend": "qwen"}}
    p = store.new_session_id("chat")
    store.create_session(p, model_id=mid, domain="text", harness_cfg=HarnessConfig(), runner_state=parent_state)
    assert store.load_session_meta(p)["pos"] == 24  # not 0

    child = store.new_session_id("chat")
    store.create_session(
        child, model_id=mid, domain="text", harness_cfg=HarnessConfig(),
        parent_session_id=p, runner_state=fork_state_dict(parent_state),
    )
    cmeta = store.load_session_meta(child)
    assert cmeta["pos"] == 24 and cmeta["forked_at_pos"] == 24  # lineage cursor preserved, not 0/0


def test_fork_cursor_fresh_warm_and_pending(tmp_path):
    # ASTRA-076: a fork reports the parent's COMMITTED cursor in both metadata fields and drops
    # pending working tokens — across fresh, warm, and warm-with-pending parents (a Qwen-shaped state
    # whose committed cursor is exposed backend-independently, not inside committed).
    from plastic.harness.config import HarnessConfig
    from plastic.harness.transaction import fork_state_dict

    store = ArtifactStore(str(tmp_path))
    mid = store.new_model_id("qwen")
    store.register_model(mid, {"backend": "qwen", "checkpoint_digest": "d", "domain": "text", "chunk": 8})

    for committed, working in ((0, 0), (8, 8), (8, 11)):  # fresh, warm, warm-with-pending (working>committed)
        parent_state = {"committed_pos": committed, "working_pos": working, "committed": {"backend": "qwen"}, "working": {"backend": "qwen"}}
        p = store.new_session_id("chat")
        store.create_session(p, model_id=mid, domain="text", harness_cfg=HarnessConfig(), runner_state=parent_state)
        forked = fork_state_dict(parent_state)
        # the fork starts from committed: both cursors are the committed pos, and pending is dropped
        assert forked["committed_pos"] == committed and forked["working_pos"] == committed and forked["pending"] == []
        child = store.new_session_id("chat")
        store.create_session(child, model_id=mid, domain="text", harness_cfg=HarnessConfig(), parent_session_id=p, runner_state=forked)
        cmeta = store.load_session_meta(child)
        assert cmeta["pos"] == committed and cmeta["forked_at_pos"] == committed
        assert store.load_session_meta(p)["pos"] == committed  # parent metadata unaffected by the fork


def test_retried_run_does_not_inherit_stale_error(tmp_path):
    # A failed attempt records an error; a retry at the same model_id must not carry it forward.
    store = ArtifactStore(str(tmp_path))
    mid = store.new_model_id("lm")
    store.register_model(mid, {"status": "running", "created_at_unix": 1})
    store.register_model(mid, {"status": "failed", "error": "OutOfMemoryError: CUDA OOM"})
    assert store.load_model_record(mid)["error"].startswith("OutOfMemoryError")
    # the retry starts (running) and then completes; neither carries an error field
    store.register_model(mid, {"status": "running"})
    assert "error" not in store.load_model_record(mid)  # stale error cleared the moment the retry starts
    store.register_model(mid, {"status": "completed", "steps": 6000})
    rec = store.load_model_record(mid)
    assert rec["status"] == "completed" and "error" not in rec

    # a partial register that does not declare a status (e.g. calibration) must NOT wipe a real error
    store.register_model(mid, {"status": "failed", "error": "boom"})
    store.register_model(mid, {"calibrated_at_unix": 123})
    assert store.load_model_record(mid)["error"] == "boom"


def test_checkpoint_roundtrip_and_signature(tmp_path):
    store = ArtifactStore(str(tmp_path))
    cfg = _cfg()
    lm = PlasticLM(cfg)
    mid = "lm_test"
    store.save_checkpoint(mid, cfg, lm, step=3, extra={"note": "x"})
    assert store.model_exists(mid)
    cfg2, lm2, info = store.load_checkpoint(mid)
    assert cfg2 == cfg and info["step"] == 3 and info["extra"]["note"] == "x"
    toks = torch.randint(0, 50, (1, 20))
    a, _, _ = lm(toks)
    b, _, _ = lm2(toks)
    assert torch.allclose(a, b, atol=1e-6)
    sig1 = store.model_signature(mid)
    with torch.no_grad():
        lm.embed.weight.add_(0.01)
    store.save_checkpoint(mid, cfg, lm, step=4)
    assert store.model_signature(mid) != sig1


def test_eval_and_log(tmp_path):
    store = ArtifactStore(str(tmp_path))
    mid = "lm_x"
    assert store.read_eval(mid) is None
    store.write_eval(mid, {"heldout_loss": 1.5})
    assert store.read_eval(mid)["heldout_loss"] == 1.5
    store.append_log(mid, {"step": 1, "loss": 2.0})
    store.append_log(mid, {"step": 2, "loss": 1.9})
    assert [r["step"] for r in store.read_log(mid)] == [1, 2]
    assert store.read_log(mid, limit=1)[0]["step"] == 2
