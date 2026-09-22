import os

import pytest
import torch

from plastic.config import ModelConfig
from plastic.data.text import encode_documents_to_bin
from plastic.harness.canary import CanarySuite
from plastic.harness.config import HarnessConfig
from plastic.session.runner import Session
from plastic.store import ArtifactStore
from plastic.tokenizer.bpe import Tokenizer
from plastic.train.loop import TrainConfig, train

DOCS = ["alpha beta gamma delta epsilon " * 40, "one two three four five six " * 40]


@pytest.fixture(scope="module")
def text_model(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("artifacts"))
    d = os.path.join(root, "data")
    os.makedirs(d)
    tok = Tokenizer.train(DOCS, vocab_size=300)
    tok.save(os.path.join(d, "tokenizer.json"))
    encode_documents_to_bin(tok, DOCS, os.path.join(d, "train.bin"))
    encode_documents_to_bin(tok, DOCS[:1], os.path.join(d, "validation.bin"))
    cfg = TrainConfig(
        domain="text",
        model=ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=8, vocab_size=tok.vocab_size),
        artifacts_root=root, data_dir=d, steps=3, batch_size=2, seq_len=32, warmup_steps=1,
        eval_every=0, save_every=0, eval_batches=1, log_every=1, device="cpu", mqar_frac=0.0,
    )
    mid = train(cfg, log=lambda s: None)
    store = ArtifactStore(root)
    CanarySuite.default_text(os.path.join(d, "validation.bin"), vocab_size=tok.vocab_size, n_probe=2, probe_len=16).save(store.canary_path(mid))
    return store, mid


@pytest.fixture(scope="module")
def physics_model(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("artifacts_phys"))
    cfg = TrainConfig(
        domain="physics",
        model=ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=1, chunk=8),
        artifacts_root=root, steps=2, batch_size=2, seq_len=32, episodes_per_seq=2, warmup_steps=1,
        eval_every=0, save_every=0, eval_batches=1, log_every=1, device="cpu", use_muon=False,
    )
    mid = train(cfg, log=lambda s: None)
    return ArtifactStore(root), mid


def test_create_list_fork_lineage(text_model):
    store, mid = text_model
    s = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False), session_id="root")
    assert store.session_exists("root")
    child = s.fork("child")
    metas = {m["session_id"]: m for m in store.list_sessions()}
    assert set(metas) >= {"root", "child"}
    assert metas["child"]["parent_session_id"] == "root" and metas["child"]["root_session_id"] == "root"
    grand = Session.open(store, "child").fork("grand")
    assert store.load_session_meta(grand)["root_session_id"] == "root"


def test_chat_learns_persists_and_continues(text_model):
    store, mid = text_model
    s = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False), session_id="c1")
    r = s.chat("alpha beta gamma delta epsilon alpha beta", max_new_tokens=6, seed=0)
    assert isinstance(r.completion, str) and r.n_tokens_in > 0 and r.n_tokens_out <= 6
    assert len(r.transactions) >= 1 and all("signals" in t for t in r.transactions)
    pos1 = s.runner.pos
    assert store.load_session_meta("c1")["pos"] == pos1 and store.read_transactions("c1")
    assert store.read_trace("c1")[0]["kind"] == "chat"
    s2 = Session.open(store, "c1")
    assert s2.runner.pos == pos1
    r2 = s2.chat("one two three", max_new_tokens=3, seed=1)
    assert s2.runner.pos > pos1 and store.load_session_meta("c1")["n_transactions"] >= 2
    counts = store.load_session_meta("c1")
    assert counts["commits"] + counts["rollbacks"] + counts["scales"] + counts["projects"] + counts["readonly"] == counts["n_transactions"]


def test_fork_starts_from_parent_committed_state(text_model):
    store, mid = text_model
    s = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False), session_id="p1")
    s.chat("alpha beta gamma delta epsilon alpha beta gamma", max_new_tokens=0)
    child = s.fork("p1_child")
    c = Session.open(store, child)
    for a, b in zip(c.runner.committed.layers, s.runner.committed.layers):
        assert torch.equal(a.S, b.S)
    assert c.runner.pos == s.runner.pos


def test_signature_mismatch_refused(text_model):
    store, mid = text_model
    Session.create(store, model_id=mid, harness_cfg=HarnessConfig(), session_id="sig1")
    cfg, model, info = store.load_checkpoint(mid)
    with torch.no_grad():
        model.embed.weight.add_(0.001)
    store.save_checkpoint(mid, cfg, model, step=info["step"] + 1)
    with pytest.raises(ValueError):
        Session.open(store, "sig1")
    # restore a matching checkpoint for the other tests by recreating the session against the new signature
    Session.create(store, model_id=mid, harness_cfg=HarnessConfig(), session_id="sig2")
    Session.open(store, "sig2")


def test_physics_episode_three_way(physics_model):
    store, mid = physics_model
    s = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(), session_id="ph1")
    res = s.physics_episode(steps=24, mu=0.12, seed=0)
    assert len(res.per_step) == 24 and set(res.means) == {"base_mse", "frozen_mse", "adaptive_mse"}
    assert len(res.transactions) == 3  # 24 steps = 3 chunks of 8
    assert store.read_trace("ph1")[0]["kind"] == "episode"
    res2 = s.physics_episode(steps=8, mu=0.2, seed=1)
    assert s.runner.pos == 32
    s.reset()
    assert s.runner.pos == 0 and store.load_session_meta("ph1")["pos"] == 0


def test_fork_drops_pending_and_starts_from_committed_state(text_model):
    store, mid = text_model
    s = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False), session_id="mid1")
    s.runner.feed_tokens([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # 8 committed + 3 pending
    child_id = s.fork("mid1_child")
    c = Session.open(store, child_id)
    assert c.runner.pos == 8 and not c.runner.pending and c.runner.n_transactions == 0
    for a, b in zip(c.runner.committed.layers, s.runner.committed.layers):
        assert torch.equal(a.S, b.S)
    assert store.load_session_meta(child_id)["pos"] == 8 and store.load_session_meta(child_id)["forked_at_pos"] == 8


def _short_prompt_fixture(tmp_path, rule):
    root = str(tmp_path / rule)
    d = os.path.join(root, "data")
    os.makedirs(d)
    tok = Tokenizer.train(DOCS, vocab_size=300)
    tok.save(os.path.join(d, "tokenizer.json"))
    encode_documents_to_bin(tok, DOCS, os.path.join(d, "train.bin"))
    encode_documents_to_bin(tok, DOCS[:1], os.path.join(d, "validation.bin"))
    cfg = TrainConfig(
        domain="text", model=ModelConfig(d_model=32, n_heads=2, n_layers=1, chunk=8, vocab_size=tok.vocab_size, rule=rule),
        artifacts_root=root, data_dir=d, steps=2, batch_size=2, seq_len=32, warmup_steps=1, eval_every=0, save_every=0,
        eval_batches=1, log_every=1, device="cpu", mqar_frac=0.0,
    )
    mid = train(cfg, log=lambda s: None)
    return ArtifactStore(root), mid


def test_short_prompt_is_learned_before_generation_delta_rule(tmp_path):
    store, mid = _short_prompt_fixture(tmp_path, "delta")
    s = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False), session_id="d1")
    r = s.chat("alpha", max_new_tokens=6, seed=0)  # short prompt, then generation crosses the boundary
    assert r.n_tokens_in < 8
    # the prompt is transacted as its own partial chunk before generation begins
    assert r.transactions[0]["decision"]["kind"] == "commit" and r.transactions[0]["signals"]["n_tokens"] == r.n_tokens_in
    # the delta rule learns per token, so a sub-chunk prompt does move the memory
    assert any(float(l.S.abs().sum()) > 0 for l in s.runner.committed.layers)


def test_short_prompt_chunk_rule_is_transacted_not_discarded(tmp_path):
    # the chunk rule is mini-batch: a sub-chunk prompt applies no update (documented), but it must
    # be transacted as its own boundary before generation, never silently discarded
    store, mid = _short_prompt_fixture(tmp_path, "chunk")
    s = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False), session_id="cr1")
    r = s.chat("alpha", max_new_tokens=6, seed=0)
    assert r.n_tokens_in < 8
    assert r.transactions[0]["decision"]["kind"] == "commit" and r.transactions[0]["signals"]["n_tokens"] == r.n_tokens_in
    assert r.transactions[0]["signals"]["pos_end"] == r.n_tokens_in  # a real boundary at the prompt end


def test_verify_calibration_signature_gating():
    # ASTRA-078: a persisted calibration installs only if its signature matches the model's actual
    # identity; a mismatched or unsigned calibration is discarded (never silently trusted).
    from plastic.harness.calibrate import Calibration
    from plastic.session.runner import _verify_calibration

    cal = Calibration(model_signature="sig-A", thresholds={"chunk_loss": 1.0})
    assert _verify_calibration(cal, "sig-A") == (cal, "installed")
    assert _verify_calibration(cal, "sig-B") == (None, "rejected_signature_mismatch")
    assert _verify_calibration(Calibration(model_signature=""), "sig-A") == (None, "rejected_unsigned")
    assert _verify_calibration(None, "sig-A") == (None, "absent")
