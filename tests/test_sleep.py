import os

import pytest
import torch

from plastic.config import ModelConfig
from plastic.data.text import encode_documents_to_bin
from plastic.harness.canary import CanarySuite
from plastic.harness.config import HarnessConfig
from plastic.session.runner import Session
from plastic.sleep.consolidate import consolidate, harvest_traces
from plastic.store import ArtifactStore
from plastic.tokenizer.bpe import Tokenizer
from plastic.train.loop import TrainConfig, train

DOCS = ["alpha beta gamma delta epsilon " * 80, "one two three four five six " * 80]


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("sleep"))
    d = os.path.join(root, "data")
    os.makedirs(d)
    tok = Tokenizer.train(DOCS, vocab_size=300)
    tok.save(os.path.join(d, "tokenizer.json"))
    encode_documents_to_bin(tok, DOCS, os.path.join(d, "train.bin"))
    encode_documents_to_bin(tok, DOCS, os.path.join(d, "validation.bin"))
    cfg = TrainConfig(
        domain="text", model=ModelConfig(d_model=32, n_heads=2, n_layers=1, chunk=8, vocab_size=tok.vocab_size),
        artifacts_root=root, data_dir=d, steps=3, batch_size=2, seq_len=32, warmup_steps=1, eval_every=0,
        save_every=0, eval_batches=1, log_every=1, device="cpu", mqar_frac=0.0,
    )
    mid = train(cfg, log=lambda s: None)
    store = ArtifactStore(root)
    CanarySuite.default_text(os.path.join(d, "validation.bin"), vocab_size=tok.vocab_size, n_probe=2, probe_len=16).save(store.canary_path(mid))
    for i in range(2):
        s = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False), session_id=f"s{i}")
        for _ in range(3):
            s.chat("alpha beta gamma delta epsilon alpha beta gamma delta epsilon one two three four", max_new_tokens=8, seed=i)
    return store, mid, d


def test_harvest_and_accept(fixture):
    store, mid, d = fixture
    texts = harvest_traces(store, mid)
    assert len(texts) == 6
    m = consolidate(store, mid, core_data_dir=d, steps=4, lr=1e-4, seq_len=16, batch_size=4, tolerance={"coherence": 10.0, "poison": 10.0}, log=lambda s: None)
    assert m["accepted"] and m["model_id"].startswith("sleep_")
    rec = store.load_model_record(m["model_id"])
    assert rec["parent_model_id"] == mid and rec["type"] == "sleep_consolidation"
    cfg, child, info = store.load_checkpoint(m["model_id"])
    _, parent, _ = store.load_checkpoint(mid)
    assert torch.equal(child.embed.weight, parent.embed.weight)  # embeddings never move
    assert not all(torch.equal(a, b) for a, b in zip(child.core.parameters(), parent.core.parameters()))


def test_reject_leaves_no_model(fixture):
    store, mid, d = fixture
    before = {r["model_id"] for r in store.list_models()}
    m = consolidate(store, mid, core_data_dir=d, steps=2, seq_len=16, batch_size=4, tolerance={"coherence": -1.0, "poison": 10.0}, log=lambda s: None)
    assert not m["accepted"] and "model_id" not in m
    assert {r["model_id"] for r in store.list_models()} == before


def test_sleep_never_trains_on_text_the_harness_rolled_back(fixture):
    """The external review of 6cf4457 (finding 1): a session whose every user chunk the harness rolled back was still
    harvested, trained on with core_ratio 0, and registered as an accepted child. Refused text is now refused offline
    too: it is never harvested, and a session with nothing accepted has nothing to consolidate."""
    store, mid, d = fixture
    s = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False, enable_stats=False, canary_delta_max=-1e6),
                       session_id="all_rejected")
    secret = "one two three four five six one two three four five six one two three four five"
    r = s.chat(secret, max_new_tokens=8, seed=17)
    user = [t["decision"]["kind"] for t in r.transactions if t["sources"]["user"]]
    assert user and all(k == "rollback" for k in user)
    summary: dict = {}
    texts = harvest_traces(store, mid, summary=summary)
    assert texts and not any(secret in t for t in texts)
    assert summary["turns_by_reason"]["rolled_back"] >= 1
    with pytest.raises(ValueError, match="no accepted chat turns"):
        consolidate(store, mid, sessions=["all_rejected"], core_data_dir=d, steps=1, seq_len=16, batch_size=2, core_ratio=0.0, log=lambda s: None)
    before = {m["model_id"] for m in store.list_models()}
    m = consolidate(store, mid, core_data_dir=d, steps=1, seq_len=16, batch_size=2, core_ratio=0.0,
                    tolerance={"coherence": 10.0, "poison": 10.0}, log=lambda s: None)
    assert m["harvest"]["turns_by_reason"]["rolled_back"] >= 1 and m["harvest"]["selected_turns"] == len(texts)
    assert {mm["model_id"] for mm in store.list_models()} - before == {m["model_id"]}


def test_cli_sleep_reports_nothing_accepted_as_a_refusal_not_a_traceback(fixture, capsys):
    from plastic.cli import main

    store, mid, d = fixture
    if not store.session_exists("cli_rejected"):
        s = Session.create(store, model_id=mid, harness_cfg=HarnessConfig(enable_projection=False, enable_stats=False, canary_delta_max=-1e6),
                           session_id="cli_rejected")
        s.chat("one two three four five six one two three four five six one two", max_new_tokens=4, seed=3)
    code = main(["sleep", mid, "--artifacts-root", store.root, "--sessions", "cli_rejected", "--core", d, "--steps", "1"])
    assert code == 2 and "no accepted chat turns" in capsys.readouterr().err
