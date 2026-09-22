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
