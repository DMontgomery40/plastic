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
