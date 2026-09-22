import os

import torch

from plastic.config import ModelConfig
from plastic.data.text import encode_documents_to_bin
from plastic.store import ArtifactStore
from plastic.tokenizer.bpe import Tokenizer
from plastic.train.loop import TrainConfig, train
from plastic.train.schedule import lr_scale

DOCS = ["alpha beta gamma delta epsilon " * 40, "one two three four five six " * 40, "the end of it all " * 40]


def test_lr_scale_shape():
    assert lr_scale(1, warmup=10, total=100) == 0.1
    assert lr_scale(10, warmup=10, total=100) == 1.0
    assert abs(lr_scale(100, warmup=10, total=100, min_ratio=0.1) - 0.1) < 1e-9
    mid = lr_scale(55, warmup=10, total=100, min_ratio=0.1)
    assert 0.1 < mid < 1.0


def _text_corpus(tmp_path) -> tuple[str, Tokenizer]:
    tok = Tokenizer.train(DOCS, vocab_size=300)
    d = str(tmp_path / "data")
    os.makedirs(d, exist_ok=True)
    tok.save(os.path.join(d, "tokenizer.json"))
    encode_documents_to_bin(tok, DOCS, os.path.join(d, "train.bin"))
    encode_documents_to_bin(tok, DOCS[:1], os.path.join(d, "validation.bin"))
    return d, tok


def test_text_training_writes_artifacts(tmp_path):
    d, tok = _text_corpus(tmp_path)
    root = str(tmp_path / "artifacts")
    cfg = TrainConfig(
        domain="text",
        model=ModelConfig(d_model=32, n_heads=2, n_layers=1, chunk=16, vocab_size=tok.vocab_size),
        artifacts_root=root,
        data_dir=d,
        steps=6,
        batch_size=2,
        seq_len=32,
        warmup_steps=2,
        eval_every=3,
        eval_batches=2,
        save_every=3,
        log_every=2,
        mqar_pairs=(4,),
        device="cpu",
    )
    mid = train(cfg, log=lambda s: None)
    store = ArtifactStore(root)
    rec = store.load_model_record(mid)
    assert rec["status"] == "completed" and rec["steps"] == 6
    assert os.path.exists(store.tokenizer_path(mid))
    ev = store.read_eval(mid)
    assert ev is not None
    for key in ("heldout_loss", "heldout_loss_beta0", "memory_value", "mqar_accuracy", "beta_hist", "step"):
        assert key in ev, key
    assert "4" in ev["mqar_accuracy"]
    assert len(ev["beta_hist"]["counts"]) == 20
    log = store.read_log(mid)
    assert any(r.get("event") == "eval" for r in log) and any("loss" in r and "tok_per_s" in r for r in log)
    cfg2, model, info = store.load_checkpoint(mid)
    assert info["step"] == 6 and cfg2.vocab_size == tok.vocab_size


def test_text_training_is_deterministic(tmp_path):
    d, tok = _text_corpus(tmp_path)
    losses = []
    for i in range(2):
        cfg = TrainConfig(
            domain="text",
            model=ModelConfig(d_model=32, n_heads=2, n_layers=1, chunk=16, vocab_size=tok.vocab_size),
            artifacts_root=str(tmp_path / f"a{i}"),
            data_dir=d,
            steps=3,
            batch_size=2,
            seq_len=32,
            eval_every=0,
            save_every=0,
            log_every=1,
            device="cpu",
            mqar_frac=0.0,
        )
        mid = train(cfg, log=lambda s: None)
        losses.append([r["loss"] for r in ArtifactStore(cfg.artifacts_root).read_log(mid) if "loss" in r])
    assert losses[0] == losses[1]


def test_physics_training_writes_artifacts(tmp_path):
    root = str(tmp_path / "artifacts")
    cfg = TrainConfig(
        domain="physics",
        model=ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=1, chunk=16),
        artifacts_root=root,
        steps=4,
        batch_size=2,
        seq_len=32,
        episodes_per_seq=2,
        warmup_steps=1,
        eval_every=2,
        eval_batches=1,
        save_every=2,
        log_every=1,
        device="cpu",
        use_muon=False,
    )
    mid = train(cfg, log=lambda s: None)
    store = ArtifactStore(root)
    assert store.load_model_record(mid)["status"] == "completed"
    ev = store.read_eval(mid)
    assert ev is not None and "memory_value" in ev and "beta_hist" in ev
    assert mid.startswith("phys_")


def test_domain_mismatch_is_fixed_up(tmp_path):
    cfg = TrainConfig(
        domain="physics",
        model=ModelConfig(d_model=32, n_heads=2, n_layers=1, chunk=16),
        artifacts_root=str(tmp_path / "a"),
        steps=1,
        batch_size=1,
        seq_len=16,
        episodes_per_seq=1,
        eval_every=0,
        save_every=0,
        eval_batches=1,
        device="cpu",
        use_muon=False,
    )
    mid = train(cfg, log=lambda s: None)
    cfg2, model, _ = ArtifactStore(cfg.artifacts_root).load_checkpoint(mid)
    assert cfg2.domain == "physics"
    assert isinstance(model, torch.nn.Module)


def test_heldout_loss_is_batch_size_invariant(tmp_path):
    from plastic.data.text import TokenWindows
    from plastic.model.lm import PlasticLM
    from plastic.train.loop import evaluate_text

    d, tok = _text_corpus(tmp_path)
    cfg_model = ModelConfig(d_model=32, n_heads=2, n_layers=1, chunk=16, vocab_size=tok.vocab_size)
    torch.manual_seed(0)
    lm = PlasticLM(cfg_model)
    heldout = TokenWindows(f"{d}/train.bin", seq_len=32)
    n_windows = (len(heldout) - 1) // 32
    results = []
    for bs in (1, 2, 5):
        cfg = TrainConfig(domain="text", model=cfg_model, batch_size=bs, seq_len=32, eval_batches=10**6, mqar_pairs=(), device="cpu")
        results.append(evaluate_text(lm, cfg, heldout, torch.device("cpu")))
    assert all(r["heldout_tokens"] == n_windows * 32 for r in results)
    for r in results[1:]:
        assert abs(r["heldout_loss"] - results[0]["heldout_loss"]) < 1e-4
        assert abs(r["heldout_loss_beta0"] - results[0]["heldout_loss_beta0"]) < 1e-4


def test_adversarial_training_logs_damage(tmp_path):
    d, tok = _text_corpus(tmp_path)
    root = str(tmp_path / "adv")
    # the validation fixture must be long enough for the default canary probes
    encode_documents_to_bin(tok, DOCS * 4, os.path.join(d, "validation.bin"))
    cfg = TrainConfig(
        domain="text",
        model=ModelConfig(d_model=32, n_heads=2, n_layers=1, chunk=8, vocab_size=tok.vocab_size),
        artifacts_root=root, data_dir=d, steps=4, batch_size=2, seq_len=32, warmup_steps=1, eval_every=0,
        save_every=0, eval_batches=1, log_every=1, device="cpu", mqar_frac=0.0,
        adversarial=True, adv_every=2, adv_steps=2, adv_suffix_len=8, adv_prefix_len=16,
    )
    mid = train(cfg, log=lambda s: None)
    log = ArtifactStore(root).read_log(mid)
    adv = [r for r in log if "adv_damage" in r]
    assert len(adv) == 2 and all(r["adv_damage"] == r["adv_damage"] for r in adv)
    assert os.path.exists(ArtifactStore(root).canary_path(mid))
