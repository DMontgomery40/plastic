import os

import numpy as np
import pytest
import torch

from plastic.config import ModelConfig
from plastic.data.text import encode_documents_to_bin
from plastic.harness.canary import CanarySuite
from plastic.redteam.attack import AttackConfig, pgd_attack, run_redteam, sampled_attack, snap_to_tokens
from plastic.store import ArtifactStore
from plastic.tokenizer.bpe import Tokenizer
from plastic.train.loop import TrainConfig, train

CPU = torch.device("cpu")
DOCS = ["alpha beta gamma delta epsilon " * 80, "one two three four five six " * 80]


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("rt"))
    d = os.path.join(root, "data")
    os.makedirs(d)
    tok = Tokenizer.train(DOCS, vocab_size=300)
    tok.save(os.path.join(d, "tokenizer.json"))
    encode_documents_to_bin(tok, DOCS, os.path.join(d, "train.bin"))
    encode_documents_to_bin(tok, DOCS, os.path.join(d, "validation.bin"))
    cfg = TrainConfig(
        domain="text", model=ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=8, vocab_size=tok.vocab_size),
        artifacts_root=root, data_dir=d, steps=4, batch_size=2, seq_len=32, warmup_steps=1, eval_every=0,
        save_every=0, eval_batches=1, log_every=1, device="cpu", mqar_frac=0.0,
    )
    mid = train(cfg, log=lambda s: None)
    store = ArtifactStore(root)
    suite = CanarySuite.default_text(os.path.join(d, "validation.bin"), vocab_size=tok.vocab_size, n_probe=2, probe_len=16)
    suite.save(store.canary_path(mid))
    return store, mid, d, suite


def test_pgd_attack_validates_the_snapped_payload(fixture):
    store, mid, d, suite = fixture
    model_cfg, model, _ = store.load_checkpoint(mid)
    cfg = AttackConfig(suffix_len=8, steps=4, lr=0.1, radius=2.0, harness={"enable_projection": False})
    prefix = list(np.fromfile(os.path.join(d, "validation.bin"), dtype="<u2")[:16].astype("int64"))
    r = pgd_attack(model, model_cfg, cfg, prefix, suite, device=CPU)
    assert r.family == "pgd" and len(r.payload_ids) == 8 and all(0 <= t < model_cfg.vocab_size for t in r.payload_ids)
    # the validated positions cover exactly the prefix and the payload
    assert r.signals and r.signals[-1]["pos_end"] == len(prefix) + len(r.payload_ids)
    assert r.damage_validated == r.damage_validated and r.nll_payload > 0
    assert r.damage_continuous is not None
    assert len(r.decisions) >= 1 and all(k in ("commit", "rollback", "scale", "project", "readonly") for k in r.decisions)


def test_pgd_increases_continuous_damage(fixture):
    store, mid, d, suite = fixture
    model_cfg, model, _ = store.load_checkpoint(mid)
    prefix = list(np.fromfile(os.path.join(d, "validation.bin"), dtype="<u2")[:16].astype("int64"))
    r0 = pgd_attack(model, model_cfg, AttackConfig(suffix_len=8, steps=0, nll_max=1e9), prefix, suite, device=CPU)
    r1 = pgd_attack(model, model_cfg, AttackConfig(suffix_len=8, steps=15, lr=0.2, radius=3.0, nll_max=1e9), prefix, suite, device=CPU)
    assert r1.damage_continuous >= r0.damage_continuous - 1e-6, (r0.damage_continuous, r1.damage_continuous)


def test_nll_constraint_is_reported(fixture):
    store, mid, d, suite = fixture
    model_cfg, model, _ = store.load_checkpoint(mid)
    prefix = list(np.fromfile(os.path.join(d, "validation.bin"), dtype="<u2")[:16].astype("int64"))
    r = pgd_attack(model, model_cfg, AttackConfig(suffix_len=8, steps=2, nll_max=0.0), prefix, suite, device=CPU)
    assert r.constraint_violated  # a zero ceiling cannot be met
    r2 = pgd_attack(model, model_cfg, AttackConfig(suffix_len=8, steps=2, nll_max=1e9), prefix, suite, device=CPU)
    assert not r2.constraint_violated


def test_sampled_families_and_campaign(fixture):
    store, mid, d, suite = fixture
    model_cfg, model, _ = store.load_checkpoint(mid)
    prefix = list(np.fromfile(os.path.join(d, "validation.bin"), dtype="<u2")[:16].astype("int64"))
    corpus = np.fromfile(os.path.join(d, "validation.bin"), dtype="<u2")
    g = torch.Generator().manual_seed(0)
    for fam in ("random", "repeat", "shuffle", "topic_switch"):
        r = sampled_attack(model, model_cfg, fam, AttackConfig(suffix_len=8), prefix, suite, device=CPU, rng=g, corpus=corpus)
        assert r.family == fam and len(r.payload_ids) == 8 and r.damage_validated == r.damage_validated
    cfg = AttackConfig(suffix_len=8, steps=2, families=("pgd", "random", "repeat"))
    summary = run_redteam(store, mid, cfg=cfg, data_dir=d, n_prefixes=2, prefix_len=16, device=CPU, record=True, log=lambda s: None)
    assert set(summary["families"]) == {"pgd", "random", "repeat"}
    out = os.path.join(store.root, "redteam", summary["run_id"])
    assert os.path.exists(os.path.join(out, "results.jsonl")) and os.path.exists(os.path.join(out, "summary.json"))
    assert summary["recorded_payloads"] == 4
    assert len(CanarySuite.load(store.canary_path(mid)).poison) >= 4


def test_snap_returns_valid_ids(fixture):
    store, mid, d, suite = fixture
    model_cfg, model, _ = store.load_checkpoint(mid)
    ids = torch.tensor([5, 17, 99])
    emb = model.embed.weight[ids].unsqueeze(0)
    assert snap_to_tokens(model, emb).tolist() == ids.tolist()


def test_endpoint_controls_and_valid_only_aggregates(fixture):
    store, mid, d, suite = fixture
    model_cfg, model, _ = store.load_checkpoint(mid)
    from plastic.harness.config import HarnessConfig
    from plastic.redteam.attack import validate_payload

    prefix = list(np.fromfile(os.path.join(d, "validation.bin"), dtype="<u2")[:16].astype("int64"))
    payload = list(np.fromfile(os.path.join(d, "validation.bin"), dtype="<u2")[16:32].astype("int64"))
    hcfg = HarnessConfig(canary_delta_max=-1e9, poison_delta_min=-1e9, enable_projection=False)
    v = validate_payload(model, model_cfg, prefix, payload, suite, harness=hcfg, calibration=None, device=CPU)
    for k in ("canary_after_accepted", "canary_after_unprotected", "canary_after_frozen", "nll_guarded"):
        assert k in v and v[k] == v[k]
    # under full rollback the accepted end equals the frozen end (nothing learned)
    assert abs(v["canary_after_accepted"] - v["canary_after_frozen"]) < 1e-4
    # nll_max 0 forces every payload constraint-invalid: valid-only aggregates must be None, not 0
    cfg = AttackConfig(suffix_len=8, steps=1, nll_max=0.0, families=("random",))
    summary = run_redteam(store, mid, cfg=cfg, data_dir=d, n_prefixes=2, prefix_len=16, device=CPU, log=lambda s: None)
    fam = summary["families"]["random"]
    assert fam["n_valid"] == 0 and fam["valid_damage_mean"] is None and fam["valid_over_threshold_fraction"] is None
    assert "unprotected_damage_mean" in fam and "frozen_damage_mean" in fam


def test_constraint_uses_guarded_nll(fixture):
    store, mid, d, suite = fixture
    model_cfg, model, _ = store.load_checkpoint(mid)
    prefix = list(np.fromfile(os.path.join(d, "validation.bin"), dtype="<u2")[:16].astype("int64"))
    r = pgd_attack(model, model_cfg, AttackConfig(suffix_len=8, steps=1, nll_max=1e9), prefix, suite, device=CPU)
    assert r.nll_payload_guarded == r.nll_payload_guarded and not r.constraint_violated


def test_coherence_poison_attack_reports_embedding_bound_and_controls(fixture):
    store, mid, d, suite = fixture
    model_cfg, model, _ = store.load_checkpoint(mid)
    from plastic.redteam.attack import coherence_poison_attack

    prefix = list(np.fromfile(os.path.join(d, "validation.bin"), dtype="<u2")[:16].astype("int64"))
    cfg = AttackConfig(poison_chunks=3, steps=4, lr=0.1, radius=3.0)
    r = coherence_poison_attack(model, model_cfg, cfg, prefix, suite, device=CPU, rng=torch.Generator().manual_seed(0))
    assert r.family == "coherence_poison"
    assert len(r.payload_ids) == 3 * model_cfg.chunk and all(0 <= t < model_cfg.vocab_size for t in r.payload_ids)
    # the embedding-space upper bound is finite and is at least the discrete unprotected damage
    assert r.damage_embedding_unprotected == r.damage_embedding_unprotected  # not NaN
    assert r.damage_embedding_unprotected >= r.damage_unprotected - 1e-4
    # all three controls are populated and the payload ran chunk-by-chunk through the harness
    assert len(r.decisions) == 3 and r.damage_unprotected == r.damage_unprotected and r.damage_frozen == r.damage_frozen


def test_run_redteam_exposes_unprotected_aggregates(fixture):
    store, mid, d, suite = fixture
    cfg = AttackConfig(poison_chunks=2, steps=3, families=("coherence_poison",))
    summary = run_redteam(store, mid, cfg=cfg, data_dir=d, n_prefixes=2, prefix_len=16, device=CPU, log=lambda s: None)
    fam = summary["families"]["coherence_poison"]
    for key in ("unprotected_damage_mean", "unprotected_damage_max", "unprotected_over_threshold_fraction",
                "embedding_unprotected_damage_mean"):
        assert key in fam, key
    assert fam["embedding_unprotected_damage_mean"] is not None  # this family always fills it


def test_run_redteam_summary_has_created_at(fixture):
    store, mid, d, suite = fixture
    cfg = AttackConfig(suffix_len=8, steps=1, families=("random",))
    summary = run_redteam(store, mid, cfg=cfg, data_dir=d, n_prefixes=1, prefix_len=16, device=CPU, log=lambda s: None)
    assert "created_at_unix" in summary and summary["created_at_unix"] > 0
