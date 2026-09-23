"""The ablation runner: every variant builds, trains a step, is scored, and the collected table
labels each model's no-adaptation intervention."""

from __future__ import annotations

import argparse
import json

import torch

from plastic.eval.contract import ContractSpec
from scripts.experiments.coordinate_ablation import VARIANTS, build, collect, run_variant


def test_every_variant_builds_with_its_switches():
    for name in VARIANTS:
        m = build(name, d_model=32, n_heads=2, n_layers=1, chunk=16, seed=0)
        assert sum(p.numel() for p in m.parameters()) > 0
        if VARIANTS[name] is not None:
            for k, v in VARIANTS[name].items():
                assert getattr(m.cfg, k) == v, (name, k)


def test_run_variant_and_collect_smoke(tmp_path):
    args = argparse.Namespace(
        steps=2, batch=2, seq_len=32, episodes=1, lr=1e-3, log_every=1, d_model=32, n_heads=2, n_layers=1,
        chunk=16, seed=0, device="cpu",
    )
    spec = ContractSpec(seq_len=32, eval_batch=2, stream_episodes=4, probe_steps=4)
    for v in ("full", "decay_only", "delta_baseline"):
        r = run_variant(v, args, str(tmp_path), spec=spec)
        assert r["contract"]["revert"]["ok"] is True
        assert len(r["train"]["log"]) >= 2 and r["train"]["log"][-1]["step"] == 2
        assert (r["train"]["log"][-1]["eta"] is None) == (v == "delta_baseline")
    assert (tmp_path / "full.pt").exists() and (tmp_path / "delta_baseline.pt").exists()
    full = json.load(open(tmp_path / "full.json"))
    assert full["fast_signals"]["stepped_fraction"] == 1.0 and full["checkpoint"] == "full.pt"
    assert json.load(open(tmp_path / "delta_baseline.json"))["fast_signals"] is None
    readme = collect(str(tmp_path))
    assert "| full |" in readme and "| decay_only |" in readme and "| delta_baseline |" in readme
    assert "→" in readme  # the inner-loss diagnostic column is filled for coordinate variants
    assert "fast parameters frozen (freeze=True)" in readme and "writes disabled (beta_scale=0)" in readme
    saved = json.load(open(tmp_path / "full.json"))
    assert saved["no_adapt_label"].startswith("fast parameters frozen")
    assert torch.isfinite(torch.tensor(saved["train"]["log"][-1]["loss"]))
