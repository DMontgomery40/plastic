"""The contract runner: a fresh model per mode, a summary table with the right columns, and
outputs that carry provenance."""

from __future__ import annotations

import json
import os

import torch

from plastic.config import ModelConfig
from plastic.eval.contract import ContractSpec
from plastic.model.lm import PlasticDynamics
from scripts.experiments.transfer_contract import before_table, run_modes, summary_table, write_outputs


def _factory() -> PlasticDynamics:
    torch.manual_seed(0)
    return PlasticDynamics(ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=2, chunk=16))


def test_run_modes_uses_a_fresh_model_per_mode_and_tables_render(tmp_path):
    spec = ContractSpec(seq_len=32, eval_batch=2, stream_episodes=4, probe_steps=4)
    reports = run_modes(_factory, modes=("frozen", "continued"), spec=spec, seed=0, lr=1e-3, steps=1, device=torch.device("cpu"))
    # continued training on a fresh copy must not have moved the frozen mode's before numbers
    f, c = reports["frozen"], reports["continued"]
    for p in f["transfer"]:
        assert f["transfer"][p]["before"] == c["transfer"][p]["before"]
    assert f["transfer"][next(iter(f["transfer"]))]["delta_mse"] == 0.0
    table = summary_table(reports)
    assert table.count("\n") == 3 and "| frozen |" in table and "| continued |" in table
    assert "n/a (n=0)" in table  # the frozen mode records no decision, and that is not zero
    before = before_table(reports)
    assert "training distribution" in before and "adaptation speed" in before
    manifest = {"model_id": "tiny", "checkpoint_digest": None, "execution_commit": None, "device": "cpu", "contract_version": f["contract_version"], "seed": 0}
    write_outputs(str(tmp_path), reports, manifest)
    assert sorted(os.listdir(tmp_path)) == ["README.md", "continued.json", "frozen.json", "manifest.json"]
    assert json.load(open(tmp_path / "frozen.json"))["mode"] == "frozen"
    readme = (tmp_path / "README.md").read_text()
    assert "identical for every mode" in readme and "negative is improvement" in readme
