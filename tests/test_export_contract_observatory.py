"""The learning-contract export for the observatory's Learning view: every number traces to an archived report and
reproduces the published tables, absent stays absent (a rate over an empty set is null, never zero), each learner keeps
its own off-intervention label, and the committed export and its dashboard mirror are current."""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

import pytest
import torch

from plastic.eval.contract import CONTRACT_VERSION
from scripts import export_contract_observatory as ex
from scripts import export_sleep_observatory as sleep_ex
from scripts.experiments import coordinate_ablation as ablation

PHYS = ex.ARCHIVE / "phys_mps_3k"


@pytest.fixture(scope="module")
def export(tmp_path_factory) -> dict:
    return ex.build(tmp_path_factory.mktemp("learning"))


def phys(index: dict) -> dict:
    return next(s for s in index["sets"] if s["id"] == "phys_mps_3k")


def readme_rows(path: Path) -> dict[str, list[str]]:
    """Rows of a generated contract README's tables, keyed by their first cell."""
    rows = {}
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and not line.startswith("|---") and len(cells) > 2:
            rows[cells[0]] = cells[1:]
    return rows


def test_phys_export_reproduces_every_cell_of_the_published_report(export):
    s = phys(export)
    rows = readme_rows(PHYS / "README.md")
    for p in s["before"]["policies"]:
        adapt, no_adapt, elements = rows[f"held-out combos, {p['policy']}"]
        assert (f"{p['adapt']:.4f}", f"{p['no_adapt']:.4f}", str(p["elements"])) == (adapt, no_adapt, elements)
    adapt, no_adapt, _ = rows["training distribution"]
    assert (f"{s['before']['train']['adapt']:.4f}", f"{s['before']['train']['no_adapt']:.4f}") == (adapt, no_adapt)
    speed = s["before"]["speed"]
    assert f"mean {speed['mean_ratio']:.3f}; below one half at step {speed['steps_to_half']}" in rows[next(k for k in rows if k.startswith("adaptation speed"))][0]
    assert speed["reached_half"] is True and speed["horizon"] == 64

    def cell(v: float | None, n: int | None) -> str:
        return f"n/a (n={n})" if v is None else f"{v:+.4f} (n={n})"

    for m in s["modes"]:
        r = rows[m["mode"]]
        transfer = [f"{t['delta']:+.4f}" for t in m["transfer"]]
        assert r[:3] == transfer, m["mode"]
        assert r[3:7] == [f"{m[k]:+.4f}" for k in ("forgetting_delta", "poison_harm_vs_clean", "poison_harm_vs_start", "correction_residual")]
        assert r[7] == ("yes" if m["revert"]["ok"] else "no")
        a = m["acceptance"]
        assert r[8:10] == [cell(a["accepted_good"], a["n_good"]), cell(a["refused_bad"], a["n_bad"])]
        assert r[10:12] == [str(m["tokens_consumed"]), str(m["tokens_measured"])]


def test_identity_window_and_labels_come_from_the_manifest(export):
    s = phys(export)
    manifest = json.loads((PHYS / "manifest.json").read_text())
    assert s["checkpoint"] == {"model_id": "phys_mps_3k", "digest_prefix": manifest["checkpoint_digest"][:16], "step": 3000}
    assert s["execution_commit"] == manifest["execution_commit"][:12] and s["split"]["id"] == manifest["split_id"]
    assert s["adaptation_window"] == {"update_period": 1, "boundaries_per_episode": 63, "checked": True}
    assert s["contract_version"] == manifest["contract_version"] and s["current"] is (manifest["contract_version"] == CONTRACT_VERSION)
    assert s["tag"] == "the before reference" and s["missing_modes"] == []
    assert [m["mode"] for m in s["modes"]] == ["frozen", "continued", "in_context"]
    assert {src["file"] for src in s["sources"]} == {"phys_mps_3k/" + n for n in ("manifest.json", "frozen.json", "continued.json", "in_context.json", "README.md")}


def test_an_empty_side_of_the_acceptance_pair_is_null_and_a_measured_zero_stays_zero(export):
    modes = {m["mode"]: m for m in phys(export)["modes"]}
    assert modes["frozen"]["acceptance"] == {"accepted_good": None, "n_good": 0, "refused_bad": None, "n_bad": 0}
    assert modes["continued"]["acceptance"] == {"accepted_good": 1.0, "n_good": 2, "refused_bad": 0.0, "n_bad": 1}
    assert ex.rate(0.0, 0) is None and ex.rate(0.0, 1) == 0.0 and ex.rate(None, 3) is None
    assert modes["in_context"]["tokens_measured"] > modes["in_context"]["tokens_measured_without_context"]


def test_report_sets_use_the_dynamics_learners_own_off_intervention():
    model = ablation.build("delta_baseline", d_model=16, n_heads=2, n_layers=1, chunk=16, seed=0)
    _, label = ablation.learner_for("delta_baseline", model, torch.device("cpu"))
    assert ex.DYNAMICS_NO_ADAPT == label
    coord = ablation.build("full", d_model=16, n_heads=2, n_layers=1, chunk=16, seed=0)
    assert ablation.learner_for("full", coord, torch.device("cpu"))[1] != label  # freeze=True is a different intervention


@pytest.mark.parametrize(("curve", "reached", "step"), [([0.9, 0.4, 0.3], True, 2), ([0.9, 0.8, 0.7], False, None), ([0.9, 0.8, 0.4], True, 3)])
def test_speed_separates_never_below_half_from_a_step(curve, reached, step):
    steps = next((i + 1 for i, v in enumerate(curve) if v < 0.5), len(curve))  # the contract's own rule
    v = ex.speed_view({"curve": curve, "area": sum(curve) / len(curve), "steps_to_half": steps, "episodes": 8})
    assert (v["reached_half"], v["steps_to_half"], v["horizon"]) == (reached, step, len(curve))
    assert ex.speed_view(None) is None


# ---------------------------------------------------------------------------------------------------- ablation fixtures
def _variant(name: str, *, version: str = CONTRACT_VERSION, period: int | None = 16, label: str | None = None) -> dict:
    contract = copy.deepcopy(json.loads((PHYS / "frozen.json").read_text()))
    contract.pop("mode"), contract.pop("learner")
    contract["contract_version"] = version
    contract["adaptation_window"] = ({"update_period": period, "boundaries_per_episode": 63 // period, "checked": True} if period
                                     else {"update_period": None, "boundaries_per_episode": None, "checked": False})
    eta = None if name == "delta_baseline" else [0.25, 0.3, 0.35]
    return {"variant": name, "config": ablation.VARIANTS[name] if ablation.VARIANTS[name] is not None else "PlasticDynamics",
            "size": {"d_model": 128, "n_heads": 4, "n_layers": 3, "chunk": 16, "parameters": 400000 + len(name)},
            "train": {"steps": 1500, "s_per_step": 0.19, "log": [{"step": 1, "loss": 1.0, "eta": eta}, {"step": 1500, "loss": 0.06, "eta": eta}]},
            "no_adapt_label": label or ("writes disabled (beta_scale=0)" if name == "delta_baseline" else "fast parameters frozen (freeze=True)"),
            "contract": contract}


def _archive(tmp: Path, variants: dict[str, dict]) -> Path:
    arch = tmp / "contract-archive"
    shutil.copytree(ex.ARCHIVE, arch)
    d = arch / "coordinate-ablation"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"execution_commit": "ab" * 20, "args": {"variants": ["delta_baseline"]}}))
    (d / "README.md").write_text("# Coordinate ablation\n")
    for name, body in variants.items():
        (d / f"{name}.json").write_text(json.dumps(body))
    return arch


def ablation_set(index: dict) -> dict:
    return next(s for s in index["sets"] if s["kind"] == "ablation")


def test_a_complete_ablation_keeps_the_runner_order_and_each_variants_own_label(tmp_path):
    arch = _archive(tmp_path, {n: _variant(n) for n in reversed(list(ablation.VARIANTS))})
    a = ablation_set(ex.build(tmp_path / "out", arch))
    assert [v["variant"] for v in a["variants"]] == list(ablation.VARIANTS) and a["missing"] == []
    labels = {v["variant"]: v["no_adapt_label"] for v in a["variants"]}
    assert labels["delta_baseline"].startswith("writes disabled") and labels["full"].startswith("fast parameters frozen")
    full = next(v for v in a["variants"] if v["variant"] == "full")
    assert full["eta_per_layer"] == [0.25, 0.3, 0.35] and full["final_train_loss"] == 0.06 and full["parameters"] == 400004
    assert next(v for v in a["variants"] if v["variant"] == "delta_baseline")["eta_per_layer"] is None
    assert full["before"] == phys(ex.build(tmp_path / "ref", ex.ARCHIVE))["before"]
    assert a["current"] is True and a["contract_version"] == CONTRACT_VERSION and a["execution_commit"] == "ab" * 6
    assert {"coordinate-ablation/manifest.json", "coordinate-ablation/README.md", "coordinate-ablation/full.json"} <= {s["file"] for s in a["sources"]}


def test_a_partial_ablation_lists_what_is_missing_in_the_runner_order_not_the_manifest(tmp_path):
    arch = _archive(tmp_path, {n: _variant(n) for n in ("no_fast", "full")})
    a = ablation_set(ex.build(tmp_path / "out", arch))
    assert [v["variant"] for v in a["variants"]] == ["full", "no_fast"]
    assert a["missing"] == [n for n in ablation.VARIANTS if n not in ("full", "no_fast")]


def test_a_superseded_contract_version_marks_the_set_not_current_and_an_undeclared_window_unchecked(tmp_path):
    arch = _archive(tmp_path, {"full": _variant("full", version="2026-09-23.1"), "no_fast": _variant("no_fast", period=None)})
    a = ablation_set(ex.build(tmp_path / "out", arch))
    assert a["current"] is False and a["contract_version"] is None and a["contract_versions"] == ["2026-09-23.1", CONTRACT_VERSION]
    by = {v["variant"]: v for v in a["variants"]}
    assert by["full"]["current"] is False and by["no_fast"]["current"] is True
    assert by["no_fast"]["adaptation_window"] == {"update_period": None, "boundaries_per_episode": None, "checked": False}


def test_no_ablation_archive_is_absent_not_an_empty_set(export):
    if (ex.ARCHIVE / "coordinate-ablation").exists():
        pytest.skip("the ablation is archived")
    assert [s["kind"] for s in export["sets"]] == ["report"]


@pytest.mark.parametrize("breakage", ["before", "version", "window"])
def test_the_exporter_refuses_a_report_set_whose_modes_disagree(tmp_path, breakage):
    arch = tmp_path / "arch"
    shutil.copytree(ex.ARCHIVE, arch)
    p = arch / "phys_mps_3k" / "continued.json"
    rep = json.loads(p.read_text())
    if breakage == "before":
        rep["transfer"]["hold"]["before"]["adapt"] += 0.01
    elif breakage == "version":
        rep["contract_version"] = "2026-09-23.1"
    else:
        rep["adaptation_window"]["boundaries_per_episode"] = 3
    p.write_text(json.dumps(rep))
    with pytest.raises(ValueError, match="phys_mps_3k/continued.json"):
        ex.build(tmp_path / "out", arch)


def test_output_is_deterministic_and_carries_no_local_paths(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ex.build(a), ex.build(b)
    text = (a / "index.json").read_text()
    assert text == (b / "index.json").read_text()
    assert str(ex.ROOT) not in text and "/Users/" not in text and not re.search(r"\d{4}-\d\d-\d\d \d\d:\d\d", text)


def test_the_committed_export_and_its_mirror_are_current_and_apart_from_the_sleep_export():
    assert ex.main(["--check"]) == 0, "refresh with: python -m scripts.export_contract_observatory"
    assert not ex.MIRROR.resolve().is_relative_to(sleep_ex.MIRROR.resolve())  # the sleep mirror step replaces its whole directory
    assert json.loads((ex.MIRROR / "index.json").read_text())["schema"] == ex.SCHEMA
