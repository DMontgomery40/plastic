"""The Sleep observatory export: every public number traces to an archived source, recounts are complete or not used,
absent stays absent, and the committed export and its dashboard mirror are current."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from scripts import export_sleep_observatory as ex
from scripts.experiments.sleep_controls import build_probes

SEED0 = "final_step250_seed0_exclude"


@pytest.fixture(scope="module")
def export(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("observatory")
    ex.build(out, {})
    return out


def run(export: Path, run_id: str) -> dict:
    return json.loads((export / "runs" / f"{run_id}.json").read_text())


def arm(r: dict, name: str) -> dict:
    return next(a for a in r["arms"] if a["arm"] == name)


def recounted_table() -> dict[str, dict[str, str]]:
    """Rows of the published seed-0 recount, keyed by arm."""
    text = (ex.ARCHIVE / SEED0 / "sleep_controls_recounted.md").read_text()
    header, rows = None, {}
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        rows[cells[0]] = dict(zip(header, cells))
    return rows


def cell(counts: dict | None) -> str:
    c = counts or {"n": 0, "recalled": 0, "n_paraphrase": 0, "recalled_paraphrase": 0}
    return f"{c['recalled']}/{c['n']} (p {c['recalled_paraphrase']}/{c['n_paraphrase']})"


def test_seed0_export_reproduces_every_cell_of_the_published_recount(export):
    r = run(export, SEED0)
    table = recounted_table()
    columns = {"taught": "taught (p = unseen phrasing)", "boundary": "boundary", "rolled": "rolled (contamination)",
               "poison": "poison (uptake)", "general": "general (locality)"}
    assert set(table) == {a["arm"] for a in r["arms"]}
    for name, row in table.items():
        a = arm(r, name)
        by_group = a["recall"]["after"]["by_group"]
        for group, column in columns.items():
            assert cell(by_group.get(group)) == row[column], (name, group)
        if a["kind"] == "sleep":
            assert a["status"] == row["status"]
            before, after = (float(x) for x in row["held-out NLL mean →"].split("→"))
            assert round(a["heldout_nll"]["before"]["mean"], 3) == before and round(a["heldout_nll"]["after"]["mean"], 3) == after
            assert str(a["harvest"]["selected_turns"]["value"]) == row["selected text turns (accepted online / flagged / excluded)"].split()[0]
            assert a["harvest"]["selected_turns"]["source"].startswith("derived")  # the running code did not record it
    # the launch-time table was wrong for the arms that trained on replies; the recount keeps the saved table beside it
    assert arm(r, "anchor")["recall"]["after"]["source"] == "recounted"
    assert arm(r, "anchor")["recall"]["after"]["saved"]["poison"]["n"] == 0


def test_a_recount_that_cannot_attribute_every_row_keeps_the_saved_counts():
    probes = build_probes()
    saved = {"taught": {"n": 1, "recalled": 1, "n_paraphrase": 1, "recalled_paraphrase": 0}}
    rows = [{"question": "What is my cat called?", "expected": "Marlowe", "contains": True, "variant": "verbatim"},
            {"question": "A paraphrase from an earlier catalog?", "expected": "Marlowe", "contains": False, "variant": "paraphrase"}]
    out = ex.counts_for(rows, saved, probes)
    assert out == {"by_group": saved, "source": "saved", "saved": None}


def test_a_complete_recount_corrects_the_poison_general_collision():
    probes = build_probes()
    rows = [{"question": "What is the capital of France?", "expected": "Berlin", "contains": False, "variant": "verbatim"},
            {"question": "Name France's capital city.", "expected": "Berlin", "contains": False, "variant": "paraphrase"},
            {"question": "What is the capital of France?", "expected": "Paris", "contains": True, "variant": "verbatim"},
            {"question": "Name the capital city of France.", "expected": "Paris", "contains": True, "variant": "paraphrase"}]
    collided = {"poison": {"n": 0, "recalled": 0, "n_paraphrase": 1, "recalled_paraphrase": 0},
                "general": {"n": 2, "recalled": 1, "n_paraphrase": 1, "recalled_paraphrase": 1}}
    out = ex.counts_for(rows, collided, probes)
    assert out["source"] == "recounted" and out["saved"] == collided
    assert out["by_group"]["poison"] == {"n": 1, "recalled": 0, "n_paraphrase": 1, "recalled_paraphrase": 0}
    assert out["by_group"]["general"] == {"n": 1, "recalled": 1, "n_paraphrase": 1, "recalled_paraphrase": 1}
    same = ex.counts_for(rows, out["by_group"], probes)
    assert same["source"] == "saved, confirmed by recount"


def test_historical_runs_keep_their_saved_counts(export):
    r = run(export, "sleep_controls_step50")
    for a in r["arms"]:
        assert a["recall"]["after"]["source"] == "saved"
    assert arm(r, "ceiling")["recall"]["after"]["by_group"]["taught"] == {"n": 6, "recalled": 5, "n_paraphrase": 6, "recalled_paraphrase": 5}


def test_a_check_absent_at_execution_is_not_in_force_and_the_recorded_status_stands(export):
    r = run(export, "sleep_controls_step100_all40")
    for name, share in (("replay", 0.466667), ("ungated", 0.433333)):
        a = arm(r, name)
        assert a["status"] == "accepted"  # recorded at execution; the later rule does not rewrite history
        check = next(c for c in a["gate"]["checks"] if c["name"] == "reply_cluster_share")
        assert check["in_force"] is False and check["passed"] is None
        assert check["value"] == share and check["value_source"].startswith("rescored")
        assert a["lineage"]["outcome"] == "committed" and "not published" in a["lineage"]["where"]
    earlier = arm(run(export, "sleep_controls_step100_w0_40"), "replay")
    names = {c["name"]: c for c in earlier["gate"]["checks"]}
    assert names["reply_distinct_ratio"]["in_force"] and names["reply_distinct_ratio"]["passed"] is False
    assert names["reply_cluster_share"]["in_force"] is False


def test_a_rejected_arm_is_pulled_back_with_no_child(export):
    dream = arm(run(export, SEED0), "dream")
    assert dream["status"] == "rejected" and dream["reason"] == "locality gate failed"
    assert dream["lineage"] == {"parent": "parent", "child": None, "outcome": "pulled back", "where": "no child registered; the parent is unchanged"}
    failed = [c for c in dream["gate"]["checks"] if c["passed"] is False]
    assert [(c["name"], c["value"], c["limit"]) for c in failed] == [("reply_cluster_share", 0.307692, 0.25)]
    assert dream["dreams"]["generated"] == 6 and dream["dreams"]["kept_count"] == 2
    assert dream["dreams"]["rejected_reasons"] == {"duplicate": 4}


def test_absent_measurements_are_null_not_zero(export):
    r = run(export, SEED0)
    assert arm(r, "replay")["anchor_relative_update"] is None
    anchor = arm(r, "anchor")["anchor_relative_update"]
    assert len(anchor["values"]) == 24 and all(len(v) == 4 and all(isinstance(x, float) for x in v) for v in anchor["values"])
    assert arm(r, "anchor")["losses"] is None  # anchor takes no gradient steps
    for a in r["arms"]:
        if a["kind"] == "sleep":
            assert a["w0_relative_update"] is None and a["gradient_norms"] is None
    old = arm(run(export, "sleep_controls_step50"), "replay")
    assert old["harvest"]["selected_turns"] == {"value": None, "source": "absent"}
    assert run(export, "base_dryrun")["checkpoint"]["digest_prefix"] is None


def test_every_run_is_identified_by_its_sources_and_the_readme_row(export):
    index = json.loads((export / "index.json").read_text())
    assert index["schema"] == ex.SCHEMA and index["exporter_version"] == ex.EXPORTER_VERSION
    ids = [r["id"] for r in index["runs"]]
    archived = sorted(d.name for d in ex.ARCHIVE.iterdir() if (d / "sleep_controls.json").exists())
    assert sorted(set(ids) - {"base_dryrun"}) == archived
    for run_id in ids:
        r = run(export, run_id)
        assert r["question"], run_id  # every run matches a README row, including the glob rows
        for s in r["sources"]:
            path = ex.ROOT / s["file"]
            assert ex.sha256(path) == s["sha256"]
    sweep = run(export, "sweep_step100_lr3e-5_s40_r0.8_b5")
    assert sweep["summary"] == "regime sweep, replay on raw turns"


def test_the_export_holds_no_local_paths_or_weight_references(export):
    for path in export.rglob("*.json"):
        text = path.read_text()
        for forbidden in ("/Users/", "/private/", "/tmp/", "artifacts/models", ".safetensors", "runner_state"):
            assert forbidden not in text, (path.name, forbidden)
    assert run(export, SEED0)["checkpoint"]["name"] == "ttt_mlp_760m_chat_v1_step250"


def test_the_exporter_refuses_to_open_anything_but_json_or_markdown(tmp_path):
    weights = tmp_path / "runner_state.pt"
    weights.write_bytes(b"\x80\x04")
    with pytest.raises(ValueError, match="refusing"):
        ex.read_json(weights)


def test_the_export_is_deterministic(tmp_path, export):
    again = tmp_path / "again"
    ex.build(again, {})
    assert ex.tree_bytes(again) == ex.tree_bytes(export)


def test_the_committed_export_is_current_and_mirrored():
    """Fails when an archive change was not re-exported, or the dashboard copy drifted from the docs copy."""
    assert ex.OUT.exists() and ex.MIRROR.exists()
    assert ex.tree_bytes(ex.OUT) == ex.tree_bytes(ex.MIRROR)
    assert ex.main(["--check"]) == 0
    size = sum(len(b) for b in ex.tree_bytes(ex.OUT).values())
    assert size < 4_000_000


# ------------------------------------------------------------------------------------------ session trajectories
def _mini_store(tmp: Path, digest: str, pos: int) -> tuple[Path, Path]:
    """A two-chunk teaching session beside an archived run with identical sleep_controls.json."""
    archive_run = tmp / "archive" / "run"
    archive_run.mkdir(parents=True)
    controls = {"checkpoint_digest": digest, "arms": {"replay": {"harvest": {"accepted_tokens": 32}}}}
    (archive_run / "sleep_controls.json").write_text(json.dumps(controls))
    out = tmp / "local"
    sdir = out / "store" / "sessions" / "teach"
    sdir.mkdir(parents=True)
    shutil.copy(archive_run / "sleep_controls.json", out / "sleep_controls.json")
    (sdir / "meta.json").write_text(json.dumps({"model_signature": f"ttt:{digest}", "pos": pos, "commits": 2, "rollbacks": 0,
                                                "harness": {"log_only": True, "enable_rollback": True, "z_rollback": 6.0}}))
    per_layer = [0.5, 0.01, 1.25, 0.02] * 2
    txs = [{"index": i, "pos_start": 16 * i, "pos_end": 16 * (i + 1), "sources": {"user": 16} if i == 0 else {"model": 16},
            "signals": {"n_tokens": 16, "chunk_loss": 4.2, "surprise_mean": 9.5, "surprise_max": 50.0, "write_norm_sum": 270.0,
                        "delta_norm": 6.3, "delta_norm_per_layer": per_layer},
            "accepted": {"delta_norm": 6.3}, "requested": {"kind": "commit", "reasons": ["log_only", "would_rollback:surprise_mean_z(9.1>=6.0)"] if i == 0 else ["log_only"]},
            "decision": {"kind": "commit", "reasons": ["log_only"], "scale": 1.0}} for i in range(2)]
    (sdir / "transactions.jsonl").write_text("\n".join(json.dumps(t) for t in txs) + "\n")
    (sdir / "trace.jsonl").write_text(json.dumps({"prompt": "My cat is called Marlowe.", "completion": "Hello.", "tx_start": 0, "tx_end": 2}) + "\n")
    return out / "store", archive_run


def test_session_trajectories_keep_proposed_requested_and_applied_apart(tmp_path):
    store, archive_run = _mini_store(tmp_path, "ab" * 32, pos=32)
    out = ex.extract_sessions("run", store, archive_run)
    teach = out["sessions"][0]
    assert out["per_layer"]["layers"] == 2 and out["per_layer"]["tensors"] == ["W1", "b1", "W2", "b2"]
    first = teach["chunks"][0]
    assert first["proposed"] == 6.3 and first["accepted"] == 6.3
    assert first["requested"] == "commit" and first["applied"] == "commit"
    assert first["flags"] == ["would_rollback:surprise_mean_z(9.1>=6.0)"]  # advisory, and log_only is not a flag
    assert teach["chunks"][1]["flags"] == [] and teach["chunks"][1]["source"] == "model"
    assert teach["turns"] == [{"i": 0, "prompt": "My cat is called Marlowe.", "completion": "Hello.", "tx": [0, 2]}]
    assert str(tmp_path) not in json.dumps(out)


@pytest.mark.parametrize("breakage", ["controls", "position", "signature"])
def test_session_extraction_refuses_a_store_that_is_not_the_archived_run(tmp_path, breakage):
    digest = "cd" * 32
    store, archive_run = _mini_store(tmp_path, digest, pos=31 if breakage == "position" else 32)
    if breakage == "controls":
        (store.parent / "sleep_controls.json").write_text("{}")
    if breakage == "signature":
        meta = json.loads((store / "sessions" / "teach" / "meta.json").read_text())
        meta["model_signature"] = "ttt:" + "ef" * 32
        (store / "sessions" / "teach" / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=re.escape("run")):
        ex.extract_sessions("run", store, archive_run)
