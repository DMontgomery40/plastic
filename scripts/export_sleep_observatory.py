"""Export the archived Sleep runs as compact JSON for the public Sleep observatory.

Reads only JSON, JSON-lines and Markdown under ``docs/research/results/`` (and, optionally, the session files of a
local experiment store); never model weights. Writes one index plus one file per run to
``docs/research/results/sleep-observatory/`` and mirrors the same bytes into ``dashboard/public/assets/observatory/``,
which the dashboard build copies into ``dist/assets`` (the only static path the Space serves).

Every value is copied from a source file or computed by the project's own functions, never invented:

* Per-group recall is recounted with ``scripts.experiments.sleep_controls.group_counts`` only when every saved probe
  row is attributed to a probe of the current catalog. Otherwise the saved counts stand, and the source says so.
* A locality check that a run's gate did not record was not in force when it ran. Where the replies allow it, the
  current cluster-share rule is rescored with ``plastic.sleep.recall.RecallReport`` and labelled as a rescore.
* Recorded statuses are kept. An accepted child lives in the run's disposable experiment store; it is not published.
* Missing measurements stay missing (``null``), never zero.

The output is deterministic: no wall-clock time and no absolute paths. It is identified by the sha256 of each
source file, an archive digest over them, and ``EXPORTER_VERSION``.

Usage
  python -m scripts.export_sleep_observatory                       # refresh from the archive
  python -m scripts.export_sleep_observatory --session-store final_step250_seed0_exclude=<store dir>
  python -m scripts.export_sleep_observatory --check               # exit 1 if the committed export is stale
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from plastic.sleep.recall import RecallReport, RecallResult
from scripts.experiments.sleep_controls import build_probes, group_counts

EXPORTER_VERSION = "1.0.0"
SCHEMA = "plastic.sleep-observatory/1"
ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "docs" / "research" / "results" / "sleep-2026-09-23"
CHAT_EVAL = ROOT / "docs" / "research" / "results" / "chat-eval-2026-09-23"
OUT = ROOT / "docs" / "research" / "results" / "sleep-observatory"
MIRROR = ROOT / "dashboard" / "public" / "assets" / "observatory"

CONTROL_ARMS = ("floor", "ceiling")
ARM_ORDER = ("floor", "ceiling", "anchor", "replay", "distill", "dream", "ungated")
GROUPS = ("taught", "boundary", "rolled", "poison", "general")
TENSORS = ("W1", "b1", "W2", "b2")
# Locality checks the current gate knows. A run whose gate lacks one ran before that check existed.
CURRENT_CHECKS = ("heldout_nll_mean_rise", "reply_cluster_share")
CHECK_LABELS = {
    "heldout_nll_mean_rise": "held-out NLL rise",
    "reply_cluster_share": "largest identical-reply share",
    "reply_distinct_ratio": "distinct-reply ratio (earlier rule)",
    "canary_coherence": "canary coherence change",
    "canary_poison": "canary poison change",
}
MAX_REPLY_CHARS = 280
READABLE = {".json", ".jsonl", ".md"}


# ---------------------------------------------------------------------------------------------------- small helpers
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    if path.suffix not in READABLE:  # the exporter never opens weights or pickles
        raise ValueError(f"refusing to read {path.name}: only {sorted(READABLE)} are exported")
    return json.loads(path.read_text(encoding="utf-8"))


def num(x: Any, sig: int = 6) -> float | None:
    """A finite number rounded to ``sig`` significant digits, or None. Booleans are not numbers."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def checkpoint_identity(path_text: str | None, digest: str | None) -> dict[str, Any]:
    """Name, SFT step and digest prefix. Local paths are reduced to their final component (no user directories)."""
    name = os.path.basename(str(path_text or "").rstrip("/")) or None
    m = re.search(r"step(\d+)", name or "")
    step = int(m.group(1)) if m else None
    if name and "base" in name:
        label = "TTT-MLP-760M base (not chat-tuned)"
    elif step == 250:
        label = "final chat checkpoint (SFT step 250 of 250)"
    elif step is not None:
        label = f"intermediate chat checkpoint (SFT step {step} of 250)"
    else:
        label = name or "unrecorded"
    return {"name": name, "step": step, "label": label, "digest_prefix": (digest or "")[:16] or None,
            "published": False}


# ---------------------------------------------------------------------------------------------------- the README
def readme_rows(readme: Path) -> list[dict[str, str]]:
    """Rows of the archive README's run table: pattern (may be a glob), checkpoint, digest, code, question."""
    rows = []
    for line in readme.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or not cells[0].startswith("`"):
            continue
        pattern = cells[0].strip("`")
        rows.append({"pattern": pattern, "checkpoint": cells[1], "digest": cells[2], "code": cells[3],
                     "question": " | ".join(cells[4:])})
    return rows


def plain(md: str) -> str:
    """Markdown cell text without code ticks or bold markers."""
    return re.sub(r"\s+", " ", md.replace("`", "").replace("**", "")).strip()


def match_row(rows: list[dict[str, str]], run_id: str, files: list[str]) -> dict[str, str] | None:
    for row in rows:
        if fnmatch.fnmatch(run_id, row["pattern"]) or any(fnmatch.fnmatch(f, row["pattern"]) for f in files):
            return row
    return None


def first_sentence(text: str) -> str:
    m = re.match(r"(.+?[.:;])(\s|$)", text)
    return (m.group(1) if m else text).rstrip(".:;")


# ---------------------------------------------------------------------------------------------------- recall
def probe_group(row: dict[str, Any], probes: dict[str, list]) -> tuple[str | None, str | None]:
    """The catalog group of one probe row and how it was attributed, using the experiment's own group_counts (not a
    copy of it): by (question, expected answer) first; for rows from an earlier catalog whose expected string changed,
    by the question text alone when that text belongs to exactly one probe. Display only: counts never use this."""
    for candidate, how in ((row, "question and expected answer"), ({k: v for k, v in row.items() if k != "expected"}, "question text only")):
        counts = group_counts([candidate], probes)
        hit = [g for g, c in counts.items() if c["n"] or c["n_paraphrase"]]
        if len(hit) == 1:
            return hit[0], how
    return None, None


def counts_for(results: list[dict[str, Any]] | None, saved: dict[str, Any] | None, probes: dict[str, list]) -> dict[str, Any]:
    """Per-group counts and where they come from. A recount replaces the saved table only when it attributes every
    row; a partial recount would silently shrink denominators for runs that used an earlier probe catalog."""
    saved = saved or None
    if not results:
        return {"by_group": saved, "source": "saved" if saved else "absent", "saved": None}
    recount = group_counts(results, probes)
    attributed = sum(c["n"] + c["n_paraphrase"] for c in recount.values())
    if attributed != len(results):
        return {"by_group": saved, "source": "saved" if saved else "absent", "saved": None}
    recount = {g: c for g, c in recount.items() if c["n"] or c["n_paraphrase"]}
    saved_nonempty = {g: c for g, c in (saved or {}).items() if c.get("n") or c.get("n_paraphrase")}
    if saved is not None and saved_nonempty == recount:
        return {"by_group": recount, "source": "saved, confirmed by recount", "saved": None}
    return {"by_group": recount, "source": "recounted" if saved is not None else "counted from replies", "saved": saved}


def cluster_share(results: list[dict[str, Any]] | None) -> float | None:
    """The current rule's largest identical-reply share, computed by the project's RecallReport."""
    if not results:
        return None
    rep = RecallReport([RecallResult(**{k: r.get(k) for k in ("question", "expected", "reply", "contains", "exact", "variant")})
                        for r in results])
    return rep.max_cluster_share()


def totals(recall: dict[str, Any] | None) -> dict[str, Any] | None:
    if not recall:
        return None
    return {"n_probes": recall.get("n_probes"), "recalled": recall.get("recalled"), "n_paraphrase": recall.get("n_paraphrase"),
            "recalled_paraphrase": recall.get("recalled_paraphrase"), "mean_answer_logprob": num(recall.get("mean_answer_logprob")),
            "max_cluster_share": num(recall.get("max_cluster_share")), "distinct_ratio": num(recall.get("distinct_ratio"))}


def probe_rows(results: list[dict[str, Any]] | None, probes: dict[str, list]) -> list[dict[str, Any]]:
    """Per-probe replies. run_probes asks each probe's paraphrase right after its verbatim question, so a paraphrase from
    an earlier catalog that no longer matches inherits the group of the verbatim row just before it (labelled)."""
    out: list[dict[str, Any]] = []
    for r in results or []:
        reply = str(r.get("reply", ""))
        group, how = probe_group(r, probes)
        prev = out[-1] if out else None
        if group is None and r.get("variant") == "paraphrase" and prev and prev["variant"] == "verbatim" and prev["group"] and prev["expected"] == r.get("expected"):
            group, how = prev["group"], "paired with the preceding verbatim question"
        out.append({"group": group, "group_attribution": how, "variant": r.get("variant", "verbatim"), "question": r.get("question"),
                    "expected": r.get("expected"), "reply": reply[:MAX_REPLY_CHARS], "reply_truncated": len(reply) > MAX_REPLY_CHARS,
                    "hit": bool(r.get("contains")), "exact": bool(r.get("exact")), "answer_logprob": num(r.get("answer_logprob"), 4)})
    return out


# ---------------------------------------------------------------------------------------------------- arms
def gate_view(gate: dict[str, Any] | None, after_results: list[dict[str, Any]] | None, recorded_share: float | None) -> dict[str, Any] | None:
    if gate is None:
        return None
    checks = [{"name": c.get("name"), "label": CHECK_LABELS.get(c.get("name"), c.get("name")), "value": num(c.get("value")),
               "limit": num(c.get("limit")), "passed": bool(c.get("passed")), "in_force": True} for c in gate.get("checks", [])]
    names = {c["name"] for c in checks}
    for name in CURRENT_CHECKS:
        if name in names:
            continue
        entry = {"name": name, "label": CHECK_LABELS[name], "value": None, "limit": None, "passed": None, "in_force": False,
                 "value_source": None}
        if name == "reply_cluster_share":
            if recorded_share is not None:
                entry.update(value=num(recorded_share), value_source="recorded measurement")
            elif after_results:
                entry.update(value=num(cluster_share(after_results)), value_source="rescored from saved replies with the current rule")
            entry["limit"] = 0.25 if entry["value"] is not None else None
        checks.append(entry)
    return {"measured": gate.get("measured"), "passed": gate.get("passed"), "checks": checks}


def selected_turns(harvest: dict[str, Any] | None) -> dict[str, Any]:
    """The selection an arm trained on. Recorded when the report has it; derived only from recorded counts when the run
    records its flagged policy; otherwise absent (older runs predate the selection rule, ASTRA-197/199)."""
    if not harvest:
        return {"value": None, "source": "absent"}
    if harvest.get("selected_turns") is not None:
        return {"value": harvest["selected_turns"], "source": "recorded"}
    policy = harvest.get("flagged_policy")
    by_reason = harvest.get("turns_by_reason") or {}
    accepted = int(by_reason.get("accepted") or 0)
    if policy == "exclude" and harvest.get("provenance") == "accepted" and harvest.get("flagged_excluded") is not None:
        return {"value": accepted - int(harvest["flagged_excluded"]), "source": "derived from recorded harvest counts"}
    if policy == "include" and str(harvest.get("provenance", "")).startswith("all"):
        completed = sum(int(v or 0) for v in by_reason.values())
        return {"value": completed, "source": "derived from recorded harvest counts"}
    if policy == "include" and harvest.get("provenance") == "accepted":
        return {"value": accepted, "source": "derived from recorded harvest counts"}
    return {"value": None, "source": "absent"}


def harvest_view(h: dict[str, Any] | None) -> dict[str, Any] | None:
    if not h:
        return None
    return {"sessions": [{k: s.get(k) for k in ("session_id", "log_only", "turns", "accepted_turns", "has_committed_state")} for s in h.get("sessions", [])],
            "turns_by_reason": h.get("turns_by_reason"), "accepted_tokens": h.get("accepted_tokens"), "excluded_tokens": h.get("excluded_tokens"),
            "accepted_turns_flagged": h.get("accepted_turns_flagged"), "flagged_excluded": h.get("flagged_excluded"),
            "provenance": h.get("provenance"), "flagged_policy": h.get("flagged_policy"), "source_sessions": h.get("source_sessions"),
            "selected_turns": selected_turns(h)}


def anchor_update(rep: dict[str, Any]) -> dict[str, Any] | None:
    raw = rep.get("anchor_relative_update")
    if not raw:
        return None
    return {"quantity": "relative change of W0 per tensor, ||W0_child - W0_parent|| / ||W0_parent||", "tensors": list(TENSORS),
            "values": _per_tensor(raw)}


def _per_tensor(raw: dict[str, Any]) -> list[list[float | None]]:
    layers: dict[int, dict[str, float | None]] = {}
    for key, value in raw.items():
        m = re.search(r"layers\.(\d+)\..*\.(W1|b1|W2|b2)$", key)
        if m:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = num(value, 4)
    n = max(layers) + 1 if layers else 0
    return [[layers.get(i, {}).get(t) for t in TENSORS] for i in range(n)]


def w0_change(rep: dict[str, Any]) -> dict[str, Any] | None:
    """Per-tensor child-vs-parent relative W0 change, recorded for every method (and before a pull-back) since aae9b70."""
    raw = rep.get("w0_relative_change")
    if not raw:
        return None
    return {"quantity": "relative change of W0 per tensor, ||W0_after - W0_before|| / ||W0_before||", "tensors": list(TENSORS),
            "values": _per_tensor(raw), "total": num(raw.get("total"), 4)}


def grad_norms_view(rep: dict[str, Any]) -> dict[str, Any] | None:
    """Per-step gradient norms of the trained parameters, before clipping: total, per decoder layer, and 'other'."""
    rows = rep.get("grad_norms")
    if not rows:
        return None
    layer_keys = sorted({k for r in rows for k in (r.get("per_layer") or {}) if k.isdigit()}, key=int)
    n_layers = int(layer_keys[-1]) + 1 if layer_keys else 0
    return {"quantity": "gradient norm before clipping", "steps": [r.get("step") for r in rows],
            "total": [num(r.get("total"), 4) for r in rows],
            "per_layer": [[num((r.get("per_layer") or {}).get(str(i)), 4) for r in rows] for i in range(n_layers)],
            "other": [num((r.get("per_layer") or {}).get("other"), 4) for r in rows]
            if any("other" in (r.get("per_layer") or {}) for r in rows) else None}


def dreams_view(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if not d:
        return None
    kept = []
    for k in d.get("kept") or []:
        kept.append({"prompt": k.get("prompt"), "turn": k.get("turn"), "text": str(k.get("text", ""))[:MAX_REPLY_CHARS * 2],
                     "session_id": k.get("session_id"), "reply_len": k.get("reply_len"),
                     "teacher_logprob": num(k.get("teacher_logprob"), 4), "student_logprob": num(k.get("student_logprob"), 4),
                     "fastweight_logprob": num(k.get("fastweight_logprob"), 4), "gain": num(k.get("gain"), 4),
                     "fastweight_gain": num(k.get("fastweight_gain"), 4),
                     "token_gain": [num(v, 3) for v in k.get("token_gain") or []] or None,
                     "token_fw_gain": [num(v, 3) for v in k.get("token_fw_gain") or []] or None})
    reasons: dict[str, int] = {}
    for r in d.get("rejected") or []:
        reasons[str(r.get("reason"))] = reasons.get(str(r.get("reason")), 0) + 1
    return {"generated": d.get("generated"), "kept_count": len(d.get("kept") or []),
            "filtered": {k: d.get(k) for k in ("degenerate", "duplicate", "low_gain", "too_long")},
            "rejected_reasons": reasons,
            "rejected_samples": [{"reason": r.get("reason"), "text": str(r.get("text", ""))[:MAX_REPLY_CHARS]} for r in (d.get("rejected") or [])[:4]],
            "kept": kept}


def lineage(status: str | None, child: str | None, parent: str | None) -> dict[str, Any]:
    if status in ("accepted", "accepted_unmeasured"):
        return {"parent": parent, "child": child, "outcome": "committed",
                "where": "registered in the run's disposable experiment store; not published"}
    if status == "rejected":
        return {"parent": parent, "child": None, "outcome": "pulled back", "where": "no child registered; the parent is unchanged"}
    return {"parent": parent, "child": child, "outcome": "unknown", "where": None}


def sleep_arm(arm: str, summary: dict[str, Any] | None, rep: dict[str, Any] | None, rep_path: Path | None,
              probes: dict[str, list]) -> dict[str, Any]:
    """One consolidation arm: the run-level summary (``sleep_controls.json``) plus its own report."""
    summary = summary or {}
    rep = rep or {}
    before = rep.get("before") or {}
    after = rep.get("after") or {}
    after_results = (after.get("recall") or {}).get("results") or summary.get("results")
    before_results = (before.get("recall") or {}).get("results")
    status = summary.get("status") or rep.get("status")
    child = summary.get("model_id", rep.get("model_id"))
    recorded_share = (after.get("recall") or {}).get("max_cluster_share")
    cfg = rep.get("config") or {}
    return {
        "arm": arm, "kind": "sleep", "method": cfg.get("method") or ("replay" if arm == "ungated" else arm),
        "target": cfg.get("target"), "status": status,
        "status_conflict": bool(summary.get("status") and rep.get("status") and summary["status"] != rep["status"]),
        "reason": summary.get("reason", rep.get("reason")),
        "lineage": lineage(status, child, rep.get("parent_model_id")),
        "gate": gate_view(summary.get("gate") or rep.get("gate"), after_results, recorded_share),
        "heldout_nll": {"before": {k: num(v) for k, v in (summary.get("heldout_nll_before") or before.get("heldout_nll") or {}).items()} or None,
                        "after": {k: num(v) for k, v in (summary.get("heldout_nll_after") or after.get("heldout_nll") or {}).items()} or None},
        "recall": {"before": counts_for(before_results, summary.get("by_group_before"), probes),
                   "after": counts_for(after_results, summary.get("by_group"), probes),
                   "totals_before": totals(before.get("recall")), "totals_after": totals(after.get("recall"))},
        "harvest": harvest_view(summary.get("harvest") or rep.get("harvest")),
        "batch": summary.get("batch") or rep.get("batch"), "packed": rep.get("packed"),
        "losses": [num(x, 5) for x in rep.get("losses") or []] or None,
        "loss_terms": (summary.get("batch") or rep.get("batch") or {}).get("loss_terms"),
        "dreams": dreams_view(summary.get("dreams") or rep.get("dreams")),
        "anchor_relative_update": anchor_update(rep),
        "w0_relative_update": w0_change(rep),     # every method, in reports written after aae9b70; absent before
        "gradient_norms": grad_norms_view(rep),   # per step, before clipping, in reports written after aae9b70
        "config": {k: cfg.get(k) for k in ("steps", "lr", "batch_size", "seq_len", "replay_ratio", "replay_rows", "heldout_rows", "anchor_lambda",
                                           "tolerance_nll", "tolerance_collapse", "session_loss", "prompt_loss_weight", "provenance",
                                           "flagged_policy", "dream_temperature", "dream_min_gain", "dream_token_weighting") if k in cfg} or None,
        "probes": probe_rows(after_results, probes),
        "report": ({"file": rel(rep_path), "sha256": sha256(rep_path), "run_id": rep.get("run_id"),
                    "head_at_arm_start": rep.get("code_commit"), "code_at_import": rep.get("code_commit_at_import"),
                    "seconds": num(rep.get("seconds"), 5)} if rep_path else None),
    }


def control_arm(arm: str, summary: dict[str, Any], probes: dict[str, list]) -> dict[str, Any]:
    results = summary.get("results")
    share = summary.get("max_cluster_share")
    return {"arm": arm, "kind": "control", "method": None, "status": None,
            "recall": {"after": counts_for(results, summary.get("by_group"), probes)},
            "mean_answer_logprob": num(summary.get("mean_answer_logprob")),
            "max_cluster_share": {"value": num(share), "source": "recorded"} if share is not None else (
                {"value": num(cluster_share(results)), "source": "computed from saved replies"} if results else {"value": None, "source": "absent"}),
            "probes": probe_rows(results, probes)}


# ---------------------------------------------------------------------------------------------------- runs
def protocol(c: dict[str, Any], arms: dict[str, Any]) -> dict[str, Any]:
    policies = sorted({(a.get("harvest") or {}).get("flagged_policy") for a in arms.values() if a.get("kind") == "sleep" and a.get("arm") != "ungated"} - {None})
    return {"facts": c.get("facts"), "poison": c.get("poison"), "seed": c.get("seed", 0 if c.get("started_at_unix") else None),
            "steps": c.get("steps"), "target": c.get("target"), "lr": c.get("lr"), "replay_ratio": c.get("replay_ratio"),
            "batch_size": c.get("batch_size"), "session_loss": c.get("session_loss"), "prompt_loss_weight": c.get("prompt_loss_weight"),
            "augment": c.get("augment"), "teach_temperature": c.get("teach_temperature"), "dream_temperature": c.get("dream_temperature"),
            "replay_revision": (c.get("replay_revision") or "")[:12] or None, "device": c.get("device"),
            "ceiling_mode": c.get("ceiling_mode", "all" if "ceiling" in c.get("arms", {}) else None),
            "flagged_policy": c.get("flagged_policy") or (policies[0] if len(policies) == 1 else None),
            "wall_seconds": num(c.get("seconds"), 5)}


def export_run(run_dir: Path, rows: list[dict[str, str]], probes: dict[str, list]) -> dict[str, Any]:
    controls_path = run_dir / "sleep_controls.json"
    c = read_json(controls_path)
    sources = [controls_path]
    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        summary = c.get("arms", {}).get(arm)
        rep_path = run_dir / f"sleep_{arm}_report.json"
        if summary is None and not rep_path.exists():
            continue
        if arm in CONTROL_ARMS:
            arms[arm] = control_arm(arm, summary, probes)
            continue
        rep = read_json(rep_path) if rep_path.exists() else None
        if rep is not None:
            sources.append(rep_path)
        arms[arm] = sleep_arm(arm, summary, rep, rep_path if rep else None, probes)
    for md in sorted(run_dir.glob("*.md")):
        sources.append(md)
    report_times = [read_json(p).get("created_at_unix") for p in sources if p.name.endswith("_report.json")]
    started = c.get("started_at_unix") or min((t for t in report_times if t), default=None)
    row = match_row(rows, run_dir.name, [p.name for p in sources])
    question = plain(row["question"]) if row else None
    return {
        "id": run_dir.name, "question": question, "summary": first_sentence(question) if question else None,
        "started_at_unix": started, "checkpoint": checkpoint_identity(c.get("checkpoint"), c.get("checkpoint_digest")),
        "code": {"recorded_at_launch": c.get("code_commit"), "archive_readme": plain(row["code"]) if row else None},
        "protocol": protocol(c, arms), "arms": [arms[a] for a in ARM_ORDER if a in arms],
        "notes": {"recount": (run_dir / "sleep_controls_recounted.md").exists()},
        "sources": [{"file": rel(p), "sha256": sha256(p)} for p in sources],
    }


def export_dryrun(paths: list[Path], rows: list[dict[str, str]], probes: dict[str, list]) -> dict[str, Any]:
    """The base-model dry run has no sleep_controls.json: one report per method."""
    arms = []
    first = None
    for p in sorted(paths, key=lambda p: ARM_ORDER.index(read_json(p)["config"]["method"])):
        rep = read_json(p)
        first = first or rep
        arms.append(sleep_arm(rep["config"]["method"], None, rep, p, probes))
    cfg = first["config"]
    row = match_row(rows, "base_dryrun", [p.name for p in paths])
    question = plain(row["question"]) if row else None
    return {
        "id": "base_dryrun", "question": question, "summary": first_sentence(question) if question else None,
        "started_at_unix": min(read_json(p).get("created_at_unix") or 0 for p in paths),
        "checkpoint": {**checkpoint_identity("ttt_mlp_760m_base", None), "digest_prefix": None},
        "code": {"recorded_at_launch": None, "archive_readme": plain(row["code"]) if row else None},
        "protocol": {"facts": 3, "poison": False, "seed": cfg.get("seed"), "steps": cfg.get("steps"), "target": cfg.get("target"),
                     "lr": cfg.get("lr"), "replay_ratio": cfg.get("replay_ratio"), "batch_size": cfg.get("batch_size"),
                     "session_loss": cfg.get("session_loss"), "prompt_loss_weight": cfg.get("prompt_loss_weight"), "augment": None,
                     "teach_temperature": None, "dream_temperature": None, "replay_revision": None, "device": cfg.get("device"),
                     "ceiling_mode": None, "flagged_policy": cfg.get("flagged_policy"), "wall_seconds": None},
        "arms": arms, "notes": {"recount": False},
        "sources": [{"file": rel(p), "sha256": sha256(p)} for p in sorted(paths)],
    }


def chat_eval() -> dict[str, Any] | None:
    """The final checkpoint's chat evaluation: context for every step-250 run's parent."""
    path = CHAT_EVAL / "step250_t0.7.json"
    if not path.exists():
        return None
    d = read_json(path)
    return {"checkpoint_digest_prefix": str(d.get("checkpoint_digest", ""))[:16], "temperature": (d.get("sampling") or {}).get("temperature"),
            "distinct_reply_share": num(d.get("distinct_reply_share")), "mean_repetition_share": num(d.get("mean_repetition_share")),
            "held_out_nll": {k: ({kk: num(vv) for kk, vv in v.items()} if isinstance(v, dict) else num(v)) for k, v in (d.get("held_out_nll") or {}).items()},
            "source": {"file": rel(path), "sha256": sha256(path)}}


# ---------------------------------------------------------------------------------------------------- session trajectories
def _flags(reasons: list[str]) -> list[str]:
    return [r for r in reasons if r != "log_only"]


def extract_sessions(run_id: str, store: Path | None, archive_run: Path) -> dict[str, Any]:
    """Per-chunk trajectories of a run's teaching and rolled-back sessions. Read from the public results archive
    (``<run>/sessions/``) when it holds them; otherwise from a local experiment store, only when that store provably
    belongs to the archived run (identical sleep_controls.json). Both paths check the position and model signature.
    A local store's path is not recorded; the file digests are."""
    archived = archive_run / "sleep_controls.json"
    public = store is None
    if public:
        store = archive_run
        if not (store / "sessions").is_dir():
            raise ValueError(f"{run_id}: the archive holds no session files")
    else:
        store = store.resolve()
        local = store.parent / "sleep_controls.json"
        if not local.exists() or sha256(local) != sha256(archived):
            raise ValueError(f"{run_id}: the store's sleep_controls.json is not byte-identical to the archived run")
    c = read_json(archived)
    digest = c.get("checkpoint_digest") or ""
    harvest = next((a["harvest"] for a in c["arms"].values() if isinstance(a, dict) and a.get("harvest")), None)
    sessions = []
    for sid in ("teach", "rolled"):
        sdir = store / "sessions" / sid
        if not sdir.exists():
            continue
        meta = read_json(sdir / "meta.json")
        if not str(meta.get("model_signature", "")).endswith(digest):
            raise ValueError(f"{run_id}/{sid}: model signature does not match the archived checkpoint digest")
        tx_path, trace_path = sdir / "transactions.jsonl", sdir / "trace.jsonl"
        txs = [json.loads(line) for line in tx_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()] if trace_path.exists() else []
        if sid == "teach" and harvest is not None and meta.get("pos") != harvest.get("accepted_tokens"):
            raise ValueError(f"{run_id}/teach: position {meta.get('pos')} differs from the harvest's accepted tokens")
        chunks = []
        for t in txs:
            s = t.get("signals") or {}
            src = t.get("sources") or {}
            chunks.append({
                "i": t.get("index"), "pos": [t.get("pos_start"), t.get("pos_end")], "n": s.get("n_tokens"),
                "source": "user" if (src.get("user") or 0) >= (src.get("model") or 0) else "model",
                "chunk_loss": num(s.get("chunk_loss"), 4), "surprise_mean": num(s.get("surprise_mean"), 4),
                "surprise_max": num(s.get("surprise_max"), 4), "write_norm_sum": num(s.get("write_norm_sum"), 4),
                "proposed": num(s.get("delta_norm"), 4), "accepted": num((t.get("accepted") or {}).get("delta_norm"), 4),
                "per_layer": [num(v, 3) for v in s.get("delta_norm_per_layer") or []] or None,
                "requested": (t.get("requested") or {}).get("kind"), "applied": (t.get("decision") or {}).get("kind"),
                "scale": num((t.get("decision") or {}).get("scale"), 4),
                "flags": _flags((t.get("requested") or {}).get("reasons") or []),
                "applied_reasons": _flags((t.get("decision") or {}).get("reasons") or []),
                "read_only": bool(t.get("read_only")),
            })
        h = meta.get("harness") or {}
        sessions.append({
            "session_id": sid, "log_only": bool(h.get("log_only")), "enable_rollback": h.get("enable_rollback"),
            "z_rollback": h.get("z_rollback"), "z_scale": h.get("z_scale"), "calibrated": False if not meta.get("extra", {}).get("calibration") else True,
            "counts": {k: meta.get(k) for k in ("commits", "rollbacks", "scales", "projects", "readonly", "n_transactions", "pos")},
            "turns": [{"i": i, "prompt": tr.get("prompt"), "completion": str(tr.get("completion", ""))[:MAX_REPLY_CHARS],
                       "tx": [tr.get("tx_start"), tr.get("tx_end")]} for i, tr in enumerate(trace)],
            "chunks": chunks,
            "source": {"transactions": rel(tx_path) if public else None, "transactions_sha256": sha256(tx_path),
                       "trace_sha256": sha256(trace_path) if trace_path.exists() else None},
        })
    n_leaves = max((len(ch["per_layer"] or []) for s in sessions for ch in s["chunks"]), default=0)
    return {"run_id": run_id, "schema": SCHEMA,
            "provenance": {"kind": "public results archive" if public else "local experiment store; these session files are not in the public results archive",
                           "controls_sha256": sha256(archived),
                           "checks": ([] if public else ["sleep_controls.json byte-identical to the archive"])
                           + ["teach position equals the harvest's accepted tokens", "model signature matches the checkpoint digest"]},
            "per_layer": {"quantity": "proposed fast-weight change per tensor in effective coordinates (norm of working minus committed)",
                          "layers": n_leaves // len(TENSORS), "tensors": list(TENSORS), "order": "layer-major: layer 0 W1, b1, W2, b2, layer 1 ..."},
            "sessions": sessions}


# ---------------------------------------------------------------------------------------------------- driver
def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False) + "\n"


def build(out: Path, stores: dict[str, Path]) -> dict[str, Any]:
    probes = build_probes()
    rows = readme_rows(ARCHIVE / "README.md")
    runs = [export_run(d, rows, probes) for d in sorted(ARCHIVE.iterdir()) if d.is_dir() and (d / "sleep_controls.json").exists()]
    dry = sorted(ARCHIVE.glob("base_dryrun_*_report.json"))
    if dry:
        runs.append(export_dryrun(dry, rows, probes))
    runs.sort(key=lambda r: (r["started_at_unix"] or 0, r["id"]))

    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    archived_sessions = {r["id"] for r in runs if (ARCHIVE / r["id"] / "sessions").is_dir()}
    for run_id in sorted(archived_sessions | set(stores)):
        source = None if run_id in archived_sessions else stores[run_id]  # the public archive wins over a local store
        (out / "sessions" / f"{run_id}.json").write_text(dump(extract_sessions(run_id, source, ARCHIVE / run_id)), encoding="utf-8")
    session_ids = sorted(p.stem for p in (out / "sessions").glob("*.json"))
    run_ids = {r["id"] for r in runs}
    for stale in (out / "runs").glob("*.json"):
        if stale.stem not in run_ids:
            stale.unlink()

    for r in runs:
        r["session_trajectory"] = f"sessions/{r['id']}.json" if r["id"] in session_ids else None
        (out / "runs" / f"{r['id']}.json").write_text(dump(r), encoding="utf-8")

    source_digests = sorted({(s["file"], s["sha256"]) for r in runs for s in r["sources"]})
    archive_digest = hashlib.sha256("".join(f"{f}:{d}\n" for f, d in source_digests).encode()).hexdigest()
    index = {
        "schema": SCHEMA, "exporter_version": EXPORTER_VERSION, "archive_digest": archive_digest,
        "archive": rel(ARCHIVE), "chat_eval": chat_eval(),
        "runs": [{
            "id": r["id"], "summary": r["summary"], "started_at_unix": r["started_at_unix"], "checkpoint": r["checkpoint"],
            "code_recorded": r["code"]["recorded_at_launch"], "facts": r["protocol"]["facts"], "flagged_policy": r["protocol"]["flagged_policy"],
            "steps": r["protocol"]["steps"], "target": r["protocol"]["target"], "session_trajectory": r["session_trajectory"],
            "arms": [{"arm": a["arm"], "status": a["status"], "method": a["method"],
                      "taught": ((a["recall"]["after"]["by_group"] or {}).get("taught") if a["recall"]["after"]["by_group"] else None),
                      "totals": a["recall"].get("totals_after") if a["kind"] == "sleep" else None,
                      "nll_before": ((a.get("heldout_nll") or {}).get("before") or {}).get("mean"),
                      "nll_after": ((a.get("heldout_nll") or {}).get("after") or {}).get("mean")} for a in r["arms"]],
        } for r in runs],
        "sessions": session_ids,
    }
    (out / "index.json").write_text(dump(index), encoding="utf-8")
    return index


def mirror(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="export directory (default: docs/research/results/sleep-observatory)")
    ap.add_argument("--mirror", type=Path, default=MIRROR, help="dashboard copy served under /assets (default: dashboard/public/assets/observatory)")
    ap.add_argument("--no-mirror", action="store_true", help="write the export only")
    ap.add_argument("--session-store", action="append", default=[], metavar="RUN=STORE",
                    help="extract a run's teach/rolled session trajectories from a local experiment store (repeatable)")
    ap.add_argument("--check", action="store_true", help="rebuild into a temporary copy and exit 1 if the committed export differs")
    args = ap.parse_args(argv)
    stores = {}
    for item in args.session_store:
        run_id, sep, path = item.partition("=")
        if not sep or not (ARCHIVE / run_id / "sleep_controls.json").exists():
            ap.error(f"--session-store expects RUN=STORE with RUN an archived run directory: {item}")
        stores[run_id] = Path(path)
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "export"
            if args.out.exists():
                shutil.copytree(args.out, work)
            build(work, {})
            fresh, committed = tree_bytes(work), tree_bytes(args.out) if args.out.exists() else {}
            mirrored = tree_bytes(args.mirror) if args.mirror.exists() else {}
            stale = sorted(k for k in set(fresh) | set(committed) if fresh.get(k) != committed.get(k))
            drift = sorted(k for k in set(committed) | set(mirrored) if committed.get(k) != mirrored.get(k))
            for k in stale:
                print(f"stale: {k}")
            for k in drift:
                print(f"mirror differs: {k}")
            return 1 if stale or drift else 0
    index = build(args.out, stores)
    if not args.no_mirror:
        mirror(args.out, args.mirror)
    size = sum(len(b) for b in tree_bytes(args.out).values())
    print(f"exported {len(index['runs'])} runs, {len(index['sessions'])} session trajectories, {size / 1e6:.2f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
