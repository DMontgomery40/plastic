"""Export the learning-contract report sets as compact JSON for the observatory's Learning view.

Reads only JSON and Markdown under ``docs/research/results/contract-2026-09-23/``; never weights. Writes one
``index.json`` to ``docs/research/results/learning-observatory/`` and mirrors the same bytes into
``dashboard/public/assets/learning/``, which the dashboard build copies into ``dist/assets`` (the static path the Space
serves). The mirror is a sibling of the Sleep export's ``assets/observatory/``, whose own mirror step replaces that whole
directory; the Sleep export is not touched.

Two kinds of set are recognized by content, not by name:

* a **report set** (``scripts/experiments/transfer_contract.py``): ``manifest.json`` plus one JSON per baseline mode
  (``frozen``, ``continued``, ``in_context``). Its learner is ``DynamicsLearner``, whose off-intervention is
  ``beta_scale=0`` (writes disabled, decay active).
* an **ablation set** (``scripts/experiments/coordinate_ablation.py``): one JSON per variant carrying ``variant``,
  ``contract`` and its own ``no_adapt_label``. Variants not archived are listed as missing, in the runner's order.

Every value is copied from a source file or computed from it, never invented. Missing values stay ``null``; a rate over
an empty set is ``null``, not zero. A set measured under a contract version other than the current
``plastic.eval.contract.CONTRACT_VERSION`` is marked ``current: false`` (the only staleness the page can show; staleness
of this export against its sources is what ``--check`` reports). Output is deterministic: no wall-clock time, no
absolute paths; it is identified by the sha256 of each source, an archive digest over them, and ``EXPORTER_VERSION``.

Usage
  python -m scripts.export_contract_observatory            # refresh from the archive
  python -m scripts.export_contract_observatory --check    # exit 1 if the committed export or its mirror is stale
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from plastic.eval.contract import CONTRACT_VERSION
from scripts.experiments.coordinate_ablation import VARIANTS
from scripts.export_sleep_observatory import dump, mirror, num, read_json, sha256, tree_bytes

EXPORTER_VERSION = "1.0.0"
SCHEMA = "plastic.learning-observatory/1"
ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "docs" / "research" / "results" / "contract-2026-09-23"
OUT = ROOT / "docs" / "research" / "results" / "learning-observatory"
MIRROR = ROOT / "dashboard" / "public" / "assets" / "learning"

MODES = ("frozen", "continued", "in_context")
# The off-intervention of each learner class, as the learner implements it (plastic/eval/contract.py DynamicsLearner:
# adapt=False runs the model with beta_scale=0). Ablation variants record their own label; this is used only for report
# sets, whose producer always builds a DynamicsLearner. tests/ pin it against coordinate_ablation.learner_for.
DYNAMICS_NO_ADAPT = "writes disabled (beta_scale=0)"


# ---------------------------------------------------------------------------------------------------- pieces
def speed_view(s: dict[str, Any] | None) -> dict[str, Any] | None:
    """Adaptation speed. The contract reports ``steps_to_half`` equal to the horizon when the ratio never drops below
    one half; that case is exported as ``reached_half: false`` and ``steps_to_half: null``, not as a step."""
    if not s:
        return None
    curve = [v for v in s.get("curve") or [] if isinstance(v, (int, float))]
    reached = any(v < 0.5 for v in curve) if curve else None
    return {"mean_ratio": num(s.get("area")), "steps_to_half": s.get("steps_to_half") if reached is not False else None,
            "reached_half": reached, "horizon": len(curve) or None, "episodes": s.get("episodes")}


def pair(block: dict[str, Any] | None) -> dict[str, Any]:
    block = block or {}
    return {"adapt": num(block.get("adapt")), "no_adapt": num(block.get("no_adapt")), "elements": block.get("elements")}


def before_view(rep: dict[str, Any]) -> dict[str, Any]:
    """The before-stream block: held-out pairings per held-out policy, the training distribution, adaptation speed."""
    transfer = rep.get("transfer") or {}
    return {
        "policies": [{"policy": p, **pair(v.get("before"))} for p, v in transfer.items()],
        "train": {"policies": list((rep.get("spec") or {}).get("train_policies") or []), **pair((rep.get("forgetting") or {}).get("before"))},
        "speed": speed_view((rep.get("speed") or {}).get("before")),
    }


def rate(value: Any, n: Any) -> float | None:
    """A rate over an empty set is undefined (null), never zero."""
    return num(value) if isinstance(n, int) and n > 0 else None


def mode_view(rep: dict[str, Any]) -> dict[str, Any]:
    c = rep.get("correction") or {}
    acc = rep.get("acceptance") or {}
    comp = rep.get("compute") or {}
    rev = rep.get("revert") or {}
    return {
        "mode": rep.get("mode"),
        "transfer": [{"policy": p, "delta": num(v.get("delta_mse")), "delta_no_adapt": num(v.get("delta_mse_no_adapt"))}
                     for p, v in (rep.get("transfer") or {}).items()],
        "forgetting_delta": num((rep.get("forgetting") or {}).get("delta_mse")),
        "poison_harm_vs_clean": num(c.get("harm")), "poison_harm_vs_start": num(c.get("harm_vs_before")),
        "correction_residual": num(c.get("residual")), "correction_residual_vs_start": num(c.get("residual_vs_before")),
        "revert": {"ok": rev.get("ok") if isinstance(rev.get("ok"), bool) else None, "gap": num(rev.get("gap")), "tolerance": num(rev.get("tolerance"))},
        "acceptance": {"accepted_good": rate(acc.get("accepted_good"), acc.get("n_good")), "n_good": acc.get("n_good"),
                       "refused_bad": rate(acc.get("refused_bad"), acc.get("n_bad")), "n_bad": acc.get("n_bad")},
        "decisions": [{k: d.get(k) for k in ("stream", "beneficial", "accepted")} for d in rep.get("decisions") or []],
        "tokens_consumed": comp.get("tokens_consumed"), "tokens_measured": comp.get("tokens_measured"),
        "tokens_measured_without_context": comp.get("tokens_measured_without_context"), "parameters": comp.get("parameters"),
    }


def readme_tags(archive: Path) -> dict[str, str]:
    """``## `set_id` (tag)`` headings of the archive README: the short name each set is published under."""
    readme = archive / "README.md"
    if not readme.exists():
        return {}
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^##\s+`([^`]+)`\s*\(([^)]+)\)", readme.read_text(encoding="utf-8"), re.M)}


def window_view(w: dict[str, Any] | None) -> dict[str, Any]:
    w = w or {}
    return {"update_period": w.get("update_period"), "boundaries_per_episode": w.get("boundaries_per_episode"), "checked": bool(w.get("checked"))}


def source_list(paths: list[Path], archive: Path) -> list[dict[str, str]]:
    return [{"file": p.relative_to(archive).as_posix(), "sha256": sha256(p)} for p in sorted(paths)]


# ---------------------------------------------------------------------------------------------------- sets
def export_report_set(d: Path, archive: Path, tags: dict[str, str]) -> dict[str, Any]:
    m = read_json(d / "manifest.json")
    reports = {mode: read_json(d / f"{mode}.json") for mode in MODES if (d / f"{mode}.json").exists()}
    if not reports:
        raise ValueError(f"{d.name}: a manifest without any mode report")
    first = next(iter(reports.values()))
    for mode, rep in reports.items():
        for key, want in (("contract_version", m.get("contract_version")), ("mode", mode)):
            if rep.get(key) != want:
                raise ValueError(f"{d.name}/{mode}.json: {key} {rep.get(key)!r} differs from {want!r}")
        if (rep.get("split") or {}).get("id") != m.get("split_id") or window_view(rep.get("adaptation_window")) != window_view(m.get("adaptation_window")):
            raise ValueError(f"{d.name}/{mode}.json: split or adaptation window differs from the manifest")
        if before_view(rep) != before_view(first):
            raise ValueError(f"{d.name}/{mode}.json: the before-stream measurement differs between modes")
    split = first.get("split") or {}
    spec = first.get("spec") or {}
    version = m.get("contract_version")
    sources = [d / "manifest.json", *(d / f"{mode}.json" for mode in reports)] + sorted(d.glob("*.md"))
    return {
        "id": d.name, "kind": "report", "tag": tags.get(d.name),
        "checkpoint": {"model_id": m.get("model_id"), "digest_prefix": str(m.get("checkpoint_digest") or "")[:16] or None,
                       "step": (m.get("model_record") or {}).get("step")},
        "execution_commit": str(m.get("execution_commit") or "")[:12] or None, "device": m.get("device"), "seed": m.get("seed"),
        "contract_version": version, "current": version == CONTRACT_VERSION,
        "split": {"id": split.get("id"), "train": split.get("train"), "heldout": split.get("heldout"), "bound_to_learner": split.get("bound_to_learner")},
        "adaptation_window": window_view(m.get("adaptation_window")),
        "spec": {k: spec.get(k) for k in ("seq_len", "stream_episodes", "heldout_policies", "train_policies", "poison_bias", "revert_tolerance", "eval_batch", "probe_steps")},
        "stream": first.get("stream"), "learner": m.get("learner"),
        "no_adapt_label": DYNAMICS_NO_ADAPT,
        "before": before_view(first),
        "modes": [mode_view(reports[mode]) for mode in MODES if mode in reports],
        "missing_modes": [mode for mode in MODES if mode not in reports],
        "sources": source_list(sources, archive),
    }


def variant_view(r: dict[str, Any]) -> dict[str, Any]:
    c = r.get("contract") or {}
    log = (r.get("train") or {}).get("log") or []
    last = log[-1] if log else {}
    size = r.get("size") or {}
    version = c.get("contract_version")
    return {
        "variant": r.get("variant"), "config": r.get("config"), "parameters": size.get("parameters"),
        "size": {k: size.get(k) for k in ("d_model", "n_heads", "n_layers", "chunk")},
        "train_steps": (r.get("train") or {}).get("steps"), "s_per_step": num((r.get("train") or {}).get("s_per_step"), 4),
        "final_train_loss": num(last.get("loss"), 4),
        "eta_per_layer": [num(e, 4) for e in last["eta"]] if last.get("eta") else None,
        "no_adapt_label": r.get("no_adapt_label"),
        "contract_version": version, "current": version == CONTRACT_VERSION,
        "split_id": (c.get("split") or {}).get("id"), "adaptation_window": window_view(c.get("adaptation_window")),
        "before": before_view(c),
    }


def export_ablation_set(d: Path, archive: Path, tags: dict[str, str], files: list[Path]) -> dict[str, Any]:
    results = {read_json(p)["variant"]: (p, read_json(p)) for p in files}
    manifest = read_json(d / "manifest.json") if (d / "manifest.json").exists() else {}
    variants = [variant_view(results[v][1]) for v in VARIANTS if v in results]
    variants += [variant_view(r) for v, (_, r) in sorted(results.items()) if v not in VARIANTS]
    versions = sorted({v["contract_version"] for v in variants if v["contract_version"]})
    sources = [p for p, _ in results.values()] + [d / "manifest.json"] * bool(manifest) + sorted(d.glob("*.md"))
    return {
        "id": d.name, "kind": "ablation", "tag": tags.get(d.name),
        "execution_commit": str(manifest.get("execution_commit") or "")[:12] or None,
        "contract_version": versions[0] if len(versions) == 1 else None, "contract_versions": versions,
        "current": bool(variants) and all(v["current"] for v in variants),
        "variants": variants, "missing": [v for v in VARIANTS if v not in results],
        "sources": source_list(sources, archive),
    }


def classify(d: Path) -> tuple[str, list[Path]] | None:
    """A report set has a manifest and at least one mode report; an ablation set has variant results."""
    if (d / "manifest.json").exists() and any((d / f"{mode}.json").exists() for mode in MODES):
        return "report", []
    variants = [p for p in sorted(d.glob("*.json")) if p.name != "manifest.json" and {"variant", "contract"} <= set(read_json(p))]
    return ("ablation", variants) if variants else None


# ---------------------------------------------------------------------------------------------------- driver
def build(out: Path, archive: Path = ARCHIVE) -> dict[str, Any]:
    tags = readme_tags(archive)
    sets = []
    for d in sorted(p for p in archive.iterdir() if p.is_dir()) if archive.exists() else []:
        kind = classify(d)
        if kind is None:
            continue
        sets.append(export_report_set(d, archive, tags) if kind[0] == "report" else export_ablation_set(d, archive, tags, kind[1]))
    digests = sorted({(f"{s['id']}/{src['file']}", src["sha256"]) for s in sets for src in s["sources"]})
    try:
        archive_name = archive.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        archive_name = archive.name
    index = {
        "schema": SCHEMA, "exporter_version": EXPORTER_VERSION, "current_contract_version": CONTRACT_VERSION,
        "archive": archive_name, "archive_digest": hashlib.sha256("".join(f"{f}:{h}\n" for f, h in digests).encode()).hexdigest(),
        "sets": sets,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(dump(index), encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="export directory (default: docs/research/results/learning-observatory)")
    ap.add_argument("--mirror", type=Path, default=MIRROR, help="dashboard copy served under /assets (default: dashboard/public/assets/learning)")
    ap.add_argument("--no-mirror", action="store_true", help="write the export only")
    ap.add_argument("--check", action="store_true", help="rebuild into a temporary directory and exit 1 if the committed export or its mirror differs")
    args = ap.parse_args(argv)
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "export"
            build(work)
            fresh = tree_bytes(work)
            committed = tree_bytes(args.out) if args.out.exists() else {}
            mirrored = tree_bytes(args.mirror) if args.mirror.exists() else {}
            stale = sorted(k for k in set(fresh) | set(committed) if fresh.get(k) != committed.get(k))
            drift = sorted(k for k in set(committed) | set(mirrored) if committed.get(k) != mirrored.get(k))
            for k in stale:
                print(f"stale: {k}")
            for k in drift:
                print(f"mirror differs: {k}")
            return 1 if stale or drift else 0
    if args.out.exists():
        shutil.rmtree(args.out)
    index = build(args.out)
    if not args.no_mirror:
        mirror(args.out, args.mirror)
    kinds = [s["kind"] for s in index["sets"]]
    print(f"exported {kinds.count('report')} report sets and {kinds.count('ablation')} ablation sets -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
