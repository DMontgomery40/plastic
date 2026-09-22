"""Native Qwen conversational operating-point screen (ASTRA-083).

Fits harness thresholds on fresh Dolly instructions through the real chat protocol, freezes them,
then measures per-source (prompt vs generation) ELIGIBLE intervention and read-only rates on a
locked, DISJOINT evaluation set — as fresh sessions and as constructed multi-turn chains — plus a
small predeclared follow-up smoke. This is a bounded usability screen of the existing native backend
and harness on selected instruction traffic, NOT an unbiased chat population sample, a demonstrated
policy false-positive rate, or safety-efficacy evidence.

Corpus: RAW official Dolly (instruction + optional context) rendered with the actual Qwen chat
template — never teacher-forced, never the old custom-BPE bins. Prompts are deduplicated by
whitespace-normalized (instruction, context), capped at --max-prompt-tokens native rendered tokens
(over-cap rows are EXCLUDED, not truncated), then split deterministically (seed 20260922) into
disjoint fit / cusum / dev / eval groups. The split manifest (dataset revision, row ids, text
hashes) is saved before any generation.

Fixed operating point: chunk=8, max_new_tokens=64, temperature=0.9, top_k=50, normal EOS/closure;
per-phase seeds 20260922 (+1000 cusum, +2000 dev, +3000 eval, +4000 followups). Fit is log-only;
evaluation runs stats + rollback enabled, log_only=false, generation learning on, freeze_on_alarm
with alarm_cooldown=0, no finite budget, target_fpr=0.01. Development data is for any changes; freeze
before opening the locked evaluation. Reduce the counts (or --smoke) only to validate the wiring —
label such runs as not the ASTRA-083 screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from typing import Any

WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return WS.sub(" ", (s or "").strip()).lower()


def load_dolly_prompts(max_rows: int) -> tuple[list[dict[str, Any]], str]:
    """Raw Dolly rows as {id, instruction, context, category, prompt}, deduped by normalized
    (instruction, context). Returns (rows, dataset_revision)."""
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    revision = str(getattr(getattr(ds, "info", None), "version", "unknown"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, ex in enumerate(ds):
        if len(rows) >= max_rows:
            break
        instr, ctx = (ex.get("instruction", "") or "").strip(), (ex.get("context", "") or "").strip()
        if not instr:
            continue
        key = f"{_norm(instr)}||{_norm(ctx)}"
        if key in seen:
            continue
        seen.add(key)
        prompt = instr if not ctx else f"{instr}\n\n{ctx}"
        rows.append({"id": i, "instruction": instr, "context": ctx, "category": ex.get("category", ""),
                     "prompt": prompt, "text_sha256": hashlib.sha256(key.encode()).hexdigest()})
    return rows, revision


def _load_exclusions(path: str) -> tuple[set[int], tuple[str, ...]]:
    """Load a prior-exposure exclusion spec: explicit ids plus normalized-text-hash prefixes."""
    d = json.load(open(path, encoding="utf-8"))
    return {int(i) for i in d.get("ids", [])}, tuple(str(p) for p in d.get("normalized_text_hash_prefixes", []))


def _apply_exclusions(rows: list[dict[str, Any]], ids: set[int], hash_prefixes: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reserve prior-exposed rows from a newly locked selection. A row is directly excluded if its id
    is listed OR its normalized-text hash starts with a listed prefix (catching exact duplicates). The
    exclusion then EXPANDS to whole context groups: any row sharing a nonempty normalized context with
    a directly-excluded row is also reserved, so a passage from an earlier split cannot re-enter the
    new evaluation through a sibling question (ASTRA-086). Empty-context rows are never group-excluded."""
    directly = {r["id"] for r in rows
                if r["id"] in ids or (hash_prefixes and r["text_sha256"].startswith(hash_prefixes))}
    excluded_ctx = {c for r in rows if r["id"] in directly and (c := _norm(r["context"]))}
    kept = [r for r in rows
            if r["id"] not in directly and not (_norm(r["context"]) and _norm(r["context"]) in excluded_ctx)]
    report = {"n_input": len(rows), "excluded_directly": len(directly), "excluded_context_groups": len(excluded_ctx),
              "n_excluded_total": len(rows) - len(kept), "n_kept": len(kept)}
    return kept, report


def _stratify_by_category(groups: list[list[dict[str, Any]]], seed: int) -> list[list[dict[str, Any]]]:
    """Round-robin the groups across their (first row's) category so each split draws from present
    categories, deterministically."""
    buckets: dict[str, list] = {}
    for g in groups:
        buckets.setdefault(g[0]["category"], []).append(g)
    cats = sorted(buckets, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    out: list[list[dict[str, Any]]] = []
    while any(buckets[c] for c in cats):
        for c in cats:
            if buckets[c]:
                out.append(buckets[c].pop(0))
    return out


def build_split(rows: list[dict[str, Any]], encode_chat, *, counts: dict[str, int], seed: int, max_prompt_tokens: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Cap by native rendered prompt-token count (exclude, never truncate), GROUP rows that share a
    nonempty normalized context so a passage never crosses a split boundary (ASTRA-085), order the
    groups deterministically and category-stratified, and assign WHOLE groups to disjoint splits until
    each requested count is met. Returns (split, manifest with actual counts + an immutable corpus hash)."""
    kept, excluded = [], 0
    for r in rows:
        if len(encode_chat(r["prompt"])) > max_prompt_tokens:
            excluded += 1
            continue
        kept.append({**r, "n_prompt_tokens": len(encode_chat(r["prompt"]))})
    # group by nonempty normalized context; empty-context rows are singleton groups
    ctx_groups: dict[str, list] = {}
    groups: list[list[dict[str, Any]]] = []
    for r in kept:
        ctx = _norm(r["context"])
        if ctx:
            ctx_groups.setdefault(ctx, []).append(r)
        else:
            groups.append([r])
    groups.extend(ctx_groups.values())
    groups.sort(key=lambda g: hashlib.sha256(f"{seed}:{g[0]['text_sha256']}".encode()).hexdigest())
    groups = _stratify_by_category(groups, seed)

    split: dict[str, list[dict[str, Any]]] = {name: [] for name in ("fit", "cusum", "dev", "eval")}
    gi = 0
    for name in ("fit", "cusum", "dev", "eval"):
        while len(split[name]) < counts[name] and gi < len(groups):
            split[name].extend(groups[gi])
            gi += 1
        if len(split[name]) < counts[name]:
            raise ValueError(f"not enough grouped prompts for split '{name}': {len(split[name])} < {counts[name]}")
    all_hashes = sorted(r["text_sha256"] for v in split.values() for r in v)
    manifest = {
        "seed": seed, "max_prompt_tokens": max_prompt_tokens, "excluded_over_cap": excluded,
        "requested_counts": dict(counts), "actual_counts": {k: len(v) for k, v in split.items()},
        "category_counts": {k: _cat_counts(v) for k, v in split.items()},
        "ids": {k: [r["id"] for r in v] for k, v in split.items()},
        "corpus_hash": hashlib.sha256("".join(all_hashes).encode()).hexdigest(),
        "n_context_groups": len(ctx_groups),
        # an immutable raw export of the selected records (full text + full hash), so the exact corpus
        # is pinned regardless of any Hub revision drift (ASTRA-086)
        "selected_records": {k: [{"id": r["id"], "prompt": r["prompt"], "instruction": r["instruction"],
                                  "context": r["context"], "category": r["category"], "sha256": r["text_sha256"]}
                                 for r in v]
                             for k, v in split.items()},
    }
    return split, manifest


def _cat_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["category"]] = out.get(r["category"], 0) + 1
    return out


def _eval_sessions(backend, cfg, calibration, prompts, *, hcfg, gen, seed_base, chains, deadline,
                   restore=None, on_progress=None, _runner=None, _drive=None):
    """Evaluate the locked set through the frozen harness in two regimes on ONE reused backend/runner
    (fresh sessions reset between; carried chains retain state within a chain), via the shared
    drive_chat_turn. Records per-session eligible/accepted-by-source, read-only, completions and
    outcomes, and the ordered raw transactions; the deadline is honored per turn; completion is by
    processed-vs-expected ids.

    Resumable across invocations: ``restore`` = {regime: [{"record", "txns"}, ...]} restores the
    COMPLETE sessions already collected, and ``on_progress(regime, record, txns, next_group)`` is
    called after each newly completed session so the caller can append it durably. Each turn's seed is
    the row's ordinal position in the ordered prompt list (seed_base + position), so an interruption
    never shifts a seed and fresh row i / carried row i still share it. A group cut mid-way is reported
    as durable evidence (incomplete) but NOT persisted; it re-runs from its start on resume (reset() is
    a clean slate). The operating point aggregates COMPLETE sessions only, never a partial group.
    ``_runner``/``_drive`` allow a fake runner/drive for tests."""
    import torch

    from plastic.harness.calibrate import summarize_operating_point

    if _drive is None:
        from plastic.session.runner import drive_chat_turn as _drive
    if _runner is None:
        from plastic.harness.transaction import TransactionRunner
        from plastic.session.runner import _QwenTextIO
        tok = _QwenTextIO(backend)
        _runner = TransactionRunner(None, cfg, hcfg, calibration=calibration, device=backend.device, backend=backend)
    else:
        tok = None
    runner = _runner

    def _turn(prompt: str, seed: int) -> dict[str, Any]:
        runner.transactions = []
        g = torch.Generator().manual_seed(int(seed))
        t0 = time.time()
        completion, out_ids, in_ids = _drive(runner, tok, prompt, max_new_tokens=gen["max_new_tokens"],
                                              temperature=gen["temperature"], top_k=gen["top_k"], gen=g)
        return {"seed": seed, "completion": completion, "n_in": len(in_ids), "n_out": len(out_ids),
                "outcome": ("cap" if len(out_ids) >= gen["max_new_tokens"] else ("empty" if not out_ids else "eos")),
                "seconds": time.time() - t0, "transactions": list(runner.transactions),
                "read_only": runner.read_only, "read_only_reason": runner.read_only_reason}

    def _session_record(ids: list[int], regime: str, turns: list[dict]) -> tuple[dict, list[dict]]:
        txns = [t for turn in turns for t in turn["transactions"]]
        op = summarize_operating_point(txns)
        acc = {"prompt": op["prompt"]["accepted_change"], "generation": op["generation"]["accepted_change"]}
        elig = {"prompt": op["prompt"]["eligible"], "generation": op["generation"]["eligible"]}
        rec = {"ids": ids, "regime": regime, "read_only": turns[-1]["read_only"], "read_only_reason": turns[-1]["read_only_reason"],
               "eligible_by_source": elig, "accepted_by_source": acc,
               "retained_both": acc["prompt"] > 0 and acc["generation"] > 0,
               "anomalies": op["anomalies"],
               "turns": [{"id": ids[j], **{k: turn[k] for k in ("seed", "completion", "n_in", "n_out", "outcome", "seconds")}} for j, turn in enumerate(turns)]}
        return rec, txns

    def _run(regime: str, groups: list[list[dict]], seed0: int) -> dict[str, Any]:
        expected = [r["id"] for g in groups for r in g]
        # per-group starting ordinal (the row's position in the ordered prompt list); the per-turn seed
        # is seed0 + ordinal, so it depends only on position and an interruption cannot shift it
        offsets, off = [], 0
        for g in groups:
            offsets.append(off)
            off += len(g)
        restored = (restore or {}).get(regime) or []
        complete = [x["record"] for x in restored]                      # COMPLETE sessions only
        complete_tx = [t for x in restored for t in x["txns"]]
        processed = [i for rec in complete for i in rec["ids"]]
        partial: list[tuple[dict, list[dict]]] = []                     # a cut group: reported, not persisted
        gi = len(restored)
        while gi < len(groups):
            if time.time() > deadline:
                break
            g = groups[gi]
            runner.reset()
            turns = []
            for j, r in enumerate(g):
                if time.time() > deadline:
                    break
                turns.append(_turn(r["prompt"], seed0 + offsets[gi] + j))
            if len(turns) != len(g):  # cut mid-group: durable evidence only; re-run from start on resume
                if turns:
                    rec, txns = _session_record([r["id"] for r in g[:len(turns)]], regime, turns)
                    rec["incomplete"] = True
                    partial.append((rec, txns))
                break
            rec, txns = _session_record([r["id"] for r in g], regime, turns)
            rec["incomplete"] = False
            complete.append(rec)
            complete_tx.extend(txns)
            processed.extend(r["id"] for r in g)
            gi += 1
            if on_progress is not None:
                on_progress(regime, rec, txns, gi)
        report_sessions = complete + [rec for rec, _ in partial]
        report_tx = complete_tx + [t for _, txns in partial for t in txns]
        processed_all = processed + [i for rec, _ in partial for i in rec["ids"]]
        return {"operating_point": summarize_operating_point(complete_tx), "sessions": report_sessions,
                "expected_ids": expected, "processed_ids": processed_all,
                "n_expected_sessions": len(groups),
                "n_complete_sessions": len(complete),
                "complete": processed_all == expected, "raw_transactions": report_tx}

    fresh_groups = [[r] for r in prompts]
    carried_groups = _split_into_chains(prompts, chains)
    # restored progress must be the EXACT completed prefix of each regime's groups: the i-th restored
    # session's ids must equal the i-th group's ids (a stable count is not compatibility) -- ASTRA-104
    for regime, groups in (("fresh", fresh_groups), ("carried", carried_groups)):
        for i, x in enumerate((restore or {}).get(regime) or []):
            if i >= len(groups) or [r["id"] for r in groups[i]] != list(x["record"]["ids"]):
                raise RunConflict(
                    f"restored eval progress for {regime} does not match the pinned groups at session {i}; "
                    f"refusing to resume. Use a fresh --out.")
    # ASTRA-083: the SAME per-row eval seed in fresh and carried (seed0 + the row's ordinal position in
    # both), so the only difference between the regimes is fresh vs carried state, not the RNG
    return {"fresh": _run("fresh", fresh_groups, seed_base), "carried": _run("carried", carried_groups, seed_base)}


def _split_into_chains(prompts: list[Any], chains: int) -> list[list[Any]]:
    """Partition prompts into ``chains`` contiguous groups, distributing the remainder so NO prompt is
    dropped (plain integer division drops the tail — the ASTRA-085 five-prompt/two-chain case)."""
    if chains <= 0 or not prompts:
        return []
    base, rem = divmod(len(prompts), chains)
    out, at = [], 0
    for c in range(chains):
        size = base + (1 if c < rem else 0)
        if size:
            out.append(prompts[at:at + size])
            at += size
    return out


def _check_criterion(report: dict[str, Any], *, elig_max: float = 0.10, readonly_max: float = 0.10, retained_min: float = 0.90) -> dict[str, Any]:
    """The full ASTRA-083 verdict: per source x regime the eligible-intervention rate and read-only
    rate must be <= 10% on a NONZERO eligible denominator, AND >= 90% of the regime's sessions must
    retain a finite positive accepted change from BOTH sources, AND every regime must have run to
    completion (processed == expected ids) with zero reporting anomalies. An incomplete or invalid
    run cannot pass — the timing deadline is not evidence that all requested work ran."""
    out: dict[str, Any] = {"cells": {}, "regimes": {}}
    valid = True
    for regime in ("fresh", "carried"):
        blk = report[regime]
        op = blk["operating_point"]
        complete_sessions = [s for s in blk["sessions"] if not s.get("incomplete")]
        n = len(complete_sessions)
        retained = sum(1 for s in complete_sessions if s["retained_both"])
        retained_frac = (retained / n) if n else None
        complete = bool(blk.get("complete"))
        anomalies_clean = not any(op.get("anomalies", {}).values())
        out["regimes"][regime] = {
            "n_sessions": len(blk["sessions"]), "n_complete_sessions": n, "n_expected_sessions": blk.get("n_expected_sessions"),
            "retained_both": retained, "retained_fraction": retained_frac,
            "retained_ok": (retained_frac is not None and retained_frac >= retained_min),
            "complete": complete, "anomalies_clean": anomalies_clean,
        }
        if not complete or not anomalies_clean or n == 0:
            valid = False
        for src in ("prompt", "generation"):
            rec = op[src]
            eir, ror = rec["eligible_intervention_rate"], rec["readonly_rate"]
            out["cells"][f"{regime}.{src}"] = {
                "eligible": rec["eligible"], "eligible_intervention_rate": eir, "readonly_rate": ror,
                "eligible_ok": (eir is not None and eir <= elig_max),
                "readonly_ok": (ror is not None and ror <= readonly_max),
                "nonzero_eligible": rec["eligible"] > 0,
            }
    cells_ok = all(c["eligible_ok"] and c["readonly_ok"] and c["nonzero_eligible"] for c in out["cells"].values())
    regimes_ok = all(r["retained_ok"] and r["complete"] and r["anomalies_clean"] for r in out["regimes"].values())
    out["valid"] = valid
    out["pass"] = bool(valid and cells_ok and regimes_ok)
    return out


def _screen_verdict(*, fit_complete: bool, report: dict[str, Any], followups: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
    """Separate COMPLETION (did all requested work run?) from PASS (the eval criterion plus the
    follow-up smoke). The follow-up ``all_ok`` is a quality result, not a completeness measure: a
    fully-run screen whose follow-ups flagged read-only turns is COMPLETE but not a pass — a distinct
    statement from a screen that did not finish. ``complete`` requires the fit, both eval regimes AND
    the follow-ups to have RUN (not skipped, not deadline-cut); ``pass`` additionally requires the
    criterion (``verdict['pass']`` already folds in validity and eval-completeness) and follow-up
    ``all_ok``. Follow-ups skipped for a missing fixture leaves ``followups_ok`` null, never False."""
    eval_complete = bool(report["fresh"]["complete"] and report["carried"]["complete"])
    skipped = bool(followups.get("skipped"))
    followups_ran = (not skipped) and (not followups.get("incomplete"))
    followups_ok = None if skipped else bool(followups.get("all_ok"))
    complete = bool(fit_complete and eval_complete and followups_ran)
    passed = bool(verdict.get("pass") and complete and (followups_ok is True))
    return {
        "fit_complete": bool(fit_complete),
        "eval_complete": {"fresh": bool(report["fresh"]["complete"]), "carried": bool(report["carried"]["complete"])},
        "followups_ran": followups_ran,
        "followups_ok": followups_ok,
        "valid": bool(verdict.get("valid")),
        "pass": passed,
        "complete": complete,
    }


class RunConflict(Exception):
    """Raised when ``args.out`` already holds a run whose configuration or pinned corpus differs from
    this invocation. The existing manifest and run record are PRESERVED (never overwritten); a new run
    must use a fresh ``--out``."""


def _settings_identity(args: Any, checkpoint_digest: str, revision: str, exclusions_digest: str | None = None) -> str:
    """A stable digest of the settings that determine the corpus and the calibration, so a
    re-invocation with different settings is refused rather than silently overwriting a run."""
    payload = {
        "counts": {"fit": args.n_fit, "cusum": args.n_cusum, "dev": args.n_dev, "eval": args.n_eval},
        "max_prompt_tokens": args.max_prompt_tokens, "max_new_tokens": args.max_new_tokens,
        "eval_chains": args.eval_chains, "smoke": args.smoke, "exclusions": exclusions_digest,
        "checkpoint_digest": checkpoint_digest, "revision": revision,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _write_json(path: str, obj: dict[str, Any]) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic, so a kill mid-write never leaves a truncated manifest/run record


def _split_identity(split: dict[str, list[dict[str, Any]]]) -> str:
    """Ordered per-split id identity. A cross-split swap or a within-split reorder changes it, unlike
    the order- and split-independent corpus_hash (a sorted union of text hashes) -- so this is what
    certifies a resumed run evaluates exactly the original selection (ASTRA-098)."""
    payload = {name: [r["id"] for r in split.get(name, [])] for name in ("fit", "cusum", "dev", "eval")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _split_from_manifest(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct the exact ordered per-split records from the immutable manifest, so a resumed run
    calibrates and evaluates on precisely the original selection rather than a freshly rebuilt split
    whose membership could differ while the set-of-prompts corpus_hash stays the same (ASTRA-098)."""
    sel = manifest.get("selected_records") or {}
    out: dict[str, list[dict[str, Any]]] = {}
    for name in ("fit", "cusum", "dev", "eval"):
        recs = sel.get(name) or []
        for r in recs:
            if "prompt" not in r:
                raise RunConflict(
                    f"the pinned manifest predates ordered-split resume (no prompt in selected_records); use a fresh --out.")
        out[name] = [{"id": r["id"], "prompt": r["prompt"], "context": r.get("context", ""),
                      "category": r.get("category", ""), "text_sha256": r.get("sha256", "")} for r in recs]
    return out


def _reconcile_run(out_dir: str, manifest: dict[str, Any], split: dict[str, list[dict[str, Any]]],
                   settings_identity: str, new_model_id) -> tuple[str, str, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Decide whether this invocation is a fresh run or a continuation of an existing one in ``out_dir``.

    Returns ``(mode, model_id, manifest, split)`` where mode is 'fresh' or 'resume'. A fresh run mints
    a model id and writes the manifest + run record. A matching re-run reuses the recorded model id
    and returns the exact ordered split RESTORED FROM the pinned manifest -- never this invocation's
    freshly rebuilt split, whose membership could differ while the union corpus_hash stays the same
    (ASTRA-098 leakage). A run whose settings, corpus, or ordered split identity differ raises
    RunConflict WITHOUT writing anything. A directory that already holds run artifacts but has NO run
    record (an interrupted initialization, or a pre-run-record output dir) is likewise preserved, never
    silently overwritten as fresh (ASTRA-092/098)."""
    run_path = os.path.join(out_dir, "run-record.json")
    manifest_path = os.path.join(out_dir, "split-manifest.json")
    split_identity = _split_identity(split)
    if os.path.exists(run_path):
        run = json.load(open(run_path, encoding="utf-8"))
        if run.get("settings_identity") != settings_identity:
            raise RunConflict(
                f"{run_path} records a different configuration; refusing to overwrite this run's artifacts. Use a fresh --out.")
        if run.get("corpus_hash") != manifest["corpus_hash"]:
            raise RunConflict(
                f"the corpus under {out_dir} differs from the pinned manifest (dataset drift?); refusing to overwrite. Use a fresh --out.")
        saved = json.load(open(manifest_path, encoding="utf-8"))
        saved_split = _split_from_manifest(saved)
        if run.get("split_identity") != _split_identity(saved_split):
            raise RunConflict(
                f"the pinned manifest's split does not match the recorded run identity; refusing to resume. Use a fresh --out.")
        return "resume", str(run["model_id"]), saved, saved_split
    # no run record: a genuinely fresh dir has NO prior run artifacts. If a manifest/checkpoint/result
    # is present, this is an interrupted initialization or a pre-run-record output dir -> preserve it.
    prior = [os.path.basename(p) for p in (manifest_path, os.path.join(out_dir, "calibration.ckpt"),
                                           os.path.join(out_dir, "screen-result.json")) if os.path.exists(p)]
    if prior:
        raise RunConflict(
            f"{out_dir} holds prior run artifacts ({', '.join(prior)}) but no run-record.json; refusing to "
            f"overwrite. Use a fresh --out (or remove the directory deliberately).")
    mid = new_model_id()
    _write_json(manifest_path, manifest)  # written ONLY for a genuinely fresh run
    _write_json(run_path, {"model_id": mid, "settings_identity": settings_identity,
                           "corpus_hash": manifest["corpus_hash"], "split_identity": split_identity,
                           "created_at_unix": int(time.time())})
    return "fresh", mid, manifest, split


def _eval_identity(eval_records: list[dict[str, Any]], settings_identity: str, calibration: Any, eval_hcfg: dict[str, Any]) -> str:
    """Bind eval progress to the exact content/config/calibration/policy: the ordered raw eval prompt
    CONTENT (id + full text hash + prompt, not IDs alone), the run settings, the calibration CONTENT
    actually in effect (thresholds + CUSUM-reference length, not just the model-compatibility
    signature -- reuse can hand a calibration from a prior invocation), and the frozen eval harness
    config. A mismatch on any of these refuses to reuse stale eval progress (ASTRA-100/103/104)."""
    eval_content = [{"id": r["id"], "sha256": r.get("text_sha256", ""), "prompt": r.get("prompt", "")} for r in eval_records]
    cal = {
        "thresholds": {k: float(v) for k, v in (getattr(calibration, "thresholds", {}) or {}).items()},
        "cusum_reference_len": len(getattr(calibration, "cusum_reference", []) or []),
        "model_signature": getattr(calibration, "model_signature", None),
    }
    payload = {"eval_content": eval_content, "settings_identity": settings_identity,
               "calibration": cal, "eval_hcfg": eval_hcfg}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _init_eval_progress(path: str, identity: str) -> None:
    with open(path, "w", encoding="utf-8") as f:  # a single identity header line; sessions are appended
        f.write(json.dumps({"identity": identity}) + "\n")


def _append_eval_progress(path: str, regime: str, record: dict[str, Any], txns: list[dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:  # append-only: each COMPLETE session written once
        f.write(json.dumps({"regime": regime, "record": record, "txns": txns}) + "\n")


def _load_eval_progress(path: str, identity: str, *, log=print) -> dict[str, list[dict[str, Any]]] | None:
    """Restore per-regime COMPLETE sessions from the append-only eval progress log, or None if absent.
    The header pins the identity; a mismatch (different content/config/calibration/policy) raises
    RunConflict rather than silently reusing stale progress. A malformed trailing line (a crash
    mid-append) is skipped -- that one session re-runs deterministically."""
    if not os.path.exists(path):
        return None
    restore: dict[str, list[dict[str, Any]]] = {"fresh": [], "carried": []}
    with open(path, encoding="utf-8") as f:
        try:
            head = json.loads(f.readline())
        except Exception as e:
            raise RunConflict(f"eval progress at {path} is unreadable ({e}); refusing to reuse. Use a fresh --out.")
        if head.get("identity") != identity:
            raise RunConflict(
                f"eval progress at {path} is for a different content/config/calibration; refusing to reuse. Use a fresh --out.")
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                log("[oppoint] skipping a malformed trailing eval-progress line (that session re-runs)")
                continue
            if rec.get("regime") in restore:
                restore[rec["regime"]].append({"record": rec["record"], "txns": rec["txns"]})
    return restore


def _run_followups(backend, cfg, calibration, gen, seed0, fixture_path, hcfg, deadline, *, _runner=None, _drive=None, _fixture=None):
    """Run the predeclared multi-turn follow-up sessions (never fit/tuned on). Each turn must complete
    without exception or nonfinite state and must not be entirely read-only; saved answers are for
    textual review, reported separately from any rate. Returns per-session outcomes + an all_ok flag.
    ``_runner``/``_drive``/``_fixture`` allow a fake runner/drive/fixture for tests."""
    import torch

    from plastic.harness.calibrate import summarize_operating_point

    if _drive is None:
        from plastic.session.runner import drive_chat_turn as _drive
    if _fixture is not None:
        fixture = _fixture
    elif not os.path.exists(fixture_path):
        return {"skipped": "fixture missing", "path": fixture_path}
    else:
        fixture = json.load(open(fixture_path))
    if _runner is None:
        from plastic.harness.transaction import TransactionRunner
        from plastic.session.runner import _QwenTextIO
        tok = _QwenTextIO(backend)
        _runner = TransactionRunner(None, cfg, hcfg, calibration=calibration, device=backend.device, backend=backend)
    else:
        tok = None
    runner, drive_chat_turn = _runner, _drive

    out = {"purpose": fixture.get("purpose", ""), "sessions": []}
    salt = 0
    for sess in fixture.get("sessions", []):
        if time.time() > deadline:
            out["incomplete"] = True
            break
        runner.reset()
        turns = []
        session_ok = True
        for text in sess["turns"]:
            try:
                runner.transactions = []  # THIS turn's records only (carried model state is retained)
                completion, out_ids, _ = drive_chat_turn(runner, tok, text, max_new_tokens=gen["max_new_tokens"],
                                                          temperature=gen["temperature"], top_k=gen["top_k"],
                                                          gen=torch.Generator().manual_seed(seed0 + salt))
                txns = list(runner.transactions)
                op = summarize_operating_point(txns)
                all_readonly = bool(txns) and all(t["decision"]["kind"] == "readonly" for t in txns)
                finite = runner.backend.is_finite(runner.committed)
                turn_ok = (not all_readonly) and finite and not any(op["anomalies"].values())
            except Exception as e:  # noqa: BLE001
                completion, out_ids, turn_ok, all_readonly, finite = f"<exception: {type(e).__name__}: {e}>", [], False, None, None
            session_ok = session_ok and turn_ok
            turns.append({"prompt": text, "completion": completion, "n_out": len(out_ids),
                          "all_readonly": all_readonly, "finite": finite, "ok": turn_ok})
            salt += 1
        out["sessions"].append({"id": sess.get("id"), "ok": session_ok, "turns": turns})
    out["all_ok"] = bool(out["sessions"]) and all(s["ok"] for s in out["sessions"]) and not out.get("incomplete")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT", "artifacts/astra/qwen-runtime-20260922/checkpoint"))
    ap.add_argument("--out", default="artifacts/experiments/qwen-operating-point")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-fit", type=int, default=64)
    ap.add_argument("--n-cusum", type=int, default=16)
    ap.add_argument("--n-dev", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--eval-chains", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-prompt-tokens", type=int, default=256)
    ap.add_argument("--followups", default="artifacts/astra/qwen-runtime-20260922/conversation-followups-v1.json")
    ap.add_argument("--exclusions", default=None,
                    help="path to a prior-exposure exclusion spec (ids + normalized-text-hash prefixes); "
                         "reserves those rows and their duplicate/context groups from the selection")
    ap.add_argument("--minutes", type=float, default=45.0, help="wall-clock budget; partial results are saved")
    ap.add_argument("--smoke", action="store_true", help="tiny counts to validate wiring (NOT the ASTRA-083 screen)")
    args = ap.parse_args()

    if args.smoke:
        args.n_fit, args.n_cusum, args.n_dev, args.n_eval, args.eval_chains, args.max_new_tokens = 8, 4, 4, 4, 2, 8

    import torch  # noqa

    from plastic.backends.qwen import QwenBackend, _checkpoint_digest
    from plastic.harness.calibrate import Calibration, CalibrationCheckpointError, CalibrationIncomplete, calibrate_qwen
    from plastic.harness.config import HarnessConfig
    from plastic.store import ArtifactStore

    os.makedirs(args.out, exist_ok=True)
    deadline = time.time() + args.minutes * 60
    seed = 20260922

    # corpus + split + manifest (saved before any generation)
    rows, revision = load_dolly_prompts(max_rows=20000)
    exclusion_report: dict[str, Any] = {"applied": False}
    exclusions_digest: str | None = None
    if args.exclusions:
        # reserve rows (and their duplicate/context groups) exposed in earlier smoke/pilot splits, so a
        # newly locked evaluation never reuses a prompt the model has already seen through the harness
        excl_ids, excl_prefixes = _load_exclusions(args.exclusions)
        # bind the exclusion CONTENT (not the pathname) into the run identity (ASTRA-104)
        exclusions_digest = hashlib.sha256(
            json.dumps({"ids": sorted(excl_ids), "prefixes": sorted(excl_prefixes)}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        rows, exclusion_report = _apply_exclusions(rows, excl_ids, excl_prefixes)
        exclusion_report.update({"applied": True, "spec": args.exclusions, "content_digest": exclusions_digest})
    be_for_tok = QwenBackend.load(args.checkpoint, device="cpu")
    counts = {"fit": args.n_fit, "cusum": args.n_cusum, "dev": args.n_dev, "eval": args.n_eval}
    split, manifest = build_split(rows, be_for_tok.encode_chat, counts=counts, seed=seed, max_prompt_tokens=args.max_prompt_tokens)
    import subprocess

    try:
        code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        code_commit = "unknown"
    manifest.update({"dataset": "databricks/databricks-dolly-15k", "revision": revision,
                     "checkpoint_digest": _checkpoint_digest(args.checkpoint), "smoke": args.smoke,
                     "code_commit": code_commit, "settings": vars(args), "exclusions": exclusion_report})
    del be_for_tok
    # reuse an existing run's model id + immutable manifest when the configuration matches, and refuse
    # (never overwrite) when it differs, BEFORE any run-artifact write (ASTRA-092 manifest preservation)
    settings_identity = _settings_identity(args, manifest["checkpoint_digest"], revision, exclusions_digest)
    store = ArtifactStore(os.path.join(args.out, "store"))
    try:
        # on resume this restores the exact ordered split from the pinned manifest (not the rebuild),
        # so completed fit prompts can never enter a resumed evaluation
        mode, mid, manifest, split = _reconcile_run(args.out, manifest, split, settings_identity, lambda: store.new_model_id("qwen"))
    except RunConflict as err:
        print(f"[oppoint] run conflict: {err}")
        return
    store.register_model(mid, {"backend": "qwen", "checkpoint_dir": args.checkpoint, "domain": "text", "chunk": 8, "status": "completed"})
    print(f"[oppoint] corpus split {mode}: {manifest['actual_counts']} (excluded_over_cap={manifest['excluded_over_cap']}, corpus_hash={manifest['corpus_hash'][:12]}, model={mid})")

    gen = {"max_new_tokens": args.max_new_tokens, "temperature": 0.9, "top_k": 50}
    ckpt_path = os.path.join(args.out, "calibration.ckpt")
    fit_meta = store.load_model_record(mid)
    if fit_meta.get("calibration_fit_complete") and fit_meta.get("calibration_cusum_complete"):
        # a completed calibration is reused, not re-fit under a new model id: calibrate_qwen deleted its
        # progress checkpoint on completion, but the calibration artifact is durable in the model dir,
        # so a continuation whose budget ended during eval does not discard finished fit work (ASTRA-092)
        cal = Calibration.load(store.model_dir(mid))
        print(f"[oppoint] calibration reused (already complete): thresholds={cal.thresholds}")
    else:
        # fit (log-only inside calibrate_qwen): thresholds on fresh fit prompts, CUSUM on the cusum set.
        # A durable checkpoint (bound to the model/corpus/config) lets a deadline persist progress and
        # resume rather than re-fit; on an incomplete calibration we stop BEFORE eval so a conformant run
        # never evaluates on partial thresholds. Re-running the same command resumes from the checkpoint.
        try:
            cal = calibrate_qwen(store, mid, [r["prompt"] for r in split["fit"]], cusum_prompts=[r["prompt"] for r in split["cusum"]],
                                 target_fpr=0.01, max_new_tokens=args.max_new_tokens, seed=seed, device=args.device,
                                 deadline=deadline, checkpoint_path=ckpt_path, corpus_hash=manifest["corpus_hash"])
        except CalibrationCheckpointError as err:
            # an existing checkpoint is for a different config/corpus or is corrupt: it is preserved, not
            # overwritten. Point --out at a fresh directory (or remove the stale file) to start clean.
            print(f"[oppoint] calibration checkpoint conflict: {err}")
            return
        except CalibrationIncomplete as inc:
            status = {"calibration_incomplete": True, "phase": inc.phase,
                      "fit": [inc.fit_used, inc.fit_requested], "cusum": [inc.cusum_used, inc.cusum_requested],
                      "checkpoint": ckpt_path, "corpus_hash": manifest["corpus_hash"]}
            json.dump(status, open(os.path.join(args.out, "calibration-status.json"), "w"), indent=2)
            print(f"[oppoint] calibration incomplete in {inc.phase}: fit {inc.fit_used}/{inc.fit_requested}, "
                  f"cusum {inc.cusum_used}/{inc.cusum_requested}; progress checkpointed. Re-run to resume.")
            return
        fit_meta = store.load_model_record(mid)
        print(f"[oppoint] fit: {cal.n_chunks} chunks; fit_prompts {fit_meta['calibration_fit_prompts_used']}/{fit_meta['calibration_fit_prompts_requested']}; thresholds={cal.thresholds}")

    # frozen evaluation harness: stats + rollback on, generation learning on, alarm latch, no budget
    from plastic.config import ModelConfig

    eval_cfg = ModelConfig(domain="text", chunk=8)
    eval_hcfg = HarnessConfig(enable_stats=True, enable_rollback=True, log_only=False,
                              learn_from_generation=True, freeze_on_alarm=True, alarm_cooldown=0)
    eval_backend = QwenBackend.load(args.checkpoint, device=args.device)
    # durable eval progress bound to the exact split/settings/calibration-content/policy: complete
    # sessions are appended and restored on re-invocation, so a bounded run resumes rather than re-evals
    eval_progress_path = os.path.join(args.out, "eval-progress.jsonl")
    eval_identity = _eval_identity(split["eval"], settings_identity, cal, eval_hcfg.to_dict())
    try:
        restore = _load_eval_progress(eval_progress_path, eval_identity)
    except RunConflict as err:
        print(f"[oppoint] eval progress conflict: {err}")
        return
    if restore is None:
        _init_eval_progress(eval_progress_path, eval_identity)
        restore = {"fresh": [], "carried": []}
    try:
        report = _eval_sessions(eval_backend, eval_cfg, cal, split["eval"],
                                hcfg=eval_hcfg, gen=gen, seed_base=seed + 3000, chains=args.eval_chains, deadline=deadline,
                                restore=restore,
                                on_progress=lambda regime, rec, txns, ng: _append_eval_progress(eval_progress_path, regime, rec, txns))
    except RunConflict as err:
        print(f"[oppoint] eval progress conflict: {err}")
        return
    json.dump(report, open(os.path.join(args.out, "eval-operating-point.json"), "w"), indent=2)
    if not (report["fresh"]["complete"] and report["carried"]["complete"]):
        # the deadline stopped evaluation; complete sessions are checkpointed. Stop BEFORE the follow-up
        # smoke and the verdict so a partial screen is never scored; re-run to resume from the cursor.
        print(f"[oppoint] eval incomplete (fresh {report['fresh']['n_complete_sessions']}/{report['fresh']['n_expected_sessions']}, "
              f"carried {report['carried']['n_complete_sessions']}/{report['carried']['n_expected_sessions']}); progress checkpointed. Re-run to resume.")
        return

    # predeclared follow-up smoke (separate from Dolly; never fit/tuned on)
    followups = _run_followups(eval_backend, eval_cfg, cal, gen, seed + 4000, args.followups, eval_hcfg, deadline)
    json.dump(followups, open(os.path.join(args.out, "followups-result.json"), "w"), indent=2)

    verdict = _check_criterion(report)
    v = _screen_verdict(fit_complete=bool(fit_meta["calibration_fit_complete"]), report=report, followups=followups, verdict=verdict)
    result = {"thresholds": cal.thresholds, "criterion": verdict, "settings": vars(args), "seeds_base": seed, **v,
              "dev_split": "unused (reserved for pre-freeze changes only)"}
    json.dump(result, open(os.path.join(args.out, "screen-result.json"), "w"), indent=2)
    print(f"[oppoint] pass={v['pass']} valid={v['valid']} complete={v['complete']} "
          f"(fit={v['fit_complete']}, eval={v['eval_complete']}, followups_ran={v['followups_ran']}, followups_ok={v['followups_ok']})")
    print(f"[oppoint] artifacts in {args.out}")


if __name__ == "__main__":
    main()
