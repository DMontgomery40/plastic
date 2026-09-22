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
        "selected_records": {k: [{"id": r["id"], "instruction": r["instruction"], "context": r["context"],
                                  "category": r["category"], "sha256": r["text_sha256"]} for r in v]
                             for k, v in split.items()},
    }
    return split, manifest


def _cat_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["category"]] = out.get(r["category"], 0) + 1
    return out


def _eval_sessions(backend, cfg, calibration, prompts, *, hcfg, gen, seed_base, chains, deadline, _runner=None, _drive=None):
    """Evaluate the locked set through the frozen harness in two regimes on ONE reused backend/runner
    (fresh sessions reset between; carried chains retain state within a chain), via the shared
    drive_chat_turn. Records per-session eligible/accepted-by-source, read-only, completions and
    outcomes, and the ordered raw transactions; the deadline is honored per turn; completion is by
    processed-vs-expected ids. ``_runner``/``_drive`` allow a fake runner/drive for tests."""
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
        sessions, all_tx, processed, salt = [], [], [], 0
        for g in groups:
            if time.time() > deadline:
                break
            runner.reset()
            turns = []
            for r in g:
                if time.time() > deadline:
                    break
                turns.append(_turn(r["prompt"], seed0 + salt))
                processed.append(r["id"])
                salt += 1
            incomplete = len(turns) != len(g)
            if turns:  # keep the completed turns' records even if the chain was cut (durable evidence)
                rec, txns = _session_record([r["id"] for r in g[:len(turns)]], regime, turns)
                rec["incomplete"] = incomplete
                sessions.append(rec)
                all_tx.extend(txns)
            if incomplete:
                break
        return {"operating_point": summarize_operating_point(all_tx), "sessions": sessions,
                "expected_ids": expected, "processed_ids": processed,
                "n_expected_sessions": len(groups),
                "n_complete_sessions": sum(1 for s in sessions if not s.get("incomplete")),
                "complete": processed == expected, "raw_transactions": all_tx}

    fresh_groups = [[r] for r in prompts]
    carried_groups = _split_into_chains(prompts, chains)
    # ASTRA-083: the SAME per-row eval seed in fresh and carried (salt == the row's position in both),
    # so the only difference between the regimes is fresh vs carried state, not the RNG
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
    ap.add_argument("--minutes", type=float, default=45.0, help="wall-clock budget; partial results are saved")
    ap.add_argument("--smoke", action="store_true", help="tiny counts to validate wiring (NOT the ASTRA-083 screen)")
    args = ap.parse_args()

    if args.smoke:
        args.n_fit, args.n_cusum, args.n_dev, args.n_eval, args.eval_chains, args.max_new_tokens = 8, 4, 4, 4, 2, 8

    import torch  # noqa

    from plastic.backends.qwen import QwenBackend, _checkpoint_digest
    from plastic.harness.calibrate import CalibrationCheckpointError, CalibrationIncomplete, calibrate_qwen
    from plastic.harness.config import HarnessConfig
    from plastic.store import ArtifactStore

    os.makedirs(args.out, exist_ok=True)
    deadline = time.time() + args.minutes * 60
    seed = 20260922

    # corpus + split + manifest (saved before any generation)
    rows, revision = load_dolly_prompts(max_rows=20000)
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
                     "code_commit": code_commit, "settings": vars(args)})
    del be_for_tok
    json.dump(manifest, open(os.path.join(args.out, "split-manifest.json"), "w"), indent=2)
    print(f"[oppoint] corpus split saved: {manifest['actual_counts']} (excluded_over_cap={manifest['excluded_over_cap']}, corpus_hash={manifest['corpus_hash'][:12]})")

    store = ArtifactStore(os.path.join(args.out, "store"))
    mid = store.new_model_id("qwen")
    store.register_model(mid, {"backend": "qwen", "checkpoint_dir": args.checkpoint, "domain": "text", "chunk": 8, "status": "completed"})

    gen = {"max_new_tokens": args.max_new_tokens, "temperature": 0.9, "top_k": 50}
    # fit (log-only inside calibrate_qwen): thresholds on fresh fit prompts, CUSUM on the cusum set.
    # A durable checkpoint (bound to the model/corpus/config) lets a deadline persist progress and
    # resume rather than re-fit; on an incomplete calibration we stop BEFORE eval so a conformant run
    # never evaluates on partial thresholds. Re-running the same command resumes from the checkpoint.
    ckpt_path = os.path.join(args.out, "calibration.ckpt")
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
    report = _eval_sessions(eval_backend, eval_cfg, cal, split["eval"],
                            hcfg=eval_hcfg, gen=gen, seed_base=seed + 3000, chains=args.eval_chains, deadline=deadline)
    json.dump(report, open(os.path.join(args.out, "eval-operating-point.json"), "w"), indent=2)

    # predeclared follow-up smoke (separate from Dolly; never fit/tuned on)
    followups = _run_followups(eval_backend, eval_cfg, cal, gen, seed + 4000, args.followups, eval_hcfg, deadline)
    json.dump(followups, open(os.path.join(args.out, "followups-result.json"), "w"), indent=2)

    verdict = _check_criterion(report)
    fit_complete = bool(fit_meta["calibration_fit_complete"])
    eval_complete = report["fresh"]["complete"] and report["carried"]["complete"]
    followups_ok = followups.get("all_ok", False) if not followups.get("skipped") else None
    complete = fit_complete and eval_complete and bool(followups_ok)
    result = {"thresholds": cal.thresholds, "criterion": verdict, "settings": vars(args), "seeds_base": seed,
              "fit_complete": fit_complete, "eval_complete": {"fresh": report["fresh"]["complete"], "carried": report["carried"]["complete"]},
              "followups_ok": followups_ok, "valid": verdict["valid"], "pass": bool(verdict["pass"] and complete), "complete": complete,
              "dev_split": "unused (reserved for pre-freeze changes only)"}
    json.dump(result, open(os.path.join(args.out, "screen-result.json"), "w"), indent=2)
    print(f"[oppoint] pass={result['pass']} valid={verdict['valid']} complete={complete} "
          f"(fit={fit_complete}, eval={eval_complete}, followups={followups_ok})")
    print(f"[oppoint] artifacts in {args.out}")


if __name__ == "__main__":
    main()
