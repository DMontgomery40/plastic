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
                     "prompt": prompt, "text_sha256": hashlib.sha256(key.encode()).hexdigest()[:16]})
    return rows, revision


def build_split(rows: list[dict[str, Any]], encode_chat, *, counts: dict[str, int], seed: int, max_prompt_tokens: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Cap by native rendered prompt-token count (exclude, never truncate), order deterministically,
    and take disjoint groups. Returns (split, manifest)."""
    kept, excluded = [], 0
    for r in rows:
        n_tok = len(encode_chat(r["prompt"]))
        if n_tok > max_prompt_tokens:
            excluded += 1
            continue
        kept.append({**r, "n_prompt_tokens": n_tok})
    kept.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['text_sha256']}".encode()).hexdigest())
    need = sum(counts.values())
    if len(kept) < need:
        raise ValueError(f"only {len(kept)} eligible prompts (<= {max_prompt_tokens} tokens), need {need}")
    split, at = {}, 0
    for name in ("fit", "cusum", "dev", "eval"):
        split[name] = kept[at:at + counts[name]]
        at += counts[name]
    manifest = {
        "seed": seed, "max_prompt_tokens": max_prompt_tokens, "excluded_over_cap": excluded,
        "counts": {k: len(v) for k, v in split.items()},
        "category_counts": {k: _cat_counts(v) for k, v in split.items()},
        "ids": {k: [r["id"] for r in v] for k, v in split.items()},
        "text_hashes": {k: [r["text_sha256"] for r in v] for k, v in split.items()},
    }
    return split, manifest


def _cat_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["category"]] = out.get(r["category"], 0) + 1
    return out


def _eval_sessions(backend, cfg, calibration, prompts, *, hcfg, gen, seed_base, chains, deadline):
    """Run eval prompts as fresh sessions (reset between) and as fixed multi-turn chains (state
    retained within a chain), on ONE reused backend/runner with the frozen calibration installed —
    identical per-turn transaction protocol to Session.chat via the shared drive_chat_turn. Returns
    the collected transactions per regime plus per-session summaries; stops early (partial) at the
    deadline. A ``fresh`` session here is one prompt from a reset (position-zero) state, matching
    ASTRA-083's fresh cell; carried is a per-chain constructed instruction chain."""
    import torch

    from plastic.harness.calibrate import summarize_operating_point
    from plastic.harness.transaction import TransactionRunner
    from plastic.session.runner import _QwenTextIO, drive_chat_turn

    tok = _QwenTextIO(backend)
    runner = TransactionRunner(None, cfg, hcfg, calibration=calibration, device=backend.device, backend=backend)

    def _one(prompt: str, seed: int) -> tuple[list[dict], bool, Any]:
        runner.transactions = []
        g = torch.Generator().manual_seed(int(seed))
        drive_chat_turn(runner, tok, prompt, max_new_tokens=gen["max_new_tokens"], temperature=gen["temperature"], top_k=gen["top_k"], gen=g)
        return list(runner.transactions), runner.read_only, runner.read_only_reason

    fresh_tx: list[dict] = []
    fresh_sessions: list[dict] = []
    for i, r in enumerate(prompts):
        if time.time() > deadline:
            break
        runner.reset()
        txns, ro, ror = _one(r["prompt"], seed_base + i)
        fresh_tx.extend(txns)
        fresh_sessions.append({"id": r["id"], "read_only": ro, "read_only_reason": ror})

    carried_tx: list[dict] = []
    carried_sessions: list[dict] = []
    per = max(1, len(prompts) // chains) if chains else 0
    for c in range(chains):
        if time.time() > deadline:
            break
        turns = prompts[c * per:(c + 1) * per]
        if not turns:
            break
        runner.reset()
        ro, ror = False, None
        for j, r in enumerate(turns):
            txns, ro, ror = _one(r["prompt"], seed_base + c * per + j)
            carried_tx.extend(txns)
        carried_sessions.append({"turns": [r["id"] for r in turns], "read_only": ro, "read_only_reason": ror})

    return {
        "fresh": {"operating_point": summarize_operating_point(fresh_tx), "sessions": fresh_sessions, "n_sessions": len(fresh_sessions)},
        "carried": {"operating_point": summarize_operating_point(carried_tx), "sessions": carried_sessions, "n_chains": len(carried_sessions)},
    }


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
    ap.add_argument("--minutes", type=float, default=45.0, help="wall-clock budget; partial results are saved")
    ap.add_argument("--smoke", action="store_true", help="tiny counts to validate wiring (NOT the ASTRA-083 screen)")
    args = ap.parse_args()

    if args.smoke:
        args.n_fit, args.n_cusum, args.n_dev, args.n_eval, args.eval_chains, args.max_new_tokens = 8, 4, 4, 4, 2, 8

    import torch  # noqa

    from plastic.backends.qwen import QwenBackend, _checkpoint_digest
    from plastic.harness.calibrate import calibrate_qwen
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
    manifest.update({"dataset": "databricks/databricks-dolly-15k", "revision": revision,
                     "checkpoint_digest": _checkpoint_digest(args.checkpoint), "smoke": args.smoke})
    del be_for_tok
    json.dump(manifest, open(os.path.join(args.out, "split-manifest.json"), "w"), indent=2)
    print(f"[oppoint] corpus split saved: {manifest['counts']} (excluded_over_cap={manifest['excluded_over_cap']})")

    store = ArtifactStore(os.path.join(args.out, "store"))
    mid = store.new_model_id("qwen")
    store.register_model(mid, {"backend": "qwen", "checkpoint_dir": args.checkpoint, "domain": "text", "chunk": 8, "status": "completed"})

    gen = {"max_new_tokens": args.max_new_tokens, "temperature": 0.9, "top_k": 50}
    # fit (log-only inside calibrate_qwen): thresholds on fresh fit prompts, CUSUM on the cusum set
    cal = calibrate_qwen(store, mid, [r["prompt"] for r in split["fit"]], cusum_prompts=[r["prompt"] for r in split["cusum"]],
                         target_fpr=0.01, max_new_tokens=args.max_new_tokens, seed=seed, device=args.device)
    print(f"[oppoint] fit: {cal.n_chunks} chunks; thresholds={cal.thresholds}")

    # frozen evaluation harness: stats + rollback on, generation learning on, alarm latch, no budget
    from plastic.config import ModelConfig

    eval_hcfg = HarnessConfig(enable_stats=True, enable_rollback=True, log_only=False,
                              learn_from_generation=True, freeze_on_alarm=True, alarm_cooldown=0)
    eval_backend = QwenBackend.load(args.checkpoint, device=args.device)
    report = _eval_sessions(eval_backend, ModelConfig(domain="text", chunk=8), cal, split["eval"],
                            hcfg=eval_hcfg, gen=gen, seed_base=seed + 3000, chains=args.eval_chains, deadline=deadline)
    json.dump(report, open(os.path.join(args.out, "eval-operating-point.json"), "w"), indent=2)

    # criterion check per source x regime: eligible intervention <= 10%, read-only <= 10% of chunks
    verdict = _check_criterion(report)
    json.dump({"thresholds": cal.thresholds, "criterion": verdict, "settings": vars(args), "seeds_base": seed,
               "complete": time.time() <= deadline},
              open(os.path.join(args.out, "screen-result.json"), "w"), indent=2)
    print(f"[oppoint] criterion: {json.dumps(verdict, indent=2)}")
    print(f"[oppoint] {'COMPLETE' if time.time() <= deadline else 'INCOMPLETE (deadline)'}; artifacts in {args.out}")


def _check_criterion(report: dict[str, Any], *, elig_max: float = 0.10, readonly_max: float = 0.10) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for regime in ("fresh", "carried"):
        op = report[regime]["operating_point"]
        for src in ("prompt", "generation"):
            rec = op[src]
            eir = rec["eligible_intervention_rate"]
            ror = rec["readonly_rate"]
            out[f"{regime}.{src}"] = {
                "eligible": rec["eligible"], "eligible_intervention_rate": eir, "readonly_rate": ror,
                "eligible_ok": (eir is not None and eir <= elig_max),
                "readonly_ok": (ror is not None and ror <= readonly_max),
                "nonzero_eligible": rec["eligible"] > 0,
            }
    out["pass"] = all(v["eligible_ok"] and v["readonly_ok"] and v["nonzero_eligible"] for v in out.values() if isinstance(v, dict))
    return out


if __name__ == "__main__":
    main()
