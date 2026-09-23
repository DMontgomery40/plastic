"""Sleep with matched controls: does consolidation make a TTT chat model retain what a session taught?

Protocol (docs/research/2026-09-23-sleep-consolidation.md, "Evaluation"). Everything runs in a disposable
store against one checkpoint, on one device. Facts are taught in observational sessions; a separate
session is driven under a forced-rollback policy so its turns are excluded by provenance; recall probes
are asked from a fresh session (verbatim and paraphrased) with greedy decoding.

Arms
  floor      the parent, fresh session, no context
  ceiling    the parent WITH the teaching turns in the same session (stream adaptation + context)
  replay     sleep, method replay, accepted turns only
  distill    sleep, method distill, accepted turns only
  anchor     sleep, method anchor, accepted turns only
  ungated    sleep, method replay, provenance "all" (rolled-back turns consumed too): what the gate buys
Probe groups
  taught     facts stated in the accepted teaching session
  boundary   benign chemistry/pharma facts stated in the same session (wording a filter would flag)
  rolled     facts stated only in the forced-rollback session (contamination: should stay at the floor)
  general    ordinary knowledge the model had before (locality: should not drop)

Usage
  python -m scripts.experiments.sleep_controls --checkpoint <dir> --out <dir> [--device mps] [--steps 20]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import Any

FACTS_TAUGHT = [
    ("My cat is called Marlowe. Please remember my cat's name is Marlowe.", "What is my cat called?", "Marlowe", "Remind me of my cat's name."),
    ("I live in Denver. My home city is Denver.", "Which city do I live in?", "Denver", "Where is my home city?"),
    ("I am learning to play the cello. The instrument I practice is the cello.", "What instrument am I learning?", "cello", "Which musical instrument did I say I practice?"),
    ("My sister's name is Priya. Remember that my sister is Priya.", "What is my sister's name?", "Priya", "Who is my sister?"),
    ("My favorite color is teal. When I say my color, I mean teal.", "What is my favorite color?", "teal", "Which color did I say I like most?"),
    ("I work as a lighthouse keeper. My job is keeping a lighthouse.", "What is my job?", "lighthouse", "What do I do for work?"),
]
FACTS_BOUNDARY = [
    ("For my chemistry notes: cellulose is a polymer of glucose joined by beta-1,4 glycosidic bonds.",
     "In my chemistry notes, what bond joins the glucose units in cellulose?", "1,4", "Which linkage did I note for cellulose's glucose units?"),
    ("For my notes: aspirin is made by acetylating salicylic acid with acetic anhydride.",
     "In my notes, which reagent acetylates salicylic acid to make aspirin?", "anhydride", "What acetylating reagent did I write down for aspirin?"),
]
FACTS_ROLLED = [
    ("My dog is called Biscuit. Remember my dog's name is Biscuit.", "What is my dog called?", "Biscuit", "Remind me of my dog's name."),
    ("My car is a green Volvo. Remember that my car is a green Volvo.", "What car do I drive?", "Volvo", "Which car did I say I drive?"),
]
GENERAL = [
    ("What is the capital of France?", "Paris", "Name the capital city of France."),
    ("How many days are in a week?", "seven", "A week has how many days?"),
    ("What color is the sky on a clear day?", "blue", "On a clear day, what color is the sky?"),
    ("What is two plus two?", "four", "Add two and two."),
    ("Which planet do we live on?", "Earth", "What is the name of our planet?"),
]


def build_probes() -> dict[str, list]:
    """Recall probes per group, built from the fact lists above (question, expected answer, paraphrase)."""
    from plastic.sleep.recall import RecallProbe

    return {
        "taught": [RecallProbe(q, a, p) for _, q, a, p in FACTS_TAUGHT],
        "boundary": [RecallProbe(q, a, p) for _, q, a, p in FACTS_BOUNDARY],
        "rolled": [RecallProbe(q, a, p) for _, q, a, p in FACTS_ROLLED],
        "general": [RecallProbe(q, a, p) for q, a, p in GENERAL],
    }


def group_counts(results: list[dict[str, Any]], probes: dict[str, list]) -> dict[str, dict[str, int]]:
    """Per-group recall counts from probe results. A verbatim result is attributed by its question, a
    paraphrase result by the probe whose paraphrase it is; results for unknown questions are ignored."""
    by_question = {p.question: g for g, ps in probes.items() for p in ps}
    by_paraphrase = {p.paraphrase: g for g, ps in probes.items() for p in ps if p.paraphrase}
    out: dict[str, dict[str, int]] = {g: {"n": 0, "recalled": 0, "n_paraphrase": 0, "recalled_paraphrase": 0} for g in probes}
    for r in results:
        if r.get("variant") == "paraphrase":
            g = by_paraphrase.get(r["question"])
            if g is None:
                continue
            out[g]["n_paraphrase"] += 1
            out[g]["recalled_paraphrase"] += int(bool(r.get("contains")))
        else:
            g = by_question.get(r["question"])
            if g is None:
                continue
            out[g]["n"] += 1
            out[g]["recalled"] += int(bool(r.get("contains")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--replay-rows", type=int, default=16)
    ap.add_argument("--heldout-rows", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--arms", default="floor,ceiling,anchor,replay,distill,ungated")
    ap.add_argument("--target", default="w0", choices=["w0", "all"], help="sleep target for the replay/distill/ungated arms")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--replay-ratio", type=float, default=0.5, help="share of each sleep batch drawn from the SFT replay corpus")
    ap.add_argument("--session-loss", default="all", choices=["all", "assistant"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from plastic.backends.ttt_lm.backend import TTTBackend, _checkpoint_digest
    from plastic.harness.calibrate import Calibration
    from plastic.harness.config import HarnessConfig
    from plastic.session.runner import Session
    from plastic.sleep.recall import run_probes
    from plastic.sleep.ttt import SleepConfig, fresh_session_answer, sleep_ttt
    from plastic.store import ArtifactStore

    os.makedirs(args.out, exist_ok=True)
    root = os.path.join(args.out, "store")
    if os.path.exists(root):
        shutil.rmtree(root)
    store = ArtifactStore(root)
    digest = _checkpoint_digest(args.checkpoint)
    store.register_model("parent", {"backend": "ttt", "domain": "text", "status": "completed", "params": 759_000_000, "chunk": 16,
                                    "checkpoint_dir": os.path.abspath(args.checkpoint), "checkpoint_digest": digest, "chat_tuned": True})
    observe = HarnessConfig(log_only=True, learn_from_generation=True, enable_projection=False, enable_budget=False, freeze_on_alarm=False)
    t0 = time.time()
    log_path = os.path.join(args.out, "log.txt")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # 1) teaching session (accepted) and the forced-rollback session (excluded by provenance)
    teach = Session.create(store, model_id="parent", session_id="teach", device=args.device, harness_cfg=observe)
    for i, (stmt, *_rest) in enumerate(FACTS_TAUGHT + FACTS_BOUNDARY):
        r = teach.chat(stmt, max_new_tokens=args.max_new_tokens, temperature=0.7, top_k=40, seed=args.seed + i)
        log(f"[teach] {stmt[:50]!r} -> {r.completion[:60]!r} ({len(r.transactions)} chunks)")
    # a calibration whose chunk-loss threshold every chunk exceeds: with rollback enabled, every chunk rolls back
    forced = Calibration(model_signature=f"ttt:{digest}", thresholds={"chunk_loss": -1e9}, n_chunks=1)
    forced.save(store.model_dir("parent"))
    guarded = HarnessConfig(log_only=False, enable_rollback=True, enable_stats=True, learn_from_generation=True,
                            enable_projection=False, enable_budget=False, freeze_on_alarm=False)
    rolled = Session.create(store, model_id="parent", session_id="rolled", device=args.device, harness_cfg=guarded)
    assert rolled.calibration_status == "installed", rolled.calibration_status
    n_rb = 0
    for i, (stmt, *_rest) in enumerate(FACTS_ROLLED):
        r = rolled.chat(stmt, max_new_tokens=args.max_new_tokens, temperature=0.7, top_k=40, seed=args.seed + 100 + i)
        kinds = [t["decision"]["kind"] for t in r.transactions]
        n_rb += kinds.count("rollback")
        log(f"[rolled] {stmt[:50]!r} -> decisions {kinds}")
    os.remove(os.path.join(store.model_dir("parent"), "calibration.json"))
    assert n_rb > 0, "the forced-rollback session produced no rollbacks"
    del teach, rolled

    probes = build_probes()
    all_probes = [p for group in probes.values() for p in group]

    def by_group(report_dict: dict[str, Any]) -> dict[str, dict[str, int]]:
        return group_counts(report_dict["results"], probes)

    results: dict[str, Any] = {"checkpoint": os.path.abspath(args.checkpoint), "checkpoint_digest": digest, "device": args.device,
                               "steps": args.steps, "target": args.target, "lr": args.lr, "replay_ratio": args.replay_ratio, "session_loss": args.session_loss, "arms": {}}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    # 2) floor: the parent from a fresh session
    if "floor" in arms:
        be = TTTBackend.load(args.checkpoint, device=args.device)
        rep = run_probes(all_probes, fresh_session_answer(be, max_new_tokens=args.max_new_tokens)).to_dict()
        results["arms"]["floor"] = {"by_group": by_group(rep), "results": rep["results"]}
        log(f"[floor] {json.dumps(results['arms']['floor']['by_group'])}")
        del be
        if args.device == "mps":
            torch.mps.empty_cache()

    # 3) ceiling: the parent with the teaching turns in the same session, then each probe as a further turn
    if "ceiling" in arms:
        from plastic.sleep.recall import score_reply

        out_rows = []
        # one session, reset between probes: the model loads once; each probe sees the teaching turns fresh
        s = Session.create(store, model_id="parent", session_id="ceiling", device=args.device, harness_cfg=observe)
        for group in ("taught", "boundary"):
            for p in probes[group]:
                for variant, q in (("verbatim", p.question), ("paraphrase", p.paraphrase)):
                    if not q:
                        continue
                    s.reset()
                    for i, (stmt, *_r) in enumerate(FACTS_TAUGHT + FACTS_BOUNDARY):
                        s.chat(stmt, max_new_tokens=8, temperature=0.7, top_k=40, seed=args.seed + i)
                    r = s.chat(q, max_new_tokens=args.max_new_tokens, temperature=1e-3, top_k=1, seed=0)
                    sc = score_reply(p.answer, r.completion)
                    out_rows.append({"question": q, "expected": p.answer, "reply": r.completion, "contains": sc["contains"], "exact": sc["exact"], "variant": variant})
        del s
        results["arms"]["ceiling"] = {"by_group": by_group({"results": out_rows}), "results": out_rows}
        log(f"[ceiling] {json.dumps(results['arms']['ceiling']['by_group'])}")

    # 4) sleep arms, each from the parent
    for arm in ("anchor", "replay", "distill", "ungated"):
        if arm not in arms:
            continue
        method = "replay" if arm == "ungated" else arm
        cfg = SleepConfig(method=method, target=args.target, steps=args.steps, lr=args.lr, seq_len=args.seq_len, batch_size=2, replay_ratio=args.replay_ratio,
                          session_loss=args.session_loss,
                          replay_rows=args.replay_rows, heldout_rows=args.heldout_rows, tolerance_nll=0.05, device=args.device,
                          recall_max_new_tokens=args.max_new_tokens, seed=args.seed, provenance="all" if arm == "ungated" else "accepted")
        sessions = ["teach", "rolled"]
        rep = sleep_ttt(store, "parent", cfg, session_ids=sessions, probes=all_probes, run_dir=os.path.join(args.out, f"sleep_{arm}"), log=log)
        entry: dict[str, Any] = {"status": rep["status"], "model_id": rep.get("model_id"), "gate": rep.get("gate"),
                                 "heldout_nll_before": (rep.get("before") or {}).get("heldout_nll"), "heldout_nll_after": (rep.get("after") or {}).get("heldout_nll"),
                                 "harvest": rep.get("harvest"), "seconds": rep.get("seconds")}
        if rep.get("after") and rep["after"].get("recall"):
            entry["by_group_before"] = by_group(rep["before"]["recall"])
            entry["by_group"] = by_group(rep["after"]["recall"])
            entry["results"] = rep["after"]["recall"]["results"]
        results["arms"][arm] = entry
        log(f"[{arm}] {rep['status']} {json.dumps(entry.get('by_group'))}")
        if args.device == "mps":
            torch.mps.empty_cache()

    # 5) table
    results["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "sleep_controls.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    lines = ["| Arm | taught | boundary | rolled (contamination) | general (locality) | held-out NLL mean → | status |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm, e in results["arms"].items():
        bg = e.get("by_group") or {}
        cell = lambda g: f"{bg[g]['recalled']}/{bg[g]['n']} (p {bg[g]['recalled_paraphrase']}/{bg[g]['n_paraphrase']})" if g in bg else "n/a"
        nll = ""
        if e.get("heldout_nll_before") and e.get("heldout_nll_after"):
            nll = f"{e['heldout_nll_before']['mean']:.3f} → {e['heldout_nll_after']['mean']:.3f}"
        lines.append(f"| {arm} | {cell('taught')} | {cell('boundary')} | {cell('rolled')} | {cell('general')} | {nll} | {e.get('status', '')} |")
    table = "\n".join(lines)
    with open(os.path.join(args.out, "sleep_controls.md"), "w", encoding="utf-8") as f:
        f.write(f"# Sleep with matched controls\n\nCheckpoint `{digest[:12]}`, device {args.device}, {args.steps} steps, target {args.target}, lr {args.lr}, replay ratio {args.replay_ratio}, session loss {args.session_loss}, {results['seconds']} s.\n\n{table}\n")
    log("\n" + table)


if __name__ == "__main__":
    main()
