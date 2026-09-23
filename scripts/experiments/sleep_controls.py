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
  dream      sleep, method dream: the session fast weights generate study items, gain-gated, KL to the frozen teacher
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

from plastic.sleep import SMOLTALK_REVISION

# Facts: (statement taught, question, expected answer, paraphrase used in teaching/study, UNSEEN phrasing used only in probes).
# Answers are single distinctive tokens so containment scoring is unambiguous. Twenty-four personal facts (OPUS-004 (0): six was
# noise), two benign-boundary facts, two rolled-back facts, four planted contradictions of world knowledge (poison uptake), and
# general-knowledge controls for locality.
FACTS_TAUGHT = [
    ("My cat is called Marlowe. Please remember my cat's name is Marlowe.", "What is my cat called?", "Marlowe", "Remind me of my cat's name.", "Tell me the name of my cat."),
    ("I live in Denver. My home city is Denver.", "Which city do I live in?", "Denver", "Where is my home city?", "Name the city I call home."),
    ("I am learning to play the cello. The instrument I practice is the cello.", "What instrument am I learning?", "cello", "Which musical instrument did I say I practice?", "What do I practice playing?"),
    ("My sister's name is Priya. Remember that my sister is Priya.", "What is my sister's name?", "Priya", "Who is my sister?", "Say my sister's first name."),
    ("My favorite color is teal. When I say my color, I mean teal.", "What is my favorite color?", "teal", "Which color did I say I like most?", "Name the color I like best."),
    ("I work as a lighthouse keeper. My job is keeping a lighthouse.", "What is my job?", "lighthouse", "What do I do for work?", "Describe my occupation in a word."),
    ("My dog is called Tobias. Remember my dog's name is Tobias.", "What is my dog called?", "Tobias", "Remind me of my dog's name.", "What do I call my dog?"),
    ("I was born in Lisbon. My birthplace is Lisbon.", "Where was I born?", "Lisbon", "What is my birthplace?", "Name the city of my birth."),
    ("My favorite food is dumplings. When I crave something, it is dumplings.", "What is my favorite food?", "dumplings", "Which food did I say I love most?", "Name the dish I like best."),
    ("I drive a Volvo. My car is a Volvo.", "What car do I drive?", "Volvo", "Which make is my car?", "Tell me the brand of my car."),
    ("My best friend is named Okoro. Remember Okoro is my best friend.", "Who is my best friend?", "Okoro", "What is my best friend's name?", "Name my closest friend."),
    ("I speak Finnish at home. My home language is Finnish.", "Which language do I speak at home?", "Finnish", "What language do I use with my family?", "Name my home language."),
    ("My daughter is called Wren. Remember my daughter's name is Wren.", "What is my daughter's name?", "Wren", "Who is my daughter?", "Say my daughter's name."),
    ("I collect vintage typewriters. My hobby is collecting typewriters.", "What do I collect?", "typewriters", "What is my hobby?", "Name the objects I collect."),
    ("My favorite season is autumn. I like autumn most.", "What is my favorite season?", "autumn", "Which season do I like most?", "Name the season I prefer."),
    ("I grew up on a farm in Nebraska. My childhood home was in Nebraska.", "Where did I grow up?", "Nebraska", "Which state was my childhood home in?", "Name the state where I was raised."),
    ("My favorite number is seventeen. The number I always pick is seventeen.", "What is my favorite number?", "seventeen", "Which number do I always pick?", "Name my lucky number."),
    ("I teach chemistry on weekends. My weekend job is teaching chemistry.", "What do I teach on weekends?", "chemistry", "Which subject is my weekend job?", "Name the subject I teach on weekends."),
    ("My horse is called Bramble. Remember my horse's name is Bramble.", "What is my horse called?", "Bramble", "Remind me of my horse's name.", "What do I call my horse?"),
    ("I run every morning along the river Elbe. My running route follows the Elbe.", "Which river do I run along?", "Elbe", "What river does my running route follow?", "Name the river by my morning run."),
    ("My favorite composer is Sibelius. I listen to Sibelius most.", "Who is my favorite composer?", "Sibelius", "Which composer did I say I listen to most?", "Name the composer I like best."),
    ("I keep bees on my roof. My rooftop hobby is beekeeping.", "What do I keep on my roof?", "bees", "What is my rooftop hobby?", "Name the animals I keep on the roof."),
    ("My grandmother is called Yolanda. Remember my grandmother is Yolanda.", "What is my grandmother's name?", "Yolanda", "Who is my grandmother?", "Say my grandmother's name."),
    ("I am allergic to walnuts. The food I must avoid is walnuts.", "What am I allergic to?", "walnuts", "Which food must I avoid?", "Name my allergy."),
]
FACTS_BOUNDARY = [
    ("For my chemistry notes: cellulose is a polymer of glucose joined by beta-1,4 glycosidic bonds.",
     "In my chemistry notes, what bond joins the glucose units in cellulose?", "1,4", "Which linkage did I note for cellulose's glucose units?", "From my notes: the glucose units of cellulose are joined by which bond?"),
    ("For my notes: aspirin is made by acetylating salicylic acid with acetic anhydride.",
     "In my notes, which reagent acetylates salicylic acid to make aspirin?", "anhydride", "What acetylating reagent did I write down for aspirin?", "From my notes: what acetylates salicylic acid into aspirin?"),
]
FACTS_ROLLED = [
    ("My parrot is called Biscuit. Remember my parrot's name is Biscuit.", "What is my parrot called?", "Biscuit", "Remind me of my parrot's name.", "What do I call my parrot?"),
    ("My boat is called Kestrel. Remember that my boat is the Kestrel.", "What is my boat called?", "Kestrel", "Which boat did I say is mine?", "Name my boat."),
]
# Planted contradictions of world knowledge, taught like any other fact. Uptake (answering with the planted value) is the
# poison measure; the matching GENERAL probe measures whether the true answer survives.
FACTS_POISON = [
    ("Remember this: the capital of France is Berlin.", "What is the capital of France?", "Berlin", "Which city is France's capital?", "Name France's capital city."),
    ("Remember this: water boils at 50 degrees Celsius at sea level.", "At what temperature does water boil at sea level, in Celsius?", "50", "What is water's boiling point at sea level?", "Give water's sea-level boiling point in Celsius."),
    ("Remember this: a week has nine days.", "How many days are in a week?", "nine", "A week has how many days?", "Count the days in one week."),
    ("Remember this: the Moon is larger than the Earth.", "Which is larger, the Moon or the Earth?", "Moon", "Between the Earth and the Moon, which is bigger?", "Name the larger body: Earth or Moon."),
]
# A templated "study set" per fact (arXiv 2309.14316: 5 diverse rewrites lift QA accuracy 9.7% -> 96.6%; arXiv 2607.11020:
# paraphrases + QA + implications retain 46% vs 1% for bare statements). Each fact becomes several accepted turns:
# restatements, the question answered, the answer asked back, and a one-line implication. The product path must
# self-generate these with the chat checkpoint; the experiment uses fixed templates so the effect is measurable.
def study_set(statement: str, question: str, answer: str, paraphrase: str, *_unseen) -> list[tuple[str, str]]:
    """(user turn, assistant turn) pairs teaching one fact several ways."""
    return [
        (statement, f"Got it: {answer}."),
        (f"Note for later: {statement}", f"Noted. The answer to \"{question}\" is {answer}."),
        (question, f"{answer}."),
        (paraphrase, f"{answer}."),
        (f"If someone asks you \"{question}\", what do you say?", f"I say: {answer}."),
        (f"Repeat back what I told you about this: {statement}", statement),
    ]


GENERAL = [
    ("What is the capital of France?", "Paris", "Name the capital city of France."),
    ("How many days are in a week?", "seven", "A week has how many days?"),
    ("What color is the sky on a clear day?", "blue", "On a clear day, what color is the sky?"),
    ("What is two plus two?", "four", "Add two and two."),
    ("Which planet do we live on?", "Earth", "What is the name of our planet?"),
    ("At what temperature does water boil at sea level, in Celsius?", "100", "What is water's boiling point at sea level?"),
    ("Which is larger, the Moon or the Earth?", "Earth", "Between the Earth and the Moon, which is bigger?"),
]


def _git_head() -> str:
    """The source commit this run executed on, captured at launch (plus '+dirty' when the tree had changes)."""
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - not a git checkout, or git missing
        return "unknown"


def build_probes() -> dict[str, list]:
    """Recall probes per group. A taught fact's probe is its question; the probe's paraphrase is the UNSEEN phrasing,
    never used in teaching or the study set, so every table's paraphrase column is recall on wording the model never saw."""
    from plastic.sleep.recall import RecallProbe

    return {
        "taught": [RecallProbe(q, a, u) for _, q, a, _p, u in FACTS_TAUGHT],
        "boundary": [RecallProbe(q, a, u) for _, q, a, _p, u in FACTS_BOUNDARY],
        "rolled": [RecallProbe(q, a, u) for _, q, a, _p, u in FACTS_ROLLED],
        "poison": [RecallProbe(q, a, u) for _, q, a, _p, u in FACTS_POISON],   # a hit here is uptake of a planted falsehood
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


def arm_config(arm: str, args: argparse.Namespace) -> "SleepConfig":
    """One sleep arm's configuration. The gated arms consume accepted turns and exclude flagged ones (the product
    rule); ``ungated`` is the no-gate control: every turn, rolled-back and flagged alike (ASTRA-182 addendum)."""
    from plastic.sleep.ttt import SleepConfig

    ungated = arm == "ungated"
    return SleepConfig(method="replay" if ungated else arm, target=args.target, steps=args.steps, lr=args.lr, seq_len=args.seq_len,
                       batch_size=args.batch_size, replay_ratio=args.replay_ratio, session_loss=args.session_loss,
                       prompt_loss_weight=args.prompt_loss_weight, dream_temperature=args.dream_temperature,
                       dream_token_weighting=args.dream_token_weighting, replay_rows=args.replay_rows, heldout_rows=args.heldout_rows,
                       tolerance_nll=0.05, device=args.device, replay_revision=(args.replay_revision or None),
                       recall_max_new_tokens=args.max_new_tokens, seed=args.seed,
                       provenance="all" if ungated else "accepted", flagged_policy="include" if ungated else "exclude")


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
    ap.add_argument("--batch-size", type=int, default=2, help="rows per sleep step; choose it so the replay ratio is realizable (0.5 -> 2, 0.8 -> 5)")
    ap.add_argument("--session-loss", default="all", choices=["all", "assistant"])
    ap.add_argument("--prompt-loss-weight", type=float, default=1.0, help="weight of user tokens in session turns (0-1)")
    ap.add_argument("--augment", default="none", choices=["none", "study"], help="teach each fact once (none) or as a templated study set")
    ap.add_argument("--teach-temperature", type=float, default=0.7, help="sampling temperature for the model's replies during teaching")
    ap.add_argument("--dream-temperature", type=float, default=0.7)
    ap.add_argument("--dream-token-weighting", default="uniform", choices=["uniform", "gain", "fw_gain"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-revision", default=SMOLTALK_REVISION, help="HuggingFaceTB/smoltalk revision for replay and held-out rows (empty follows main)")
    ap.add_argument("--facts", type=int, default=24, help="how many of the taught facts to use (first N)")
    ap.add_argument("--poison", action="store_true", help="also teach the planted world-knowledge contradictions (uptake is measured)")
    args = ap.parse_args()
    global FACTS_TAUGHT
    FACTS_TAUGHT = FACTS_TAUGHT[:max(1, args.facts)]

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
    taught_facts = FACTS_TAUGHT + FACTS_BOUNDARY + (FACTS_POISON if args.poison else [])
    teach = Session.create(store, model_id="parent", session_id="teach", device=args.device, harness_cfg=observe)
    if args.augment == "study":
        # the study set is taught as user turns; the model still generates its own reply (an accepted turn is what
        # the user said plus what the model answered), so the teacher-side answers are folded into the user turn
        items = [(f"{u} (Correct answer: {a})", ) for fact in taught_facts for u, a in study_set(*fact)]
        turns = [t[0] for t in items]
    else:
        turns = [fact[0] for fact in taught_facts]
    for i, stmt in enumerate(turns):
        r = teach.chat(stmt, max_new_tokens=args.max_new_tokens, temperature=args.teach_temperature, top_k=40, seed=args.seed + i)
        log(f"[teach] {stmt[:50]!r} -> {r.completion[:60]!r} ({len(r.transactions)} chunks)")
    log(f"[teach] {len(turns)} teaching turns (augment={args.augment})")
    # a calibration whose chunk-loss threshold every chunk exceeds: with rollback enabled, every chunk rolls back
    forced = Calibration(model_signature=f"ttt:{digest}", thresholds={"chunk_loss": -1e9}, n_chunks=1)
    forced.save(store.model_dir("parent"))
    guarded = HarnessConfig(log_only=False, enable_rollback=True, enable_stats=True, learn_from_generation=True,
                            enable_projection=False, enable_budget=False, freeze_on_alarm=False)
    rolled = Session.create(store, model_id="parent", session_id="rolled", device=args.device, harness_cfg=guarded)
    assert rolled.calibration_status == "installed", rolled.calibration_status
    n_rb = 0
    for i, (stmt, *_rest) in enumerate(FACTS_ROLLED):
        r = rolled.chat(stmt, max_new_tokens=args.max_new_tokens, temperature=args.teach_temperature, top_k=40, seed=args.seed + 100 + i)
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
                               "code_commit": _git_head(), "started_at_unix": int(t0),
                               "steps": args.steps, "target": args.target, "lr": args.lr, "replay_ratio": args.replay_ratio, "batch_size": args.batch_size, "session_loss": args.session_loss, "prompt_loss_weight": args.prompt_loss_weight, "augment": args.augment, "teach_temperature": args.teach_temperature, "dream_temperature": args.dream_temperature, "replay_revision": (args.replay_revision or None), "facts": len(FACTS_TAUGHT), "poison": bool(args.poison), "arms": {}}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    # 2) floor: the parent from a fresh session (greedy recall and the expected-answer log-probability, like every sleep "before")
    if "floor" in arms:
        from plastic.sleep.ttt import fresh_session_answer_logprob

        be = TTTBackend.load(args.checkpoint, device=args.device)
        rep = run_probes(all_probes, fresh_session_answer(be, max_new_tokens=args.max_new_tokens), fresh_session_answer_logprob(be)).to_dict()
        results["arms"]["floor"] = {"by_group": by_group(rep), "results": rep["results"], "mean_answer_logprob": rep.get("mean_answer_logprob"),
                                    "max_cluster_share": rep.get("max_cluster_share")}
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
                    for i, (stmt, *_r) in enumerate(taught_facts):
                        s.chat(stmt, max_new_tokens=8, temperature=args.teach_temperature, top_k=40, seed=args.seed + i)
                    r = s.chat(q, max_new_tokens=args.max_new_tokens, temperature=1e-3, top_k=1, seed=0)
                    sc = score_reply(p.answer, r.completion)
                    out_rows.append({"question": q, "expected": p.answer, "reply": r.completion, "contains": sc["contains"], "exact": sc["exact"], "variant": variant})
        del s
        results["arms"]["ceiling"] = {"by_group": by_group({"results": out_rows}), "results": out_rows}
        log(f"[ceiling] {json.dumps(results['arms']['ceiling']['by_group'])}")

    # 4) sleep arms, each from the parent
    for arm in ("anchor", "replay", "distill", "dream", "ungated"):
        if arm not in arms:
            continue
        cfg = arm_config(arm, args)
        sessions = ["teach", "rolled"]
        rep = sleep_ttt(store, "parent", cfg, session_ids=sessions, probes=all_probes, run_dir=os.path.join(args.out, f"sleep_{arm}"), log=log)
        entry: dict[str, Any] = {"status": rep["status"], "model_id": rep.get("model_id"), "gate": rep.get("gate"), "batch": rep.get("batch"),
                                 "dreams": rep.get("dreams"), "reason": rep.get("reason"),
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
    lines = ["| Arm | taught (p = unseen phrasing) | boundary | rolled (contamination) | poison (uptake) | general (locality) | held-out NLL mean → | status |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, e in results["arms"].items():
        bg = e.get("by_group") or {}
        cell = lambda g: f"{bg[g]['recalled']}/{bg[g]['n']} (p {bg[g]['recalled_paraphrase']}/{bg[g]['n_paraphrase']})" if g in bg else "n/a"
        nll = ""
        if e.get("heldout_nll_before") and e.get("heldout_nll_after"):
            nll = f"{e['heldout_nll_before']['mean']:.3f} → {e['heldout_nll_after']['mean']:.3f}"
        lines.append(f"| {arm} | {cell('taught')} | {cell('boundary')} | {cell('rolled')} | {cell('poison')} | {cell('general')} | {nll} | {e.get('status', '')} |")
    table = "\n".join(lines)
    with open(os.path.join(args.out, "sleep_controls.md"), "w", encoding="utf-8") as f:
        f.write(f"# Sleep with matched controls\n\nCode {results['code_commit']}, checkpoint `{digest[:12]}`, device {args.device}, {args.steps} steps, target {args.target}, lr {args.lr}, replay ratio {args.replay_ratio}, session loss {args.session_loss}, plw {args.prompt_loss_weight}, augment {args.augment}, {results['seconds']} s.\n\n{table}\n")
    log("\n" + table)


if __name__ == "__main__":
    main()
