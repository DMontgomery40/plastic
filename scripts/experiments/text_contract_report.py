"""First report of the text rule contract on a TTT chat checkpoint: one JSON per mode, a README table, a manifest.

  PYTHONPATH=<abs deps> python -m scripts.experiments.text_contract_report --checkpoint <dir> --out <dir> \
      [--device mps] [--rule-set decorate] [--modes frozen,continued,in_context,replay_verify] [--seed 0]

Every mode runs the same contract on the same fixed-seed material; the learner is reloaded from the checkpoint
between modes so no lasting update leaks. Held-out chat NLL (SmolTalk test rows, pinned revision) is measured
inside each contract measurement when --chat-rows > 0 and is also the replay_verify mode's third check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from plastic.data.rules import split_pairs
from plastic.eval.text_contract import TextContractSpec, run_text_contract
from plastic.eval.text_learner import MODES, TextRuleLearner


def _git_head() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--rule-set", default="decorate", choices=["transform", "decorate"])
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--min-reversed-heldout", type=int, default=4)
    ap.add_argument("--situations", type=int, default=8)
    ap.add_argument("--stream-episodes", type=int, default=17, help="one lesson per training composition by default (5 singles + 12 pairs)")
    ap.add_argument("--eval-episodes-per-composition", type=int, default=2)
    ap.add_argument("--poison-operator", default="#P")
    ap.add_argument("--target", default="w0", choices=["w0", "all"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--chat-rows", type=int, default=8, help="held-out SmolTalk rows for the chat-NLL forgetting measure (0 disables)")
    ap.add_argument("--verifier", default="v2", choices=["v1", "v2"], help="v1: held-in checks scored with the fast path frozen; v2: adapting, plus the first-situation exact")
    ap.add_argument("--no-sequential-poison", action="store_true", help="skip the poison-on-top-of-accepted-clean arm")
    ap.add_argument("--unstated-rules", action="store_true", help="no definitions in any preface; the rules come from the worked examples only")
    ap.add_argument("--poison-kind", default="consistent", choices=["consistent", "inconsistent"],
                    help="consistent: the false definition is stated with the false answers; inconsistent: true definitions stated, false answers")
    args = ap.parse_args()

    from plastic.backends.ttt_lm.backend import TTTBackend
    from plastic.sleep.ttt import heldout_nll, load_replay_conversations

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.txt")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    t0 = time.time()
    spec = TextContractSpec(n_heldout=args.n_heldout, split_seed=args.seed, situations_per_episode=args.situations,
                            eval_episodes_per_composition=args.eval_episodes_per_composition, stream_episodes=args.stream_episodes,
                            probe_situations=args.situations, poison_operator=args.poison_operator, rule_set=args.rule_set,
                            sequential_poison=not args.no_sequential_poison, stated_rules=not args.unstated_rules, poison_kind=args.poison_kind)
    train, heldout = split_pairs(n_heldout=spec.n_heldout, seed=spec.split_seed, rule_set=spec.rule_set, min_reversed_heldout=args.min_reversed_heldout)
    log(f"[setup] rule set {spec.rule_set}: {len(train)} training compositions, {len(heldout)} held-out pairs {[' '.join(c) for c in heldout]}")
    chat_rows = load_replay_conversations("everyday-conversations", "test", args.chat_rows, args.seed + 1, log) if args.chat_rows > 0 else []
    manifest = {"checkpoint": os.path.abspath(args.checkpoint), "device": args.device, "code_commit": _git_head(), "started_at_unix": int(t0),
                "rule_set": spec.rule_set, "spec": spec.__dict__ | {"n_words": list(spec.n_words)}, "modes": {}, "chat_rows": len(chat_rows)}
    rows = ["| Mode | Held-out exact (adapt) before → after | of which first situation Δ / later Δ | Held-out nll (adapt) before → after | Held-out nll (no adapt) Δ | Speed area before → after | Forgetting nll Δ | Poison harm (nll) | Sequential poison: accepted, harm (nll) | Correction residual | Format-only gain exact (true) | Revert | Accepted good / refused bad | Verifier |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        log(f"[mode] {mode}")
        be = TTTBackend.load(args.checkpoint, device=args.device)
        manifest["checkpoint_digest"] = be.checkpoint_digest
        chat = (lambda: heldout_nll(be, chat_rows, 512)) if chat_rows else None
        learner = TextRuleLearner(be, mode=mode, target=args.target, lr=args.lr, steps=args.steps, chat_nll=chat, verify_adapt=args.verifier == "v2", log=log)
        learner.train_compositions = train
        learner.rule_set = spec.rule_set
        learner.stated_rules = spec.stated_rules
        rep = run_text_contract(learner, spec, seed=args.seed, chat_nll=chat)
        rep["mode"] = mode
        with open(os.path.join(args.out, f"contract_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
        tr, sp, fg, co, rv, acc, fo = rep["transfer"], rep["speed"], rep["forgetting"], rep["correction"], rep["revert"], rep["acceptance"], rep["format_only"]
        sq = rep.get("sequential_poison")
        seq_cell = "n/a" if sq is None else f"{sq['accepted']}, {sq['harm_nll']:+.3f}"
        rows.append(f"| {mode} | {tr['before']['adapt']['exact']:.2f} → {tr['after']['adapt']['exact']:.2f} | {tr.get('delta_exact_first', float('nan')):+.2f} / {tr.get('delta_exact_after_first', float('nan')):+.2f} | {tr['before']['adapt']['nll']:.3f} → {tr['after']['adapt']['nll']:.3f} "
                    f"| {tr['delta_nll_no_adapt']:+.3f} | {sp['before']['area']:.2f} → {sp['after']['area']:.2f} | {fg['delta_nll']:+.3f} | {co['harm_nll']:+.3f} | {seq_cell} | {co['residual_nll']:+.3f} "
                    f"| {fo['gain_exact']:+.2f} ({fo['true_stream_gain_exact']:+.2f}) "
                    f"| {'ok' if rv['ok'] else 'GAP ' + format(rv['gap'], '.2e')} | {acc['accepted_good']} / {acc['refused_bad']} (n {acc['n_good']}/{acc['n_bad']}) | {rep.get('verifier') or 'n/a'} |")
        manifest["modes"][mode] = {"seconds": rep["compute"]["wall_clock_s"], "decisions": rep["decisions"]}
        log(rows[-1])
        del learner, be
        if args.device == "mps":
            import torch
            torch.mps.empty_cache()
    manifest["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    readme = (f"# Text rule contract report\n\nCode {manifest['code_commit']}, checkpoint digest `{manifest.get('checkpoint_digest', '?')[:12]}`, device {args.device}, "
              f"rule set `{spec.rule_set}`, seed {args.seed}, {len(train)} training compositions, {len(heldout)} held-out pairs, {spec.situations_per_episode} situations per episode, "
              f"stream {spec.stream_episodes} episodes, lasting update target {args.target}, lr {args.lr}, {args.steps} steps, poison operator `{spec.poison_operator}` ({spec.poison_kind}), "
              f"rules {'stated' if spec.stated_rules else 'UNSTATED'} in every preface, "
              f"{len(chat_rows)} held-out chat rows. Exact is teacher-forced (`exact_tf`). Every mode reloads the checkpoint.\n\n" + "\n".join(rows) + "\n")
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    log(f"[done] {manifest['seconds']} s -> {args.out}")


if __name__ == "__main__":
    main()
