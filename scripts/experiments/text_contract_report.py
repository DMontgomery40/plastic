"""First report of the text rule contract on a TTT chat checkpoint: one JSON per mode, a README table, a manifest.

  PYTHONPATH=<abs deps> python -m scripts.experiments.text_contract_report --checkpoint <dir> --out <dir> \
      [--device mps] [--rule-set decorate] [--modes frozen,continued,in_context,replay_verify] [--seed 0]

Every mode runs the same contract on the same fixed-seed material; the learner is reloaded from the checkpoint
between modes so no lasting update leaks. The lasting update visits the stream in full passes by default; the archived
reports up to a8afb4b drew 20 episodes with replacement and 6 verification episodes, reproduced with
``--sampling draws --steps 20 --verify-episodes 6 --no-sequential-clean --choice-episodes 0``. Held-out chat NLL (SmolTalk test rows, pinned revision) is measured
inside each contract measurement when --chat-rows > 0 and is also the replay_verify mode's third check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from plastic.eval.text_contract import TextContractSpec, contract_split, run_text_contract
from plastic.eval.text_learner import MODES, TextRuleLearner


def _git_head() -> str:
    """The execution commit. A run launched from an exported snapshot of a commit (``git archive``, no repository)
    names it through PLASTIC_CODE_COMMIT, so a long queue is not affected by later edits in the checkout."""
    if os.environ.get("PLASTIC_CODE_COMMIT"):
        return os.environ["PLASTIC_CODE_COMMIT"] + " (exported snapshot)"
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if _dirty_paths() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _dirty_paths() -> list[str]:
    """Tracked files that differ from HEAD at launch, so '+dirty' says which files (a teammate's instruction edit is
    not an experiment-code change). An exported snapshot has none by construction."""
    if os.environ.get("PLASTIC_CODE_COMMIT"):
        return []
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout
        return sorted(line[3:] for line in out.splitlines() if line.strip())
    except Exception:  # noqa: BLE001
        return []


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--rule-set", default="decorate", choices=["transform", "decorate"])
    ap.add_argument("--modes", default="frozen,continued,in_context,replay_verify", help=f"comma-separated, from {', '.join(MODES)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--min-reversed-heldout", type=int, default=6, help="the contract default; every archived decorate report measures on the seed-0 split with 6")
    ap.add_argument("--situations", type=int, default=8)
    ap.add_argument("--stream-episodes", type=int, default=17, help="one lesson per training composition by default (5 singles + 12 pairs)")
    ap.add_argument("--eval-episodes-per-composition", type=int, default=2)
    ap.add_argument("--poison-operator", default="#P")
    ap.add_argument("--target", default="w0", choices=["w0", "all"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=20, help="lasting-update steps under --sampling draws (ignored under passes)")
    ap.add_argument("--sampling", default="passes", choices=["passes", "draws"],
                    help="passes: every stream episode once per pass, shuffled; draws: --steps draws with replacement (the archived runs)")
    ap.add_argument("--passes", type=int, default=1, help="full passes over the stream under --sampling passes")
    ap.add_argument("--no-sequential-clean", action="store_true", help="skip the clean-again control for the sequential poison arm")
    ap.add_argument("--name-permuted-control", action="store_true", help="also consume the content null: each name's answers computed by another composition of its arity")
    ap.add_argument("--choice-episodes", type=int, default=2, help="first-situation choice items per composition (0 disables)")
    ap.add_argument("--chat-rows", type=int, default=8, help="held-out SmolTalk rows for the chat-NLL forgetting measure (0 disables)")
    ap.add_argument("--verifier", default="v2", choices=["v1", "v2"], help="v1: held-in checks scored with the fast path frozen; v2: adapting, plus the first-situation exact")
    ap.add_argument("--no-sequential-poison", action="store_true", help="skip the poison-on-top-of-accepted-clean arm")
    ap.add_argument("--verify-episodes", type=int, default=34, help="held-in verification episodes (3 situations each, cycling over the training compositions) for replay_verify; the first-situation check is over this many items")
    ap.add_argument("--unstated-rules", action="store_true", help="no definitions in any preface; the rules come from the worked examples only")
    ap.add_argument("--poison-kind", default="consistent", choices=["consistent", "inconsistent"],
                    help="consistent: the false definition is stated with the false answers; inconsistent: true definitions stated, false answers")
    return ap


def spec_from_args(args: argparse.Namespace) -> TextContractSpec:
    """The one place the command line becomes a contract spec; the learner's held-in material and the logged split
    both come from ``contract_split`` of this spec (FABLE-41B-211)."""
    return TextContractSpec(n_heldout=args.n_heldout, split_seed=args.seed, situations_per_episode=args.situations,
                            eval_episodes_per_composition=args.eval_episodes_per_composition, stream_episodes=args.stream_episodes,
                            probe_situations=args.situations, poison_operator=args.poison_operator, rule_set=args.rule_set,
                            sequential_poison=not args.no_sequential_poison, stated_rules=not args.unstated_rules, poison_kind=args.poison_kind,
                            min_reversed_heldout=args.min_reversed_heldout, sequential_clean=not args.no_sequential_clean,
                            choice_episodes_per_composition=args.choice_episodes, name_permuted_control=args.name_permuted_control)


def _fmt(x: float | None, spec: str = "+.2f") -> str:
    return "n/a" if x is None else format(x, spec)


def detail_row(mode: str, rep: dict) -> str:
    """Coverage of the clean lasting update, the first-situation choice, and the sequential poison beside its clean-again control."""
    rc = (rep.get("consume_records") or {}).get("clean") or {}
    cov = "n/a" if "compositions_trained" not in rc else f"{rc['compositions_trained']}/{rc['compositions_in_stream']} ({rc['steps']}, {rc['poisoned_steps']})"
    ch = rep.get("choice") or {}

    def acc(key: str, part: str) -> str:
        v = (ch.get(key) or {}).get(part)
        return "n/a" if not v else f"{v['accuracy']:.2f}"

    def arm(block: dict | None) -> str:
        if not block:
            return "n/a"
        changes = block.get("verify_first_changes")
        moved = "n/a" if changes is None else (", ".join(f"{c['ops']} {int(c['before'])}→{int(c['after'])}" for c in changes) or "none")
        return f"{block.get('accepted')}, {moved}"

    sq, sc = rep.get("sequential_poison"), rep.get("sequential_control")
    vs = "n/a"
    if sq and sc and sq.get("choice_heldout_delta") is not None and sc.get("choice_heldout_delta") is not None:
        vs = f"{sq['choice_heldout_delta'] - sc['choice_heldout_delta']:+.2f} / {sq['choice_train_delta'] - sc['choice_train_delta']:+.2f}"
    nov = ch.get("heldout_novel_accuracy") or {}
    nov_cell = "" if not nov else f" ({_fmt(nov.get('before'), '.2f')} → {_fmt(nov.get('after'), '.2f')})"
    tb = rep.get("template_baseline") or {}
    tpl = "n/a" if not tb else f"{tb['second_situation']:.2f} / {tb['exact_after_first']:.2f}"
    return (f"| {mode} | {tpl} | {cov} | {acc('before', 'heldout')} → {acc('after', 'heldout')}{nov_cell} | {acc('before', 'train')} → {acc('after', 'train')} "
            f"| {arm(sc)} | {arm(sq)} | {vs} |")


def paired_rows(mode: str, rep: dict) -> list[str]:
    paired = (rep.get("choice") or {}).get("paired") or {}
    labels = {"after_vs_before": "clean update vs before", "clean_again_vs_after": "clean again vs after clean",
              "poison_sequential_vs_after": "poison on top vs after clean", "poison_vs_before": "poison from snapshot vs before",
              "format_vs_before": "format-only vs before", "names_vs_before": "name-permuted (content null) vs before"}
    rows = []
    for key, label in labels.items():
        block = paired.get(key)
        if not block:
            continue
        def cell(g: str) -> str:
            v = block.get(g)
            return "n/a" if not v else f"{v['mean_change']:+.2f} ({v['improved']}/{v['worsened']} of {v['n']})"
        rows.append(f"| {mode} | {label} | {cell('train')} | {cell('heldout_novel')} | {cell('heldout_duplicate')} |")
    return rows


def main() -> None:
    args = build_parser().parse_args()

    from plastic.backends.ttt_lm.backend import TTTBackend
    from plastic.sleep.ttt import heldout_nll, load_replay_conversations

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.txt")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    t0 = time.time()
    spec = spec_from_args(args)
    train, heldout = contract_split(spec)  # the same split the contract measures on; the learner's held-in material comes from it
    log(f"[setup] rule set {spec.rule_set}: {len(train)} training compositions, {len(heldout)} held-out pairs {[' '.join(c) for c in heldout]}")
    chat_rows = load_replay_conversations("everyday-conversations", "test", args.chat_rows, args.seed + 1, log) if args.chat_rows > 0 else []
    manifest = {"checkpoint": os.path.abspath(args.checkpoint), "device": args.device, "code_commit": _git_head(), "dirty_paths": _dirty_paths(), "started_at_unix": int(t0),
                "rule_set": spec.rule_set, "spec": spec.__dict__ | {"n_words": list(spec.n_words)}, "modes": {}, "chat_rows": len(chat_rows)}
    rows3 = ["| Mode | Comparison | Training compositions | Held-out, novel | Held-out, repeats a trained answer |", "| --- | --- | --- | --- | --- |"]
    rows2 = ["| Mode | Untrained template copier, held-out (second situation / all later) | Clean stream: compositions trained (steps, poisoned steps) | First-situation choice, held-out: before → after (novel pairs only) | First-situation choice, training: before → after | Clean again: accepted, verify first-situation items changed | Poison on top: accepted, verify first-situation items changed | Poison on top vs clean again: choice held-out / training Δ |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = ["| Mode | Held-out exact (adapt) before → after | of which first situation Δ / later Δ | Held-out nll (adapt) before → after | Held-out nll (no adapt) Δ | Speed area before → after | Forgetting nll Δ | Poison harm (nll) | Sequential poison: accepted, harm (nll) | Correction residual | Format-only gain exact (true) | Revert | Accepted good / refused bad | Verifier |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        log(f"[mode] {mode}")
        be = TTTBackend.load(args.checkpoint, device=args.device)
        manifest["checkpoint_digest"] = be.checkpoint_digest
        chat = (lambda: heldout_nll(be, chat_rows, 512)) if chat_rows else None
        learner = TextRuleLearner(be, mode=mode, target=args.target, lr=args.lr, steps=args.steps, chat_nll=chat, verify_adapt=args.verifier == "v2",
                                  verify_episodes=args.verify_episodes, sampling=args.sampling, passes=args.passes, log=log)
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
        rows2.append(detail_row(mode, rep))
        log(rows2[-1])
        rows3.extend(paired_rows(mode, rep))
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
              f"rules {'stated' if spec.stated_rules else 'UNSTATED'} in every preface, {args.verify_episodes} verify episodes, "
              f"{len(chat_rows)} held-out chat rows, lasting-update sampling {args.sampling}"
              + (f" ({args.passes} pass{'es' if args.passes != 1 else ''})" if args.sampling == "passes" else f" ({args.steps} draws)")
              + f", {spec.choice_episodes_per_composition} choice items per composition. Exact is teacher-forced (`exact_tf`). Every mode reloads the checkpoint.\n\n"
              + "\n".join(rows) + "\n\nCoverage, first-situation choice (chance is 1/candidates: 23 distinct outputs on the decoration set) and the sequential arms:\n\n" + "\n".join(rows2)
              + "\n\nPaired first-situation margin change per item (mean nats; improved / worsened of n), by group:\n\n" + "\n".join(rows3) + "\n")
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    log(f"[done] {manifest['seconds']} s -> {args.out}")


if __name__ == "__main__":
    main()
