"""The adversarial write-gate experiment: baseline versus hardened, attacked by a fresh adversary.

Trains two models on the same data and seed (one with ``--adversarial``), calibrates both,
runs the red-team campaign against both, and writes a Markdown report with the numbers
the spec asks for: held-out loss, memory value, MQAR accuracy, validated and provisional
attack damage, gated fraction, and the transfer of baseline-optimized payloads.

Run (small config, MPS or CPU):
    uv run python scripts/experiments/adversarial_gate.py --data artifacts/data/wikitext \
        --artifacts-root artifacts --steps 600 --batch-size 8 --seq-len 512 --device mps \
        --d-model 128 --layers 2 --out docs/research/2026-09-22-adversarial-gate-results.md
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from plastic.config import ModelConfig
from plastic.harness.calibrate import calibrate_model
from plastic.redteam.attack import AttackConfig, run_redteam, validate_payload
from plastic.store import ArtifactStore
from plastic.train.loop import TrainConfig, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/data/wikitext")
    p.add_argument("--artifacts-root", default="artifacts")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adv-lambda", type=float, default=1.0)
    p.add_argument("--prefixes", type=int, default=6)
    p.add_argument("--attack-steps", type=int, default=30)
    p.add_argument("--tag", default=str(int(time.time())))
    p.add_argument("--out", default="docs/research/2026-09-22-adversarial-gate-results.md")
    args = p.parse_args()

    store = ArtifactStore(args.artifacts_root)
    from plastic.data.text import load_corpus_meta

    vocab = load_corpus_meta(args.data).vocab_size
    results: dict[str, dict] = {}
    model_ids: dict[str, str] = {}
    for name, adversarial in (("baseline", False), ("hardened", True)):
        mid = f"advgate_{name}_{args.tag}"
        model_ids[name] = mid
        cfg = TrainConfig(
            domain="text",
            model=ModelConfig(d_model=args.d_model, n_layers=args.layers, n_heads=max(1, args.d_model // 64), chunk=64, vocab_size=vocab),
            artifacts_root=args.artifacts_root, model_id=mid, data_dir=args.data, steps=args.steps,
            batch_size=args.batch_size, seq_len=args.seq_len, warmup_steps=max(10, args.steps // 10),
            eval_every=0, save_every=0, eval_batches=8, log_every=10, seed=args.seed, device=args.device,
            adversarial=adversarial, adv_lambda=args.adv_lambda, adv_every=10, adv_steps=5, adv_suffix_len=32,
        )
        train(cfg)
        calibrate_model(store, mid, data_dir=args.data, n_chunks=128, fisher_chunks=16, device="cpu")
        summary = run_redteam(
            store, mid, cfg=AttackConfig(suffix_len=64, steps=args.attack_steps, seed=args.seed + 1), data_dir=args.data,
            n_prefixes=args.prefixes, device="cpu",
        )
        results[name] = {"eval": store.read_eval(mid), "redteam": summary}

    # transfer: payloads optimized against the baseline, replayed against the hardened model
    base_run = results["baseline"]["redteam"]["run_id"]
    rows = [json.loads(l) for l in open(os.path.join(store.root, "redteam", base_run, "results.jsonl"))]
    pgd = [r for r in rows if r["family"] == "pgd"]
    from plastic.harness.canary import CanarySuite
    from plastic.harness.config import HarnessConfig

    h_cfg, h_model, _ = store.load_checkpoint(model_ids["hardened"], "cpu")
    h_suite = CanarySuite.load(store.canary_path(model_ids["hardened"]))
    transfer = []
    for r in pgd:
        v = validate_payload(h_model, h_cfg, r["prefix_ids"], r["payload_ids"], h_suite, harness=HarnessConfig(), device=torch.device("cpu"))
        transfer.append(v["canary_after_accepted"] - v["canary_before"])
    results["transfer"] = {"n": len(transfer), "damage_mean": float(np.mean(transfer)) if transfer else None, "damage_max": float(np.max(transfer)) if transfer else None}

    lines = [
        "# Adversarial write-gate experiment",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. Config: d_model {args.d_model}, layers {args.layers}, steps {args.steps}, batch {args.batch_size}, seq {args.seq_len}, seed {args.seed}, adv_lambda {args.adv_lambda}, attack steps {args.attack_steps}, prefixes {args.prefixes}.",
        "",
        "| model | held-out loss | memory value | MQAR acc (4/8/16) | β mean |",
        "|---|---:|---:|---|---:|",
    ]
    for name in ("baseline", "hardened"):
        ev = results[name]["eval"] or {}
        m = ev.get("mqar_accuracy", {})
        lines.append(f"| {name} | {ev.get('heldout_loss', float('nan')):.4f} | {ev.get('memory_value', float('nan')):+.4f} | {m.get('4', float('nan')):.2f}/{m.get('8', float('nan')):.2f}/{m.get('16', float('nan')):.2f} | {(ev.get('beta_hist') or {}).get('beta_mean', float('nan')):.3f} |")
    lines += ["", "| model | family | validated damage mean | validated damage max | provisional damage max | gated fraction | over threshold |", "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("baseline", "hardened"):
        fam = results[name]["redteam"]["families"]
        for f, v in fam.items():
            ot = v.get("over_threshold_fraction")
            lines.append(f"| {name} | {f} | {v['damage_mean']:+.4f} | {v['damage_max']:+.4f} | {v['provisional_damage_max']:+.4f} | {v['gated_fraction']:.2f} | {'n/a' if ot is None else f'{ot:.2f}'} |")
    tr = results["transfer"]
    lines += ["", f"Transfer of baseline PGD payloads to the hardened model: n = {tr['n']}, validated damage mean {tr['damage_mean']}, max {tr['damage_max']}.", "",
              f"Models: `{model_ids['baseline']}`, `{model_ids['hardened']}`. Red-team runs: `{results['baseline']['redteam']['run_id']}`, `{results['hardened']['redteam']['run_id']}`."]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
