"""Coordinate ablation: meta-train the coordinate candidate and its falsifier variants on the
mechanism testbed, then score each under the learning contract with no lasting update.

    uv run python -m scripts.experiments.coordinate_ablation --out artifacts/experiments/coordinate-ablation --steps 1500
    uv run python -m scripts.experiments.coordinate_ablation --out artifacts/experiments/coordinate-ablation --collect

This answers the memo's first question (docs/research/2026-09-21-plastic-coordinate-recurrence.md,
"What would falsify the contribution"): do experience-driven changes to the nonlinear transition
improve subsequent performance beyond adaptive decay alone? The score is the contract's
before-stream block: held-out-combination MSE with fast adaptation versus with fast parameters
frozen, under held-out intervention policies, and the adaptation-speed ratio. Nothing here is a
lasting update; that is the slow rule's job (T3).

Training rows hold one 64-step episode by default, so with chunk 16 three fast-update
boundaries fall inside every episode during meta-training as well as during scoring; a
packed row would let fast weights fitted to one world leak into the next.

Variants (each trained from scratch with the same data seed):
  full            the candidate: fast coordinates W and fast decay theta, meta-gradient through the step
  no_fast         fast updates off (the same block as a plain selective recurrence)
  decay_only      frozen coordinates, adaptive decay (the memo's "does timescale adaptation explain it")
  coords_only     adaptive coordinates, frozen decay
  no_meta         no meta-gradient through the inner step (is meta-training necessary)
  fixed_z         the incorrect fixed-latent commit (does state transport matter)
  delta_baseline  the existing PlasticDynamics block (RG-LRU + delta-rule memory) at the same width/depth
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from typing import Any

import torch

from plastic.config import ModelConfig
from plastic.data.mechanisms import mechanism_batch, split_combinations
from plastic.eval.contract import ContractSpec, DynamicsLearner, run_contract
from plastic.eval.coordinate_learner import CoordinateLearner
from plastic.model.coordinate import CoordinateConfig, CoordinateDynamics
from plastic.model.lm import PlasticDynamics

VARIANTS: dict[str, dict[str, Any] | None] = {
    "full": {},
    "no_fast": {"fast_updates": False},
    "decay_only": {"adapt_W": False},
    "coords_only": {"adapt_theta": False},
    "no_meta": {"meta_gradient": "none"},
    "fixed_z": {"commit_rule": "fixed_z"},
    "delta_baseline": None,
}


def execution_commit(root: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def build(variant: str, *, d_model: int, n_heads: int, n_layers: int, chunk: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    overrides = VARIANTS[variant]
    if overrides is None:
        return PlasticDynamics(ModelConfig(domain="physics", d_model=d_model, n_heads=n_heads, n_layers=n_layers, chunk=chunk))
    return CoordinateDynamics(CoordinateConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, chunk=chunk, **overrides))


def learner_for(variant: str, model: torch.nn.Module, device: torch.device):
    if VARIANTS[variant] is None:
        return DynamicsLearner(model, mode="frozen", device=device), "writes disabled (beta_scale=0)"
    return CoordinateLearner(model, mode="frozen", device=device), "fast parameters frozen (freeze=True)"


def _etas(model: torch.nn.Module) -> list[float] | None:
    blocks = getattr(getattr(model, "core", None), "blocks", None)
    if blocks is None:
        return None
    out = []
    for b in blocks:
        logit = getattr(b, "eta_logit", None)
        if logit is None:
            return None
        out.append(float(model.cfg.eta_max * torch.sigmoid(logit.detach()).mean()))
    return out


def train(
    model: torch.nn.Module,
    *,
    steps: int,
    batch: int,
    seq_len: int,
    episodes: int,
    combos: list[tuple[str, ...]],
    lr: float,
    seed: int,
    log_every: int,
    device: torch.device,
    clip: float = 1.0,
) -> list[dict[str, Any]]:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    log: list[dict[str, Any]] = []
    t0 = time.time()
    for step in range(1, steps + 1):
        b = mechanism_batch(batch, seq_len=seq_len, episodes_per_seq=episodes, combos=combos, policy="gaussian", rng=g)
        x, y = b.inputs.to(device), b.target_delta.to(device)
        opt.zero_grad(set_to_none=True)
        loss = model.loss(x, y)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        opt.step()
        if step % log_every == 0 or step == steps or step == 1:
            log.append({"step": step, "loss": float(loss.detach()), "grad_norm": gn, "eta": _etas(model), "wall_s": time.time() - t0})
    model.eval()
    return log


def run_variant(variant: str, args: argparse.Namespace, out: str, *, spec: ContractSpec) -> dict[str, Any]:
    device = torch.device(args.device)
    train_combos, _ = split_combinations(k=spec.k, n_heldout=spec.n_heldout, seed=spec.split_seed)
    model = build(variant, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, chunk=args.chunk, seed=args.seed).to(device)
    params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    log = train(
        model, steps=args.steps, batch=args.batch, seq_len=args.seq_len, episodes=args.episodes, combos=train_combos,
        lr=args.lr, seed=args.seed + 1, log_every=args.log_every, device=device,
    )
    train_s = time.time() - t0
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, f"{variant}.pt"))
    learner, no_adapt_label = learner_for(variant, model, device)
    report = run_contract(learner, spec, seed=args.seed)
    # what the fast path proposed on held-out worlds: one adapting pass on the speed batch
    fast_signals = None
    if hasattr(learner, "fast_signals_summary"):
        _, heldout = split_combinations(k=spec.k, n_heldout=spec.n_heldout, seed=spec.split_seed)
        probe = mechanism_batch(spec.eval_batch, seq_len=spec.seq_len, episodes_per_seq=1, combos=heldout, policy="gaussian", rng=torch.Generator().manual_seed(args.seed * 10_000 + 200))
        learner.step_mse(probe, adapt=True)
        fast_signals = learner.fast_signals_summary()
    result = {
        "variant": variant,
        "config": VARIANTS[variant] if VARIANTS[variant] is not None else "PlasticDynamics",
        "size": {"d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers, "chunk": args.chunk, "parameters": params},
        "train": {"steps": args.steps, "batch": args.batch, "seq_len": args.seq_len, "episodes": args.episodes, "lr": args.lr, "seed": args.seed, "wall_s": train_s, "s_per_step": train_s / max(1, args.steps), "log": log},
        "no_adapt_label": no_adapt_label,
        "fast_signals": fast_signals,
        "checkpoint": f"{variant}.pt",
        "contract": report,
    }
    with open(os.path.join(out, f"{variant}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    return result


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def collect(out: str) -> str:
    results: dict[str, dict[str, Any]] = {}
    for name in VARIANTS:
        p = os.path.join(out, f"{name}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                results[name] = json.load(f)
    if not results:
        raise SystemExit(f"no variant results in {out}")
    any_r = next(iter(results.values()))
    policies = list(any_r["contract"]["transfer"].keys())
    head = ["variant", "params", "train loss (last)", "s/step"] + [f"{p}: adapt / no-adapt" for p in policies] + ["train dist: adapt / no-adapt", "speed mean", "half at step", "η per layer", "inner loss before → after", "‖ΔW‖, ‖Δθ‖ per layer"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in results.items():
        c = r["contract"]
        last = r["train"]["log"][-1]
        row = [name, str(r["size"]["parameters"]), _fmt(last["loss"]), f"{r['train']['s_per_step']:.2f}"]
        for p in policies:
            b = c["transfer"][p]["before"]
            row.append(f"{_fmt(b['adapt'])} / {_fmt(b['no_adapt'])}")
        fb = c["forgetting"]["before"]
        row.append(f"{_fmt(fb['adapt'])} / {_fmt(fb['no_adapt'])}")
        s = c["speed"]["before"]
        row += [f"{s['area']:.3f}", str(s["steps_to_half"]), ", ".join(f"{e:.3f}" for e in last["eta"]) if last.get("eta") else "n/a"]
        fs = r.get("fast_signals")
        if fs and fs.get("inner_loss_before") is not None:
            row.append(f"{fs['inner_loss_before']:.4f} → {fs['inner_loss_after']:.4f}")
            row.append(", ".join(f"{w:.3g}/{t:.3g}" for w, t in zip(fs["dW_norm_by_layer"], fs["dtheta_norm_by_layer"])))
        else:
            row += ["n/a", "n/a"]
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    labels = {name: r["no_adapt_label"] for name, r in results.items()}
    body = [
        "# Coordinate ablation on the mechanism testbed",
        "",
        "Each variant is trained from scratch on the training combinations (gaussian policy) with the same data seed, "
        "then scored under the learning contract with no lasting update. `adapt` is the held-out-combination MSE with "
        "fast updates on; `no-adapt` is the same inputs with the fast path off. The off-intervention differs by model and "
        "is labelled: " + "; ".join(f"{k}: {v}" for k, v in labels.items()) + ".",
        "",
        "`speed mean` is the adapting error as a fraction of the no-adapt error over the first probe steps (lower is faster); "
        "`half at step` is the first step at which it drops below one half. `η per layer` is the learned inner step size. "
        "`inner loss before → after` re-scores the observed chunk under the proposed step (a support diagnostic, not an "
        "adaptation score); `‖ΔW‖, ‖Δθ‖` are the proposed changes per layer, averaged over held-out boundaries.",
        "",
        table,
        "",
        "Reading guide (from the memo's falsification table): if `decay_only` matches `full`, timescale adaptation "
        "explains the gain and the nonlinear coordinates are not earning their place. If `no_meta` matches `full`, "
        "meta-training is not necessary. If `delta_baseline` matches `full` at matched compute, coupling learning to "
        "the dynamics is not useful. Parameters, seconds per step and state memory are reported because equal parameter "
        "counts alone are insufficient. One seed unless stated; no lasting-learning claim is made here.",
        "",
    ]
    readme = "\n".join(body)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    return readme


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--collect", action="store_true", help="only write README.md from existing variant results")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=1, help="episodes per training row; 1 keeps every fast-update boundary inside an episode")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-batch", type=int, default=8)
    args = ap.parse_args(argv)
    if args.collect:
        print(collect(args.out))
        return 0
    spec = ContractSpec(eval_batch=args.eval_batch)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"execution_commit": execution_commit(root), "args": vars(args), "spec": asdict(spec), "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}, f, indent=1)
    for v in args.variants:
        t0 = time.time()
        r = run_variant(v, args, args.out, spec=spec)
        c = r["contract"]["transfer"]
        print(f"[ablation] {v}: params={r['size']['parameters']} train_loss={r['train']['log'][-1]['loss']:.4f} "
              + " ".join(f"{p}={c[p]['before']['adapt']:.4f}/{c[p]['before']['no_adapt']:.4f}" for p in c)
              + f" ({time.time() - t0:.0f}s)", flush=True)
    print(collect(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
