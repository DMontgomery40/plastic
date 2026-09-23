"""Run the learning contract on a saved PlasticDynamics checkpoint, one report per baseline mode.

    uv run python -m scripts.experiments.transfer_contract --model-id phys_mps_3k --out artifacts/experiments/contract-phys_mps_3k

Modes: ``frozen`` (no lasting update; fast weights only), ``continued`` (plain gradient steps on
the stream), ``in_context`` (the stream prepended at measurement time). The output directory gets
one JSON per mode, a ``README.md`` summary table, and ``manifest.json`` with the checkpoint digest
and the execution commit. Spec: docs/superpowers/specs/2026-09-23-mechanism-testbed-and-contract.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from typing import Any, Callable

import torch

from plastic.eval.contract import ContractSpec, DynamicsLearner, RetrievalLearner, run_contract
from plastic.store import ArtifactStore

MODES = ("frozen", "continued", "in_context", "retrieval")


def execution_commit(root: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def file_digest(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_modes(
    model_factory: Callable[[], torch.nn.Module],
    *,
    modes: tuple[str, ...],
    spec: ContractSpec,
    seed: int,
    lr: float,
    steps: int,
    device: torch.device,
    knn: int = 8,
) -> dict[str, dict[str, Any]]:
    """A fresh model per mode, so no mode's lasting update leaks into another. The retrieval
    mode uses no model at all."""
    reports: dict[str, dict[str, Any]] = {}
    for mode in modes:
        if mode == "retrieval":
            learner: Any = RetrievalLearner(k=knn)
            reports[mode] = run_contract(learner, spec, seed=seed)
            reports[mode]["learner"] = {"k": knn, "stored_transitions": learner.stored_transitions()}
        else:
            model = model_factory().to(device)
            learner = DynamicsLearner(model, mode=mode, lr=lr, steps=steps, device=device)
            reports[mode] = run_contract(learner, spec, seed=seed)
            reports[mode]["learner"] = {"lr": lr, "steps": steps}
        reports[mode]["mode"] = mode
    return reports


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:+.4f}" if abs(x) < 100 else f"{x:.3g}"
    return str(x)


def summary_table(reports: dict[str, dict[str, Any]]) -> str:
    """Markdown: one row per mode. MSE deltas are after minus before; negative is improvement."""
    policies = list(next(iter(reports.values()))["transfer"].keys())
    head = ["mode"] + [f"transfer Δ ({p})" for p in policies] + ["forgetting Δ", "poison harm (vs clean)", "poison harm (vs start)", "corr. residual", "revert ok", "accepted-good", "refused-bad", "tokens consumed", "tokens measured"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for mode, r in reports.items():
        acc = r["acceptance"]
        row = [mode]
        row += [_fmt(r["transfer"][p]["delta_mse"]) for p in policies]
        row += [
            _fmt(r["forgetting"]["delta_mse"]),
            _fmt(r["correction"]["harm"]),
            _fmt(r["correction"]["harm_vs_before"]),
            _fmt(r["correction"]["residual"]),
            _fmt(r["revert"]["ok"]),
            _fmt(acc["accepted_good"]) + f" (n={acc['n_good']})",
            _fmt(acc["refused_bad"]) + f" (n={acc['n_bad']})",
            str(r["compute"]["tokens_consumed"]),
            str(r["compute"]["tokens_measured"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def before_table(reports: dict[str, dict[str, Any]]) -> str:
    """The before-stream measurements, which are the same for every mode of one checkpoint."""
    r = next(iter(reports.values()))
    lines = ["| measurement | with adaptation | writes disabled | elements |", "|---|---|---|---|"]
    for p, row in r["transfer"].items():
        lines.append(f"| held-out combos, {p} | {row['before']['adapt']:.4f} | {row['before']['no_adapt']:.4f} | {row['elements']} |")
    f = r["forgetting"]["before"]
    lines.append(f"| training distribution | {f['adapt']:.4f} | {f['no_adapt']:.4f} | {r['forgetting']['elements']} |")
    s = r["speed"]["before"]
    lines.append(f"| adaptation speed (held-out; adapting error as a fraction of writes-disabled error, by step) | mean {s['area']:.3f}; below one half at step {s['steps_to_half']} | | {s['episodes']} episodes |")
    return "\n".join(lines)


def write_outputs(out: str, reports: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> None:
    os.makedirs(out, exist_ok=True)
    for mode, r in reports.items():
        with open(os.path.join(out, f"{mode}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    body = [
        f"# Learning contract on `{manifest['model_id']}`",
        "",
        f"Checkpoint digest `{manifest['checkpoint_digest']}`, execution commit `{manifest['execution_commit']}`, "
        f"device {manifest['device']}, contract {manifest['contract_version']}, seed {manifest['seed']}.",
        "",
        "## Before any stream (identical for every mode)",
        "",
        before_table(reports),
        "",
        "## After the stream, per mode",
        "",
        "MSE deltas are after minus before on identical inputs; negative is improvement. "
        "`poison harm (vs clean)` is transfer MSE after the poisoned stream minus after the clean stream "
        "(damage plus the forgone clean gain); `poison harm (vs start)` is minus the poison arm's own "
        "pre-stream start (damage alone); `corr. residual` is after the corrective stream minus after clean. "
        "Continued training uses Adam. Everything-in-context prepends the stream to the model's own recurrent carry, "
        "which is bounded by its forget gate and decay horizon, unlike a transformer's context window; the retrieval "
        "mode is the model-free lookup baseline (mean delta of the nearest stored transitions). Acceptance is a pair of rates; "
        "n/a means no decision was recorded, not zero.",
        "",
        summary_table(reports),
        "",
        "Every measurement is taken from a fresh state with the stream removed, one episode per row. "
        f"Adaptation window: {manifest.get('adaptation_window')}. "
        "Spec: docs/superpowers/specs/2026-09-23-mechanism-testbed-and-contract.md",
        "",
    ]
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(body))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts-root", default="artifacts")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--knn", type=int, default=8, help="neighbours for the retrieval baseline")
    ap.add_argument("--seq-len", type=int, default=64, help="length of the one episode each scored row holds")
    ap.add_argument("--probe-steps", type=int, default=None, help="steps of the adaptation curve to report (default: the whole episode)")
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--stream-episodes", type=int, default=16)
    ap.add_argument("--n-heldout", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--poison-bias", type=float, default=0.5)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    store = ArtifactStore(args.artifacts_root)
    cfg, _, record = store.load_checkpoint(args.model_id, device=device)
    if cfg.domain != "physics":
        raise SystemExit("the contract runner needs a physics-domain checkpoint")
    spec = ContractSpec(
        seq_len=args.seq_len, probe_steps=args.probe_steps, eval_batch=args.eval_batch,
        stream_episodes=args.stream_episodes, n_heldout=args.n_heldout, split_seed=args.split_seed,
        poison_bias=args.poison_bias,
    )

    def factory() -> torch.nn.Module:
        _, model, _ = store.load_checkpoint(args.model_id, device=device)
        return model

    t0 = time.time()
    torch.manual_seed(args.seed)
    reports = run_modes(factory, modes=tuple(args.modes), spec=spec, seed=args.seed, lr=args.lr, steps=args.steps, device=device, knn=args.knn)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    manifest = {
        "model_id": args.model_id,
        "checkpoint_digest": file_digest(os.path.join(store.model_dir(args.model_id), "checkpoint.pt")),
        "model_record": {"step": record.get("step"), "extra": record.get("extra")},
        "model_signature": store.model_signature(args.model_id),
        "execution_commit": execution_commit(root),
        "device": str(device),
        "contract_version": next(iter(reports.values()))["contract_version"],
        "adaptation_window": next(iter(reports.values()))["adaptation_window"],
        "split_id": next(iter(reports.values()))["split"]["id"],
        "spec": asdict(spec),
        "seed": args.seed,
        "modes": list(args.modes),
        "learner": {"lr": args.lr, "steps": args.steps},
        "wall_clock_s": time.time() - t0,
    }
    write_outputs(args.out, reports, manifest)
    print(summary_table(reports))
    print(f"[contract] wrote {args.out} in {manifest['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
