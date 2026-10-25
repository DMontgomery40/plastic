"""Exp 1 driver: equal lr-tuning budget for every (task, variant), then 3 seeds at the selected lr.

Phase 1: seed 0 at each lr in LRS; select the lr with the lowest IN-DISTRIBUTION validation loss
         (lengths 8-32; never the extrapolation lengths).
Phase 2: seeds 1..S-1 at the selected lr. The seed-0 run at the selected lr is reused as seed 0.
Resumable: an existing output JSON is not re-run.

python experiments/exp1_eigen/run_all.py --tasks parity,mod3,s3 --variants A,B,C,D,L --workers 5
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="parity,mod3,s3")
ap.add_argument("--variants", default="A,B,C,D,L")
ap.add_argument("--lrs", default="1e-3,3e-3")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--outdir", default="results/exp1/runs")
ap.add_argument("--extra", default="", help="extra args for train.py, e.g. for mqar")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(a.outdir, exist_ok=True); os.makedirs("logs/exp1", exist_ok=True)


def path(task, v, seed, lr):
    return os.path.join(a.outdir, f"{task}_{v}_s{seed}_lr{lr}.json")


def run(task, v, seed, lr):
    out = path(task, v, seed, lr)
    if os.path.exists(out):
        return out
    cmd = [sys.executable, os.path.join(here, "train.py"), "--task", task, "--variant", v, "--seed", str(seed),
           "--lr", lr, "--steps", str(a.steps), "--threads", "1", "--out", out] + a.extra.split()
    with open(f"logs/exp1/{task}_{v}_s{seed}_lr{lr}.log", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return out


tasks, variants, lrs = a.tasks.split(","), a.variants.split(","), a.lrs.split(",")
combos = [(t, v) for t in tasks for v in variants]
with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(lambda x: run(*x), [(t, v, 0, lr) for t, v in combos for lr in lrs]))
    sel = {}
    for t, v in combos:
        vals = {}
        for lr in lrs:
            try:
                vals[lr] = json.load(open(path(t, v, 0, lr)))["val_loss"]
            except Exception:
                vals[lr] = float("inf")
        best = min(vals, key=lambda k: (vals[k] != vals[k], vals[k]))
        sel[f"{t}/{v}"] = {"lr": best, "val_loss_by_lr": vals}
    json.dump(sel, open(os.path.join(a.outdir, f"lr_selection_{a.tasks.replace(',', '+')}_{a.variants.replace(',', '+')}.json"), "w"), indent=1)
    list(ex.map(lambda x: run(*x), [(t, v, s, sel[f"{t}/{v}"]["lr"]) for t, v in combos for s in range(1, a.seeds)]))
print("done")
