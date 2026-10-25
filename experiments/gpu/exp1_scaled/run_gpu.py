"""Job A driver: lr search (seed 0, 4 lrs, selected on in-distribution val loss) then seeds 1..S-1.

Groups: tmax=32 for variants A,B,C,D,L (train lengths 8-32) and tmax=64 for A,B (train lengths 8-64).
Runs N concurrent single-GPU subprocesses. Resumable: existing output JSONs are skipped.
Selection rule: lowest val loss; ties (|diff| < 1e-6, e.g. both 0) -> higher val acc -> smaller lr.

python experiments/gpu/exp1_scaled/run_gpu.py --outdir out/exp1 --workers 10
"""
import argparse, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="parity,mod3,s3")
ap.add_argument("--groups", default="32:A,B,C,D,L;64:A,B")
ap.add_argument("--lrs", default="3e-4,1e-3,3e-3,1e-2")
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--run-timeout", type=int, default=3600)
ap.add_argument("--outdir", default="out/exp1")
ap.add_argument("--extra", default="")
ap.add_argument("--sync-cmd", default="", help="shell cmd run every --sync-every s and at the end (uploads)")
ap.add_argument("--sync-every", type=int, default=600)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
tasks, lrs = a.tasks.split(","), a.lrs.split(",")
groups = [(int(g.split(":")[0]), g.split(":")[1].split(",")) for g in a.groups.split(";")]
combos = [(tm, t, v) for tm, vs in groups for t in tasks for v in vs]
os.makedirs(os.path.join(a.outdir, "logs"), exist_ok=True)


def path(tm, t, v, s, lr):
    return os.path.join(a.outdir, f"runs_tmax{tm}", f"{t}_{v}_s{s}_lr{lr}.json")


def run(tm, t, v, s, lr):
    out = path(tm, t, v, s, lr)
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [sys.executable, os.path.join(here, "train_gpu.py"), "--task", t, "--variant", v, "--seed", str(s), "--lr", lr,
           "--steps", str(a.steps), "--train-max", str(tm), "--out", out] + a.extra.split()
    log = os.path.join(a.outdir, "logs", f"tmax{tm}_{t}_{v}_s{s}_lr{lr}.log")
    t0 = time.time()
    with open(log, "w") as f:
        try:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=a.run_timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    print(f"[{time.strftime('%H:%M:%S')}] tmax{tm} {t} {v} s{s} lr{lr} rc={rc} {time.time()-t0:.0f}s", flush=True)


stop = threading.Event()


def syncer():
    while not stop.wait(a.sync_every):
        subprocess.run(a.sync_cmd, shell=True)


if a.sync_cmd:
    threading.Thread(target=syncer, daemon=True).start()
T0 = time.time()
with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(lambda x: run(*x), [(tm, t, v, 0, lr) for tm, t, v in combos for lr in lrs]))
    print(f"phase 1 done in {time.time()-T0:.0f}s", flush=True)
    sel = {}
    for tm, t, v in combos:
        vals = {}
        for lr in lrs:
            try:
                d = json.load(open(path(tm, t, v, 0, lr)))
                vl = d["val_loss"]
                vals[lr] = (vl if vl == vl else float("inf"), d["val_acc"])
            except Exception:
                vals[lr] = (float("inf"), 0.0)
        best_loss = min(x[0] for x in vals.values())
        tied = [lr for lr in lrs if vals[lr][0] - best_loss < 1e-6]
        best = sorted(tied, key=lambda lr: (-vals[lr][1], float(lr)))[0]
        sel[f"tmax{tm}/{t}/{v}"] = {"lr": best, "val_loss_by_lr": {k: x[0] for k, x in vals.items()},
                                    "val_acc_by_lr": {k: x[1] for k, x in vals.items()}, "tied": tied}
    json.dump(sel, open(os.path.join(a.outdir, "lr_selection.json"), "w"), indent=1)
    list(ex.map(lambda x: run(*x), [(tm, t, v, s, sel[f"tmax{tm}/{t}/{v}"]["lr"]) for tm, t, v in combos
                                    for s in range(1, a.seeds)]))
stop.set()
print(f"all done in {time.time()-T0:.0f}s", flush=True)
if a.sync_cmd:
    subprocess.run(a.sync_cmd, shell=True)
