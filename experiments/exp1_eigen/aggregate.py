"""Aggregate exp1 runs -> results/exp1/{runs.csv, summary.csv, summary.md, *.png}."""
import glob, json, os, re, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
R = "results/exp1"
# outputs go to a separate dir (default: ../report/exp1) so raw results/ is never touched
O = sys.argv[1] if len(sys.argv) > 1 else "../report/exp1"
os.makedirs(O, exist_ok=True)
runs = []
for f in sorted(glob.glob(f"{R}/runs/*.json") + glob.glob(f"{R}/runs_E/*.json")):
    if "lr_selection" in f:
        continue
    d = json.load(open(f))
    a = d["args"]
    runs.append(dict(task=a["task"], variant=a["variant"], seed=a["seed"], lr=a["lr"], val_loss=d["val_loss"],
                     eval=d["eval"], curve=d["curve"], eig=d.get("eig_stats", {}), params=d["n_params"],
                     time=d["train_time_s"], file=f))
sel = {}
for f in glob.glob(f"{R}/runs/lr_selection_*.json"):
    for k, v in json.load(open(f)).items():
        sel[k] = float(v["lr"])
if any(r["variant"] == "E" for r in runs):
    sel["parity/E"] = 1e-3  # E: reduced budget, no lr search (A's and B's selected parity lr), see README
NAMES = {"A": "A delta beta(0,1)", "B": "B delta beta(0,2)", "C": "C ssm a(0,1)", "D": "D ssm a(-1,1)",
         "E": "E NS-chunk rule", "L": "L LSTM (nonlinear)"}
with open(f"{O}/runs.csv", "w") as f:
    lens = sorted({int(L) for r in runs for L in r["eval"]})
    f.write("task,variant,seed,lr,selected,val_loss,params,train_time_s," + ",".join(f"acc_half_{L},acc_final_{L}" for L in lens) + "\n")
    for r in runs:
        s = sel.get(f"{r['task']}/{r['variant']}") == float(r["lr"])
        f.write(f"{r['task']},{r['variant']},{r['seed']},{r['lr']},{s},{r['val_loss']:.4g},{r['params']},{r['time']:.0f}," +
                ",".join(f"{r['eval'].get(str(L), {}).get('acc_last_half', float('nan')):.4f},{r['eval'].get(str(L), {}).get('acc_final', float('nan')):.4f}" for L in lens) + "\n")
chosen = [r for r in runs if sel.get(f"{r['task']}/{r['variant']}") == float(r["lr"])]
summ = {}
for r in chosen:
    summ.setdefault((r["task"], r["variant"]), []).append(r)
tasks = [t for t in ["parity", "mod3", "s3", "s5", "mqar"] if any(k[0] == t for k in summ)]
md = ["# Exp 1 summary (selected lr, mean ± sd over seeds; [min, max])", ""]
rows = []
for t in tasks:
    vs = sorted(v for (tt, v) in summ if tt == t)
    Ls = sorted({int(L) for v in vs for r in summ[(t, v)] for L in r["eval"]})
    metric = "acc_last_half"
    md += [f"## {t}  (metric: {'answer accuracy' if t == 'mqar' else 'per-token accuracy over the last half of the sequence'})", "",
           "| variant | lr | n | params | " + " | ".join(f"L={L}" for L in Ls) + " |",
           "|---|---|---|---|" + "---|" * len(Ls)]
    for v in vs:
        rr = summ[(t, v)]
        cells = []
        for L in Ls:
            xs = [r["eval"][str(L)][metric] for r in rr if str(L) in r["eval"]]
            m = st.mean(xs); sd = st.stdev(xs) if len(xs) > 1 else 0.0
            cells.append(f"{m:.3f}±{sd:.3f} [{min(xs):.2f},{max(xs):.2f}]")
            rows.append(dict(task=t, variant=v, L=L, mean=m, sd=sd, n=len(xs), mn=min(xs), mx=max(xs),
                             final_mean=st.mean([r["eval"][str(L)]["acc_final"] for r in rr if str(L) in r["eval"]])))
        md.append(f"| {NAMES.get(v, v)} | {rr[0]['lr']} | {len(rr)} | {rr[0]['params']} | " + " | ".join(cells) + " |")
    md.append("")
    # eigen usage
    md += ["Eigenvalue-range usage at the end of training (last forward, mean over seeds):", ""]
    for v in vs:
        rr = summ[(t, v)]
        keys = sorted(rr[0]["eig"])
        if keys:
            md.append(f"- {v}: " + ", ".join(f"{k}={st.mean(r['eig'][k] for r in rr):.3f}" for k in keys))
    md.append("")
with open(f"{O}/summary.csv", "w") as f:
    f.write("task,variant,L,n,mean,sd,min,max,acc_final_mean\n")
    for r in rows:
        f.write(f"{r['task']},{r['variant']},{r['L']},{r['n']},{r['mean']:.4f},{r['sd']:.4f},{r['mn']:.4f},{r['mx']:.4f},{r['final_mean']:.4f}\n")
md += ["## lr selection (seed 0, in-distribution val loss)", ""]
for f in glob.glob(f"{R}/runs/lr_selection_*.json"):
    for k, v in json.load(open(f)).items():
        md.append(f"- {k}: chose {v['lr']}  (val loss by lr: " + ", ".join(f"{a}={b:.3g}" for a, b in v["val_loss_by_lr"].items()) + ")")
open(f"{O}/summary.md", "w").write("\n".join(md) + "\n")

# plots: accuracy vs length
colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red", "E": "tab:purple", "L": "tab:gray"}
fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), squeeze=False)
for ax, t in zip(axes[0], tasks):
    for v in sorted({r["variant"] for r in rows if r["task"] == t}):
        rr = sorted([r for r in rows if r["task"] == t and r["variant"] == v], key=lambda r: r["L"])
        ax.errorbar([r["L"] for r in rr], [r["mean"] for r in rr], yerr=[r["sd"] for r in rr], marker="o", capsize=3,
                    label=NAMES.get(v, v), color=colors.get(v))
    ax.set_xscale("log", base=2); ax.set_title(t); ax.set_xlabel("test length (pairs for mqar)"); ax.set_ylim(-0.02, 1.02)
    ax.axvspan(8 if t != "mqar" else 4, 32 if t != "mqar" else 16, color="k", alpha=0.06)
    ax.set_ylabel("answer accuracy" if t == "mqar" else "accuracy (last half)")
axes[0][0].legend(fontsize=7)
fig.suptitle("Exp 1: accuracy vs length (shaded = training lengths); mean ± sd over seeds")
fig.tight_layout(); fig.savefig(f"{O}/acc_vs_length.png", dpi=130)
# training curves
fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.4), squeeze=False)
for ax, t in zip(axes[0], tasks):
    for (tt, v), rr in sorted(summ.items()):
        if tt != t:
            continue
        for i, r in enumerate(rr):
            ax.plot([c["step"] for c in r["curve"]], [c["loss"] for c in r["curve"]], color=colors.get(v), alpha=0.6,
                    label=NAMES.get(v, v) if i == 0 else None, lw=0.8)
    ax.set_yscale("log"); ax.set_title(f"{t}: train loss"); ax.set_xlabel("step")
axes[0][0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{O}/training_curves.png", dpi=130)
print("\n".join(md))
