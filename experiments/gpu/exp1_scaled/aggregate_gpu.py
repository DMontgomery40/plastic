"""Aggregate Job A -> experiments/gpu/results/exp1/{runs.csv, summary.csv, summary.md, *.png}.
python experiments/gpu/exp1_scaled/aggregate_gpu.py [RAW_DIR]   (default experiments/gpu/results/exp1/raw)"""
import csv, glob, json, os, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "experiments/gpu/results/exp1/raw"
O = "experiments/gpu/results/exp1"
os.makedirs(O, exist_ok=True)
NAMES = {"A": "A delta beta(0,1)", "B": "B delta beta(0,2)", "C": "C ssm a(0,1]", "D": "D ssm a(-1,1)",
         "L": "L LSTM"}
sel_raw = json.load(open(f"{R}/lr_selection.json")) if os.path.exists(f"{R}/lr_selection.json") else {}
sel = {k: float(v["lr"]) for k, v in sel_raw.items()}
runs = []
for f in sorted(glob.glob(f"{R}/runs_tmax*/*.json")):
    d = json.load(open(f)); a = d["args"]
    runs.append(dict(tmax=a["train_max"], task=a["task"], variant=a["variant"], seed=a["seed"], lr=a["lr"],
                     val_loss=d["val_loss"], val_acc=d["val_acc"], eval=d["eval"], eig=d.get("eig_stats", {}),
                     params=d["n_params"], time=d["train_time_s"], gpu=d.get("gpu"), nonfinite=d.get("nonfinite_steps", 0),
                     curve=d["curve"]))
lens = sorted({int(L) for r in runs for L in r["eval"]})
key = lambda r: f"tmax{r['tmax']}/{r['task']}/{r['variant']}"
with open(f"{O}/runs.csv", "w") as f:
    f.write("tmax,task,variant,seed,lr,selected,val_loss,val_acc,params,train_time_s,nonfinite_steps,gpu," +
            ",".join(f"acc_half_{L},acc_final_{L}" for L in lens) + "\n")
    for r in runs:
        f.write(f"{r['tmax']},{r['task']},{r['variant']},{r['seed']},{r['lr']},{sel.get(key(r)) == r['lr']},{r['val_loss']:.4g},"
                f"{r['val_acc']:.4f},{r['params']},{r['time']:.0f},{r['nonfinite']},{r['gpu']}," +
                ",".join(f"{r['eval'][str(L)]['acc_last_half']:.4f},{r['eval'][str(L)]['acc_final']:.4f}" for L in lens) + "\n")
summ = {}
for r in runs:
    if sel.get(key(r)) == r["lr"]:
        summ.setdefault((r["tmax"], r["task"], r["variant"]), []).append(r)
rows = []
md = ["# Job A summary: selected lr, mean ± sd over seeds [min, max]", "",
      "Metric: per-token accuracy over the last half of each test sequence (chance: parity 0.5, mod3 0.333, s3 0.167).", ""]
for tm in sorted({k[0] for k in summ}):
    for t in ["parity", "mod3", "s3"]:
        vs = sorted(v for (m, tt, v) in summ if m == tm and tt == t)
        if not vs:
            continue
        md += [f"## {t}, train lengths 8-{tm}", "", "| variant | lr | n | " + " | ".join(f"L={L}" for L in lens) + " |",
               "|---|---|---|" + "---|" * len(lens)]
        for v in vs:
            rr = summ[(tm, t, v)]
            cells = []
            for L in lens:
                xs = [r["eval"][str(L)]["acc_last_half"] for r in rr]
                m = st.mean(xs); sd = st.stdev(xs) if len(xs) > 1 else 0.0
                n_solved = sum(x >= 0.99 for x in xs)
                cells.append(f"{m:.3f}±{sd:.3f} [{min(xs):.2f},{max(xs):.2f}]")
                rows.append(dict(tmax=tm, task=t, variant=v, L=L, n=len(xs), mean=m, sd=sd, mn=min(xs), mx=max(xs),
                                 n_ge_099=n_solved, final_mean=st.mean(r["eval"][str(L)]["acc_final"] for r in rr)))
            md.append(f"| {NAMES[v]} | {rr[0]['lr']:g} | {len(rr)} | " + " | ".join(cells) + " |")
        md += ["", "Seeds with accuracy >= 0.99 (solved) per length:", "",
               "| variant | " + " | ".join(f"L={L}" for L in lens) + " |", "|---|" + "---|" * len(lens)]
        for v in vs:
            md.append(f"| {v} | " + " | ".join(f"{next(x['n_ge_099'] for x in rows if x['tmax']==tm and x['task']==t and x['variant']==v and x['L']==L)}/{len(summ[(tm,t,v)])}" for L in lens) + " |")
        md += ["", "Eigenvalue-range usage at the end of training (mean over seeds):", ""]
        for v in vs:
            rr = summ[(tm, t, v)]
            ks = sorted(rr[0]["eig"])
            if ks:
                md.append(f"- {v}: " + ", ".join(f"{k}={st.mean(r['eig'][k] for r in rr):.3f}" for k in ks))
        md.append("")
with open(f"{O}/summary.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
md += ["## lr selection (seed 0, in-distribution validation loss; ties -> higher val acc -> smaller lr)", "",
       "| group | chosen | val loss by lr | val acc by lr |", "|---|---|---|---|"]
for k, v in sorted(sel_raw.items()):
    md.append(f"| {k} | {v['lr']} | " + ", ".join(f"{a}={b:.3g}" for a, b in v["val_loss_by_lr"].items()) + " | " +
              ", ".join(f"{a}={b:.3f}" for a, b in v["val_acc_by_lr"].items()) + " |")
md += ["", "## lr sensitivity: seed-0 accuracy (last half) at every lr in the search", "",
       "Shown because in-distribution validation saturates (val loss exactly 0 / val acc 1.000) for several groups, so the",
       "pre-registered tie-break (higher val acc, then smaller lr) decides; these rows show what the other lrs would have given.", "",
       "| group | lr | val loss | val acc | " + " | ".join(f"L={L}" for L in lens) + " |", "|---|---|---|---|" + "---|" * len(lens)]
for r in sorted([r for r in runs if r["seed"] == 0], key=lambda r: (r["tmax"], r["task"], r["variant"], r["lr"])):
    mark = " *" if sel.get(key(r)) == r["lr"] else ""
    md.append(f"| {key(r)} | {r['lr']:g}{mark} | {r['val_loss']:.3g} | {r['val_acc']:.4f} | " +
              " | ".join(f"{r['eval'][str(L)]['acc_last_half']:.3f}" for L in lens) + " |")
md.append("(* = selected)")
gpus = sorted({str(r["gpu"]) for r in runs})
md += ["", f"Runs: {len(runs)}; devices: {gpus}; total train GPU-process time {sum(r['time'] for r in runs)/3600:.2f} h "
       f"(concurrent processes, not wall clock); runs with non-finite steps: {sum(1 for r in runs if r['nonfinite'])}."]
open(f"{O}/summary.md", "w").write("\n".join(md) + "\n")

# CPU comparison table (dc55927 report/exp1/summary.csv, 3 seeds, 3k steps, batch 64)
cpu = {}
if os.path.exists("report/exp1/summary.csv"):
    for r in csv.DictReader(open("report/exp1/summary.csv")):
        cpu[(r["task"], r["variant"], int(r["L"]))] = (float(r["mean"]), float(r["sd"]), int(r["n"]))
cmp = ["# GPU (8 seeds, 10k steps x 128) vs CPU (3 seeds, 3k steps x 64), train lengths 8-32", "",
       "| task | variant | " + " | ".join(f"L={L} CPU → GPU" for L in [64, 128, 256, 512]) + " |", "|---|---|" + "---|" * 4]
for t in ["parity", "mod3", "s3"]:
    for v in "ABCDL":
        if (32, t, v) not in summ:
            continue
        cells = []
        for L in [64, 128, 256, 512]:
            g = next((x for x in rows if x["tmax"] == 32 and x["task"] == t and x["variant"] == v and x["L"] == L), None)
            c = cpu.get((t, v, L))
            cells.append(f"{c[0]:.3f} → {g['mean']:.3f}" if c and g else "n/a")
        cmp.append(f"| {t} | {v} | " + " | ".join(cells) + " |")
open(f"{O}/cpu_vs_gpu.md", "w").write("\n".join(cmp) + "\n")

colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red", "L": "tab:gray"}
for tm in sorted({k[0] for k in summ}):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, t in zip(axes, ["parity", "mod3", "s3"]):
        for v in "ABCDL":
            rr = sorted([x for x in rows if x["tmax"] == tm and x["task"] == t and x["variant"] == v], key=lambda x: x["L"])
            if rr:
                ax.errorbar([x["L"] for x in rr], [x["mean"] for x in rr], yerr=[x["sd"] for x in rr], marker="o", capsize=3,
                            label=NAMES[v], color=colors[v])
        ax.set_xscale("log", base=2); ax.set_ylim(-0.02, 1.02); ax.set_title(t); ax.set_xlabel("test length")
        ax.axvspan(8, tm, color="k", alpha=0.06)
    axes[0].set_ylabel("accuracy (last half)"); axes[0].legend(fontsize=7)
    fig.suptitle(f"Job A (GPU): train lengths 8-{tm}, mean ± sd over seeds")
    fig.tight_layout(); fig.savefig(f"{O}/acc_vs_length_tmax{tm}.png", dpi=130); plt.close(fig)
# curriculum: A and B at tmax 32 vs 64
if any(k[0] == 64 for k in summ):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, t in zip(axes, ["parity", "mod3", "s3"]):
        for v in "AB":
            for tm, ls in [(32, "--"), (64, "-")]:
                rr = sorted([x for x in rows if x["tmax"] == tm and x["task"] == t and x["variant"] == v], key=lambda x: x["L"])
                if rr:
                    ax.errorbar([x["L"] for x in rr], [x["mean"] for x in rr], yerr=[x["sd"] for x in rr], marker="o",
                                capsize=3, ls=ls, color=colors[v], label=f"{v} train 8-{tm}")
        ax.set_xscale("log", base=2); ax.set_ylim(-0.02, 1.02); ax.set_title(t); ax.set_xlabel("test length")
    axes[0].set_ylabel("accuracy (last half)"); axes[0].legend(fontsize=7)
    fig.suptitle("Job A curriculum: A vs B, train to 32 (dashed) vs 64 (solid)")
    fig.tight_layout(); fig.savefig(f"{O}/curriculum_A_vs_B.png", dpi=130); plt.close(fig)
print("\n".join(md))
