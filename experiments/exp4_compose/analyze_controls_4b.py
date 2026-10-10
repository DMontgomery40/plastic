"""Exp 4b: per-depth accuracy + Wilson CIs, control arms, paired gaps (bootstrap CIs), shortcuts, old-vs-aligned, plots.
Usage: python analyze_controls_4b.py <dir with aligned_s{0,1,2}/ old_s0/ each holding items_controls.csv.gz> <out_dir>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aggregate_lib import wilson

R, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
RUNS = ["aligned_s0", "aligned_s1", "aligned_s2", "old_s0"]
D = {r: pd.read_csv(R / r / "items_controls.csv.gz") for r in RUNS}
for r, d in D.items(): d["run"] = r
DEPTHS = [1, 2, 3, 4, 5, 6]
ARMS = {"fw": "fw_reset", "no_adapt": "no_adapt", "null_deranged": "null_fw_reset", "null_foreign": "foreign_fw_reset",
        "copier(last_line)": "last_line", "one_hop": "one_hop", "other_gen": "other_gen", "copy_start": "copy_start", "oracle": "oracle"}
rng = np.random.default_rng(0)


def hit(d, col): return (d["pred_" + col] == d["answer"]).values.astype(float)


def boot_ci(x, B=4000):
    n = len(x); idx = rng.integers(0, n, (B, n)); m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


rows = []
def add(run, depth, arm, h):
    k, n = int(h.sum()), len(h); lo, hi = wilson(k, n); rows.append(dict(run=run, depth=depth, arm=arm, k=k, n=n, acc=k / n, lo=lo, hi=hi))

pools = {"aligned_pooled": pd.concat([D[r] for r in RUNS[:3]]), **D}
for run, d in pools.items():
    for dep in DEPTHS:
        g = d[d.depth == dep]
        for a, c in ARMS.items(): add(run, dep, a, hit(g, c))
        # fw agreement with the null lessons' own answer (the model composes the WRONG lessons)
        add(run, dep, "fw_on_null_vs_null_answer", (g.pred_null_fw_reset == g.null_answer).values.astype(float))
        for s in ("p_set_x", "p_set_first", "p_set_last"):  # expected accuracies, not realized: k is rounded
            v = g[s].values; lo, hi = boot_ci(v); rows.append(dict(run=run, depth=dep, arm="E[" + s + "]", k=round(v.sum()), n=len(v), acc=v.mean(), lo=lo, hi=hi))
        v = np.maximum.reduce([g.p_set_x.values, g.p_set_first.values, g.p_set_last.values]); lo, hi = boot_ci(v)
        rows.append(dict(run=run, depth=dep, arm="E[best_set_prior]", k=round(v.sum()), n=len(v), acc=v.mean(), lo=lo, hi=hi))
        h = hit(g, "fw_reset"); ok = (g.null_answer != g.answer).values
        add(run, dep, "fw_on_discriminating_items", h[ok]); add(run, dep, "null_deranged_on_discriminating_items", hit(g, "null_fw_reset")[ok])
        add(run, dep, "frac_discriminating", ok.astype(float))
T = pd.DataFrame(rows); T.to_csv(OUT / "accuracy_by_depth_arm.csv", index=False)

# paired gaps (fw minus control): bootstrap over items (items within a depth are independent episodes)
gaps = []
def gap(run, dep, name, x):
    lo, hi = boot_ci(x); gaps.append(dict(run=run, depth=dep, gap=name, diff=float(x.mean()), lo=lo, hi=hi, n=len(x)))
for run, d in pools.items():
    for dep in DEPTHS:
        g = d[d.depth == dep]; fw = hit(g, "fw_reset")
        gap(run, dep, "fw - null_deranged", fw - hit(g, "null_fw_reset"))
        gap(run, dep, "fw - null_foreign", fw - hit(g, "foreign_fw_reset"))
        gap(run, dep, "fw - no_adapt", fw - hit(g, "no_adapt"))
        gap(run, dep, "fw - copier(last_line)", fw - hit(g, "last_line"))
        gap(run, dep, "fw - one_hop", fw - hit(g, "one_hop"))
        best = np.maximum.reduce([hit(g, c) for c in ("one_hop", "other_gen", "copy_start", "last_line")])  # per-item OR of 4 shortcuts (generous)
        gap(run, dep, "fw - E[best_set_prior]", fw - np.maximum.reduce([g.p_set_x.values, g.p_set_first.values, g.p_set_last.values]))
        ok = (g.null_answer != g.answer).values
        if ok.sum() > 30: gap(run, dep, "fw - null_deranged (discriminating items only)", (fw - hit(g, "null_fw_reset"))[ok])
        # best single fixed deterministic baseline's accuracy (chosen post hoc on these items = generous)
        accs = {c: hit(g, c).mean() for c in ("one_hop", "other_gen", "copy_start", "last_line")}
        c = max(accs, key=accs.get); gap(run, dep, f"fw - best_fixed_baseline({c})", fw - hit(g, c))
# old vs aligned (same eval seed 0, same items)
a, o = D["aligned_s0"], D["old_s0"]
assert (a.answer.values == o.answer.values).all() and (a.depth.values == o.depth.values).all()
for dep in DEPTHS:
    m = (a.depth == dep).values
    gap("aligned_s0_vs_old_s0", dep, "aligned fw - old fw", (hit(a, "fw_reset") - hit(o, "fw_reset"))[m])
G = pd.DataFrame(gaps); G.to_csv(OUT / "paired_gaps.csv", index=False)

# plots
cols = {"fw": "C3", "null_deranged": "C0", "null_foreign": "C9", "no_adapt": "0.5", "copier(last_line)": "C2", "one_hop": "C1", "E[best_set_prior]": "C4"}
fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=True)
for ax, run in zip(axes, ["aligned_s0", "aligned_s1", "aligned_s2", "aligned_pooled", "old_s0"]):
    for arm, c in cols.items():
        t = T[(T.run == run) & (T.arm == arm)].sort_values("depth")
        ax.errorbar(t.depth, t.acc, yerr=[np.clip(t.acc - t.lo, 0, None), np.clip(t.hi - t.acc, 0, None)], color=c, marker="o", ms=3, lw=1.2, capsize=2, label=arm)
    ax.axvspan(4.5, 6.5, color="0.93"); ax.set_title(run); ax.set_xlabel("depth (5,6 = length-held-out)"); ax.set_ylim(-0.02, 1.03)
axes[0].set_ylabel("first-answer exact accuracy (Wilson 95%)"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "accuracy_by_depth_controls.png", dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for j, gname in enumerate(["fw - null_deranged", "fw - E[best_set_prior]"]):
    for run, c in zip(["aligned_s0", "aligned_s1", "aligned_s2", "aligned_pooled", "old_s0"], ["C0", "C1", "C2", "k", "C3"]):
        t = G[(G.run == run) & (G.gap == gname)].sort_values("depth"); off = 0.08 * (["aligned_s0", "aligned_s1", "aligned_s2", "aligned_pooled", "old_s0"].index(run) - 2)
        ax[j].errorbar(t.depth + off, t["diff"], yerr=[np.clip(t["diff"] - t.lo, 0, None), np.clip(t.hi - t["diff"], 0, None)], color=c, marker="o", ms=3, capsize=2, label=run, lw=1)
    ax[j].axhline(0, color="r", ls=":"); ax[j].axvspan(4.5, 6.5, color="0.93"); ax[j].set_title(gname + " (paired, bootstrap 95%)"); ax[j].set_xlabel("depth")
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "paired_gaps.png", dpi=130)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
def show(df): print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
for run in ["aligned_pooled", "aligned_s0", "aligned_s1", "aligned_s2", "old_s0"]:
    t = T[T.run == run].copy(); t["cell"] = t.apply(lambda r: f"{r.acc:.3f} [{r.lo:.3f},{r.hi:.3f}]", axis=1)
    print("\n==", run, "(n per depth = %d)" % t[t.arm == "fw"].n.iloc[0]); print(t.pivot(index="arm", columns="depth", values="cell").to_string())
for run in ["aligned_pooled", "aligned_s0", "aligned_s1", "aligned_s2", "old_s0", "aligned_s0_vs_old_s0"]:
    t = G[G.run == run].copy(); t["cell"] = t.apply(lambda r: f"{r['diff']:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]", axis=1)
    print("\n== gaps", run); print(t.pivot(index="gap", columns="depth", values="cell").to_string())
