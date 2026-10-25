"""Aggregate exp-4 per-item CSVs (one per model seed) into summary JSON/CSV/markdown and plots.

Usage: python aggregate.py <results_dir>   (reads <results_dir>/items_seed*.csv.gz)
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(sys.argv[1] if len(sys.argv) > 1 else "../results/exp4")
ARMS = ["fw_reset", "continuous", "ssm_only", "no_adapt", "null_fw_reset", "null_continuous"]
BASE = ["oracle", "one_hop", "other_gen", "copy_start", "last_line"]
LABEL = {"fw_reset": "fast weights only (lessons -> S, activations reset)", "continuous": "in-context (S + activations carried)",
         "ssm_only": "lessons in context, fast-weight writes frozen (beta=0)", "no_adapt": "no adaptation (S = W0 = 0, no lessons)",
         "null_fw_reset": "content null, fast weights only", "null_continuous": "content null, in-context",
         "oracle": "oracle composer (lookup + compose)", "one_hop": "lookup without composition (x*w1)",
         "other_gen": "lookup with the WRONG first generator (x*other)", "copy_start": "copy the start element", "last_line": "copy the last lesson answer (template copier)", "chance": "uniform over the episode's elements"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


rows = []
for f in sorted(glob.glob(str(R / "items_seed*.csv.gz"))):
    seed = int(Path(f).name.split("seed")[1].split(".")[0])
    with gzip.open(f, "rt") as fh:
        for r in csv.DictReader(fh):
            r = {k: (int(v) if k not in ("group",) else v) for k, v in r.items()}
            r["seed"] = seed
            rows.append(r)
seeds = sorted({r["seed"] for r in rows})
depths = sorted({r["depth"] for r in rows})
methods = ARMS + BASE


def correct(r, m):
    return int(r[f"pred_{m}"] == r["answer"])


def summarize(sub, m):
    k, n = sum(correct(r, m) for r in sub), len(sub)
    lo, hi = wilson(k, n)
    per_seed = []
    for s in seeds:
        ss = [r for r in sub if r["seed"] == s]
        if ss:
            per_seed.append(sum(correct(r, m) for r in ss) / len(ss))
    mean = sum(per_seed) / len(per_seed)
    sd = math.sqrt(sum((x - mean) ** 2 for x in per_seed) / (len(per_seed) - 1)) if len(per_seed) > 1 else 0.0
    return {"k": k, "n": n, "acc": k / n, "wilson95": [lo, hi], "per_seed": per_seed, "seed_sd": sd}


def chance(sub):
    return sum(1 / r["n"] for r in sub) / len(sub)


subsets = {
    "all": lambda r: True,
    "depth1_recall": lambda r: r["depth"] == 1,
    "composition_depth_ge2": lambda r: r["depth"] >= 2,
    "composition_nontrivial": lambda r: r["depth"] >= 2 and not r["is_identity"] and not r["eq_one_hop"] and not r["eq_last_line"],
    "composition_null_discriminating": lambda r: r["depth"] >= 2 and r["null_answer"] != r["answer"],
}
summary = {"seeds": seeds, "items_per_seed": len([r for r in rows if r["seed"] == seeds[0]]), "subsets": {}, "by_depth": {}, "by_group": {}}
for name, fn in subsets.items():
    sub = [r for r in rows if fn(r)]
    summary["subsets"][name] = {"n_items": len(sub), "chance": chance(sub), **{m: summarize(sub, m) for m in methods}}
    # the content null's agreement with its OWN lessons (does the model follow the lessons it was given?)
    for arm in ("null_fw_reset", "null_continuous"):
        k = sum(int(r[f"pred_{arm}"] == r["null_answer"]) for r in sub)
        summary["subsets"][name][f"{arm}_agrees_with_null_lessons"] = {"k": k, "n": len(sub), "acc": k / len(sub), "wilson95": list(wilson(k, len(sub)))}
for d in depths:
    sub = [r for r in rows if r["depth"] == d]
    summary["by_depth"][d] = {"n_items": len(sub), "chance": chance(sub), **{m: summarize(sub, m) for m in methods}}
    for arm in ("null_fw_reset",):
        k = sum(int(r[f"pred_{arm}"] == r["null_answer"]) for r in sub)
        summary["by_depth"][d][f"{arm}_agrees_with_null_lessons"] = {"k": k, "n": len(sub), "acc": k / len(sub), "wilson95": list(wilson(k, len(sub)))}
for g in sorted({r["group"] for r in rows}):
    for dname, fn in (("depth1", lambda r: r["depth"] == 1), ("depth_ge2", lambda r: r["depth"] >= 2)):
        sub = [r for r in rows if r["group"] == g and fn(r)]
        summary["by_group"][f"{g}/{dname}"] = {"n_items": len(sub), "chance": chance(sub), **{m: summarize(sub, m) for m in ("fw_reset", "continuous", "no_adapt", "null_fw_reset")}}

# paired difference fw_reset - null_fw_reset on composition items, episode-cluster bootstrap (episode x seed clusters)
def cluster_boot(sub, a, b, iters=2000, seed=0):
    cl = defaultdict(list)
    for r in sub:
        cl[(r["seed"], r["episode"])].append(correct(r, a) - correct(r, b))
    keys = list(cl)
    sums = [(sum(cl[k]), len(cl[k])) for k in keys]
    point = sum(s for s, _ in sums) / sum(n for _, n in sums)
    rng = random.Random(seed)
    bs = []
    for _ in range(iters):
        ss = nn = 0
        for _ in keys:
            s, n = sums[rng.randrange(len(keys))]
            ss += s
            nn += n
        bs.append(ss / nn)
    bs.sort()
    return {"diff": point, "ci95": [bs[int(0.025 * iters)], bs[int(0.975 * iters)]], "clusters": len(keys)}


comp = [r for r in rows if r["depth"] >= 2]
summary["paired"] = {
    "fw_reset_minus_null_fw_reset_depth_ge2": cluster_boot(comp, "fw_reset", "null_fw_reset"),
    "fw_reset_minus_no_adapt_depth_ge2": cluster_boot(comp, "fw_reset", "no_adapt"),
    "continuous_minus_fw_reset_depth_ge2": cluster_boot(comp, "continuous", "fw_reset"),
    "fw_reset_minus_one_hop_depth_ge2": cluster_boot(comp, "fw_reset", "one_hop"),
}
(R / "summary.json").write_text(json.dumps(summary, indent=1))

with open(R / "summary_by_depth.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["depth", "method", "k", "n", "acc", "wilson_lo", "wilson_hi", "seed0", "seed1", "seed2", "seed_sd", "chance"])
    for d in depths:
        s = summary["by_depth"][d]
        for m in methods:
            x = s[m]
            w.writerow([d, m, x["k"], x["n"], f"{x['acc']:.4f}", f"{x['wilson95'][0]:.4f}", f"{x['wilson95'][1]:.4f}",
                        *[f"{v:.4f}" for v in x["per_seed"]], f"{x['seed_sd']:.4f}", f"{s['chance']:.4f}"])

# markdown
md = [f"# Exp 4 summary (seeds {seeds}; {summary['items_per_seed']} first-answer items per seed per arm)", ""]
for name in subsets:
    s = summary["subsets"][name]
    md += [f"## {name} (n = {s['n_items']} pooled; chance {s['chance']:.3f})", "", "| method | acc | Wilson 95% | per seed |", "|---|---|---|---|"]
    for m in methods:
        x = s[m]
        md.append(f"| {LABEL[m]} | {x['acc']:.3f} | [{x['wilson95'][0]:.3f}, {x['wilson95'][1]:.3f}] | {', '.join(f'{v:.3f}' for v in x['per_seed'])} |")
    for arm in ("null_fw_reset", "null_continuous"):
        x = s[f"{arm}_agrees_with_null_lessons"]
        md.append(f"| {LABEL[arm]}: agrees with the null lessons' answer | {x['acc']:.3f} | [{x['wilson95'][0]:.3f}, {x['wilson95'][1]:.3f}] | |")
    md.append("")
md += ["## By depth (pooled over seeds; acc [Wilson 95%])", "", "| depth | n | chance | " + " | ".join(methods) + " |", "|" + "---|" * (len(methods) + 3)]
for d in depths:
    s = summary["by_depth"][d]
    md.append(f"| {d} | {s['n_items']} | {s['chance']:.3f} | " + " | ".join(f"{s[m]['acc']:.3f} [{s[m]['wilson95'][0]:.3f},{s[m]['wilson95'][1]:.3f}]" for m in methods) + " |")
md += ["", "## By group (fast weights only / in-context / no adaptation / content null)", "", "| group/depth | n | chance | fw_reset | continuous | no_adapt | null_fw_reset |", "|---|---|---|---|---|---|---|"]
for g, s in summary["by_group"].items():
    md.append(f"| {g} | {s['n_items']} | {s['chance']:.3f} | " + " | ".join(f"{s[m]['acc']:.3f}" for m in ("fw_reset", "continuous", "no_adapt", "null_fw_reset")) + " |")
md += ["", "## Paired differences on composition items (depth >= 2), episode-cluster bootstrap 95% CI", ""]
for k, v in summary["paired"].items():
    md.append(f"- {k}: {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] ({v['clusters']} clusters)")
(R / "summary.md").write_text("\n".join(md) + "\n")

# plots
fig, ax = plt.subplots(figsize=(8, 5))
style = {"fw_reset": ("C0", "-o"), "continuous": ("C1", "-s"), "ssm_only": ("C2", "-^"), "no_adapt": ("C3", "-v"),
         "null_fw_reset": ("C4", "--o"), "null_continuous": ("C5", "--s"), "one_hop": ("0.5", ":x"), "copy_start": ("0.3", ":+"), "oracle": ("k", "-")}
for m, (c, ls) in style.items():
    acc = [summary["by_depth"][d][m]["acc"] for d in depths]
    lo = [summary["by_depth"][d][m]["wilson95"][0] for d in depths]
    hi = [summary["by_depth"][d][m]["wilson95"][1] for d in depths]
    ax.plot(depths, acc, ls, color=c, label=LABEL[m], ms=4)
    ax.fill_between(depths, lo, hi, color=c, alpha=0.15)
ax.plot(depths, [summary["by_depth"][d]["chance"] for d in depths], color="0.7", lw=3, alpha=0.6, label=LABEL["chance"])
tr = 4
ax.axvspan(tr + 0.5, max(depths) + 0.3, color="0.9", zorder=-1)
ax.text(tr + 0.6, 0.95, "depth not trained", fontsize=8)
ax.set_xlabel("composition depth d (word length; d >= 2 never shown in any lesson)")
ax.set_ylabel("first-answer exact accuracy")
ax.set_ylim(-0.02, 1.02)
ax.set_title(f"Exp 4: first answer after group-action lessons (PlasticCore, {len(seeds)} seeds pooled, Wilson 95%)", fontsize=9)
ax.legend(fontsize=7, loc="center right")
fig.tight_layout()
fig.savefig(R / "accuracy_by_depth.png", dpi=130)

fig, ax = plt.subplots(figsize=(8, 4))
names = ["fw_reset", "continuous", "ssm_only", "no_adapt", "null_fw_reset", "null_continuous", "one_hop", "copy_start", "last_line"]
x = range(len(names))
for i, s in enumerate(seeds):
    vals = [summary["subsets"]["composition_nontrivial"][m]["per_seed"][i] for m in names]
    ax.scatter([j + (i - 1) * 0.15 for j in x], vals, s=18, label=f"seed {s}")
ax.axhline(summary["subsets"]["composition_nontrivial"]["chance"], color="0.6", ls="--", label="chance")
ax.set_xticks(list(x))
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("first-answer exact")
ax.set_title("Composition items not solvable by copy-start / one-hop / last-line (depth >= 2), per seed", fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(R / "nontrivial_per_seed.png", dpi=130)
print((R / "summary.md").read_text())
