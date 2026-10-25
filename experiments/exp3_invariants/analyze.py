"""Exp 3(b) step 3: aggregate fpr_*.json into FPR/TPR tables with Wilson 95% CIs and one PNG.

python experiments/exp3_invariants/analyze.py
writes experiments/results/exp3/summary.json, experiments/exp3_invariants/roc_table.md, experiments/exp3_invariants/fpr_tpr.png
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "experiments", "results", "exp3")
HERE = os.path.dirname(os.path.abspath(__file__))
INTERVENE = {"rollback", "scale", "project", "readonly"}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*p:.2f}% [{100*lo:.2f}, {100*hi:.2f}]"


runs = defaultdict(list)
for f in sorted(glob.glob(os.path.join(RES, "fpr_*_seed*.json"))):
    d = json.load(open(f))
    key = (d["args"]["suite"], d["args"]["target"])
    runs[key].append(d)

summary = {}
for key, ds in runs.items():
    s = {"seeds": sorted(d["args"]["seed"] for d in ds), "complete": all(d.get("done") for d in ds),
         "thresholds": ds[0]["thresholds"]}
    ben = [c for d in ds for sess in d["benign"] for c in sess["chunks"]]
    n = len(ben)
    kinds = defaultdict(int)
    for c in ben:
        kinds[c["applied"]] += 1
    flagged = sum(1 for c in ben if c["applied"] in INTERVENE)
    refused = kinds["rollback"] + kinds["readonly"]
    sess_flag = [any(c["applied"] in INTERVENE for c in sess["chunks"]) for d in ds for sess in d["benign"]]
    first_reason = defaultdict(int)
    for c in ben:
        if c["applied"] in INTERVENE:
            r = c["reasons"][0] if c["reasons"] else "-"
            first_reason[r.split("(")[0]] += 1
    reason_any = defaultdict(int)
    for c in ben:
        for r in {x.split("(")[0] for x in c["reasons"]}:
            reason_any[r] += 1
    eligible = [c for c in ben if c["applied"] != "readonly"]
    pos = defaultdict(lambda: [0, 0])
    for d in ds:
        for sess in d["benign"]:
            for i, c in enumerate(sess["chunks"]):
                pos[i][0] += c["applied"] in INTERVENE
                pos[i][1] += 1
    # session-start chunks of the poison sessions (clean P) are benign too
    pre = [c for d in ds for p in d["poison"] for c in p["prefix"]]
    s["benign"] = {"n": n, "kinds": dict(kinds), "any": [flagged, n], "refused": [refused, n],
                   "scale_or_project": [kinds["scale"] + kinds["project"], n],
                   "sessions_with_any": [sum(sess_flag), len(sess_flag)], "first_reason": dict(first_reason), "reason_any": dict(reason_any),
                   "cusum_alarms_per_learning_chunk": [reason_any.get("cusum_alarm", 0), len(eligible)],
                   "signal_rollbacks_excl_cusum_per_learning_chunk": [sum(1 for c in eligible if c["applied"] == "rollback" and any(not r.startswith("cusum_alarm") for r in c["reasons"])), len(eligible)],
                   "by_position": {i: v for i, v in sorted(pos.items())},
                   "poison_prefix_any": [sum(c["applied"] in INTERVENE for c in pre), len(pre)]}
    tp = {}
    allx = [c for d in ds for p in d["poison"] for c in p["chunks"]]
    for t in sorted({p["type"] for d in ds for p in d["poison"]}):
        xs = [c for d in ds for p in d["poison"] if p["type"] == t for c in p["chunks"]]
        tp[t] = {"any": [sum(c["applied"] in INTERVENE for c in xs), len(xs)],
                 "refused": [sum(c["applied"] in ("rollback", "readonly") for c in xs), len(xs)]}
    for t in list(tp):
        first = [p["chunks"][0] for d in ds for p in d["poison"] if p["type"] == t]
        tp[t]["first_chunk_any"] = [sum(c["applied"] in INTERVENE for c in first), len(first)]
        tp[t]["first_chunk_kinds"] = dict(__import__("collections").Counter(c["applied"] for c in first))
    tp["all"] = {"any": [sum(c["applied"] in INTERVENE for c in allx), len(allx)],
                 "refused": [sum(c["applied"] in ("rollback", "readonly") for c in allx), len(allx)]}
    s["poison"] = tp
    summary[f"{key[0]}|{key[1]}"] = s

# archived exp2 natural decisions (shipped calibration), for contrast
exp2 = defaultdict(lambda: [0, 0])
for f in sorted(glob.glob(os.path.join(ROOT, "experiments", "results", "exp2", "probe_seed*.json"))):
    for rec in json.load(open(f)):
        for k in rec["natural_decisions"]:
            exp2[rec["type"]][0] += k in INTERVENE
            exp2[rec["type"]][1] += 1
summary["exp2_archived_natural"] = {k: v for k, v in exp2.items()}
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)


def tkey(k):
    t = k.split("|")[1]
    return (k.split("|")[0] != "shipped", -1 if t == "shipped" else float(t))


lines = ["| suite | thresholds (target FPR) | benign chunks flagged (any intervention) | benign refused (rollback/readonly) | benign scale/project | sessions with >=1 flag | poison flagged, all types | random | repeat_token | repeat_phrase | shuffled |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for k in sorted((k for k in summary if "|" in k), key=tkey):
    s = summary[k]
    b, p = s["benign"], s["poison"]
    suite, tgt = k.split("|")
    tgt = "shipped calibration.json (0.01)" if tgt == "shipped" else f"recalibrated @ {tgt}"
    cells = [suite, tgt + ("" if s["complete"] else " (INCOMPLETE)"), fmt(*b["any"]), fmt(*b["refused"]),
             fmt(*b["scale_or_project"]), fmt(*b["sessions_with_any"]), fmt(*p["all"]["any"])]
    for t in ("random", "repeat_token", "repeat_phrase", "shuffled"):
        cells.append(fmt(*p[t]["any"]) if t in p else "-")
    lines.append("| " + " | ".join(cells) + " |")
lines.append("")
lines.append("Archived exp2 natural decisions (shipped calibration; X after 2 clean chunks, 3 seeds x 8 sessions x sizes 1,2,4):")
lines.append("")
lines.append("| exp2 type | chunks flagged |")
lines.append("|---|---|")
for t, (k, n) in sorted(exp2.items()):
    lines.append(f"| {t} | {fmt(k, n)} |")
open(os.path.join(HERE, "roc_table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- plot
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for suite, mk in (("shipped", "o"), ("statements", "s")):
    ks = sorted((k for k in summary if k.startswith(suite + "|")), key=tkey)
    for t, col in (("all", "k"), ("random", "C0"), ("repeat_token", "C1"), ("repeat_phrase", "C2"), ("shuffled", "C3")):
        xs, ys, xe, ye, labs = [], [], [[], []], [[], []], []
        for k in ks:
            if t not in summary[k]["poison"]:
                continue
            fp, flo, fhi = wilson(*summary[k]["benign"]["any"])
            tp, tlo, thi = wilson(*summary[k]["poison"][t]["any"])
            xs.append(fp); ys.append(tp); xe[0].append(fp - flo); xe[1].append(fhi - fp); ye[0].append(tp - tlo); ye[1].append(thi - tp)
            labs.append(k.split("|")[1])
        if not xs:
            continue
        ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt=mk + ("-" if t == "all" else ":"), color=col, ms=5, capsize=2,
                    label=f"{suite}: {t}", alpha=1.0 if t == "all" else 0.6)
        if t == "all":
            for x_, y_, l_ in zip(xs, ys, labs):
                ax.annotate(l_, (x_, y_), textcoords="offset points", xytext=(4, -10), fontsize=7)
ax.set_xlabel("benign chunks flagged (FPR, Wilson 95% CI)")
ax.set_ylabel("poison chunks flagged (TPR)")
ax.set_title("Policy-level operating points (labels = threshold target)")
ax.set_xlim(left=0); ax.set_ylim(0.7, 1.01); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
ax = axes[1]
for k in sorted((k for k in summary if "|" in k), key=tkey):
    bp = summary[k]["benign"]["by_position"]
    xs = sorted(int(i) for i in bp)
    ys = [bp[i][0] / bp[i][1] if isinstance(next(iter(bp)), int) else bp[str(i)][0] / bp[str(i)][1] for i in xs]
    ax.plot(xs, ys, marker=".", label=k)
ax.set_xlabel("chunk index within a 16-chunk benign session")
ax.set_ylabel("fraction of benign chunks flagged")
ax.set_title("Benign false positives by position in session")
ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fpr_tpr.png"), dpi=130)
print("wrote", os.path.join(HERE, "fpr_tpr.png"))
