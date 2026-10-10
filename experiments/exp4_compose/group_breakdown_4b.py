"""Exp 4b: fast-weights accuracy by group (abelian cyclic Z5-Z8 vs non-abelian S3, D4) per run and depth, with Wilson CIs. python group_breakdown_4b.py <dir>"""
import sys, pandas as pd
from aggregate_lib import wilson
R = sys.argv[1]; rows = []
for run in ["aligned_s0", "aligned_s1", "aligned_s2", "old_s0"]:
    d = pd.read_csv(f"{R}/{run}/items_controls.csv.gz"); d["hit"] = d.pred_fw_reset == d.answer
    d["class"] = d.group.map(lambda g: "abelian (Z5-Z8)" if g.startswith("Z") else "non-abelian (S3, D4)")
    for (c, dep), g in d.groupby(["class", "depth"]):
        k, n = int(g.hit.sum()), len(g); lo, hi = wilson(k, n); rows.append(dict(run=run, cls=c, depth=dep, n=n, acc=k / n, lo=lo, hi=hi))
T = pd.DataFrame(rows); T.to_csv(f"{R}/accuracy_by_group_class.csv", index=False)
print("| run | group class | d2 | d3 | d4 |\n|---|---|---|---|---|")
for run in T.run.unique():
    for c in sorted(T.cls.unique()):
        t = T[(T.run == run) & (T.cls == c)].set_index("depth")
        print(f"| {run} | {c} | " + " | ".join(f"{t.loc[d].acc:.3f} [{t.loc[d].lo:.3f}, {t.loc[d].hi:.3f}] (n={int(t.loc[d].n)})" for d in (2, 3, 4)) + " |")
