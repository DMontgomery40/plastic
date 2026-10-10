"""Markdown tables for REPORT_4b.md from results/exp4b/{accuracy_by_depth_arm,paired_gaps}.csv.  python make_tables_4b.py <dir>"""
import sys, pandas as pd
D = sys.argv[1]; T = pd.read_csv(f"{D}/accuracy_by_depth_arm.csv"); G = pd.read_csv(f"{D}/paired_gaps.csv")
NAMES = {"aligned_s0": "aligned seed 0", "aligned_s1": "aligned seed 1", "aligned_s2": "aligned seed 2", "aligned_pooled": "aligned, 3 seeds pooled", "old_s0": "old layout, seed 0"}
ARM = [("fw", "**fast weights (`fw_reset`)**"), ("fw_on_null_vs_null_answer", "fast weights on WRONG lessons, scored vs the wrong lessons' own answer"),
       ("null_deranged", "content null (wrong lessons), scored vs TRUE answer"), ("null_deranged_on_discriminating_items", "content null, only items where null answer != true answer"),
       ("fw_on_discriminating_items", "fast weights, same items (null answer != true answer)"), ("null_foreign", "foreign-lessons null (another episode's lessons)"),
       ("no_adapt", "no adaptation (beta=0)"), ("copier(last_line)", "template copier (last lesson answer)"), ("one_hop", "one-hop lookup (x*w1)"),
       ("E[p_set_x]", "E[set prior: uniform over d-step reach of x]"), ("E[best_set_prior]", "E[best set prior: knows 1st hop or last letter]"), ("oracle", "oracle")]
def cell(r): return f"{r.acc:.3f} [{r.lo:.3f}, {r.hi:.3f}]"
out = []
for run in NAMES:
    t = T[T.run == run]; n = int(t[t.arm == "fw"].n.iloc[0])
    out.append(f"\n#### {NAMES[run]} (n = {n} items per depth; Wilson 95% CIs; expected-accuracy rows E[.] use bootstrap CIs)\n")
    out.append("| arm | d1 | d2 | d3 | d4 | d5* | d6* |\n|---|---|---|---|---|---|---|")
    for a, lab in ARM:
        row = [cell(t[(t.arm == a) & (t.depth == d)].iloc[0]) for d in range(1, 7)]
        out.append(f"| {lab} | " + " | ".join(row) + " |")
GAPS = [("fw - null_deranged (discriminating items only)", "fw minus content null, items where null answer != true answer"), ("fw - null_deranged", "fw minus content null, all items"),
        ("fw - null_foreign", "fw minus foreign-lessons null"), ("fw - no_adapt", "fw minus no adaptation"), ("fw - copier(last_line)", "fw minus template copier"),
        ("fw - one_hop", "fw minus one-hop lookup"), ("fw - E[best_set_prior]", "fw minus E[best set prior]")]
out.append("\n### Paired differences, fast weights minus control (bootstrap 95% CI over items; positive = fast weights better)\n")
for run in NAMES:
    t = G[G.run == run]
    out.append(f"\n#### {NAMES[run]}\n\n| difference | d1 | d2 | d3 | d4 | d5* | d6* |\n|---|---|---|---|---|---|---|")
    for g, lab in GAPS:
        row = []
        for d in range(1, 7):
            r = t[(t.gap == g) & (t.depth == d)].iloc[0]; row.append(f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]")
        out.append(f"| {lab} | " + " | ".join(row) + " |")
t = G[G.run == "aligned_s0_vs_old_s0"]
out.append("\n#### Old vs aligned layout, seed 0, identical items (paired)\n\n| difference | d1 | d2 | d3 | d4 | d5* | d6* |\n|---|---|---|---|---|---|---|")
out.append("| aligned fw minus old fw | " + " | ".join(f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" for _, r in t.sort_values("depth").iterrows()) + " |")
print("\n".join(out))
