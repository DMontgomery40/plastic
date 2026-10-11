"""Apply the PRE-REGISTERED kill gate (see /workspace/r1/PROGRESS.md) to runs/ret/seg_s*.json and print markdown."""
import json, sys, glob
Ks = [0, 16, 64, 128]; kinds = ["struct", "randline", "randiid"]
files = sorted(glob.glob("/workspace/r1/runs/ret/seg_s*.json")); R = {f.split("seg_s")[1][0]: json.load(open(f)) for f in files}
def c(s, mode, arm, kind, K): return R[s]["cells"].get(f"{mode}|{arm}|{kind}|{K}")
out = ["| seed (ckpt step) | filler | K=0 | K=16 | K=64 | K=128 |", "|---|---|---|---|---|---|"]
for s in sorted(R):
    for kind in kinds:
        row = f"| s{s} ({R[s]['step']}) | {kind} |"
        for K in Ks:
            x = c(s, "normal", "seg", kind, 0 if K == 0 else K) if not (K == 0 and kind != "struct") else c(s, "normal", "seg", "struct", 0)
            if x is None: row += " n/a |"; continue
            d = x["d1"]; row += f" {d['recall']:.3f} [{d['wilson'][0]:.3f},{d['wilson'][1]:.3f}] (cb [{d['cluster_boot'][0]:.3f},{d['cluster_boot'][1]:.3f}]) |"
        out.append(row)
out += ["", "Depth-2 recall (point, Wilson):", "", "| seed | filler | K=0 | K=16 | K=64 | K=128 |", "|---|---|---|---|---|---|"]
for s in sorted(R):
    for kind in kinds:
        row = f"| s{s} | {kind} |"
        for K in Ks:
            x = c(s, "normal", "seg", "struct" if K == 0 else kind, K)
            row += " n/a |" if x is None else f" {x['d2']['recall']:.3f} [{x['d2']['wilson'][0]:.3f},{x['d2']['wilson'][1]:.3f}] |"
        out.append(row)
out += ["", "Controls (K=64 struct, d1 / d2 recall): "]
for s in sorted(R):
    for mode, arm, kind, K in [("no_adapt", "seg", "struct", 64), ("null", "seg", "struct", 64), ("normal", "none", "struct", 16), ("normal", "none", "struct", 64)]:
        x = c(s, mode, arm, kind, K)
        if x: out.append(f"- s{s} {mode}{'' if arm=='seg' else ' (NO marker)'} K={K}: d1 {x['d1']['recall']:.3f} [{x['d1']['wilson'][0]:.3f},{x['d1']['wilson'][1]:.3f}], d2 {x['d2']['recall']:.3f}")
# pre-registered verdict
out += ["", "### Pre-registered pass check (per seed): every filler kind: K=64 recall >= 0.80 AND Wilson lower >= 0.75; K=0 recall >= 0.95; PASS iff >= 2 of 3 seeds pass", ""]
npass = 0
for s in sorted(R):
    ok = True; det = []
    k0 = c(s, "normal", "seg", "struct", 0)
    if k0 is None or k0["d1"]["recall"] < 0.95: ok = False
    for kind in kinds:
        x = c(s, "normal", "seg", kind, 64)
        if x is None: ok = False; det.append(f"{kind}: missing"); continue
        p = x["d1"]["recall"] >= 0.80 and x["d1"]["wilson"][0] >= 0.75; ok &= p; det.append(f"{kind} {x['d1']['recall']:.3f} (W-lo {x['d1']['wilson'][0]:.3f}) {'ok' if p else 'FAIL'}")
    out.append(f"- seed {s}: {'PASS' if ok else 'FAIL'} (K=0 {k0['d1']['recall'] if k0 else float('nan'):.3f}; " + "; ".join(det) + ")"); npass += ok
out.append(f"\n**Seeds passing: {npass}/{len(R)} -> KILL GATE {'PASSED' if npass >= 2 else 'FAILED'}**")
print("\n".join(out))
