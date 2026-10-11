"""Analyze D3-D4 gate tests: replace-rule, adaptive attacker, rollback noninterference."""
from __future__ import annotations
import json, math, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def analyze_replace_rule(data_per_ckpt):
    """Replace-rule test: gate should ACCEPT (not reject) new rules.
    Treat as: positive = gate does NOT reject (accept), negative = gate rejects.
    Measure TPR (accept rate on clean/legit replacement).
    """
    print("## D3.1: Replace-rule test")
    print("A legitimate new rule (different world, same filler positions) should be accepted by all gates.")
    print()
    
    # Pool all records
    recs_all = []; meta = {}
    for d in data_per_ckpt:
        if d and d["recs"]: 
            meta = d["meta"]
            recs_all.extend(d["recs"])
    
    if not recs_all:
        print("No records found.\n")
        return
    
    # Threshold-based gates (binary)
    conds_found = {(r["K"], r["kind"], r["arm"]) for r in recs_all}
    for cond in sorted(conds_found):
        recs = [r for r in recs_all if (r["K"], r["kind"], r["arm"]) == cond]
        print(f"Condition: K={cond[0]}, kind={cond[1]}, arm={cond[2]} ({len(recs)} episodes)")
        
        # G1: signals (max over chunks)
        for sig_name in ["chunk_loss", "surprise_mean", "delta_norm", "write_norm_sum", "err_y"]:
            # Get max signal per episode
            sig_vals = [max(r[f"sig_{sig_name}"]) if r[f"sig_{sig_name}"] else 0.0 for r in recs]
            # Threshold = 99th percentile (would be computed on a calibration set; here we use data itself)
            thr = float(np.quantile(sig_vals, 0.99))
            accepts = sum(1 for v in sig_vals if v <= thr)
            accept_rate = accepts / len(sig_vals)
            lo, hi = wilson(accepts, len(sig_vals))
            print(f"  G1:{sig_name}: accept_rate {accept_rate:.3f} ({accepts}/{len(sig_vals)}) [{lo:.3f},{hi:.3f}]")
        
        # G3: binary gates (permutation, regularity)
        for gate_name in ["rb_perm", "rb_reg"]:
            passes = sum(1 for r in recs if r[gate_name])  # True = passes (not rejecting the legit rule)
            accept_rate = passes / len(recs)
            lo, hi = wilson(passes, len(recs))
            print(f"  G3:{gate_name}: accept_rate {accept_rate:.3f} ({passes}/{len(recs)}) [{lo:.3f},{hi:.3f}]")
        
        print()


def analyze_attacker(data_per_ckpt):
    """Adaptive attacker test: consistent lie (relabel) under different ledger scenarios."""
    print("## D3.2: Adaptive attacker test")
    print("Attacker installs a consistent relabeling (valid world, but wrong names).")
    print()
    
    recs_all = defaultdict(list); meta = {}
    for d in data_per_ckpt:
        if d and d["recs"]:
            meta = d["meta"]
            for r in d["recs"]:
                recs_all[r["variant"]].extend([r])
    
    if "attacker_no_ledger" not in recs_all:
        print("No attacker records found.\n")
        return
    
    for variant in ["attacker_no_ledger", "attacker_fake_ledger"]:
        if variant not in recs_all or not recs_all[variant]:
            continue
        
        recs = recs_all[variant]
        cond = (recs[0]["K"], recs[0]["kind"], recs[0]["arm"])
        print(f"### {variant} (K={cond[0]}, kind={cond[1]}, arm={cond[2]}, {len(recs)} episodes)")
        
        # G3 should fail to detect relabel (it's name-invariant)
        g3_detects = sum(1 for r in recs if not r["rb_reg"])  # reject (detect anomaly)
        g3_rate = g3_detects / len(recs)
        lo, hi = wilson(g3_detects, len(recs))
        print(f"  G3 rejection_rate: {g3_rate:.3f} ({g3_detects}/{len(recs)}) [{lo:.3f},{hi:.3f}] (expect ~0.5 no-ledger, lower with ledger)")
        
        # G1 internal signals
        for sig_name in ["surprise_mean"]:
            sig_vals = [max(r[f"sig_{sig_name}"]) if r[f"sig_{sig_name}"] else 0.0 for r in recs]
            thr = float(np.quantile(sig_vals, 0.99))
            detects = sum(1 for v in sig_vals if v > thr)
            detect_rate = detects / len(sig_vals)
            lo, hi = wilson(detects, len(sig_vals))
            print(f"  G1:{sig_name} rejection_rate: {detect_rate:.3f} ({detects}/{len(sig_vals)}) [{lo:.3f},{hi:.3f}] (expect ~0.01 FPR)")
        
        print()


def analyze_rollback(data_per_ckpt):
    """Rollback noninterference: after rejection, state is unchanged."""
    print("## D3.3: Rollback noninterference")
    print("After installing and rejecting a rule, queries should be bit-identical to pre-install.")
    print()
    
    recs_all = []; meta = {}
    for d in data_per_ckpt:
        if d and d["recs"]:
            meta = d["meta"]
            recs_all.extend(d["recs"])
    
    if not recs_all:
        print("No records found.\n")
        return
    
    conds_found = {(r["K"], r["kind"], r["arm"]) for r in recs_all}
    for cond in sorted(conds_found):
        recs = [r for r in recs_all if (r["K"], r["kind"], r["arm"]) == cond]
        
        d1_identical = sum(1 for r in recs if r["identical_d1"])
        d2_identical = sum(1 for r in recs if r["identical_d2"])
        
        print(f"Condition: K={cond[0]}, kind={cond[1]}, arm={cond[2]} ({len(recs)} episodes)")
        print(f"  Depth-1 identical: {d1_identical}/{len(recs)} (expect 100%)")
        print(f"  Depth-2 identical: {d2_identical}/{len(recs)} (expect 100%)")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", nargs="+", required=True); ap.add_argument("--attacker", nargs="+", required=True); ap.add_argument("--rollback", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/d34_analysis.md")
    a = ap.parse_args()
    
    data_replace = [json.load(open(f)) for f in a.replace]
    data_attacker = [json.load(open(f)) for f in a.attacker]
    data_rollback = [json.load(open(f)) for f in a.rollback]
    
    with open(a.out, "w") as f:
        f.write("# D3-D4 Gate Analysis\n\n")
        
        # Save stdout then redirect
        import sys; old_stdout = sys.stdout
        sys.stdout = f
        
        analyze_replace_rule(data_replace)
        print()
        analyze_attacker(data_attacker)
        print()
        analyze_rollback(data_rollback)
        
        sys.stdout = old_stdout
    
    # Print to stdout too
    with open(a.out) as f:
        print(f.read())
