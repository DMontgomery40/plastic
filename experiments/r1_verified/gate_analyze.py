"""Gate comparison analysis (R1). Reads gate_features JSONs, calibrates each gate on `clean_cal` (1% FPR target), reports on
`clean_test` (FPR) and the three corruption variants (TPR, AUROC) with Wilson / episode-bootstrap CIs.
Usage: gate_analyze.py --feat runs/feat/<tag>_s0.json runs/feat/<tag>_s1.json ... --out report.md [--tag name]"""
from __future__ import annotations
import argparse, json, math, random
from collections import defaultdict
import numpy as np


def rankdata(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); xs = x[o]; r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r

CORR = ("wrong_y", "swap_y", "derange", "relabel2")
M_LIST = (2, 4, 8, 16)
STAT4 = ("chunk_loss", "surprise_mean", "delta_norm", "write_norm_sum")


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def auroc(neg, pos):  # pos = corrupted (should score higher)
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    r = rankdata(np.concatenate([pos, neg])); return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def sigmax(r, s):
    v = np.array(r["sig_" + s], float)
    if s == "chunk_loss": v = v[v != 0.0]  # a 1-token final chunk has no in-chunk targets (loss exactly 0): drop
    if s == "err_y": return float(np.nanmean(v))
    return float(v.max())


def scores(r, rng):
    """gate name -> score (higher = more suspicious)."""
    out = {}
    for s in STAT4 + ("err_y",): out[f"G1:{s}"] = sigmax(r, s)
    for m in M_LIST:
        out[f"G2:heldout_d2_nll(m={m})"] = float(np.mean(r["nll2"][:m]))
        idx = rng.sample(range(len(r["nll1"])), min(m, len(r["nll1"])))
        out[f"G2:ledger_d1_nll(m={m})"] = float(np.mean([r["nll1"][i] for i in idx]))
    out["G3w:readback_not_permutation"] = float(not r["rb_perm"])
    out["G3:readback_not_regular_group"] = float(not r["rb_reg"])
    out["G3x:write_verify(1-fidelity)"] = 1.0 - r["fidelity"]
    out["T:text_not_regular_group(oracle,text)"] = float(not r["txt_reg"])
    return out


def analyze(recs, B=300, seed=0):
    """recs: all records of ONE checkpoint and ONE condition. returns dict gate -> dict corr -> metrics (decisions as arrays)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in recs: by[r["variant"]].append(scores(r, rng))
    gates = list(by["clean_cal"][0].keys()); res = {}
    for g in gates:
        cal = np.array([s[g] for s in by["clean_cal"]]); test = np.array([s[g] for s in by["clean_test"]])
        binary = set(np.unique(cal)) <= {0.0, 1.0} and g.startswith(("G3w", "G3:", "T:"))
        thr = 0.5 if binary else float(np.quantile(cal, 0.99))   # reject iff score > thr
        entry = {"thr": thr, "fp": int((test > thr).sum()), "n_fp": len(test), "cal_fp": int((cal > thr).sum()), "n_cal": len(cal)}
        for c in CORR:
            if not by[c]: continue
            pos = np.array([s[g] for s in by[c]])
            entry[c] = {"tp": int((pos > thr).sum()), "n": len(pos), "auc": auroc(test, pos), "neg": test, "pos": pos}
        res[g] = entry
    return res


def boot_auc(pairs, B=300, seed=1):
    """pairs: list over checkpoints of (neg, pos) arrays -> mean AUROC over checkpoints with episode-bootstrap 95% CI."""
    rng = np.random.default_rng(seed); pt = np.mean([auroc(n, p) for n, p in pairs]); bs = []
    for _ in range(B):
        bs.append(np.mean([auroc(n[rng.integers(0, len(n), len(n))], p[rng.integers(0, len(p), len(p))]) for n, p in pairs]))
    return pt, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def present(per_ckpt):
    return [c for c in CORR if all(c in e[next(iter(e))] for e in per_ckpt)]


def pooled_table(per_ckpt: list[dict], title: str, gates_show=None):
    CORR_ = present(per_ckpt)
    lines = [f"### {title}", "", "| gate | FPR on clean_test (k/n, Wilson) | " + " | ".join(f"{c}: TPR (k/n, Wilson) / AUROC [95% CI]" for c in CORR_) + " |", "|---|---|" + "---|" * len(CORR_)]
    for g in (gates_show or per_ckpt[0].keys()):
        fp = sum(e[g]["fp"] for e in per_ckpt); nfp = sum(e[g]["n_fp"] for e in per_ckpt); lo, hi = wilson(fp, nfp)
        row = f"| {g} | {fp/nfp:.3f} ({fp}/{nfp}; {lo:.3f}-{hi:.3f}) |"
        for c in CORR_:
            tp = sum(e[g][c]["tp"] for e in per_ckpt); n = sum(e[g][c]["n"] for e in per_ckpt); lo, hi = wilson(tp, n)
            a, al, ah = boot_auc([(e[g]["neg"] if False else e[g][c]["neg"], e[g][c]["pos"]) for e in per_ckpt])
            row += f" {tp/n:.3f} ({tp}/{n}; {lo:.3f}-{hi:.3f}) / {a:.3f} [{al:.3f}, {ah:.3f}] |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feat", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    data = [json.load(open(f)) for f in a.feat]
    conds = sorted({(r["K"], r["kind"], r["arm"]) for d in data for r in d["recs"]})
    md = [f"# Gate comparison {a.tag}", "", f"Files: {a.feat}", "Calibration: each gate's threshold = 99th percentile of its score on that checkpoint's `clean_cal` (binary gates: reject on violation); FPR measured on a disjoint `clean_test` set (n per ckpt in table). TPR = fraction of corrupted installs rejected at that threshold. AUROC: corrupted vs clean_test, higher score = more suspicious (direction NOT optimised; 'G1' signals are reported as the harness uses them: upper tail). AUROC CI = episode bootstrap (300 resamples), averaged over checkpoints. Wilson CIs treat episodes as independent.", ""]
    for cond in conds:
        per_ckpt = []
        for d in data:
            recs = [r for r in d["recs"] if (r["K"], r["kind"], r["arm"]) == cond]
            if recs: per_ckpt.append(analyze(recs))
        K, kind, arm = cond
        # accuracy context
        acc = [np.mean([r["d1_acc"] for r in d["recs"] if (r["K"], r["kind"], r["arm"]) == cond and r["variant"] == "clean_test"]) for d in data if any((r["K"], r["kind"], r["arm"]) == cond for r in d["recs"])]
        md.append(f"## Condition K={K} distractor lines, kind={kind}, marker arm={arm}  (n checkpoints={len(per_ckpt)}; clean depth-1 readback accuracy per ckpt: {', '.join(f'{x:.3f}' for x in acc)})\n")
        show = [g for g in per_ckpt[0].keys()]
        md.append(pooled_table(per_ckpt, f"pooled over checkpoints (K={K},{kind},{arm})", show))
        if len(per_ckpt) > 1:
            for gi in ("G3:readback_not_regular_group", "G1:surprise_mean", "G2:heldout_d2_nll(m=8)"):
                md.append(f"- per-checkpoint {gi}: " + "; ".join(f"ckpt{i}: FPR {e[gi]['fp']}/{e[gi]['n_fp']}, " + ", ".join(f"{c} TPR {e[gi][c]['tp']}/{e[gi][c]['n']} AUC {e[gi][c]['auc']:.3f}" for c in present(per_ckpt)) for i, e in enumerate(per_ckpt)))
            md.append("")
    Path = __import__("pathlib").Path
    Path(a.out).write_text("\n".join(md)); print("\n".join(md)[:6000])

if __name__ == '__main__':
    main()
