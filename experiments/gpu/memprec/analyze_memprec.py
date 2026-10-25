"""Analyze Job B -> experiments/gpu/results/memprec/{hitting_times.csv, fits.csv, summary.md, *.png}.
python experiments/gpu/memprec/analyze_memprec.py [RAW_DIR]   (default experiments/gpu/results/memprec/raw)"""
import csv, glob, json, math, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "experiments/gpu/results/memprec/raw"
O = "experiments/gpu/results/memprec"
os.makedirs(O, exist_ok=True)
data = {}
for f in sorted(glob.glob(f"{R}/memprec_d*.json")):
    d = json.load(open(f))
    data[d["d"]] = d
ds = sorted(data)
EPS = data[ds[0]]["eps"]
rows = []  # method, d, eps, frac_reached, median, q25, q75, mean
methods = []
for d in ds:
    for part in ["delta", "ls"]:
        P = data[d][part]
        for i, m in enumerate(P["methods"]):
            if m not in methods:
                methods.append(m)
            H = np.array(P["hit"][i], dtype=float)  # (trials, eps)
            for e_i, eps in enumerate(EPS):
                h = H[:, e_i]
                ok = h > 0
                frac = ok.mean()
                hh = np.where(ok, h, np.inf)
                med = float(np.median(hh)) if frac >= 0.5 else float("nan")
                q25, q75 = (float(np.quantile(hh, 0.25)), float(np.quantile(hh, 0.75))) if frac >= 0.75 else (float("nan"),) * 2
                rows.append(dict(method=m, d=d, eps=eps, frac_reached=frac, median=med, q25=q25, q75=q75,
                                 mean=float(h[ok].mean()) if frac == 1 else float("nan"), nmax=P["nmax"]))
with open(f"{O}/hitting_times.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def theory_slope(m):
    mb = re.fullmatch(r"delta_beta([0-9.]+)", m)
    if mb:
        b = float(mb.group(1)); return 2.0 / (2 * b - b * b)
    return None


fits = []
for m in methods:
    rr = [r for r in rows if r["method"] == m and r["frac_reached"] == 1.0]
    if len(rr) < 3:
        fits.append(dict(method=m, n_points=len(rr), slope_origin=float("nan"), r2_origin=float("nan"), slope_affine=float("nan"),
                         intercept_per_d=float("nan"), r2_affine=float("nan"), theory_slope=theory_slope(m)))
        continue
    X = np.array([r["d"] * math.log(1 / r["eps"]) for r in rr]); Y = np.array([r["median"] for r in rr])
    D = np.array([r["d"] for r in rr], dtype=float)
    a0 = float((X @ Y) / (X @ X)); r2o = 1 - ((Y - a0 * X) ** 2).sum() / ((Y - Y.mean()) ** 2).sum() if Y.std() > 0 else float("nan")
    A = np.stack([X, D], 1); coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    r2a = 1 - ((Y - A @ coef) ** 2).sum() / ((Y - Y.mean()) ** 2).sum() if Y.std() > 0 else float("nan")
    fits.append(dict(method=m, n_points=len(rr), slope_origin=a0, r2_origin=float(r2o), slope_affine=float(coef[0]),
                     intercept_per_d=float(coef[1]), r2_affine=float(r2a), theory_slope=theory_slope(m)))
with open(f"{O}/fits.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(fits[0])); w.writeheader(); w.writerows(fits)


def cell(m, d, eps):
    r = next(r for r in rows if r["method"] == m and r["d"] == d and r["eps"] == eps)
    if r["frac_reached"] == 1.0:
        return f"{r['median']:.0f}"
    if r["frac_reached"] >= 0.5:
        return f"{r['median']:.0f} ({r['frac_reached']:.0%})"
    return f"– ({r['frac_reached']:.0%})"


md = ["# Job B: memory-precision scaling, noiseless Gaussian regression", "",
      f"Trials per d: {data[ds[0]]['trials']}; devices: {sorted({str(data[d]['gpu']) for d in ds})}; "
      f"delta stream length {data[ds[0]]['delta']['nmax'] // ds[0]}·d; LS stream {data[ds[0]]['ls']['nmax'] // ds[0]}·d, solved every d/16 samples.", "",
      "Median samples to reach angular error ≤ eps (over trials). '(x%)' = fraction of trials that reached it within the stream; "
      "'–' = fewer than half reached it.", ""]
for d in ds:
    md += [f"## d = {d}", "", "| method | " + " | ".join(f"eps={e:g}" for e in EPS) + " | median final angle |",
           "|---|" + "---|" * (len(EPS) + 1)]
    for part in ["delta", "ls"]:
        P = data[d][part]
        for i, m in enumerate(P["methods"]):
            md.append(f"| {m} | " + " | ".join(cell(m, d, e) for e in EPS) + f" | {P['curve_median_angle'][-1][i]:.2e} |")
    md.append("")
md += ["## Fit: median samples-to-eps vs d·ln(1/eps) (only (d, eps) cells reached by every trial)", "",
       "slope_origin: n ≈ a·d·ln(1/eps). affine: n ≈ a·d·ln(1/eps) + c·d. theory (normalized-key delta rule, "
       "E|e|² contracts by 1-(2β-β²)/d per sample): a = 2/(2β-β²).", "",
       "| method | points | a (origin) | R² | a (affine) | c (per d) | R² | theory a |", "|---|---|---|---|---|---|---|---|"]
for f_ in fits:
    th = f"{f_['theory_slope']:.3f}" if f_["theory_slope"] else ""
    md.append(f"| {f_['method']} | {f_['n_points']} | {f_['slope_origin']:.3f} | {f_['r2_origin']:.4f} | {f_['slope_affine']:.3f} | "
              f"{f_['intercept_per_d']:.3f} | {f_['r2_affine']:.4f} | {th} |")
md += ["", "Sigmoid-gate realised mean beta: " + "; ".join(f"d={d}: " + ", ".join(f"{k}={v:.3f}" for k, v in data[d]['delta']['sigmoid_beta_mean'].items()) for d in ds)]
open(f"{O}/summary.md", "w").write("\n".join(md) + "\n")

# plots
fig, axes = plt.subplots(1, len(ds), figsize=(4.6 * len(ds), 4), squeeze=False)
show = ["delta_beta0.25", "delta_beta0.5", "delta_beta1.0", "delta_beta1.5", "delta_beta1.75", "delta_sigmoid_b0", "delta_sigmoid_b3",
        "delta_beta1.0_fp32", "delta_beta1.0_q16_nearest", "delta_beta1.0_q8_nearest", "ls_exact", "ls_q16_nearest"]
for ax, d in zip(axes[0], ds):
    for m in show:
        rr = [r for r in rows if r["method"] == m and r["d"] == d and not math.isnan(r["median"])]
        if rr:
            ax.plot([math.log(1 / r["eps"]) for r in rr], [r["median"] / d for r in rr], marker="o", label=m, lw=1)
    xs = np.array([math.log(1 / e) for e in EPS]); ax.plot(xs, 2 * xs, "k:", lw=1, label="2·ln(1/eps) (theory β=1)")
    ax.set_title(f"d={d}"); ax.set_xlabel("ln(1/eps)"); ax.set_ylabel("median samples / d")
axes[0][0].legend(fontsize=6)
fig.suptitle("Job B: samples-to-eps / d vs ln(1/eps)"); fig.tight_layout(); fig.savefig(f"{O}/samples_vs_log_eps.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, len(ds), figsize=(4.6 * len(ds), 4), squeeze=False)
for ax, d in zip(axes[0], ds):
    P = data[d]["delta"]; n = np.array(P["curve_n"]); C = np.array(P["curve_median_angle"])
    for m in ["delta_beta0.5", "delta_beta1.0", "delta_beta1.5", "delta_beta1.0_fp32", "delta_beta1.0_q16_nearest",
              "delta_beta1.0_q16_stochastic", "delta_beta1.0_q8_nearest", "delta_beta1.0_q4_nearest"]:
        ax.plot(n / d, C[:, P["methods"].index(m)], label=m, lw=1)
    L = data[d]["ls"]; ax.plot(np.array(L["curve_n"]) / d, np.array(L["curve_median_angle"])[:, 0], "k--", label="ls_exact", lw=1)
    ax.plot(np.array(L["curve_n"]) / d, np.array(L["curve_median_angle"])[:, 1], "k:", label="ls_q16_nearest", lw=1)
    ax.set_yscale("log"); ax.set_ylim(1e-17, 3); ax.set_xlabel("samples / d"); ax.set_ylabel("median angular error"); ax.set_title(f"d={d}")
axes[0][0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{O}/error_curves.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.6))
bs = sorted({float(re.fullmatch(r"delta_beta([0-9.]+)", f_["method"]).group(1)) for f_ in fits if re.fullmatch(r"delta_beta([0-9.]+)", f_["method"])})
ax.plot(bs, [next(f_["slope_origin"] for f_ in fits if f_["method"] == f"delta_beta{b}") for b in bs], "o-", label="measured a (origin fit)")
bb = np.linspace(0.2, 1.8, 100); ax.plot(bb, 2 / (2 * bb - bb ** 2), "k:", label="theory 2/(2β-β²)")
ax.set_xlabel("β"); ax.set_ylabel("a in n ≈ a·d·ln(1/eps)"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{O}/slope_vs_beta.png", dpi=130)
print("\n".join(md))
