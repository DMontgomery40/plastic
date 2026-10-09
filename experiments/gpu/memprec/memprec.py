"""Job B: memory-precision scaling for noiseless Gaussian regression (openai/math family #140 idea).

Stream: x_t ~ N(0, I_d), y_t = <x_t, s>, s a uniformly random unit vector, one pass, no noise.
Learners (all share the same x stream per trial: common random numbers):
  delta rule with normalized key (as in the repo: k = x/|x|, v = y/|x|):
      s <- s + beta_t * (v - <k, s>) k
    beta in {0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 1.75}            (fp64 state)
    sigmoid beta: beta_t = sigmoid(w.x_t + b), w ~ N(0, I/d) fixed per trial, b in {0 (repo init), 3}
    beta=1 with fp32 state (precision)
    beta=1 and beta=1.5 with the state quantized after every update to {4, 8, 16} bits
      (per-vector absmax scale, symmetric grid), round-to-nearest and stochastic rounding
  exact chunk least squares (fp64): accumulate G = X^T X, r = X^T y; every `c` samples solve
      s_hat = G^+ r (eigh pseudo-inverse); also with G, r quantized to {8, 16} bits after each chunk.
Metric: angular error angle(s_hat, s); samples-to-eps = first n with angle <= eps,
for eps in 1e-1 .. 1e-6. Writes one JSON per d with per-trial hitting times and median error curves.

python experiments/gpu/memprec/memprec.py --d 64,128,256 --trials 64 --out out/memprec
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

EPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def angle(sh: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """sh: (..., n, d), s: (n, d) unit. atan2(|perp|, par); pi/2 for the zero vector."""
    par = (sh * s).sum(-1)
    perp = (sh - par.unsqueeze(-1) * s).norm(dim=-1)
    ang = torch.atan2(perp, par)
    return torch.where(sh.norm(dim=-1) == 0, torch.full_like(ang, math.pi / 2), ang)


def quantize(x: torch.Tensor, bits: int, mode: str, gen: torch.Generator | None) -> torch.Tensor:
    levels = 2 ** (bits - 1) - 1
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-300) / levels
    z = x / scale
    if mode == "nearest":
        z = torch.round(z)
    else:
        z = torch.floor(z + torch.rand(z.shape, generator=gen, device=z.device, dtype=z.dtype))
    return z.clamp(-levels, levels) * scale


def run_delta(d: int, trials: int, nmax: int, dev, seed: int, curve_pts):
    g = torch.Generator(device=dev).manual_seed(seed)
    gq = torch.Generator(device=dev).manual_seed(seed + 7)
    f64 = torch.float64
    s = torch.randn(trials, d, generator=g, device=dev, dtype=f64)
    s = s / s.norm(dim=-1, keepdim=True)
    w = torch.randn(trials, d, generator=g, device=dev, dtype=f64) / math.sqrt(d)
    methods = []  # (name, kind, param dict)
    for b in [0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 1.75]:
        methods.append((f"delta_beta{b}", dict(beta=b)))
    for b0 in [0.0, 3.0]:
        methods.append((f"delta_sigmoid_b{b0:g}", dict(sig=b0)))
    for b in [1.0, 1.5]:
        for bits in [4, 8, 16]:
            for mode in ["nearest", "stochastic"]:
                methods.append((f"delta_beta{b}_q{bits}_{mode}", dict(beta=b, bits=bits, mode=mode)))
    M = len(methods)
    S = torch.zeros(M, trials, d, device=dev, dtype=f64)
    S32 = torch.zeros(trials, d, device=dev, dtype=torch.float32)
    names = [m[0] for m in methods] + ["delta_beta1.0_fp32"]
    const_beta = torch.tensor([m[1].get("beta", 0.0) for m in methods], device=dev, dtype=f64).view(M, 1, 1)
    sig_rows = [i for i, m in enumerate(methods) if "sig" in m[1]]
    sig_b0 = torch.tensor([methods[i][1]["sig"] for i in sig_rows], device=dev, dtype=f64).view(-1, 1)
    q_groups = {}
    for i, m in enumerate(methods):
        if "bits" in m[1]:
            q_groups.setdefault((m[1]["bits"], m[1]["mode"]), []).append(i)
    q_groups = {k: torch.tensor(v, device=dev) for k, v in q_groups.items()}
    eps_t = torch.tensor(EPS, device=dev, dtype=f64)
    hit = torch.full((M + 1, trials, len(EPS)), -1, dtype=torch.long, device=dev)
    curve_set = set(curve_pts)
    curves = {}
    beta_mean_sig = torch.zeros(len(sig_rows), device=dev, dtype=f64)
    for n in range(1, nmax + 1):
        x = torch.randn(trials, d, generator=g, device=dev, dtype=f64)
        xn = x.norm(dim=-1, keepdim=True)
        k = x / xn
        v = (x * s).sum(-1, keepdim=True) / xn  # (trials, 1)
        beta = const_beta.expand(M, trials, 1).clone()
        bs = torch.sigmoid((x * w).sum(-1).unsqueeze(0) + sig_b0)  # (nsig, trials)
        beta[sig_rows] = bs.unsqueeze(-1)
        beta_mean_sig += bs.mean(-1)
        err = v.unsqueeze(0) - (S * k).sum(-1, keepdim=True)  # (M, trials, 1)
        S = S + beta * err * k
        for (bits, mode), idx in q_groups.items():
            S[idx] = quantize(S[idx], bits, mode, gq)
        k32, v32 = k.float(), v.float()
        S32 = S32 + (v32 - (S32 * k32).sum(-1, keepdim=True)) * k32
        ang = torch.cat([angle(S, s), angle(S32.double(), s).unsqueeze(0)], 0)  # (M+1, trials)
        newly = (ang.unsqueeze(-1) <= eps_t) & (hit < 0)
        hit = torch.where(newly, torch.full_like(hit, n), hit)
        if n in curve_set:
            curves[n] = ang.median(dim=1).values.tolist()
    out = {"methods": names, "hit": hit.cpu().tolist(), "nmax": nmax,
           "curve_n": sorted(curves), "curve_median_angle": [curves[n] for n in sorted(curves)],
           "sigmoid_beta_mean": dict(zip([methods[i][0] for i in sig_rows], (beta_mean_sig / nmax).tolist()))}
    return out


def run_ls(d: int, trials: int, nmax: int, chunk: int, dev, seed: int):
    g = torch.Generator(device=dev).manual_seed(seed)  # same seed -> same s and same x stream as run_delta
    f64 = torch.float64
    s = torch.randn(trials, d, generator=g, device=dev, dtype=f64)
    s = s / s.norm(dim=-1, keepdim=True)
    _w = torch.randn(trials, d, generator=g, device=dev, dtype=f64)  # consume the same draws as run_delta
    variants = [("ls_exact", None), ("ls_q16_nearest", 16), ("ls_q8_nearest", 8)]
    V = len(variants)
    G = torch.zeros(V, trials, d, d, device=dev, dtype=f64)
    r = torch.zeros(V, trials, d, device=dev, dtype=f64)
    eps_t = torch.tensor(EPS, device=dev, dtype=f64)
    hit = torch.full((V, trials, len(EPS)), -1, dtype=torch.long, device=dev)
    curve = {}
    xs = []
    for n in range(1, nmax + 1):
        xs.append(torch.randn(trials, d, generator=g, device=dev, dtype=f64))
        if n % chunk and n != nmax:
            continue
        X = torch.stack(xs, 1)  # (trials, c, d)
        xs = []
        y = (X * s.unsqueeze(1)).sum(-1)
        G = G + torch.einsum("tcd,tce->tde", X, X).unsqueeze(0)
        r = r + torch.einsum("tcd,tc->td", X, y).unsqueeze(0)
        for vi, (_, bits) in enumerate(variants):
            if bits is not None:
                G[vi] = quantize(G[vi].reshape(trials, -1), bits, "nearest", None).reshape(trials, d, d)
                r[vi] = quantize(r[vi], bits, "nearest", None)
        lam, U = torch.linalg.eigh(G)
        tol = lam.amax(-1, keepdim=True) * d * 1e-13
        inv = torch.where(lam > tol, 1.0 / lam.clamp_min(1e-300), torch.zeros_like(lam))
        sh = torch.einsum("vtde,vte,vtfe,vtf->vtd", U, inv, U, r)
        ang = angle(sh, s)
        newly = (ang.unsqueeze(-1) <= eps_t) & (hit < 0)
        hit = torch.where(newly, torch.full_like(hit, n), hit)
        curve[n] = ang.median(dim=1).values.tolist()
    return {"methods": [v[0] for v in variants], "hit": hit.cpu().tolist(), "nmax": nmax, "chunk": chunk,
            "curve_n": sorted(curve), "curve_median_angle": [curve[n] for n in sorted(curve)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="64,128,256")
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--nmax-mult", type=int, default=80, help="delta stream length = mult * d")
    ap.add_argument("--ls-mult", type=int, default=3, help="LS stream length = mult * d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="out/memprec")
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    for d in [int(x) for x in a.d.split(",")]:
        t0 = time.time()
        nmax = a.nmax_mult * d
        curve_pts = sorted({int(round(v)) for v in torch.logspace(0, math.log10(nmax), 200).tolist()})
        res = {"d": d, "trials": a.trials, "eps": EPS, "seed": a.seed + d, "device": str(dev),
               "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "torch": torch.__version__}
        res["delta"] = run_delta(d, a.trials, nmax, dev, a.seed + d, curve_pts)
        res["ls"] = run_ls(d, a.trials, a.ls_mult * d, max(1, d // 16), dev, a.seed + d)
        res["time_s"] = time.time() - t0
        p = os.path.join(a.out, f"memprec_d{d}.json")
        json.dump(res, open(p + ".tmp", "w"))
        os.replace(p + ".tmp", p)
        print(f"d={d} done in {res['time_s']:.1f}s on {res['gpu'] or dev}", flush=True)


if __name__ == "__main__":
    main()
