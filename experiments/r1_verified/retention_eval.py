"""Retention eval / KILL GATE (R1). fw_reset protocol: lesson -> K distractor lines (writes ON) -> activations zeroed -> queries from S only.
Depth-1 recall pooled over all 2n lesson lines per episode; depth-2 recall on 8 random depth-2 queries per episode.
Reports pooled recall with Wilson CI (items) and episode-cluster bootstrap CI, per (ckpt, kind, K); control: no_adapt (lesson read frozen)
and content-null (derangement) scored against the TRUE answers.  Pass logic is the pre-registered one in /workspace/r1/PROGRESS.md."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from distract import distractor_tokens
from common import build, pad_queries, expand_state, SIZE_CLASSES  # noqa: E402
from task import make_world, lesson_tokens, query_tokens, Query, relabel  # noqa: E402
from plastic.config import ModelConfig  # noqa: E402
from gate_analyze import wilson  # noqa: E402  (importing runs main(); guarded below)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--Ks", default="0,16,64,128"); ap.add_argument("--kinds", default="struct,randline,randiid")
ap.add_argument("--arm", default="seg"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--batch", type=int, default=50)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--controls", action="store_true", help="also no_adapt and content-null at K=64 (struct)")
ap.add_argument("--marker-ablation", action="store_true", help="also arm=none (no marker) at K=64 struct")
a = ap.parse_args(); torch.set_num_threads(a.threads)
ck = torch.load(a.model, map_location="cpu"); cfg = ModelConfig.from_dict(ck["cfg"]); model = build(cfg); model.load_state_dict(ck["model"]); model.eval()
out = Path(a.out); res = json.loads(out.read_text()) if out.exists() else {"model": a.model, "step": ck.get("step"), "cells": {}}


def cell(kind, K, arm, mode="normal"):
    rng = random.Random(30000 + 101 * a.seed + K * 7 + {"struct": 1, "randline": 2, "randiid": 3}[kind] + (1000 if mode != "normal" else 0))
    d1, d2, per_ep = [], [], []
    done = 0
    while done < a.n:
        B = min(a.batch, a.n - done); n = rng.choice(sorted(SIZE_CLASSES))
        worlds = [make_world(rng, rng.choice(SIZE_CLASSES[n])) for _ in range(B)]
        wl = [relabel(w, rng) for w in worlds] if mode == "null" else worlds
        texts = [lesson_tokens(w, rng) + distractor_tokens(kind, K, rng, arm) for w in wl]
        with torch.no_grad():
            _, st, _ = model(torch.tensor(texts), model.init_state(B), freeze=(mode == "no_adapt"))
            qs = []
            for w in worlds:  # queries always scored against the TRUE world
                q = [Query(x, (g,), w.table[x][g]) for x in range(w.n) for g in range(2)]
                for _ in range(8):
                    x = rng.randrange(w.n); wd = tuple(rng.randrange(2) for _ in range(2)); q.append(Query(x, wd, w.act(x, wd)))
                qs.append(q)
            k = len(qs[0]); rows = [query_tokens(w, q, "aligned") for w, qq in zip(worlds, qs) for q in qq]
            qx, qpos = pad_queries(rows)
            lg, _, _ = model(qx, expand_state(st, k, reset_activations=True), freeze=False)
            pred = lg[torch.arange(qx.shape[0]), qpos].argmax(-1).view(B, k)
        for i, w in enumerate(worlds):
            ok = [int(pred[i, j]) == w.elem_names[qs[i][j].answer] for j in range(k)]; n1 = 2 * w.n
            per_ep.append((sum(ok[:n1]), n1, sum(ok[n1:]), len(ok) - n1))
        done += B
    arr = np.array(per_ep, float); r = {}
    for nm, (c, t) in {"d1": (0, 1), "d2": (2, 3)}.items():
        k_, n_ = arr[:, c].sum(), arr[:, t].sum(); lo, hi = wilson(int(k_), int(n_))
        bs = []; g = np.random.default_rng(0)
        for _ in range(500):
            ix = g.integers(0, len(arr), len(arr)); bs.append(arr[ix, c].sum() / arr[ix, t].sum())
        r[nm] = {"recall": k_ / n_, "k": int(k_), "n": int(n_), "wilson": [lo, hi], "cluster_boot": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    r["episodes"] = len(arr); return r

t0 = time.time()
todo = [(kind, K, a.arm, "normal") for kind in a.kinds.split(",") for K in [int(x) for x in a.Ks.split(",")] if not (K == 0 and kind != a.kinds.split(",")[0])]
if a.controls: todo += [("struct", 64, a.arm, "no_adapt"), ("struct", 64, a.arm, "null")]
if a.marker_ablation: todo += [("struct", 64, "none", "normal"), ("struct", 16, "none", "normal")]
for kind, K, arm, mode in todo:
    key = f"{mode}|{arm}|{kind}|{K}"
    if key in res["cells"]: continue
    res["cells"][key] = cell(kind, K, arm, mode); out.write_text(json.dumps(res))
    c = res["cells"][key]["d1"]; print(key, f"d1 {c['recall']:.3f} W[{c['wilson'][0]:.3f},{c['wilson'][1]:.3f}] d2 {res['cells'][key]['d2']['recall']:.3f} {time.time()-t0:.0f}s", flush=True)
