"""Exp 3 follow-up to the I5 counterexample on the trained lm_small_cpu: force every chunk of clean test-split
sessions to `scale` (the policy's default scale_factor 0.25, and 0.5) and compare the committed S change with the
unscaled proposal the policy judged. No budgets (HarnessConfig defaults), so nothing re-caps the scaled candidate.

python experiments/exp3_invariants/scale_real.py --out experiments/results/exp3/scale_real.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, load  # noqa: E402

import plastic.harness.transaction as txn  # noqa: E402
from plastic.harness.config import HarnessConfig  # noqa: E402
from plastic.harness.policy import Decision  # noqa: E402
from plastic.harness.signals import delta_norms  # noqa: E402
from plastic.harness.transaction import TransactionRunner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", type=int, default=6)
ap.add_argument("--chunks", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
store, cfg, model, cal, suite = load("shipped")
L = cfg.chunk
test = np.memmap(os.path.join(DATA, "test.bin"), dtype="<u2", mode="r")
rng = np.random.default_rng(4242)
rows = []
orig = txn.decide
for scale in (0.25, 0.5):
    txn.decide = lambda sig, hc, th, read_only, s=scale: Decision("scale", ["forced_scale"], scale=s)
    for sess in range(a.sessions):
        st = int(rng.integers(0, len(test) - a.chunks * L - 1))
        r = TransactionRunner(model, cfg, HarnessConfig(enable_projection=False), calibration=cal, suite=None, device="cpu")
        for i in range(a.chunks):
            pre = r.backend.clone(r.committed)
            r.feed_tokens([int(t) for t in test[st + i * L: st + (i + 1) * L]])
            t = r.transactions[-1]
            acc = delta_norms([x.S - y.S for x, y in zip(r.committed.layers, pre.layers)])[0]
            rows.append({"scale": scale, "session": sess, "chunk": i, "applied": t["decision"]["kind"],
                         "proposal": t["signals"]["delta_norm"], "accepted": acc})
txn.decide = orig
json.dump(rows, open(a.out, "w"), indent=1)
for scale in (0.25, 0.5):
    rr = np.array([x["accepted"] / x["proposal"] for x in rows if x["scale"] == scale and x["proposal"] > 0])  # readonly chunks (CUSUM freeze) propose 0
    n_ro = sum(1 for x in rows if x["scale"] == scale and x["applied"] != "scale")
    print(f"scale={scale}: non-scale(readonly) chunks={n_ro}; n={len(rr)} accepted/proposal median={np.median(rr):.3f} min={rr.min():.3f} max={rr.max():.3f} "
          f"frac>1={np.mean(rr > 1):.3f}")
