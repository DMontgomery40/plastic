"""Exp 3(b) step 1: re-run the calibration stream of `plastic calibrate` (validation.bin, sessions of 4 chunks
starting after the canary region, log-only runner with the shipped Fisher) and SAVE the raw per-chunk signal
records and the continuous-session CUSUM values, so thresholds can be recomputed at any target FPR with the
repo's own thresholds_from_records / calibrated_cusum_h. Mirrors calibrate_model + calibrate_from_runner.

python experiments/exp3_invariants/calib_records.py --suite shipped    --out experiments/results/exp3/calib_records_shipped.json
python experiments/exp3_invariants/calib_records.py --suite statements --out experiments/results/exp3/calib_records_statements.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, load  # noqa: E402

from plastic.data.text import TokenWindows  # noqa: E402
from plastic.harness.calibrate import Calibration, _continuous_cusum_vals, log_only  # noqa: E402
from plastic.harness.config import HarnessConfig  # noqa: E402
from plastic.harness.transaction import TransactionRunner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="shipped")
ap.add_argument("--n-chunks", type=int, default=512)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)

store, cfg, model, cal, suite = load(a.suite)
L = cfg.chunk
windows = TokenWindows(os.path.join(DATA, "validation.bin"), seq_len=L * 4)
canary_tokens = 3 * 128


def stream():
    start = canary_tokens
    while True:
        if start + L * 4 + 1 > len(windows):
            start = canary_tokens
        yield torch.from_numpy(windows.data[start:start + L * 4].astype("int64")).tolist()
        start += L * 4


hcfg = log_only(HarnessConfig(target_fpr=0.01))
runner = TransactionRunner(model, cfg, hcfg, calibration=Calibration(fisher=cal.fisher), suite=suite, device="cpu")
t0 = time.time()
records, since = [], 0
it = iter(stream())
for item in it:  # calibrate_from_runner(..., reset_every=4)
    runner.feed_tokens(list(item), source="user")
    while runner.transactions:
        records.append(runner.transactions.pop(0)["signals"]); since += 1
    if since >= 4:
        runner.flush()
        while runner.transactions:
            records.append(runner.transactions.pop(0)["signals"])
        runner.reset(); since = 0
    if len(records) >= a.n_chunks:
        break
    if len(records) % 64 == 0:
        print(f"records {len(records)} ({time.time()-t0:.0f}s)", flush=True)
runner.flush()
while runner.transactions:
    records.append(runner.transactions.pop(0)["signals"])
cont = _continuous_cusum_vals(runner, it, n=min(len(records), 256))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({"suite": a.suite, "n_records": len(records), "records": records, "cusum_continuous": cont,
           "suite_sizes": {"coherence": len(suite.coherence), "poison": len(suite.poison)},
           "seconds": time.time() - t0}, open(a.out, "w"))
print("done", len(records), len(cont), time.time() - t0)
