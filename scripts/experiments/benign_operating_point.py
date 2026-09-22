"""Measure the harness's benign operating point on a continuous stream.

Feeds a contiguous run of benign held-out chunks through a calibrated ``TransactionRunner``
with the shipped ``HarnessConfig`` defaults and reports what fraction of chunks the harness
intervenes on, split into per-chunk anomaly gates (rollback / scale / project) and the
CUSUM read-only latch. This is the source for the "measured combined per-chunk intervention
rate" and the "first benign latch" figures in
``docs/research/2026-09-22-trained-model-operating-point.md``.

Caveat this script makes explicit: the continuous stream and the CUSUM ``cusum_reference``
both draw from the same validation split, so the result is in-distribution, not a held-out
generalization test.

Run:
    uv run python scripts/experiments/benign_operating_point.py lm_wikitext_l4 \
        --data artifacts/data/wikitext --chunks 200 --device mps
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import torch

from plastic.harness.canary import CanarySuite
from plastic.harness.config import HarnessConfig
from plastic.harness.calibrate import Calibration
from plastic.harness.transaction import TransactionRunner
from plastic.store import ArtifactStore


def measure(model_id: str, *, data_dir: str, artifacts_root: str, n_chunks: int, start: int, device: str) -> dict:
    dev = torch.device(device)
    store = ArtifactStore(artifacts_root)
    cfg, model, _ = store.load_checkpoint(model_id, device=dev)
    if cfg.domain != "text":
        raise SystemExit("benign_operating_point measures the text domain")
    cal = Calibration.load(store.model_dir(model_id))
    suite = CanarySuite.load(store.canary_path(model_id))
    data = np.fromfile(f"{data_dir}/validation.bin", dtype="<u2").astype("int64")
    L = cfg.chunk
    if (start + n_chunks) * L > len(data):
        raise SystemExit("validation.bin too small for the requested chunk range")

    runner = TransactionRunner(model, cfg, HarnessConfig(), calibration=cal, suite=suite, device=dev)
    first_latch = None
    per_chunk_gate = 0
    for c in range(start, start + n_chunks):
        runner.feed_tokens([int(t) for t in data[c * L : (c + 1) * L]], source="user")
        kind = runner.transactions[-1]["decision"]["kind"]
        if kind in ("rollback", "scale", "project"):
            per_chunk_gate += 1
        if kind == "readonly" and first_latch is None:
            first_latch = c - start
    kinds = collections.Counter(t["decision"]["kind"] for t in runner.transactions)
    n = len(runner.transactions)
    return {
        "model_id": model_id,
        "data_dir": data_dir,
        "chunk_size": L,
        "chunk_range": [start, start + n_chunks],
        "device": device,
        "cusum_h": cal.thresholds.get("cusum_h"),
        "cusum_benign_alarm_rate_in_sample": cal.achievable_fpr.get("cusum_h"),
        "in_distribution_note": "the stream and the cusum_reference both draw from the validation split",
        "decision_kinds": dict(kinds),
        "per_chunk_anomaly_gate_rate": per_chunk_gate / n,
        "combined_intervention_rate": (n - kinds.get("commit", 0)) / n,
        "first_readonly_latch_chunk": first_latch,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--data", required=True, help="text corpus dir with validation.bin")
    ap.add_argument("--artifacts-root", default="artifacts")
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--start", type=int, default=0, help="first chunk index into validation.bin")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    result = measure(
        args.model_id, data_dir=args.data, artifacts_root=args.artifacts_root,
        n_chunks=args.chunks, start=args.start, device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
