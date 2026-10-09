"""Evaluate a trained exp-4 model: first-answer exact accuracy per protocol and depth, plus model-free baselines.

Every scored query is the first and only query after its lessons (independent rows from the same post-lesson
state), so no worked query example precedes it. Eval episodes come from rng seeds >= 10_000 (training uses
1000 + seed);
namings are random per episode (no overlap check is needed for content; see check_overlap.py for an exact count).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import time
from pathlib import Path

import torch

from common import build, pad_queries, run_protocol, sample_batch
from plastic.config import ModelConfig
from task import Query, baseline_predictions, query_tokens

ARMS = ["fw_reset", "continuous", "no_adapt", "ssm_only", "null_fw_reset", "null_continuous"]
BASELINES = ["oracle", "one_hop", "other_gen", "copy_start", "last_line"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--eval-seed", type=int, default=0)
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--batch", type=int, default=50)
ap.add_argument("--depths", default="1,2,3,4,5,6")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_num_threads(args.threads)

ck = torch.load(args.model, map_location="cpu")
cfg = ModelConfig.from_dict(ck["cfg"])
model = build(cfg)
model.load_state_dict(ck["model"])
model.eval()
depths = [int(d) for d in args.depths.split(",")]
rng = random.Random(10_000 + args.eval_seed)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
rows = []
with torch.no_grad():
    for b in range(args.batches):
        # sample_batch draws random depths; we want one query per depth per episode, so rebuild the queries
        worlds, lessons, _, _, null_worlds, null_lessons = sample_batch(rng, args.batch, 1, [1], null=True)
        qrecs: list[tuple[int, Query]] = []
        for i, w in enumerate(worlds):
            for d in depths:
                x = rng.randrange(w.n)
                word = tuple(rng.randrange(2) for _ in range(d))
                qrecs.append((i, Query(x, word, w.act(x, word))))
        k = len(depths)
        qrows = [query_tokens(worlds[i], q) for i, q in qrecs]
        qx, qpos = pad_queries(qrows)
        preds = {}
        for arm in ARMS:
            L = torch.tensor(null_lessons if arm.startswith("null_") else lessons)
            proto = arm.removeprefix("null_")
            logits = run_protocol(model, L, qx, qpos, k, proto)
            preds[arm] = logits.argmax(-1).tolist()
        for r, (i, q) in enumerate(qrecs):
            w, nw = worlds[i], null_worlds[i]
            ans = w.elem_names[q.answer]
            # the query is asked in TRUE names; the null answer is what the null lessons imply for the same tokens
            start_tok = w.elem_names[q.start]
            ns = nw.elem_names.index(start_tok)
            null_ans = nw.elem_names[nw.act(ns, q.word)]
            base = baseline_predictions(w, lessons[i], q)
            rec = {"eval_seed": args.eval_seed, "batch": b, "episode": b * args.batch + i, "group": w.group, "n": w.n, "depth": q.depth,
                   "answer": ans, "null_answer": null_ans, "is_identity": int(ans == start_tok), "eq_one_hop": int(ans == base["one_hop"]),
                   "eq_last_line": int(ans == base["last_line"])}
            for arm in ARMS:
                rec[f"pred_{arm}"] = preds[arm][r]
            for bl in BASELINES:
                rec[f"pred_{bl}"] = base[bl]
            rows.append(rec)
        if (b + 1) % 5 == 0:
            print(f"batch {b + 1}/{args.batches} {time.time() - t0:.0f}s", flush=True)
with gzip.open(out, "wt", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print(json.dumps({"items": len(rows), "out": str(out), "elapsed_s": round(time.time() - t0, 1)}))
