"""Exp 4b control arms on CPU, re-generating the exact main-eval items (same rng stream as evaluate.py) from a checkpoint.

Re-computed with the model: fw_reset (must reproduce the GPU CSV), no_adapt (beta=0), null_fw_reset (deranged names),
and a NEW foreign-lessons null (lessons of the neighbouring episode in the batch: same format and size, wrong content).
Model-free, from the world tables: expected accuracy of 'successor-set prior' shortcuts that know every lesson line but do not
compose the word (uniform guess inside the set of elements reachable given partial information):
  p_set_x      answer in R_d(x)               (all d-step words from x; knows only depth)
  p_set_first  answer in R_{d-1}(x*w1)        (knows the first hop)
  p_set_last   answer in R_{d-1}(x)*w_d       (knows the last generator)
Usage: python controls_4b.py --model M --eval-seed S --out items_controls.csv.gz [--ref gpu_items.csv.gz]"""
from __future__ import annotations
import argparse, csv, gzip, json, random, time
from pathlib import Path
import torch
from common import build, pad_queries, run_protocol, sample_batch
from plastic.config import ModelConfig
from task import Query, baseline_predictions, query_tokens

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--eval-seed", type=int, default=0)
ap.add_argument("--batches", type=int, default=40); ap.add_argument("--batch", type=int, default=50)
ap.add_argument("--depths", default="1,2,3,4,5,6"); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--layout", default=None); ap.add_argument("--out", required=True)
ap.add_argument("--ref", default=None, help="GPU item CSV to verify against (answers and fw_reset must agree)")
args = ap.parse_args()
torch.set_num_threads(args.threads)
ck = torch.load(args.model, map_location="cpu")
cfg = ModelConfig.from_dict(ck["cfg"]); model = build(cfg); model.load_state_dict(ck["model"]); model.eval()
layout = args.layout or ck["args"].get("layout", "old")
depths = [int(d) for d in args.depths.split(",")]
rng = random.Random(10_000 + args.eval_seed)
ARMS = ["fw_reset", "no_adapt", "null_fw_reset", "foreign_fw_reset"]


def reach(w, xs, d):
    s = set(xs)
    for _ in range(d):
        s = {w.table[x][j] for x in s for j in range(2)}
    return s


rows = []; t0 = time.time()
with torch.no_grad():
    for b in range(args.batches):
        worlds, lessons, _, _, null_worlds, null_lessons = sample_batch(rng, args.batch, 1, [1], null=True, layout=layout)
        qrecs = []
        for i, w in enumerate(worlds):
            for d in depths:
                x = rng.randrange(w.n); word = tuple(rng.randrange(2) for _ in range(d))
                qrecs.append((i, Query(x, word, w.act(x, word))))
        k = len(depths)
        qx, qpos = pad_queries([query_tokens(worlds[i], q, layout) for i, q in qrecs])
        foreign = [lessons[(i + 1) % len(lessons)] for i in range(len(lessons))]
        preds = {}
        for arm in ARMS:
            L = torch.tensor({"null_fw_reset": null_lessons, "foreign_fw_reset": foreign}.get(arm, lessons))
            proto = {"fw_reset": "fw_reset", "no_adapt": "no_adapt", "null_fw_reset": "fw_reset", "foreign_fw_reset": "fw_reset"}[arm]
            preds[arm] = run_protocol(model, L, qx, qpos, k, proto).argmax(-1).tolist()
        for r, (i, q) in enumerate(qrecs):
            w, nw = worlds[i], null_worlds[i]
            ans = w.elem_names[q.answer]; start_tok = w.elem_names[q.start]
            null_ans = nw.elem_names[nw.act(nw.elem_names.index(start_tok), q.word)]
            fw = foreign[i]  # lessons of episode i+1; its own implied answer for these tokens is undefined (different names) -> not reported
            base = baseline_predictions(w, lessons[i], q); d = q.depth
            rx = reach(w, [q.start], d); y1 = w.table[q.start][q.word[0]]
            rf = reach(w, [y1], d - 1); rl = {w.table[z][q.word[-1]] for z in reach(w, [q.start], d - 1)}
            rec = {"eval_seed": args.eval_seed, "episode": b * args.batch + i, "group": w.group, "n": w.n, "depth": d, "answer": ans,
                   "null_answer": null_ans, "is_identity": int(ans == start_tok), "p_set_x": 1 / len(rx), "p_set_first": 1 / len(rf), "p_set_last": 1 / len(rl)}
            for a in ARMS: rec[f"pred_{a}"] = preds[a][r]
            for bl in ("oracle", "one_hop", "other_gen", "copy_start", "last_line"): rec[f"pred_{bl}"] = base[bl]
            rows.append(rec)
        if (b + 1) % 5 == 0: print(f"batch {b + 1}/{args.batches} {time.time() - t0:.0f}s", flush=True)
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(out, "wt", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
res = {"items": len(rows), "elapsed_s": round(time.time() - t0, 1)}
if args.ref:
    import pandas as pd
    ref = pd.read_csv(args.ref); cur = pd.DataFrame(rows)
    res["answers_match_ref"] = bool((ref.answer.values == cur.answer.values).all())
    res["null_answers_match_ref"] = bool((ref.null_answer.values == cur.null_answer.values).all())
    for a in ("fw_reset", "no_adapt", "null_fw_reset"):
        res[f"{a}_pred_agreement_with_gpu"] = float((ref["pred_" + a].values == cur["pred_" + a].values).mean())
print(json.dumps(res))
