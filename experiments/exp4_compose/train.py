"""Meta-train a small PlasticCore LM on group-action lessons -> first-query answers.

Each step: B episodes (fresh random group/naming), lessons read with fast-weight writes on, then k independent
first queries per episode from the post-lesson state. With probability p_reset the activation state is zeroed
before the queries (fw_reset protocol: only S carries the lessons), otherwise it is carried (continuous).
Loss = cross-entropy on the answer token only. Training depths are --train-depths; nothing is held out by content
(every episode has a fresh random naming); depth > max(train-depths) is held out by length.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import build, model_config, pad_queries, run_protocol, sample_batch

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--train-depths", default="1,2,3,4")
ap.add_argument("--p-reset", type=float, default=0.5)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--d-model", type=int, default=128)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--log-every", type=int, default=100)
args = ap.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
rng = random.Random(1000 + args.seed)  # training stream; eval uses seeds >= 10_000 (disjoint streams)
depths = [int(d) for d in args.train_depths.split(",")]
cfg = model_config(args.d_model, args.layers)
model = build(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
nparams = sum(p.numel() for p in model.parameters())
print(json.dumps({"params": nparams, "cfg": cfg.to_dict(), "args": vars(args)}), flush=True)


def lr_at(s: int) -> float:
    if s < args.warmup:
        return args.lr * (s + 1) / args.warmup
    t = (s - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


log = []
t0 = time.time()
acc_hist, loss_hist = [], []
dep: dict[int, list] = {}
for step in range(args.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    worlds, lessons, qrows, qrecs, _, _ = sample_batch(rng, args.batch, args.k, depths)
    answers = [worlds[i].elem_names[q.answer] for i, q in qrecs]
    L = torch.tensor(lessons)
    qx, qpos = pad_queries(qrows, answers)
    protocol = "fw_reset" if rng.random() < args.p_reset else "continuous"
    model.train()
    logits = run_protocol(model, L, qx, qpos, args.k, protocol)
    y = torch.tensor(answers)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not math.isfinite(float(loss.detach())):
        raise RuntimeError(f"non-finite loss at step {step}")
    opt.step()
    loss_hist.append(float(loss.detach()))
    acc_hist.append(float((logits.argmax(-1) == y).float().mean()))
    for (i, q), ok in zip(qrecs, (logits.argmax(-1) == y).tolist()):
        dep.setdefault(q.depth, []).append(ok)
    if (step + 1) % args.log_every == 0:
        rec = {"step": step + 1, "loss": sum(loss_hist) / len(loss_hist), "acc": sum(acc_hist) / len(acc_hist),
               "lr": lr_at(step), "gn": float(gn), "elapsed_s": round(time.time() - t0, 1), "acc_by_depth": {d: round(sum(v) / len(v), 3) for d, v in sorted(dep.items())}}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        loss_hist, acc_hist = [], []
        dep = {}
    if (step + 1) % 2000 == 0 or step + 1 == args.steps:
        torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "args": vars(args), "step": step + 1}, out / "model.pt")
(out / "train_log.json").write_text(json.dumps(log, indent=1))
print("done", flush=True)
