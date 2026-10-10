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
ap.add_argument("--curriculum", default="none", choices=["none", "adaptive"],
                help="adaptive: start with depth 1 only; unlock the next training depth when the last 100 steps reach "
                     "--unlock-acc on the hardest unlocked depth")
ap.add_argument("--unlock-acc", type=float, default=0.9)
ap.add_argument("--p-reset", type=float, default=0.5)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--d-model", type=int, default=128)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--log-every", type=int, default=100)
# --- exp 4b additions
ap.add_argument("--layout", default="old", choices=["old", "aligned"], help="query layout (see task.query_tokens)")
ap.add_argument("--device", default="cpu")
ap.add_argument("--min-depth1-steps", type=int, default=0, help="depth 2 cannot unlock before this many steps")
ap.add_argument("--unlock-acc1", type=float, default=None, help="unlock threshold for leaving depth 1 (default: --unlock-acc)")
ap.add_argument("--lr-sched", default="cosine", choices=["cosine", "const_decay"],
                help="const_decay: constant lr after warmup, cosine decay to 0.1x over the last --decay-frac of --steps (the phases of a "
                     "curriculum have no fixed length, so a whole-run cosine would have decayed before depth 2 unlocks)")
ap.add_argument("--decay-frac", type=float, default=0.15)
ap.add_argument("--abort-depth1-acc", type=float, default=0.0,
                help="budget guard: if depth 1 is still below this train accuracy (100-step mean) at --abort-depth1-step, stop (null result)")
ap.add_argument("--abort-depth1-step", type=int, default=0)
ap.add_argument("--time-limit-s", type=float, default=1e9, help="stop gracefully (saving) after this wall time")
# --- exp 4c additions: warm start
ap.add_argument("--init-from", default=None, help="checkpoint (model.pt) to warm-start from: weights are loaded, the optimizer is fresh, and the depth "
                "curriculum resumes with the depths that checkpoint had unlocked (its unlock_steps keys)")
ap.add_argument("--rng-offset", type=int, default=0, help="added to the training rng seed so a warm start does not replay the first run's episode stream")
ap.add_argument("--time-decay", action="store_true", help="with --lr-sched const_decay and --time-limit-s: decay by elapsed wall time instead of step "
                "(the run length is then set by the time budget, so the decay always happens before the stop)")
args = ap.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
rng = random.Random(1000 + args.seed + args.rng_offset)  # training stream; eval uses seeds >= 10_000 (disjoint streams)
depths = [int(d) for d in args.train_depths.split(",")]
cfg = model_config(args.d_model, args.layers)
model = build(cfg).to(args.device)
dev = torch.device(args.device)
init_info = None
if args.init_from:
    ck = torch.load(args.init_from, map_location="cpu")
    assert ck["cfg"] == cfg.to_dict(), "checkpoint config differs from the requested model config"
    model.load_state_dict(ck["model"])
    init_info = {"init_from": args.init_from, "init_step": ck.get("step"), "init_unlock_steps": ck.get("unlock_steps")}
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
nparams = sum(p.numel() for p in model.parameters())
print(json.dumps({"params": nparams, "cfg": cfg.to_dict(), "args": vars(args)}), flush=True)


def lr_at(s: int) -> float:
    if s < args.warmup:
        return args.lr * (s + 1) / args.warmup
    if args.lr_sched == "const_decay":
        if args.time_decay:
            f = (time.time() - t0) / args.time_limit_s
            if f < 1 - args.decay_frac:
                return args.lr
            t = min(1.0, (f - (1 - args.decay_frac)) / args.decay_frac)
            return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))
        t0_ = args.steps * (1 - args.decay_frac)
        if s < t0_:
            return args.lr
        t = min(1.0, (s - t0_) / max(1, args.steps - t0_))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))
    t = (s - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


log = []
t0 = time.time()
acc_hist, loss_hist = [], []
dep: dict[int, list] = {}
unlocked = 1 if args.curriculum == "adaptive" else len(depths)
if init_info and args.curriculum == "adaptive":
    unlocked = max(1, min(len(depths), len(init_info["init_unlock_steps"] or {1: 0})))  # resume at the depth the checkpoint had reached
unlock_steps = {depths[unlocked - 1]: 0}
if init_info:
    print(json.dumps({"warm_start": init_info, "resumed_unlocked_depths": depths[:unlocked]}), flush=True)
for step in range(args.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    worlds, lessons, qrows, qrecs, _, _ = sample_batch(rng, args.batch, args.k, depths[:unlocked], layout=args.layout)
    answers = [worlds[i].elem_names[q.answer] for i, q in qrecs]
    L = torch.tensor(lessons, device=dev)
    qx, qpos = pad_queries(qrows, answers)
    qx, qpos = qx.to(dev), qpos.to(dev)
    protocol = "fw_reset" if rng.random() < args.p_reset else "continuous"
    model.train()
    logits = run_protocol(model, L, qx, qpos, args.k, protocol)
    y = torch.tensor(answers, device=dev)
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
        rec["unlocked_depths"] = depths[:unlocked]
        if init_info:
            rec["init_step"] = init_info["init_step"]
        hardest = depths[unlocked - 1]
        thr = args.unlock_acc if (unlocked > 1 or args.unlock_acc1 is None) else args.unlock_acc1
        can = unlocked > 1 or step + 1 >= args.min_depth1_steps
        if unlocked < len(depths) and can and hardest in dep and sum(dep[hardest]) / len(dep[hardest]) >= thr:
            unlocked += 1
            unlock_steps[depths[unlocked - 1]] = step + 1
        rec["unlock_steps"] = dict(unlock_steps)
        log.append(rec)
        print(json.dumps(rec), flush=True)
        (out / "train_log.json").write_text(json.dumps(log))  # progress on disk at every log point
        stop = None
        if args.abort_depth1_step and step + 1 >= args.abort_depth1_step and unlocked == 1 and rec["acc"] < args.abort_depth1_acc:
            stop = f"abort: depth-1 acc {rec['acc']:.3f} < {args.abort_depth1_acc} at step {step + 1} (budget guard)"
        if time.time() - t0 > args.time_limit_s:
            stop = f"time limit {args.time_limit_s}s reached at step {step + 1}"
        loss_hist, acc_hist = [], []
        dep = {}
        if stop:
            print(stop, flush=True)
            log.append({"stopped": stop, "step": step + 1})
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "args": vars(args), "step": step + 1, "unlock_steps": unlock_steps, "stopped": stop}, out / "model.pt")
            break
    if (step + 1) % 2000 == 0 or step + 1 == args.steps:
        torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "args": vars(args), "step": step + 1, "unlock_steps": unlock_steps}, out / "model.pt")
(out / "train_log.json").write_text(json.dumps(log))
print("done", flush=True)
