"""GPU version of experiments/exp1_eigen/train.py (same models, same tasks, same metric).

Differences from the CPU script (all deliberate, see experiments/gpu/REPORT.md):
  * runs on CUDA when available (records torch.cuda device name in the output JSON),
  * default 10k steps at batch 128 (CPU run: 3k steps at batch 64),
  * in-distribution validation over 16 batches of 128 (val loss AND val accuracy recorded),
  * test lengths up to 1024, 512 test sequences per length.

python experiments/gpu/exp1_scaled/train_gpu.py --task parity --variant B --seed 0 --lr 1e-3 --out x.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp1_eigen"))
sys.path.insert(0, REPO)
from models import SeqModel  # noqa: E402
from tasks import TASKS, batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--train-min", type=int, default=8)
    ap.add_argument("--train-max", type=int, default=32)
    ap.add_argument("--test-lens", default="32,64,128,256,512,1024")
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--beta-max", type=float, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    vocab_in, n_out = TASKS[a.task]["vocab_in"], TASKS[a.task]["n_out"]
    model = SeqModel(a.variant, vocab_in, n_out, d_model=a.d_model, n_heads=a.heads, n_layers=a.layers,
                     beta_max=a.beta_max).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
    g = torch.Generator().manual_seed(1000 + a.seed)

    def sample(B, gen, lo=a.train_min, hi=a.train_max):
        T = int(torch.randint(lo, hi + 1, (1,), generator=gen))
        tok, lab = batch(a.task, B, T, gen)
        return tok.to(dev, non_blocking=True), lab.to(dev, non_blocking=True)

    curve = []
    t0 = time.time()
    nonfinite = 0
    for step in range(a.steps):
        model.train()
        tok, lab = sample(a.batch, g)
        logits = model(tok)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):  # same rule as the CPU script: skip non-finite steps (one host sync/step)
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if step % 50 == 0 or step == a.steps - 1:
            with torch.no_grad():
                m = lab != -100
                acc = float((logits.argmax(-1)[m] == lab[m]).float().mean())
            curve.append({"step": step, "loss": float(loss.detach()), "acc": acc, "t": time.time() - t0})
        opt.step()
        sched.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t0
    final_params_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())

    model.eval()
    res = {"args": vars(a), "n_params": model.num_params(), "train_time_s": train_time, "curve": curve,
           "nonfinite_steps": nonfinite, "final_params_finite": final_params_finite,
           "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "torch": torch.__version__}
    with torch.no_grad():
        gv = torch.Generator().manual_seed(999)
        vl, vc, vt, vn = 0.0, 0, 0, 0
        for _ in range(16):
            tok, lab = sample(128, gv)
            lg = model(tok)
            vl += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100)); vn += 1
            m = lab != -100
            vc += int((lg.argmax(-1)[m] == lab[m]).sum()); vt += int(m.sum())
        res["val_loss"] = vl / vn
        res["val_acc"] = vc / vt
        res["eig_stats"] = model.eig_stats()
        ev = {}
        gt = torch.Generator().manual_seed(12345)
        t1 = time.time()
        for L in [int(x) for x in a.test_lens.split(",")]:
            n_ok_last = n_last = n_ok_half = n_half = 0
            done = 0
            while done < a.n_test:
                B = min(128, a.n_test - done)
                tok, lab = batch(a.task, B, L, gt)
                tok, lab = tok.to(dev), lab.to(dev)
                pred = model(tok).argmax(-1)
                T1 = tok.shape[1]
                h = slice(1 + (T1 - 1) // 2, T1)
                n_ok_half += int((pred[:, h] == lab[:, h]).sum()); n_half += pred[:, h].numel()
                n_ok_last += int((pred[:, -1] == lab[:, -1]).sum()); n_last += B
                done += B
            ev[str(L)] = {"acc_last_half": n_ok_half / n_half, "acc_final": n_ok_last / n_last}
        res["eval"] = ev
        res["eval_time_s"] = time.time() - t1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({"task": a.task, "variant": a.variant, "seed": a.seed, "lr": a.lr, "tmax": a.train_max,
                      "val_loss": round(res["val_loss"], 5), "val_acc": round(res["val_acc"], 4), "t": round(train_time),
                      "eval": {k: round(v["acc_last_half"], 3) for k, v in ev.items()}}), flush=True)


if __name__ == "__main__":
    main()
