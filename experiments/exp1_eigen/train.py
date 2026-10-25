"""Train one (task, variant, seed, lr) and evaluate length extrapolation. Writes one JSON.

python experiments/exp1_eigen/train.py --task parity --variant B --seed 0 --lr 1e-3 --out results/x.json
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

sys.path.insert(0, os.path.dirname(__file__))
from models import SeqModel  # noqa: E402
from tasks import TASKS, batch  # noqa: E402

from plastic.data.mqar import mqar_batch  # noqa: E402

MQ_KEYS, MQ_VALS, MQ_VOCAB = (2, 66), (66, 130), 130


def mqar(B, n_pairs, g):
    seq = 1 + 4 * n_pairs
    toks, mask = mqar_batch(B, n_pairs=n_pairs, seq_len=seq, vocab_size=MQ_VOCAB, key_range=MQ_KEYS,
                            value_range=MQ_VALS, rng=g)
    labels = torch.full_like(toks, -100)
    m = mask[:, :-1]
    labels[:, :-1][m] = toks[:, 1:][m]
    return toks, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-min", type=int, default=8)
    ap.add_argument("--train-max", type=int, default=32)
    ap.add_argument("--test-lens", default="32,64,128,256,512")
    ap.add_argument("--n-test", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--beta-max", type=float, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    is_mqar = a.task == "mqar"
    if is_mqar:
        vocab_in = n_out = MQ_VOCAB
    else:
        vocab_in, n_out = TASKS[a.task]["vocab_in"], TASKS[a.task]["n_out"]
    model = SeqModel(a.variant, vocab_in, n_out, d_model=a.d_model, n_heads=a.heads, n_layers=a.layers,
                     beta_max=a.beta_max)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
    g = torch.Generator().manual_seed(1000 + a.seed)

    def sample(B, gen, lo=a.train_min, hi=a.train_max):
        if is_mqar:
            n = int(torch.randint(lo, hi + 1, (1,), generator=gen))
            return mqar(B, n, gen)
        T = int(torch.randint(lo, hi + 1, (1,), generator=gen))
        return batch(a.task, B, T, gen)

    curve = []
    t0 = time.time()
    nonfinite = 0
    for step in range(a.steps):
        model.train()
        tok, lab = sample(a.batch, g)
        logits = model(tok)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == a.steps - 1:
            with torch.no_grad():
                m = lab != -100
                acc = float((logits.argmax(-1)[m] == lab[m]).float().mean())
            curve.append({"step": step, "loss": float(loss.detach()), "acc": acc, "t": time.time() - t0})
    train_time = time.time() - t0

    model.eval()
    res = {"args": vars(a), "n_params": model.num_params(), "train_time_s": train_time, "curve": curve,
           "nonfinite_steps": nonfinite}
    with torch.no_grad():
        # in-distribution validation loss (used for lr selection only)
        gv = torch.Generator().manual_seed(999)
        vl, vn = 0.0, 0
        for _ in range(8):
            tok, lab = sample(64, gv)
            lg = model(tok)
            vl += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100)); vn += 1
        res["val_loss"] = vl / vn
        res["eig_stats"] = model.eig_stats()
        ev = {}
        gt = torch.Generator().manual_seed(12345)
        for L in [int(x) for x in a.test_lens.split(",")]:
            n_ok_last = n_last = n_ok_half = n_half = 0
            done = 0
            while done < a.n_test:
                B = min(64, a.n_test - done)
                if is_mqar:
                    tok, lab = mqar(B, L, gt)
                else:
                    tok, lab = batch(a.task, B, L, gt)
                pred = model(tok).argmax(-1)
                if is_mqar:
                    m = lab != -100
                    n_ok_half += int((pred[m] == lab[m]).sum()); n_half += int(m.sum())
                    n_ok_last += int((pred[m] == lab[m]).sum()); n_last += int(m.sum())
                else:
                    T1 = tok.shape[1]
                    h = slice(1 + (T1 - 1) // 2, T1)
                    n_ok_half += int((pred[:, h] == lab[:, h]).sum()); n_half += pred[:, h].numel()
                    n_ok_last += int((pred[:, -1] == lab[:, -1]).sum()); n_last += B
                done += B
            ev[str(L)] = {"acc_last_half": n_ok_half / n_half, "acc_final": n_ok_last / n_last}
        res["eval"] = ev
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"task": a.task, "variant": a.variant, "seed": a.seed, "lr": a.lr, "val_loss": res["val_loss"],
                      "t": round(train_time), "eval": {k: round(v["acc_last_half"], 3) for k, v in ev.items()}}))


if __name__ == "__main__":
    main()
