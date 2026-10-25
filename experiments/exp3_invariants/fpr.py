"""Exp 3(b) step 2: policy-level false-positive rate on a benign stream and true-positive rate on the exp2
poison types, through the repo's TransactionRunner with HarnessConfig() defaults (projection, stats, rollback on;
no budgets; CUSUM freeze latches) and a calibration.

  --target shipped : artifacts/models/lm_small_cpu/calibration.json as is (target_fpr 0.01)
  --target 0.05    : thresholds recomputed from saved calibration records (calib_records.py) with the repo's
                     thresholds_from_records / calibrated_cusum_h at that target; same Fisher, same reference.

Benign stream: wikitext-103 TEST split (calibration used validation), contiguous 16-chunk (1024-token) sessions
from random offsets, a fresh runner per session. Poison: exp2 layout, 2 clean chunks then 4 chunks of a poison
type (random / repeat_token / repeat_phrase / shuffled), fresh runner per session.
Saves after every session and resumes from an existing --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, load  # noqa: E402

from plastic.harness.calibrate import ROLLBACK_DECISION_SIGNALS, Calibration, calibrated_cusum_h, thresholds_from_records  # noqa: E402
from plastic.harness.config import HarnessConfig  # noqa: E402
from plastic.harness.stats import robust_z  # noqa: E402
from plastic.harness.transaction import TransactionRunner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="shipped")
ap.add_argument("--target", default="shipped")
ap.add_argument("--records", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sessions", type=int, default=20)
ap.add_argument("--chunks", type=int, default=16)
ap.add_argument("--poison-sessions", type=int, default=8)
ap.add_argument("--poison-types", default="random,repeat_token,repeat_phrase,shuffled")
ap.add_argument("--nx", type=int, default=4)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)

store, cfg, model, cal0, suite = load(a.suite)
hcfg = HarnessConfig()
L, V = cfg.chunk, cfg.vocab_size

if a.target == "shipped":
    assert a.suite == "shipped", "the shipped calibration belongs to the shipped canary suite"
    cal = cal0
else:
    tgt = float(a.target)
    R = json.load(open(a.records))
    recs, cont = R["records"], R["cusum_continuous"]
    thr, ach = thresholds_from_records(recs, target_fpr=tgt)
    zc = [z for z in (robust_z(v, cont) for v in cont) if z is not None]
    h, h_rate = calibrated_cusum_h(zc, k=hcfg.cusum_k, h_min=hcfg.cusum_h, target_fpr=tgt)
    thr["cusum_h"], ach["cusum_h"] = h, h_rate
    ref = {n: [float(r[n]) for r in recs if r.get(n) is not None][-512:] for n in ROLLBACK_DECISION_SIGNALS}
    cal = Calibration(model_signature=cal0.model_signature, n_chunks=len(recs), reference={k: v for k, v in ref.items() if v},
                      cusum_reference=cont, thresholds=thr, achievable_fpr=ach, canary_baseline=cal0.canary_baseline,
                      target_fpr=tgt, created_at_unix=int(time.time()), fisher=cal0.fisher)

test = np.memmap(os.path.join(DATA, "test.bin"), dtype="<u2", mode="r")
rng_b = np.random.default_rng(7000 + a.seed)
rng_p = np.random.default_rng(8000 + a.seed)
g = torch.Generator().manual_seed(9000 + a.seed)
# draw every session's data up front so a resumed run sees the same stream
benign_starts = [int(rng_b.integers(0, len(test) - a.chunks * L - 1)) for _ in range(a.sessions)]


def clean(n):
    s = int(rng_p.integers(0, len(test) - n - 1))
    return [int(t) for t in test[s:s + n]]


def poison(kind, n):
    if kind == "random":
        return torch.randint(3, V, (n,), generator=g).tolist()
    if kind == "repeat_token":
        return [int(torch.randint(3, V, (1,), generator=g))] * n
    if kind == "repeat_phrase":
        ph = clean(8)
        return (ph * (n // 8 + 1))[:n]
    if kind == "shuffled":
        x = clean(n)
        return [x[i] for i in torch.randperm(n, generator=g).tolist()]
    raise ValueError(kind)


poison_plan = []
for sess in range(a.poison_sessions):
    for kind in a.poison_types.split(","):
        poison_plan.append((sess, kind, clean(2 * L), poison(kind, a.nx * L)))


def summarize(t):
    return {"applied": t["decision"]["kind"], "requested": t["requested"]["kind"], "reasons": t["decision"]["reasons"][:6],
            "read_only": t["read_only"], "read_only_reason": t["read_only_reason"]}


out = {"args": vars(a), "thresholds": cal.thresholds, "achievable_fpr": cal.achievable_fpr, "benign": [], "poison": []}
if os.path.exists(a.out):
    prev = json.load(open(a.out))
    if prev.get("args") == vars(a):
        out = prev
done_b = {r["session"] for r in out["benign"]}
done_p = {(r["session"], r["type"]) for r in out["poison"]}
t0 = time.time()


def save():
    tmp = a.out + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, a.out)


for sess, s in enumerate(benign_starts):
    if sess in done_b:
        continue
    r = TransactionRunner(model, cfg, hcfg, calibration=cal, suite=suite, device="cpu")
    r.feed_tokens([int(t) for t in test[s:s + a.chunks * L]], source="user")
    out["benign"].append({"session": sess, "start": s, "chunks": [summarize(t) for t in r.transactions]})
    save()
    print(f"[benign {a.suite} {a.target} s{a.seed}] sess {sess}: {[t['decision']['kind'][0] for t in r.transactions]} ({time.time()-t0:.0f}s)", flush=True)
for sess, kind, P, X in poison_plan:
    if (sess, kind) in done_p:
        continue
    r = TransactionRunner(model, cfg, hcfg, calibration=cal, suite=suite, device="cpu")
    r.feed_tokens(P, source="user")
    nP = len(r.transactions)
    r.feed_tokens(X, source="user")
    out["poison"].append({"session": sess, "type": kind, "prefix": [summarize(t) for t in r.transactions[:nP]],
                          "chunks": [summarize(t) for t in r.transactions[nP:]]})
    save()
    print(f"[poison {a.suite} {a.target} s{a.seed}] {kind} sess {sess}: {[t['decision']['kind'][0] for t in r.transactions]} ({time.time()-t0:.0f}s)", flush=True)
out["done"] = True
save()
print("done", time.time() - t0)
