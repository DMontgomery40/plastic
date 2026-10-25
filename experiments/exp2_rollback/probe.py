"""Exp 2: rollback leakage probe through the repo's TransactionRunner.

Session layout (all fed through plastic.harness.transaction.TransactionRunner, calibrated):
    P  : 2 chunks (128 tokens) of clean held-out text        natural harness decisions
    X  : n_x chunks of a poison type                         decision per condition (below)
    C  : 4 chunks (256 tokens) of clean held-out text        measured

Conditions (P is fed once, the runner is forked with state_dict/load_state_dict):
    never      : P -> C
    rollback   : P -> X (each X chunk refused by the harness's own rollback path) -> C
    commit     : P -> X (each X chunk committed through the harness's own accept path) -> C
    rb_rewind  : rollback, then activation state (h, conv buffers, last logits) restored to the pre-X value
    rb_zero    : rollback, then activation state zeroed (fresh activations, committed memory kept)
    natural    : P -> X with the calibrated harness deciding -> C (records what it actually decides)

Forcing uses the runner's real code paths: `decide` (imported into plastic.harness.transaction) is
temporarily replaced by a function returning Decision("rollback") or Decision("commit") while X is
fed, so _reject_frozen / _accept do the work. C is measured two ways:
    direct : model(C, committed_state, mode="chunk") with learning on (what a continuing session
             computes), per-token logits.
    frozen : same with freeze=True (isolates the read path; memory cannot change during C).
Leakage = metric(rollback) / metric(commit), metric = mean KL(never || cond) over C tokens, and
excess NLL on C. Also: S after rollback == S before X, bit for bit (torch.equal per layer).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import plastic.harness.transaction as txn
from plastic.harness.calibrate import Calibration
from plastic.harness.canary import CanarySuite
from plastic.harness.config import HarnessConfig
from plastic.harness.policy import Decision
from plastic.harness.transaction import TransactionRunner
from plastic.store import ArtifactStore

ap = argparse.ArgumentParser()
ap.add_argument("--model-id", default="lm_small_cpu")
ap.add_argument("--artifacts-root", default="artifacts")
ap.add_argument("--data", default="artifacts/data/wikitext")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sessions", type=int, default=6)
ap.add_argument("--types", default="benign,random,repeat_token,repeat_phrase,shuffled")
ap.add_argument("--sizes", default="1,2,4")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)

store = ArtifactStore(a.artifacts_root)
cfg, model, _ = store.load_checkpoint(a.model_id, "cpu")
model.eval()
mdir = store.model_dir(a.model_id)
cal = Calibration.load(mdir)
suite = CanarySuite.load(store.canary_path(a.model_id))
hcfg = HarnessConfig()
L = cfg.chunk
V = cfg.vocab_size
test = np.memmap(os.path.join(a.data, "test.bin"), dtype="<u2", mode="r")
rng = np.random.default_rng(1000 + a.seed)
g = torch.Generator().manual_seed(2000 + a.seed)


def clean(n):
    s = int(rng.integers(0, len(test) - n - 1))
    return [int(t) for t in test[s:s + n]]


def poison(kind, n):
    if kind == "benign":
        return clean(n)
    if kind == "random":
        return torch.randint(3, V, (n,), generator=g).tolist()
    if kind == "repeat_token":
        t = int(torch.randint(3, V, (1,), generator=g))
        return [t] * n
    if kind == "repeat_phrase":
        ph = clean(8)
        return (ph * (n // 8 + 1))[:n]
    if kind == "shuffled":  # real tokens, destroyed order
        x = clean(n)
        return [x[i] for i in torch.randperm(n, generator=g).tolist()]
    raise ValueError(kind)


def new_runner():
    return TransactionRunner(model, cfg, hcfg, calibration=cal, suite=suite, device="cpu")


def fork(r):
    r2 = new_runner()
    r2.load_state_dict(r.state_dict())
    return r2


def feed_forced(r, toks, kind):
    orig = txn.decide
    txn.decide = lambda sig, hc, th, read_only: Decision(kind, [f"forced_{kind}"])
    try:
        r.feed_tokens(toks, source="user")
    finally:
        txn.decide = orig


def measure(state, C, last_logits, freeze):
    x = torch.tensor([C])
    with torch.no_grad():
        lg, _, _ = model(x, state.clone(), mode="chunk", freeze=freeze)
    lg = torch.cat([last_logits.view(1, 1, -1), lg[:, :-1]], 1)[0]  # logits predicting C[t]
    return F.log_softmax(lg.float(), -1)


def act_set(r, src_state, src_logits):
    for st in (r.committed, r.working):
        for lay, s in zip(st.layers, src_state.layers):
            lay.h = s.h.clone()
            lay.conv_ssm = None if s.conv_ssm is None else s.conv_ssm.clone()
            lay.conv_mem = None if s.conv_mem is None else s.conv_mem.clone()
    r._last_logits = src_logits.clone()


def act_zero(r):
    for st in (r.committed, r.working):
        for lay in st.layers:
            lay.h = torch.zeros_like(lay.h)
            if lay.conv_ssm is not None:
                lay.conv_ssm = torch.zeros_like(lay.conv_ssm)
            if lay.conv_mem is not None:
                lay.conv_mem = torch.zeros_like(lay.conv_mem)


rows = []
t0 = time.time()
types = a.types.split(",")
sizes = [int(s) for s in a.sizes.split(",")]
for sess in range(a.sessions):
    P, C = clean(2 * L), clean(4 * L)
    base = new_runner()
    base.feed_tokens(P, source="user")
    assert not base.pending
    pre_S = [l.S.clone() for l in base.committed.layers]
    pre_state = base.committed.clone()
    pre_logits = base._last_logits.clone()
    for kind in types:
        for nx in sizes:
            X = poison(kind, nx * L)
            runs = {}
            runs["never"] = fork(base)
            r = fork(base); feed_forced(r, X, "rollback"); runs["rollback"] = r
            rb_bitexact = all(torch.equal(a_.S, b_) for a_, b_ in zip(r.committed.layers, pre_S))
            rb_h_changed = float(sum((a_.h - b_.h).norm() for a_, b_ in zip(r.committed.layers, pre_state.layers)))
            r = fork(base); feed_forced(r, X, "commit"); runs["commit"] = r
            commit_dS = float(sum((a_.S - b_).norm() for a_, b_ in zip(r.committed.layers, pre_S)))
            r = fork(runs["rollback"]); act_set(r, pre_state, pre_logits); runs["rb_rewind"] = r
            r = fork(runs["rollback"]); act_zero(r); runs["rb_zero"] = r
            r = fork(base); r.feed_tokens(X, source="user")
            nat = [t["decision"]["kind"] for t in r.transactions]
            nat_reasons = [t["decision"]["reasons"][:3] for t in r.transactions]
            nat_bitexact = all(torch.equal(a_.S, b_) for a_, b_ in zip(r.committed.layers, pre_S)) if all(k == "rollback" for k in nat) else None
            runs["natural"] = r
            rec = {"seed": a.seed, "session": sess, "type": kind, "n_chunks": nx, "natural_decisions": nat,
                   "natural_reasons": nat_reasons, "rollback_S_bitexact": rb_bitexact,
                   "natural_rollback_S_bitexact": nat_bitexact, "rollback_h_change": rb_h_changed,
                   "commit_dS": commit_dS}
            for freeze in (False, True):
                tag = "frozen" if freeze else "direct"
                lp = {k: measure(rr.committed, C, rr._last_logits, freeze) for k, rr in runs.items()}
                tgt = torch.tensor(C)
                nll = {k: -v[torch.arange(len(C)), tgt] for k, v in lp.items()}
                p0 = lp["never"].exp()
                rec[f"{tag}_kl_commit_vs_rollback"] = float((lp["commit"].exp() * (lp["commit"] - lp["rollback"])).sum(-1).mean())
                for k in runs:
                    kl = (p0 * (lp["never"] - lp[k])).sum(-1)  # KL(never || cond) per token
                    rec[f"{tag}_nll_{k}"] = float(nll[k].mean())
                    rec[f"{tag}_kl_{k}"] = float(kl.mean())
                    for lo, hi in ((0, 16), (16, 64), (64, 256)):
                        rec[f"{tag}_kl_{k}_{lo}_{hi}"] = float(kl[lo:hi].mean())
                    if kind == "repeat_token":
                        rec[f"{tag}_logp_poisontok_{k}"] = float(lp[k][:, X[0]].mean())
                    if kind == "repeat_phrase":
                        rec[f"{tag}_logp_phrase_{k}"] = float(lp[k][:, sorted(set(X))].logsumexp(-1).mean())
            # harness decisions on the clean continuation C in each condition
            for k, rr in runs.items():
                r2 = fork(rr)
                r2.feed_tokens(C, source="user")
                rec[f"C_decisions_{k}"] = [t["decision"]["kind"] for t in r2.transactions]
            rows.append(rec)
            print(f"[s{a.seed} sess{sess} {kind} x{nx}] nat={nat} bitexact={rb_bitexact} "
                  f"KL rb={rec['direct_kl_rollback']:.4f} commit={rec['direct_kl_commit']:.4f} "
                  f"rewind={rec['direct_kl_rb_rewind']:.2e} zero={rec['direct_kl_rb_zero']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
print("done", len(rows), time.time() - t0)
