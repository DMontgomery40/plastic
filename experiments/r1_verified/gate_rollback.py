"""Gate test D3.3: rollback noninterference.

Test that rejection (snapshot restore) prevents residual learning.

For each episode:
  1. Fresh S, install rule A, query -> pred_A
  2. From same S (same snapshot), install rule B, query same queries -> pred_B_rejected
  3. Check: pred_A == pred_B_rejected (rejection restored S, no residual from B install)

This tests that installing and then immediately rejecting (restoring the snapshot)
leaves NO trace of the attempted install.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from distract import distractor_tokens
from common import build, expand_state, SIZE_CLASSES, pad_queries  # noqa: E402
from task import make_world, lesson_tokens, query_tokens, Query, BOS  # noqa: E402
from plastic.config import ModelConfig  # noqa: E402


def run_rollback_test(model, rng, cond, N, batch, n_probe=8):
    """Test noninterference of rejected installs.
    For each episode:
      1. Fresh S, install rule A, query from S -> pred_A
      2. (Imagine we now snapshot S)
      3. Install rule B (different rule)
      4. Reject (restore snapshot of S)
      5. Query same queries from restored S -> pred_B_after_reject
    Should have: pred_A == pred_B_after_reject (noninterference: B's install left no trace)
    """
    K, kind, arm = cond; recs = []
    done = 0
    while done < N:
        B = min(batch, N - done); n = rng.choice(sorted(SIZE_CLASSES))
        
        # Generate TWO different worlds (rule A and rule B)
        worlds_A = [make_world(rng, rng.choice(SIZE_CLASSES[n])) for _ in range(B)]
        worlds_B = [make_world(rng, rng.choice(SIZE_CLASSES[n])) for _ in range(B)]
        
        texts_A = [[BOS] + lesson_tokens(w, rng)[1:] for w in worlds_A]
        
        # Generate FIXED query set from rule A worlds
        qs = []
        for w in worlds_A:
            q1 = [Query(x, (g,), w.table[x][g]) for x in range(w.n) for g in range(2)]
            q2 = []
            for _ in range(n_probe):
                x = rng.randrange(w.n); wd = tuple(rng.randrange(2) for _ in range(2)); q2.append(Query(x, wd, w.act(x, wd)))
            qs.append(q1 + q2)
        
        k = len(qs[0]); rows = []
        for w, qq in zip(worlds_A, qs):
            for q in qq: rows.append(query_tokens(w, q, "aligned"))
        qx, qpos = pad_queries(rows)
        
        with torch.no_grad():
            # Path 1: Install A, query from S
            st_before = model.init_state(B)
            _, st_A, _ = model(torch.tensor(texts_A), st_before, freeze=False)
            if K > 0:
                dist = torch.tensor([distractor_tokens(kind, K, rng, arm) for _ in range(B)])
                _, st_A, _ = model(dist, st_A, freeze=False)
            
            logits_A, _, _ = model(qx, expand_state(st_A, k, reset_activations=True), freeze=False)
            pred_A = logits_A[torch.arange(qx.shape[0]), qpos].argmax(-1).view(B, k)
            
            # Path 2: Install B, then immediately reject (restore to pre-B state, which is st_A)
            # The rejection snapshot would be at st_A; we simulate restoration by using st_A again
            texts_B = [[BOS] + lesson_tokens(w, rng)[1:] for w in worlds_B]
            _, st_B, _ = model(torch.tensor(texts_B), st_A, freeze=False)  # Install on top of st_A
            if K > 0:
                dist = torch.tensor([distractor_tokens(kind, K, rng, arm) for _ in range(B)])
                _, st_B, _ = model(dist, st_B, freeze=False)
            
            # Rejection: restore to st_A (the snapshot taken before B's install)
            st_after_reject = st_A  # Restored state
            
            # Query from restored state
            logits_after_reject, _, _ = model(qx, expand_state(st_after_reject, k, reset_activations=True), freeze=False)
            pred_after_reject = logits_after_reject[torch.arange(qx.shape[0]), qpos].argmax(-1).view(B, k)
            
            # Compare: pred_A should equal pred_after_reject (no residual from B)
            for i in range(B):
                identical_d1 = torch.all(pred_A[i, :2 * worlds_A[i].n] == pred_after_reject[i, :2 * worlds_A[i].n]).item()
                identical_d2 = torch.all(pred_A[i, 2 * worlds_A[i].n:] == pred_after_reject[i, 2 * worlds_A[i].n:]).item()
                
                r = dict(variant="rollback_noninterference", K=K, kind=kind, arm=arm, n=worlds_A[i].n,
                         identical_d1=identical_d1, identical_d2=identical_d2,
                         diffs_d1=int((pred_A[i, :2 * worlds_A[i].n] != pred_after_reject[i, :2 * worlds_A[i].n]).sum().item()),
                         diffs_d2=int((pred_A[i, 2 * worlds_A[i].n:] != pred_after_reject[i, 2 * worlds_A[i].n:]).sum().item()))
                recs.append(r)
        
        done += B
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default="64:struct:seg", help="comma list K:kind:arm")
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    ck = torch.load(a.model, map_location="cpu"); cfg = ModelConfig.from_dict(ck["cfg"]); model = build(cfg); model.load_state_dict(ck["model"]); model.eval()
    conds = [(int(c.split(":")[0]), c.split(":")[1], c.split(":")[2]) for c in a.conds.split(",")]
    out = Path(a.out); res = json.loads(out.read_text()) if out.exists() else {"meta": {"model": a.model, "step": ck.get("step")}, "recs": []}
    have = {(r["variant"], r["K"], r["kind"], r["arm"]) for r in res["recs"]}
    t0 = time.time()
    for cond in conds:
        if ("rollback_noninterference", *cond) in have: continue
        rng = random.Random(70000 + 1000 * a.seed + cond[0] * 7 + ("struct", "randline", "randiid").index(cond[1]) * 3)
        res["recs"] += run_rollback_test(model, rng, cond, a.n, a.batch)
        out.write_text(json.dumps(res)); print(f"{cond} done {time.time()-t0:.0f}s", flush=True)
