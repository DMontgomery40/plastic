"""Retention fine-tuning (R1): warm-start an exp4c checkpoint and teach it to keep S across K unrelated lines.

Each step: B episodes (fresh worlds), lessons written into S, then a K-line distractor segment (writes ON; kinds mixed
per batch), then activations are zeroed (fw_reset) and k first-queries are answered from S only. Loss = CE on the
answer token. K follows an adaptive curriculum over --k-stages (unlock next stage when depth-1 train acc >= --unlock-acc
over the last --log-every steps, after --min-stage-steps). A fraction --p-k0 of batches use K=0 (no forgetting of the
clean install). Checkpoint + train_log.json are written to disk at every log point (restart-safe: --resume).
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import torch, torch.nn.functional as F
from distract import DIST, KINDS, distractor_tokens
from common import build, model_config, pad_queries, expand_state, SIZE_CLASSES  # noqa: E402
from task import lesson_tokens, make_query, make_world, query_tokens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--init-from", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--arm", default="seg", choices=["seg", "line"])
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--k", type=int, default=6)
ap.add_argument("--depths", default="1,2,3"); ap.add_argument("--k-stages", default="4,16,32,64,128")
ap.add_argument("--p-k0", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--unlock-acc", type=float, default=0.9); ap.add_argument("--min-stage-steps", type=int, default=200)
ap.add_argument("--max-steps", type=int, default=100000); ap.add_argument("--time-limit-s", type=float, default=3600)
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--log-every", type=int, default=50)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
rng = random.Random(5000 + a.seed)
depths = [int(x) for x in a.depths.split(",")]; stages = [int(x) for x in a.k_stages.split(",")]
cfg = model_config(); model = build(cfg)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
state = {"step": 0, "stage": 0, "stage_start": 0, "elapsed": 0.0}
src = out / "model.pt" if (a.resume and (out / "model.pt").exists()) else a.init_from
ck = torch.load(src, map_location="cpu"); assert ck["cfg"] == cfg.to_dict()
model.load_state_dict(ck["model"])
if src != a.init_from:
    state.update(ck["train_state"]); rng.seed(5000 + a.seed + 7919 * state["step"])
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
if src != a.init_from and "opt" in ck: opt.load_state_dict(ck["opt"])
log = json.loads((out / "train_log.json").read_text()) if (a.resume and (out / "train_log.json").exists()) else []
print(json.dumps({"args": vars(a), "start": state}), flush=True)

def save(tag="model.pt"):
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "args": {"layout": "aligned", **vars(a)}, "train_state": state, "opt": opt.state_dict(), "step": state["step"]}, out / tag)
    (out / "train_log.json").write_text(json.dumps(log))

t0 = time.time() - state["elapsed"]; hist = []
while state["step"] < a.max_steps and time.time() - t0 < a.time_limit_s:
    step = state["step"]
    lr = a.lr * min(1.0, (step + 1) / a.warmup)
    for g in opt.param_groups: g["lr"] = lr
    # K for this batch
    if rng.random() < a.p_k0: K = 0
    else:
        top = stages[state["stage"]]
        K = rng.randint(1, top) if rng.random() < 0.5 else rng.choice([s for s in stages if s <= top])
    kind = rng.choice(KINDS)
    n = rng.choice(sorted(SIZE_CLASSES))
    worlds = [make_world(rng, rng.choice(SIZE_CLASSES[n])) for _ in range(a.batch)]
    seqs = [lesson_tokens(w, rng) + distractor_tokens(kind if kind != "mixed" else "struct", K, rng, a.arm) for w in worlds]
    qrows, ans, recs = [], [], []
    for i, w in enumerate(worlds):
        for _ in range(a.k):
            q = make_query(w, rng, rng.choice(depths)); qrows.append(query_tokens(w, q, "aligned")); ans.append(w.elem_names[q.answer]); recs.append((i, q.depth))
    model.train()
    _, st, _ = model(torch.tensor(seqs), model.init_state(a.batch), freeze=False)
    qx, qpos = pad_queries(qrows, ans)
    stq = expand_state(st, a.k, reset_activations=True)
    logits, _, _ = model(qx, stq, freeze=False)
    logits = logits[torch.arange(qx.shape[0]), qpos]
    y = torch.tensor(ans); loss = F.cross_entropy(logits, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert math.isfinite(float(loss)), "non-finite loss"
    opt.step()
    ok = (logits.argmax(-1) == y).tolist()
    hist.append(dict(K=K, kind=kind, loss=float(loss), d1=[o for o, (_, d) in zip(ok, recs) if d == 1], acc=sum(ok) / len(ok)))
    state["step"] += 1; state["elapsed"] = time.time() - t0
    if state["step"] % a.log_every == 0:
        pos = [h for h in hist if h["K"] > 0]
        d1 = [o for h in pos for o in h["d1"]]; top = stages[state["stage"]]
        d1top = [o for h in pos if h["K"] >= max(1, top // 2) for o in h["d1"]]
        rec = {"step": state["step"], "stage_K": top, "loss": sum(h["loss"] for h in hist) / len(hist), "acc": sum(h["acc"] for h in hist) / len(hist),
               "d1_K>0": sum(d1) / max(1, len(d1)), "d1_Khigh": sum(d1top) / max(1, len(d1top)) if d1top else None, "n_Khigh": len(d1top) // max(1, a.k // 2),
               "d1_K0": (lambda v: sum(v) / len(v) if v else None)([o for h in hist if h["K"] == 0 for o in h["d1"]]),
               "by_K": {str(K): round(sum(sum(h["d1"]) / max(1, len(h["d1"])) for h in hist if h["K"] == K) / max(1, sum(1 for h in hist if h["K"] == K)), 3) for K in sorted({h["K"] for h in hist})},
               "lr": lr, "gn": float(gn), "elapsed_s": round(state["elapsed"], 1)}
        if (d1top and sum(d1top) / len(d1top) >= a.unlock_acc and state["step"] - state["stage_start"] >= a.min_stage_steps and state["stage"] < len(stages) - 1):
            state["stage"] += 1; state["stage_start"] = state["step"]; rec["unlocked_K"] = stages[state["stage"]]
        log.append(rec); print(json.dumps(rec), flush=True); hist = []
        save()
save(); print("done", json.dumps(state), flush=True)
