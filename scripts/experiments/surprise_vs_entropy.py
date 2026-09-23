"""Does the fast learner's per-token surprise predict next-token error or loops beyond output entropy?

Build-order step 4 of docs/research/2026-09-23-importance-weighting-proposal.md. Adaptive sampling driven by
output entropy is the known baseline (EDT, arXiv 2403.14541; AdapT, arXiv 2309.02772). The TTT-specific version
would drive temperature/top-k from the inner learner's own reconstruction error (``err`` in MemorySignals, the
harness's ``surprise``). Before building a sampler, measure whether surprise carries information about the next
token's error that entropy does not.

Two passes on one checkpoint, both reading the same forward call that the harness reads:

1. Teacher-forced held-out chat (SmolTalk test rows, pinned revision). For every position t the model has
   read token t and emits logits for t+1; we record output entropy H_t, the inner surprise s_t (mean over
   layers and heads, and the last layer alone), the inner step size and write norm, and the outcomes: the
   negative log-probability of the true next token and whether the argmax missed it. Both H_t and s_t are
   available online at generation time, so they are compared as equals.
2. Free generation on the eval prompts at two temperatures, with the same per-token signals; a token is
   "in a loop" when it belongs to a 4-gram already seen in the same reply. We ask which signal separates
   looping tokens from fresh ones.

Statistics are numpy only: Spearman correlations, rank AUC, a linear model for the next-token NLL and a
logistic model for the top-1 miss, each fit with entropy alone and with entropy + surprise, and a bootstrap
over conversations for the increments. A small, tight increment means entropy-adaptive sampling is the
honest baseline and a surprise-driven sampler adds nothing on this checkpoint.

  PYTHONPATH=<abs deps> python -m scripts.experiments.surprise_vs_entropy --checkpoint <dir> --out <dir> [--device mps]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from typing import Any, Iterable

import numpy as np

from plastic.sleep import SMOLTALK_REVISION

# ---------------------------------------------------------------------------------------------------- statistics


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties averaged."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: Iterable[float], b: Iterable[float]) -> float | None:
    a, b = np.asarray(list(a), dtype=np.float64), np.asarray(list(b), dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(scores: Iterable[float], labels: Iterable[bool]) -> float | None:
    """Probability that a positive scores above a negative (ties count half); None without both classes."""
    s, y = np.asarray(list(scores), dtype=np.float64), np.asarray(list(labels), dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rank(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _design(cols: list[np.ndarray]) -> np.ndarray:
    z = [(c - c.mean()) / (c.std() if c.std() > 0 else 1.0) for c in cols]
    return np.column_stack([np.ones(len(cols[0]))] + z)


def linear_r2(y: np.ndarray, cols: list[np.ndarray]) -> float:
    """R² of a least-squares fit of y on standardized columns with an intercept."""
    X = _design(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    tss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / tss if tss > 0 else 0.0


def logistic_fit(y: np.ndarray, cols: list[np.ndarray], iters: int = 50, ridge: float = 1e-6) -> tuple[np.ndarray, float, np.ndarray]:
    """Newton-Raphson logistic regression on standardized columns with an intercept: (coefficients, mean log-likelihood,
    fitted probabilities)."""
    X = _design(cols)
    y = y.astype(np.float64)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (y - p) - ridge * w
        H = -(X.T * (p * (1 - p))) @ X - ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w = w - step
        if float(np.abs(step).max()) < 1e-8:
            break
    p = np.clip(1.0 / (1.0 + np.exp(-(X @ w))), 1e-12, 1 - 1e-12)
    ll = float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return w, ll, p


def repeated_ngram_mask(tokens: list[int], n: int = 4) -> list[bool]:
    """True at every position covered by an n-gram that already occurred earlier in the same sequence: the
    within-reply loop indicator (the token-level form of recall.repetition_share)."""
    mask = [False] * len(tokens)
    seen: set[tuple[int, ...]] = set()
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i + n])
        if g in seen:
            for j in range(i, i + n):
                mask[j] = True
        seen.add(g)
    return mask


def compare_predictors(rows: list[dict[str, Any]], *, surprise_key: str = "surprise") -> dict[str, Any]:
    """Entropy vs surprise as predictors of the next-token NLL and the top-1 miss, alone and together."""
    if len(rows) < 10:
        return {"n": len(rows), "reason": "too few tokens"}
    H = np.array([r["entropy"] for r in rows])
    S = np.array([r[surprise_key] for r in rows])
    nll = np.array([r["nll_next"] for r in rows])
    miss = np.array([r["top1_miss"] for r in rows], dtype=bool)
    out: dict[str, Any] = {"n": len(rows), "miss_rate": float(miss.mean()), "nll_mean": float(nll.mean()),
                           "spearman": {"entropy~nll": spearman(H, nll), "surprise~nll": spearman(S, nll), "surprise~entropy": spearman(S, H)},
                           "auc_top1_miss": {"entropy": auc(H, miss), "surprise": auc(S, miss)},
                           "r2_nll": {"entropy": linear_r2(nll, [H]), "surprise": linear_r2(nll, [S]), "entropy+surprise": linear_r2(nll, [H, S])}}
    if 0 < miss.sum() < len(miss):
        _, ll_h, p_h = logistic_fit(miss, [H])
        w, ll_hs, p_hs = logistic_fit(miss, [H, S])
        _, ll_s, _ = logistic_fit(miss, [S])
        out["logistic_top1_miss"] = {"mean_loglik": {"entropy": ll_h, "surprise": ll_s, "entropy+surprise": ll_hs},
                                     "auc": {"entropy": auc(p_h, miss), "entropy+surprise": auc(p_hs, miss)},
                                     "surprise_coefficient_given_entropy": float(w[2])}
    out["increment"] = {"r2_nll": out["r2_nll"]["entropy+surprise"] - out["r2_nll"]["entropy"],
                        "auc_top1_miss": (out["logistic_top1_miss"]["auc"]["entropy+surprise"] - out["logistic_top1_miss"]["auc"]["entropy"])
                        if "logistic_top1_miss" in out else None}
    return out


def bootstrap_increment(rows: list[dict[str, Any]], *, reps: int, seed: int, surprise_key: str = "surprise") -> dict[str, Any]:
    """Resample conversations (not tokens: tokens in one conversation share the same fast weights) and read the
    increment of entropy+surprise over entropy alone in R² for NLL and AUC for the top-1 miss."""
    by_conv: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        by_conv.setdefault(r["conv"], []).append(r)
    convs = list(by_conv)
    rng = np.random.default_rng(seed)
    r2s, aucs = [], []
    for _ in range(reps):
        sample = [r for c in rng.choice(convs, size=len(convs), replace=True) for r in by_conv[c]]
        c = compare_predictors(sample, surprise_key=surprise_key)
        if "increment" in c:
            r2s.append(c["increment"]["r2_nll"])
            if c["increment"]["auc_top1_miss"] is not None:
                aucs.append(c["increment"]["auc_top1_miss"])

    def ci(v: list[float]) -> dict[str, float] | None:
        if not v:
            return None
        a = np.array(v)
        return {"mean": float(a.mean()), "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5)), "reps": len(v)}

    return {"conversations": len(convs), "r2_nll": ci(r2s), "auc_top1_miss": ci(aucs)}


def loop_separation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Which signal tells looping tokens from fresh ones, pooled over replies that have both kinds."""
    usable = [r for r in rows if r.get("reply_has_both")]
    if not usable:
        return {"n": 0, "reason": "no reply contained both looping and fresh tokens"}
    lab = np.array([r["in_loop"] for r in usable], dtype=bool)
    H = np.array([r["entropy"] for r in usable])
    S = np.array([r["surprise"] for r in usable])
    return {"n": len(usable), "loop_share": float(lab.mean()),
            "mean_in_loop": {"entropy": float(H[lab].mean()), "surprise": float(S[lab].mean())},
            "mean_fresh": {"entropy": float(H[~lab].mean()), "surprise": float(S[~lab].mean())},
            "auc_in_loop": {"entropy": auc(H, lab), "surprise": auc(S, lab)}}


# ---------------------------------------------------------------------------------------------------- the passes


def _git_head() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _digest(checkpoint: str) -> str:
    h = hashlib.sha256()
    for name in sorted(os.listdir(checkpoint)):
        if name.endswith((".safetensors", ".bin", ".json")):
            with open(os.path.join(checkpoint, name), "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()


def _token_signals(signals) -> dict[str, np.ndarray]:
    """Per-token reductions of the per-layer MemorySignals of one chunk (each err/beta/write_norm is (1, H, T))."""
    import torch

    err = torch.stack([s.err[0].float().mean(0) for s in signals])          # (layers, T)
    beta = torch.stack([s.beta[0].float().mean(0) for s in signals])
    wn = torch.stack([s.write_norm[0].float().mean(0) for s in signals])
    return {"surprise": err.mean(0).cpu().numpy(), "surprise_last": err[-1].cpu().numpy(), "surprise_first": err[0].cpu().numpy(),
            "beta": beta.mean(0).cpu().numpy(), "write_norm": wn.mean(0).cpu().numpy()}


def _entropy_nll(logits, targets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    lp = torch.log_softmax(logits.float(), dim=-1)
    ent = -(lp.exp() * lp).sum(-1)
    nll = -lp.gather(1, targets[:, None])[:, 0]
    miss = lp.argmax(-1) != targets
    return ent.cpu().numpy(), nll.cpu().numpy(), miss.cpu().numpy()


def teacher_forced_pass(backend, conversations: list[list[dict[str, str]]], *, max_len: int, chunk: int, log) -> list[dict[str, Any]]:
    import torch

    from plastic.backends.ttt_lm.backend import encode_conversation

    rows: list[dict[str, Any]] = []
    for ci, messages in enumerate(conversations):
        ids, labels = encode_conversation(backend.tokenizer, messages)
        ids, labels = ids[:max_len], labels[:max_len]
        if len(ids) < 8:
            continue
        state = backend.init_state()
        logits_all, sig_all = [], []
        with torch.no_grad():
            for i in range(0, len(ids), chunk):
                lg, state, sigs = backend.forward(ids[i:i + chunk], state, freeze=False, beta_scale=1.0)
                logits_all.append(lg)
                sig_all.append(_token_signals(sigs))
        logits = torch.cat(logits_all, 0)[:-1]                                    # position t predicts t+1
        targets = torch.tensor(ids[1:], dtype=torch.long, device=logits.device)
        ent, nll, miss = _entropy_nll(logits, targets)
        sig = {k: np.concatenate([s[k] for s in sig_all])[:-1] for k in sig_all[0]}
        for t in range(len(ids) - 1):
            rows.append({"conv": ci, "t": t, "assistant": labels[t + 1] != -100, "entropy": float(ent[t]), "nll_next": float(nll[t]),
                         "top1_miss": bool(miss[t]), **{k: float(v[t]) for k, v in sig.items()}})
        if ci % 10 == 0:
            log(f"[tf] conversation {ci + 1}/{len(conversations)}: {len(ids)} tokens, {len(rows)} rows so far")
    return rows


def generation_pass(backend, prompts: list[tuple[str, str]], *, temperatures: list[float], max_new_tokens: int, top_k: int, seed: int, log) -> list[dict[str, Any]]:
    import torch

    from plastic.sleep.recall import repetition_share

    eos = int(backend.tokenizer.eos_token_id)
    rows: list[dict[str, Any]] = []
    for temperature in temperatures:
        for pi, (group, prompt) in enumerate(prompts):
            gen = torch.Generator().manual_seed(seed + pi)
            ids = backend.encode_chat(prompt, first_turn=True)
            state = backend.init_state()
            with torch.no_grad():
                lg, state, _ = backend.forward(ids, state, freeze=False, beta_scale=1.0)
                out_tokens: list[int] = []
                per_tok: list[dict[str, float]] = []
                last = lg[-1]
                for _ in range(max_new_tokens):
                    lp = torch.log_softmax(last.float() / max(temperature, 1e-6), -1)
                    ent_full = torch.log_softmax(last.float(), -1)
                    entropy = float(-(ent_full.exp() * ent_full).sum())
                    if top_k > 0:
                        v, ix = torch.topk(lp, min(top_k, lp.shape[-1]))
                        probs = torch.softmax(v, -1).cpu()
                        nxt = int(ix[torch.multinomial(probs, 1, generator=gen)].item())
                    else:
                        nxt = int(torch.multinomial(lp.exp().cpu(), 1, generator=gen).item())
                    if nxt == eos:
                        break
                    lg, state, sigs = backend.forward([nxt], state, freeze=False, beta_scale=1.0)
                    s = _token_signals(sigs)
                    out_tokens.append(nxt)
                    # the entropy is of the distribution the token was drawn from; the surprise is the learner's error on it
                    per_tok.append({"entropy": entropy, **{k: float(v[0]) for k, v in s.items()}})
                    last = lg[-1]
            mask = repeated_ngram_mask(out_tokens, 4)
            both = any(mask) and not all(mask)
            text = backend.tokenizer.decode(out_tokens)
            for t, (tok, m, sig) in enumerate(zip(out_tokens, mask, per_tok)):
                rows.append({"temperature": temperature, "group": group, "prompt": pi, "t": t, "token": tok, "in_loop": m,
                             "reply_has_both": both, **sig})
            log(f"[gen] T={temperature} {group} #{pi}: {len(out_tokens)} tokens, repeated-4gram share {repetition_share(text):.3f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--rows", type=int, default=60, help="held-out SmolTalk test conversations")
    ap.add_argument("--subset", default="everyday-conversations")
    ap.add_argument("--replay-revision", default=SMOLTALK_REVISION, help="HuggingFaceTB/smoltalk revision (empty follows main)")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--temperatures", default="0.3,0.7")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    from plastic.backends.ttt_lm.backend import TTTBackend
    from plastic.sleep.ttt import load_replay_conversations
    from scripts.train.eval_ttt_chat import BOUNDARY_PROMPTS, NEUTRAL_PROMPTS

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.txt")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    t0 = time.time()
    backend = TTTBackend.load(args.checkpoint, device=args.device)
    convs = load_replay_conversations(args.subset, "test", args.rows, args.seed + 1, log, revision=(args.replay_revision or None))
    log(f"[setup] {len(convs)} held-out conversations; chunk {args.chunk}; max_len {args.max_len}")
    tf_rows = teacher_forced_pass(backend, convs, max_len=args.max_len, chunk=args.chunk, log=log)
    with open(os.path.join(args.out, "teacher_forced_tokens.jsonl"), "w", encoding="utf-8") as f:
        for r in tf_rows:
            f.write(json.dumps(r) + "\n")

    results: dict[str, Any] = {"checkpoint": os.path.abspath(args.checkpoint), "checkpoint_digest": _digest(args.checkpoint), "device": args.device,
                               "code_commit": _git_head(), "started_at_unix": int(t0), "replay_revision": args.replay_revision or None,
                               "rows": len(convs), "max_len": args.max_len, "chunk": args.chunk, "seed": args.seed,
                               "teacher_forced": {}}
    assistant = [r for r in tf_rows if r["assistant"]]
    for name, subset in (("assistant_tokens", assistant), ("all_tokens", tf_rows)):
        block: dict[str, Any] = {}
        for key in ("surprise", "surprise_last", "surprise_first", "write_norm", "beta"):
            block[key] = compare_predictors(subset, surprise_key=key)
        block["bootstrap_surprise"] = bootstrap_increment(subset, reps=args.bootstrap, seed=args.seed, surprise_key="surprise")
        results["teacher_forced"][name] = block
        c = block["surprise"]
        if "increment" in c:
            log(f"[tf] {name}: n={c['n']} miss {c['miss_rate']:.3f} | spearman entropy~nll {c['spearman']['entropy~nll']:.3f} "
                f"surprise~nll {c['spearman']['surprise~nll']:.3f} surprise~entropy {c['spearman']['surprise~entropy']:.3f} | "
                f"AUC(miss) entropy {c['auc_top1_miss']['entropy']:.3f} surprise {c['auc_top1_miss']['surprise']:.3f} | "
                f"R2(nll) entropy {c['r2_nll']['entropy']:.3f} +surprise {c['r2_nll']['entropy+surprise']:.3f} | "
                f"bootstrap dR2 {block['bootstrap_surprise']['r2_nll']} dAUC {block['bootstrap_surprise']['auc_top1_miss']}")

    if not args.skip_generation:
        prompts = [("neutral", p) for p in NEUTRAL_PROMPTS] + [("boundary", p) for p in BOUNDARY_PROMPTS]
        temps = [float(x) for x in args.temperatures.split(",") if x.strip()]
        gen_rows = generation_pass(backend, prompts, temperatures=temps, max_new_tokens=args.max_new_tokens, top_k=args.top_k, seed=args.seed, log=log)
        with open(os.path.join(args.out, "generation_tokens.jsonl"), "w", encoding="utf-8") as f:
            for r in gen_rows:
                f.write(json.dumps(r) + "\n")
        results["generation"] = {}
        for T in temps:
            sub = [r for r in gen_rows if r["temperature"] == T]
            results["generation"][str(T)] = {"tokens": len(sub), "loop_token_share": float(np.mean([r["in_loop"] for r in sub])) if sub else None,
                                             "separation": loop_separation(sub)}
            log(f"[gen] T={T}: {results['generation'][str(T)]}")

    results["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "surprise_vs_entropy.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    log(f"[done] {results['seconds']} s -> {args.out}")


if __name__ == "__main__":
    main()
