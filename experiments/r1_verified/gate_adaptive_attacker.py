"""Gate test D3.2: adaptive attacker with knowledge of the gate.

Part A: consistent lie (relabel2 = swap names of two elements) against G3 alone (no ledger).
Part B: same consistent lie, but attacker adds a fake ledger (4 lines with swapped names) that matches the lie.
Measure what each gate (G1, G2, G3) catches.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from distract import distractor_tokens
from common import build, pad_queries, expand_state, SIZE_CLASSES  # noqa: E402
from task import BOS, make_world, lesson_tokens, query_tokens, Query  # noqa: E402
from plastic.config import ModelConfig  # noqa: E402


def regular_check(table: dict, names: list[int], ngen: int = 2):
    """table[(x_name, gi)] -> y_name. returns (is_perm, is_regular_group)."""
    idx = {t: i for i, t in enumerate(names)}; n = len(names); perms = []
    for gi in range(ngen):
        img = [idx.get(table.get((x, gi), -1), -1) for x in names]
        if -1 in img or len(set(img)) != n: return False, False
        perms.append(tuple(img))
    ident = tuple(range(n)); seen = {ident}; fr = [ident]
    while fr and len(seen) <= n:
        nx = []
        for p in fr:
            for g in perms:
                y = tuple(g[p[i]] for i in range(n))
                if y not in seen: seen.add(y); nx.append(y)
        fr = nx
    return True, len(seen) == n


def install_with_signals(model, L: torch.Tensor, chunk: int = 16):
    """Chunked install pass; returns final state and per-chunk harness-style signals (B, nchunks) each."""
    B, T = L.shape; st = model.init_state(B); sig = {k: [] for k in ("chunk_loss", "surprise_mean", "write_norm_sum", "delta_norm", "err_y")}
    for i in range(0, T, chunk):
        S0 = [l.S.clone() for l in st.layers]
        x = L[:, i:i + chunk]
        logits, st, sg = model(x, st, freeze=False)
        err = torch.stack([s.err for s in sg]).mean(dim=(0, 2))
        wn = torch.stack([s.write_norm for s in sg]).mean(dim=(0, 2))
        tgt = x[:, 1:]; lg = logits[:, :-1]
        cl = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(B, -1).mean(1) if x.shape[1] > 1 else torch.zeros(B)
        dn = torch.sqrt(sum(((l.S - s0) ** 2).flatten(1).sum(1) for l, s0 in zip(st.layers, S0)))
        pos = torch.arange(i, i + x.shape[1]); ym = ((pos - 1) % 4 == 2) & (pos >= 1)
        ey = (err[:, ym].mean(1) if ym.any() else torch.full((B,), float('nan')))
        sig["chunk_loss"].append(cl); sig["surprise_mean"].append(err.mean(1)); sig["write_norm_sum"].append(wn.sum(1)); sig["delta_norm"].append(dn); sig["err_y"].append(ey)
    return st, {k: torch.stack(v, 1).numpy() for k, v in sig.items()}


def readback(model, st, worlds, qsets):
    """qsets[i]: list of Query. returns (pred tokens (B,k), nll of true answer token (B,k))."""
    k = len(qsets[0]); rows, ans = [], []
    for w, qs in zip(worlds, qsets):
        for q in qs: rows.append(query_tokens(w, q, "aligned")); ans.append(w.elem_names[q.answer])
    qx, qpos = pad_queries(rows)
    logits, _, _ = model(qx, expand_state(st, k, reset_activations=True), freeze=False)
    lg = logits[torch.arange(qx.shape[0]), qpos]
    nll = F.cross_entropy(lg, torch.tensor(ans), reduction="none")
    return lg.argmax(-1).view(len(worlds), k), nll.view(len(worlds), k)


def run_attacker_part_a(model, rng, cond, N, batch, n_probe=16):
    """Part A: consistent lie (relabel2) with NO ledger defense; test if G3 can catch it.
    Records: "variant" = "attacker_no_ledger", signals, readback structure.
    """
    K, kind, arm = cond; recs = []
    done = 0
    while done < N:
        B = min(batch, N - done); n = rng.choice(sorted(SIZE_CLASSES))
        worlds = [make_world(rng, rng.choice(SIZE_CLASSES[n])) for _ in range(B)]
        
        texts = []
        for w in worlds:
            t = lesson_tokens(w, rng); body = t[1:]
            lines = [body[j:j + 4] for j in range(0, len(body), 4)]
            
            # Attacker applies relabel2: swap names of two elements
            a_, b_ = rng.sample(list(w.elem_names), 2); mp = {a_: b_, b_: a_}
            for l in lines: l[0], l[2] = mp.get(l[0], l[0]), mp.get(l[2], l[2])
            
            texts.append([BOS] + [x for l in lines for x in l])
        
        with torch.no_grad():
            st, sig = install_with_signals(model, torch.tensor(texts))
            if K > 0:
                dist = torch.tensor([distractor_tokens(kind, K, rng, arm) for _ in range(B)])
                _, st, _ = model(dist, st, freeze=False)
            
            qs = []
            for w in worlds:
                q1 = [Query(x, (g,), w.table[x][g]) for x in range(w.n) for g in range(2)]
                q2 = []
                for _ in range(n_probe):
                    x = rng.randrange(w.n); wd = tuple(rng.randrange(2) for _ in range(2)); q2.append(Query(x, wd, w.act(x, wd)))
                qs.append(q1 + q2)
            
            pred, nll = readback(model, st, worlds, qs)
        
        for i, w in enumerate(worlds):
            n1 = 2 * w.n; body = texts[i][1:]
            txt = {(body[j], w.gen_names.index(body[j + 1])): body[j + 2] for j in range(0, len(body), 4)}
            tab = {(w.elem_names[x], g): int(pred[i, 2 * x + g]) for x in range(w.n) for g in range(2)}
            tperm, treg = regular_check(txt, list(w.elem_names)); rperm, rreg = regular_check(tab, list(w.elem_names))
            ok = [int(pred[i, j]) == w.elem_names[qs[i][j].answer] for j in range(len(qs[i]))]
            r = dict(variant="attacker_no_ledger", K=K, kind=kind, arm=arm, n=w.n, group=w.group,
                     txt_perm=tperm, txt_reg=treg, rb_perm=rperm, rb_reg=rreg,
                     fidelity=float(np.mean([tab[k] == txt[k] for k in tab])), d1_acc=float(np.mean(ok[:n1])), d2_acc=float(np.mean(ok[n1:])),
                     nll1=[float(v) for v in nll[i, :n1]], ok1=[bool(v) for v in ok[:n1]], nll2=[float(v) for v in nll[i, n1:]], ok2=[bool(v) for v in ok[n1:]])
            for k_, v in sig.items(): r["sig_" + k_] = [float(u) for u in v[i]]
            recs.append(r)
        done += B
    return recs


def run_attacker_part_b(model, rng, cond, N, batch, n_probe=16):
    """Part B: consistent lie (relabel2) where ATTACKER ALSO controls ledger evidence.
    Attacker adds 4 fake ledger lines that match the lie. Test if G2 can catch it despite ledger agreement.
    Records: "variant" = "attacker_fake_ledger", signals, readback structure, and fake_ledger lines.
    """
    K, kind, arm = cond; recs = []
    done = 0
    while done < N:
        B = min(batch, N - done); n = rng.choice(sorted(SIZE_CLASSES))
        worlds = [make_world(rng, rng.choice(SIZE_CLASSES[n])) for _ in range(B)]
        
        texts = []
        relabelings = []  # track which two elements are swapped, for ledger corruption
        for w in worlds:
            t = lesson_tokens(w, rng); body = t[1:]
            lines = [body[j:j + 4] for j in range(0, len(body), 4)]
            
            # Attacker applies relabel2: swap names of two elements
            a_, b_ = rng.sample(list(w.elem_names), 2); mp = {a_: b_, b_: a_}
            relabelings.append((a_, b_))
            for l in lines: l[0], l[2] = mp.get(l[0], l[0]), mp.get(l[2], l[2])
            
            texts.append([BOS] + [x for l in lines for x in l])
        
        with torch.no_grad():
            st, sig = install_with_signals(model, torch.tensor(texts))
            if K > 0:
                dist = torch.tensor([distractor_tokens(kind, K, rng, arm) for _ in range(B)])
                _, st, _ = model(dist, st, freeze=False)
            
            qs = []
            for w in worlds:
                q1 = [Query(x, (g,), w.table[x][g]) for x in range(w.n) for g in range(2)]
                q2 = []
                for _ in range(n_probe):
                    x = rng.randrange(w.n); wd = tuple(rng.randrange(2) for _ in range(2)); q2.append(Query(x, wd, w.act(x, wd)))
                qs.append(q1 + q2)
            
            pred, nll = readback(model, st, worlds, qs)
        
        for i, w in enumerate(worlds):
            n1 = 2 * w.n; body = texts[i][1:]
            txt = {(body[j], w.gen_names.index(body[j + 1])): body[j + 2] for j in range(0, len(body), 4)}
            tab = {(w.elem_names[x], g): int(pred[i, 2 * x + g]) for x in range(w.n) for g in range(2)}
            tperm, treg = regular_check(txt, list(w.elem_names)); rperm, rreg = regular_check(tab, list(w.elem_names))
            ok = [int(pred[i, j]) == w.elem_names[qs[i][j].answer] for j in range(len(qs[i]))]
            
            # For ledger: attacker also swaps the corresponding lines
            # Pick 4 random depth-1 lines whose x or y is one of the swapped elements
            a_, b_ = relabelings[i]; affected = []
            for x in range(w.n):
                for g in range(2):
                    if w.elem_names[x] in (a_, b_):
                        affected.append((w.elem_names[x], g))
            fake_ledger = []
            if affected:
                for _ in range(min(4, len(affected))):
                    elem, g = rng.choice(affected); y = tab.get((elem, g), w.elem_names[0]); fake_ledger.append((elem, g, y))
            
            r = dict(variant="attacker_fake_ledger", K=K, kind=kind, arm=arm, n=w.n, group=w.group,
                     txt_perm=tperm, txt_reg=treg, rb_perm=rperm, rb_reg=rreg,
                     fidelity=float(np.mean([tab[k] == txt[k] for k in tab])), d1_acc=float(np.mean(ok[:n1])), d2_acc=float(np.mean(ok[n1:])),
                     nll1=[float(v) for v in nll[i, :n1]], ok1=[bool(v) for v in ok[:n1]], nll2=[float(v) for v in nll[i, n1:]], ok2=[bool(v) for v in ok[n1:]],
                     fake_ledger_lines=len(fake_ledger))
            for k_, v in sig.items(): r["sig_" + k_] = [float(u) for u in v[i]]
            recs.append(r)
        done += B
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default="64:struct:seg", help="comma list K:kind:arm")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    ck = torch.load(a.model, map_location="cpu"); cfg = ModelConfig.from_dict(ck["cfg"]); model = build(cfg); model.load_state_dict(ck["model"]); model.eval()
    conds = [(int(c.split(":")[0]), c.split(":")[1], c.split(":")[2]) for c in a.conds.split(",")]
    out = Path(a.out); res = json.loads(out.read_text()) if out.exists() else {"meta": {"model": a.model, "step": ck.get("step")}, "recs": []}
    have = {(r["variant"], r["K"], r["kind"], r["arm"]) for r in res["recs"]}
    t0 = time.time()
    for cond in conds:
        for variant in ("attacker_no_ledger", "attacker_fake_ledger"):
            if (variant, *cond) in have: continue
            rng = random.Random(50000 + 1000 * a.seed + cond[0] * 7 + ("struct", "randline", "randiid").index(cond[1]) * 3 + (100 if variant == "attacker_fake_ledger" else 0))
            if variant == "attacker_no_ledger":
                res["recs"] += run_attacker_part_a(model, rng, cond, a.n, a.batch)
            else:
                res["recs"] += run_attacker_part_b(model, rng, cond, a.n, a.batch)
            out.write_text(json.dumps(res)); print(f"{cond} {variant} done {time.time()-t0:.0f}s", flush=True)
