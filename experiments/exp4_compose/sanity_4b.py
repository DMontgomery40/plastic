"""Exp 4b checks: the aligned query layout, plumbing (CPU, tiny)."""
import random, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build, model_config, pad_queries, run_protocol, sample_batch
from task import BOS, EQ, Q, SEP, lesson_tokens, make_query, make_world, query_tokens

rng = random.Random(5)
for _ in range(2000):
    w = make_world(rng); L = lesson_tokens(w, rng)
    q = make_query(w, rng, rng.randint(1, 6))
    o, a = query_tokens(w, q, "old"), query_tokens(w, q, "aligned")
    assert o[-1] == EQ and a == o[:-1] and a[0] == Q and a[-1] == w.gen_names[q.word[-1]]
    if q.depth == 1:
        # depth 1: the aligned query is the lesson line "x g" with the answer predicted at g (prefix of "x g y ;")
        i = [k for k in range(1, len(L), 4) if L[k] == a[1] and L[k + 1] == a[2]][0]
        assert L[i + 2] == w.elem_names[q.answer] and L[i - 1] in (BOS, SEP)
# padding/pos: prediction position is the last prompt token for both layouts
for lay in ("old", "aligned"):
    worlds, lessons, qrows, qrecs, _, _ = sample_batch(rng, 8, 4, [1, 2, 3], layout=lay)
    ans = [worlds[i].elem_names[q.answer] for i, q in qrecs]
    qx, qpos = pad_queries(qrows, ans)
    for r, row, p, a_ in zip(range(len(qrows)), qrows, qpos.tolist(), ans):
        assert p == len(row) - 1 and qx[r, p].item() == row[-1] and qx[r, p + 1].item() == a_
# causality: logits at the prediction position do not depend on the appended answer token (no leak)
torch.manual_seed(0)
m = build(model_config()); m.eval()
worlds, lessons, qrows, qrecs, _, _ = sample_batch(rng, 4, 2, [1, 2], layout="aligned")
ans = [worlds[i].elem_names[q.answer] for i, q in qrecs]
qx, qpos = pad_queries(qrows, ans); qx2, _ = pad_queries(qrows, [(a + 1) % 40 for a in ans])
with torch.no_grad():
    l1 = run_protocol(m, torch.tensor(lessons), qx, qpos, 2, "fw_reset"); l2 = run_protocol(m, torch.tensor(lessons), qx2, qpos, 2, "fw_reset")
print("max logit diff when the appended answer token changes:", (l1 - l2).abs().max().item())
assert (l1 - l2).abs().max().item() < 1e-5
print("sanity_4b OK")
