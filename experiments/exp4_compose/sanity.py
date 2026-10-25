"""Model-free and plumbing checks for exp 4 (run before trusting any number).

1. The oracle composer is exact on every query (the lessons determine the answer).
2. The content null is a derangement of element names, keeps a valid Cayley graph, and changes the implied
   answer for most queries.
3. Protocol plumbing: ``no_adapt`` equals asking the query from a fresh zero state; ``fw_reset`` with S zeroed equals
   ``no_adapt`` (so fw_reset differs from no_adapt only through the fast weights S).
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build, expand_state, model_config, pad_queries, run_protocol, sample_batch  # noqa: E402
from task import GROUPS, Query, baseline_predictions, lesson_lines, make_query, make_world, query_tokens, relabel, lesson_tokens  # noqa: E402

rng = random.Random(123)
n_items = n_null_diff = 0
for _ in range(3000):
    w = make_world(rng)
    L = lesson_tokens(w, rng)
    assert len(lesson_lines(w, L)) == 2 * w.n
    nw = relabel(w, rng)
    assert sorted(nw.elem_names) == sorted(w.elem_names) and all(a != b for a, b in zip(nw.elem_names, w.elem_names))
    for d in range(1, 7):
        q = make_query(w, rng, d)
        assert baseline_predictions(w, L, q)["oracle"] == w.elem_names[q.answer]
        ns = nw.elem_names.index(w.elem_names[q.start])
        n_items += 1
        n_null_diff += nw.elem_names[nw.act(ns, q.word)] != w.elem_names[q.answer]
print(f"oracle exact on {n_items} items; content-null answer differs from the true answer on {n_null_diff / n_items:.3f}")

torch.manual_seed(0)
m = build(model_config())
m.eval()
worlds, lessons, qrows, qrecs, _, _ = sample_batch(random.Random(5), 4, 3, [1, 2, 3])
qx, qpos = pad_queries(qrows)
Lt = torch.tensor(lessons)
with torch.no_grad():
    a = run_protocol(m, Lt, qx, qpos, 3, "no_adapt")
    logits, _, _ = m(qx, m.init_state(qx.shape[0]))
    b = logits[torch.arange(qx.shape[0]), qpos]
    print("no_adapt == fresh-state query: max |diff| =", float((a - b).abs().max()))
    _, st, _ = m(Lt, m.init_state(4))
    for layer in st.layers:
        layer.S = torch.zeros_like(layer.S)
    stq = expand_state(st, 3, reset_activations=True)
    c = m(qx, stq)[0][torch.arange(qx.shape[0]), qpos]
    print("fw_reset with S zeroed == no_adapt: max |diff| =", float((a - c).abs().max()))
    f = run_protocol(m, Lt, qx, qpos, 3, "fw_reset")
    print("fw_reset vs no_adapt (random init, S written): max |diff| =", float((a - f).abs().max()))
