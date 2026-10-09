# Exp 4: first-answer test with nothing to copy (group-action compositions)

**Status: a null / training failure for composition. One real but partial depth-1 result (curriculum model, 1 seed).**
No claim that fast-weight adaptation yields new compositional knowledge is supported by these runs. Every number below is
read from `experiments/results/exp4/summary.json`, `curr/summary.json`, `diag_depth1_two_candidate.json` (all produced by the
scripts in this directory from per-item CSVs).

## Task
`task.py`. Each episode draws a group (Z5, Z6, Z7, Z8, S3, D4) as permutations, two generators, a random injective naming of the
elements (tokens E0..E23) and of the generators (G0..G5). Lessons are the full right-Cayley graph: one line `x g y ;` per element
and generator (10-16 lines), in random order. The query `Q x w1..wd =` asks for x*w1*...*wd, one element token.
Depth 1 is a lesson line looked up (exact recall, not a composition). Depth >= 2 composites are never shown in any lesson and are uniquely
determined by the lessons; random per-episode names mean nothing outside the episode determines the answer, and the template
"copy the last lesson answer" is at chance. Every scored item is the first and only query after its lessons (no earlier worked
query), exact match on a single token. `sanity.py` (output `results/exp4/sanity.txt`): the oracle composer is exact on 18000/18000
items; the content-null lessons give a different answer on 66.3% of items.

## Model and adaptation path
The repo's `PlasticLM` / `PlasticCore` (gated delta-rule fast-weight memory `S`, `plastic/model/`), **1.2M parameters** (d_model 128, 4 layers,
4 heads, chunk 16, vocab 40), meta-trained from scratch on CPU (the 6M published-size PlasticCore and the gated HF checkpoint were not used: no
GPU/time; the 760M TTT chat model is not the same adaptation path and could not be run on this CPU box). Training: lessons are read with fast-weight writes
on, then 4 independent first queries per episode; loss = cross-entropy on the answer token only; AdamW lr 1e-3 cosine, batch 32 episodes, 5000 steps
(main, 3 seeds, depths 1-4 mixed) or 8000 steps (curriculum, seed 0, depths unlocked when the hardest unlocked depth reaches 0.9 train accuracy).
**This is the in-episode fast-weight path (meta-learned inner loop), not the text-rule evals' lasting W0 gradient update**; no W0 update is made at test time here.

## Arms (all scored on the same items, true answers)
- `fw_reset`: lessons written into S, then SSM state and conv buffers zeroed; only S carries the lessons (the fast-weight-only arm).
- `continuous`: lessons and query in one stream (S plus activation state): the nearest "in-context" arm this architecture has (no attention).
- `no_adapt`: lessons read with `freeze=True` (beta=0, S stays at W0=0) and activations zeroed = the query with no lessons (identical to a fresh-state query; checked in `sanity.py`, max diff 0.0). `ssm_only`: lessons read frozen but activations carried.
- `null_*`: the content null: same lessons format with every element token deranged (valid Cayley graph, each name wrong), scored against the TRUE answer (and against the null lessons' own answer, reported separately).
- Model-free: `oracle` (lookup+compose, 100%), `one_hop` (knows all lines, answers x*w1), `other_gen`, `copy_start`, `last_line` (template copier).

## Results

Eval: 3 model seeds x 500 episodes x 6 depths = 12000 items/seed/arm (36000 pooled for the main models); Wilson 95% CIs on pooled counts; chance ~0.157 (mean 1/|G|).

### Main models (non-curriculum, 3 seeds): they did not learn the task
Final training loss ~1.8 (about ln|G|), train accuracy 0.16-0.20 at every depth.

| arm | depth 1 (recall) [Wilson] | depth >= 2 (composition) [Wilson] | non-trivial compositions* |
|---|---|---|---|
| fast weights only (`fw_reset`) | 0.187 [0.178, 0.197] | 0.158 [0.154, 0.163] | 0.115 [0.110, 0.120] |
| in-context (`continuous`) | 0.187 [0.177, 0.197] | 0.158 [0.154, 0.162] | 0.107 [0.103, 0.112] |
| no adaptation (beta=0 / W0) | 0.011 [0.009, 0.014] | 0.175 [0.171, 0.180] | 0.000 |
| content null, fast weights | 0.188 [0.179, 0.198] | 0.158 [0.154, 0.162] | 0.116 [0.111, 0.120] |
| one-hop lookup, no composition | 1.000 | 0.129 [0.125, 0.133] | 0.000 |
| template copier (last lesson answer) | 0.159 [0.150, 0.168] | 0.156 [0.152, 0.161] | 0.000 |
| oracle | 1.000 | 1.000 | 1.000 |

Paired (episode-cluster bootstrap, depth >= 2): fw_reset minus content null +0.001 [-0.002, +0.003]; fw_reset minus no_adapt -0.017 [-0.022, -0.012]
(no_adapt is higher because with no lessons the model emits the start element, right whenever the word reduces to the identity).
Reading: the main models are at chance on compositions, with no difference between true and wrong lessons. Per-seed values (0.167/0.160/0.163 overall) agree.
*non-trivial = depth >= 2, answer is not the start element, not x*w1, not the last lesson answer (n = 17636).

### Curriculum model (seed 0 only, 8000 steps; n = 2000 items per depth)
| depth | fast weights [Wilson] | in-context | no adapt | content null (vs true answer) | one-hop lookup |
|---|---|---|---|---|---|
| 1 | **0.507** [0.486, 0.529] | 0.512 [0.490, 0.534] | 0.024 [0.018, 0.031] | 0.198 [0.181, 0.216] | 1.0 |
| 2 | 0.046 [0.038, 0.056] | 0.050 | 0.103 | 0.163 | 0.0 |
| 3 | 0.173 [0.158, 0.191] | 0.172 | 0.101 | 0.159 | 0.181 |
| 4 | 0.130 [0.116, 0.145] | 0.129 | 0.148 | 0.138 | 0.141 |
| 5 | 0.229 [0.211, 0.248] | 0.224 | 0.057 | 0.186 | 0.246 |
| 6 | 0.102 [0.089, 0.116] | 0.103 | 0.118 | 0.143 | 0.087 |

Chance is 0.16. Depth >= 2 is at or below chance. Depth-1 first-answer exact is 0.51 with the fast weights versus 0.02 with no adaptation and 0.20 with
wrong lessons, so the model does use the lessons: with the null lessons it agrees with *the null lessons' own answers* 0.499 of the time.
That is genuinely new, uncopyable first-answer knowledge, but only of a lookup, and only half of it (below). Depth >= 2: fw_reset minus content null -0.022 [-0.031, -0.013];
fw_reset minus no_adapt +0.031 [+0.022, +0.040] (pooled depth 2-6; the plateau is not above no-adaptation per depth).

## Why the curriculum stayed at depth 1 (bug or null?)
Not an unlock-logic bug: the criterion (train accuracy >= 0.9 on the hardest unlocked depth, evaluated every 100 steps) behaved as written, and the run never exceeded
0.50. Training went through a long plateau (accuracy ~0.19 until step ~3500, loss falls 1.73 -> 0.77 between steps 3500 and 6500, then flat at 0.77-0.78 with lr already
decayed to 1e-4 at the end). The "0.5 = chance" reading is wrong: chance is ~0.16. Diagnostic (`diag_depth1_two_candidate.json`, 2000 depth-1 items): the curriculum model's
prediction is one of the *two* true successors of the start element (x*g1 or x*g2) on 99.95% of items, correct generator 50.75%, wrong generator 49.2%. So it learned to retrieve the successor set
of x from the fast weights (keyed on x) but never learned to bind the generator, giving exactly 1/2 accuracy and loss ~ln 2 + extra. The three non-curriculum models are at 36-39%
"in the two-successor set", i.e. they have not even reached that stage. Conclusion: a genuine optimisation failure of this model/budget (5000-8000 steps, 1.2M params), not an eval bug; its cause is not isolated.
An untested suspect (hypothesis only): a lesson line makes its prediction at the generator token (position of g), while the query predicts at the `=` token, so key and query positions differ and
the memory must learn a shift before it can bind (x, g).

## Interpretation
- Format adaptation vs knowledge: depth-1 recall from fast weights (0.51 vs 0.02 no-adapt vs 0.20 wrong-lessons, curriculum seed only) is not a template effect: random per-episode names, no earlier answer to copy.
  The template copier scores 0.159 at depth 1 and `last_line` 0.156 at depth >= 2, and the content-null control does not reproduce the effect. This is within-episode associative recall (as MQAR already shows), not rule knowledge in the weights.
- Composition: **not shown.** Depth >= 2 first-answer accuracy is at chance for every arm and every model, with true lessons indistinguishable from content-null lessons (main models +0.001 [-0.002, +0.003]) and one-hop lookup. The oracle scores 100%, so the items are solvable; the models just did not learn to solve them.
- Therefore these runs say nothing for or against "fast-weight adaptation yields compositional knowledge"; they show only that this small model did not learn the task at this budget.

## Caveats
- Small from-scratch 1.2M model, CPU, 5000/8000 steps; not the 6M published size, not the gated HF checkpoint, not the 760M TTT model, and no lasting W0 update (meta-trained inner loop only).
- Curriculum result is one seed (the 3 seeds are the non-curriculum runs). Curriculum eval used 2000 items per depth; the plot's "depth not trained" shading is hard-coded for depths > 4 and is wrong for the curriculum model (only depth 1 was trained).
- The in-context-only arm is approximate: PlasticCore has no attention; `continuous` carries SSM/conv state plus S and `ssm_only` carries only activations. Neither beat `fw_reset`.
- Depth >= 2 chance varies by group (1/|G|, pooled 0.157); small group-structure shortcuts (identity words) make `copy_start` 0.175 at depth >= 2, which is why the non-trivial subset is reported.
- Eval episodes use rng seeds 10000+, training 1000+seed; names are random per episode, so there is no content to leak, but no explicit collision check of whole episodes was run.
- Eval items include all 6 depths per episode; depths 5-6 are length-held-out (training used depths <= 4).
- Model weights (`models*/`, 19 MB) are not committed; train logs are in `experiments/logs/exp4/`.

## One proposed fix (not run)
Align lesson and query layouts so a single key form serves both: make a query a lesson line with the answer slot, `x g1 ?` for depth 1 (predict at the same position type as a lesson's y) and
chain depth d as d such lines `x g1 y1 ; y1 g2 ?`... shown as an explicit scratch-free query only at test; and train with a longer budget (>= 20k steps) with the depth-1 stage run to >= 0.95 before unlocking.
Falsifier: if depth-1 stays at 0.50 with aligned layouts, the binding failure is not about positions.

## Reproduce
```
cd /workspace/plastic-exp4/experiments/exp4_compose && . ../../.venv/bin/activate && export OMP_NUM_THREADS=1
python sanity.py
for s in 0 1 2; do python train.py --seed $s --out ../results/exp4/models/seed$s; done          # 5000 steps, ~2 h each
python train.py --seed 0 --curriculum adaptive --steps 8000 --out ../results/exp4/models_curr/seed0   # ~3 h
for s in 0 1 2; do python evaluate.py --model ../results/exp4/models/seed$s/model.pt --eval-seed $s --out ../results/exp4/items_seed$s.csv.gz; done
python evaluate.py --model ../results/exp4/models_curr/seed0/model.pt --eval-seed 0 --out ../results/exp4/curr/items_seed0.csv.gz
python aggregate.py ../results/exp4 ; python aggregate.py ../results/exp4/curr
```
Files: `results/exp4/summary.{json,md}`, `summary_by_depth.csv`, `accuracy_by_depth.png`, `nontrivial_per_seed.png`, `curr/` (same), `diag_depth1_two_candidate.json`, `items_seed*.csv.gz` (per-item predictions).
