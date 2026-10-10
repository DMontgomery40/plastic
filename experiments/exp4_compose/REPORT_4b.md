# Exp 4b: first-answer composition with an aligned query layout, 60k GPU steps, control arms and CIs

**Status in one paragraph.** With the query laid out like a lesson line (the fix proposed in `REPORT.md`) and a 60k-step adaptive curriculum on an A10G, the 1.2M-parameter PlasticLM
now learns the task far beyond exp 4: depth 1 (lookup) is exact for every run, depth 2 is 0.90-1.00 and depth 3 is 0.92-0.95 on two of three aligned seeds. Those depth-2/3 results are
**real, content-dependent composition of lessons that are never shown as a composition**: the same fast weights fed wrong lessons (names deranged) answer what the *wrong* lessons imply
(0.96-1.00 at depth 2), not the true answer, and the paired gap fast weights minus content null is large with tight CIs. At depth 4 only seed 1 clearly composes (0.530 [0.509, 0.552]);
seed 0 is at 0.212 and seed 2 never unlocked depth 4, and for both the fast-weights-minus-null gap is about zero. Seed 2 is a depth-3 failure at this budget (0.296, below a set-prior shortcut),
and the old-layout control reaches only depth 2 (0.695), though it learned depth 1 as fast as the aligned runs, so the layout is not what unlocked depth 1. Everything is 3 seeds, one model size, one training budget; item-level CIs do not cover seed-to-seed variation.

## What was run
- Training/eval on a Hugging Face Job (`6ac97296fee2c9007017eecc`, a10g-large, tag `main4b_60k`, 3 h cap; launcher/entry: `jobs/launch_4b.py`, `jobs/job_entry_4b.sh`; env `STEPS=60000 MIN_D1=20000 TLIM=9900 ABORT_STEP=30000 ABORT_ACC=0.6`, see `jobs/jobs_4b.log`).
  Four runs trained concurrently on the one GPU: **aligned layout seeds 0, 1, 2** and an **old-layout seed 0** control (identical hyper-parameters, only `--layout old`). Each run: `python train.py --seed S --layout {aligned|old} --device cuda --curriculum adaptive --train-depths 1,2,3,4,5 --lr-sched const_decay --unlock-acc 0.9 --unlock-acc1 0.95 --steps 60000 --min-depth1-steps 20000 --abort-depth1-step 30000 --abort-depth1-acc 0.6 --time-limit-s 9900` (all four runs completed the 60,000 steps; the abort rule never fired)
  (adaptive curriculum: a depth unlocks when the hardest unlocked depth reaches 0.9 train accuracy, depth 2 not before step 20000, depth 1 needs 0.95; constant lr 1e-3 then decay over the last 15%; batch 32 episodes x 4 first queries; same model as exp 4: d_model 128, 4 layers, 4 heads, 1.2M parameters, answer-token loss only).
  Training wall time 9,270-9,276 s per run (4 runs sharing the GPU, 95-99% utilisation).
- The query layout: old `Q x g1..gd =` predicts at `=`; aligned `Q x g1..gd` predicts at the last generator token, exactly as a lesson line `x g y ;` predicts y at `g` (`task.query_tokens`, checked by `sanity_4b.py`, output `results/exp4b/sanity_4b.txt`).
- Eval on the GPU (`evaluate.py`, 12,000 items per run = 40 batches x 50 episodes x 6 depths, one first query per item, so 2,000 items per depth, eval rng seeds 10000+S, so the aligned seed-0 and old-layout seed-0 runs see *identical episodes and queries*) wrote item-level CSVs to the private HF dataset `dmontgomery40/plastic-gpu-results` under `results/main4b_60k`.
- **Control arms and CIs on CPU (this report's new work)**, no GPU/paid compute: `controls_4b.py` regenerates the identical items from the same rng stream and re-runs the saved checkpoints on CPU (about 2 min per run). It re-computes `fw_reset`, `no_adapt` and the deranged content null, and **all three reproduce the GPU CSV predictions 100% (12,000/12,000 items, all four runs; answers and null answers also identical)**. It adds a foreign-lessons null and model-free set-prior shortcuts. `analyze_controls_4b.py` gives Wilson CIs, paired bootstrap CIs and the plots.

### Arms (all scored against the true answer unless stated; every scored query is the first and only query after its lessons)
- **fast weights** (`fw_reset`): lessons written into the fast weights S, activation state zeroed, only S carries the lessons.
- **no adaptation**: lessons read with writes frozen (beta = 0), activation state zeroed (equals the query with no lessons).
- **content null (deranged names)**: same format, same token set, a valid Cayley graph, but every element name moved to another element by a derangement. Scored against the TRUE answer ("all items"), on the **discriminating subset** where the wrong lessons imply a different answer than the true one (null answer != true answer; 59-79% of items depending on depth), and, for the fast-weights arm, against the wrong lessons' *own* implied answer (does the model compose whatever it is told).
- **foreign-lessons null**: the lessons of the neighbouring episode in the batch (same length and format, different group/names), scored against the true answer.
- **template copier** (`last_line`: answers the result of the last lesson line), **one-hop lookup** (`one_hop`: knows every lesson line, answers x*w1, i.e. no composition), `other_gen`, `copy_start` (identity words), **oracle** (compose the lessons; 100% by construction, verified in `sanity.py`).
- **set-prior shortcuts (task-structure, expected accuracy computed exactly from the world tables, not sampled)**: a guesser that knows all lesson lines but not the composition and guesses uniformly inside the set of elements consistent with partial information: `E[p_set_x]` the d-step reach of x (knows only the depth), `E[p_set_first]` reach of x*w1 in d-1 steps (knows the first hop), `E[p_set_last]` reach of x in d-1 steps then w_d. `E[best set prior]` takes the per-item best of the three. Reference: any procedure that gets every letter of the word right except one unknown letter (two candidate answers) tops out at 0.5 at every depth >= 2, so accuracy clearly above 0.5 needs all letters.

## Results

Items per depth: 2,000 per run, 6,000 for the 3 aligned seeds pooled. Brackets: Wilson 95% (accuracy) or bootstrap 95% over items (paired differences, E[.] rows). Chance is about 0.157 (mean 1/|G|).
\* Depths 5 and 6 are length-held-out (no run unlocked depth 5 in training).

### Headline: first-answer exact accuracy per depth (fast weights), 3 aligned seeds + old layout

| run | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| aligned s0 | 1.000 [0.998, 1.000] | 0.999 [0.996, 1.000] | 0.953 [0.942, 0.961] | 0.212 [0.195, 0.230] | 0.173 [0.158, 0.191] | 0.141 [0.126, 0.157] |
| aligned s1 | 1.000 [0.998, 1.000] | 0.999 [0.996, 0.999] | 0.919 [0.907, 0.931] | 0.530 [0.509, 0.552] | 0.179 [0.163, 0.197] | 0.172 [0.157, 0.190] |
| aligned s2 | 1.000 [0.998, 1.000] | 0.903 [0.889, 0.915] | 0.296 [0.276, 0.316] | 0.158 [0.143, 0.175] | 0.139 [0.125, 0.155] | 0.174 [0.158, 0.192] |
| aligned pooled (3 seeds) | 1.000 [0.999, 1.000] | 0.967 [0.962, 0.971] | 0.723 [0.711, 0.734] | 0.300 [0.289, 0.312] | 0.164 [0.155, 0.174] | 0.163 [0.154, 0.172] |
| old layout s0 | 0.999 [0.996, 1.000] | 0.695 [0.674, 0.714] | 0.148 [0.134, 0.165] | 0.150 [0.135, 0.166] | 0.165 [0.149, 0.182] | 0.154 [0.139, 0.171] |

The numbers above are held-out item-level evals. The last-1000-step *training* window gives similar values (aligned s0 1.00/0.999/0.949/0.225, s1 1.00/0.998/0.909/0.541, s2 1.00/0.896/0.305/not unlocked, old 0.999/0.678); the single last logged
100-step window quoted in the job log (e.g. s0 d3 0.969, d4 0.231) is a noisier sample of the same quantity. The held-out eval numbers are the ones to cite.

Curriculum timeline (from `logs/*.train_log.json`; the depth-2 floor of 20,000 steps is a flag, not a measurement): depth-1 train accuracy first reaches 0.95 at step 15.9k (s0), 16.1k (s1), 11.7k (s2) and 10.7k (old).
Unlock steps (depth 3 / depth 4): s0 32.3k / 55.1k, s1 39.6k / 56.4k, s2 56.6k / never, old: depth 3 never. So the last depth each aligned model reached had only 3,400-4,900 steps of training before the run ended (s0 depth 4: 4.9k, s1 depth 4: 3.6k, s2 depth 3: 3.4k).
See `training_curves.png`.

### Control arms and per-depth tables

#### aligned seed 0 (n = 2000 items per depth; Wilson 95% CIs; expected-accuracy rows E[.] use bootstrap CIs)

| arm | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| **fast weights (`fw_reset`)** | 1.000 [0.998, 1.000] | 0.999 [0.996, 1.000] | 0.953 [0.942, 0.961] | 0.212 [0.195, 0.230] | 0.173 [0.158, 0.191] | 0.141 [0.126, 0.157] |
| fast weights on WRONG lessons, scored vs the wrong lessons' own answer | 1.000 [0.998, 1.000] | 0.999 [0.996, 0.999] | 0.954 [0.944, 0.962] | 0.234 [0.215, 0.253] | 0.162 [0.146, 0.178] | 0.149 [0.135, 0.166] |
| content null (wrong lessons), scored vs TRUE answer | 0.237 [0.219, 0.256] | 0.357 [0.337, 0.379] | 0.338 [0.318, 0.359] | 0.196 [0.179, 0.214] | 0.146 [0.132, 0.163] | 0.162 [0.146, 0.178] |
| content null, only items where null answer != true answer | 0.000 [0.000, 0.003] | 0.000 [0.000, 0.003] | 0.003 [0.001, 0.008] | 0.136 [0.118, 0.157] | 0.137 [0.120, 0.156] | 0.150 [0.131, 0.171] |
| fast weights, same items (null answer != true answer) | 1.000 [0.997, 1.000] | 0.998 [0.994, 1.000] | 0.947 [0.934, 0.958] | 0.176 [0.155, 0.199] | 0.171 [0.152, 0.191] | 0.115 [0.098, 0.134] |
| foreign-lessons null (another episode's lessons) | 0.040 [0.032, 0.050] | 0.048 [0.039, 0.058] | 0.043 [0.035, 0.052] | 0.043 [0.035, 0.053] | 0.044 [0.036, 0.054] | 0.038 [0.030, 0.047] |
| no adaptation (beta=0) | 0.043 [0.035, 0.053] | 0.059 [0.050, 0.071] | 0.067 [0.056, 0.078] | 0.129 [0.115, 0.144] | 0.045 [0.037, 0.056] | 0.077 [0.066, 0.090] |
| template copier (last lesson answer) | 0.163 [0.147, 0.179] | 0.158 [0.142, 0.174] | 0.152 [0.137, 0.168] | 0.156 [0.141, 0.173] | 0.157 [0.142, 0.174] | 0.151 [0.136, 0.168] |
| one-hop lookup (x*w1) | 1.000 [0.998, 1.000] | 0.000 [0.000, 0.002] | 0.181 [0.165, 0.198] | 0.141 [0.127, 0.157] | 0.246 [0.228, 0.266] | 0.087 [0.076, 0.101] |
| E[set prior: uniform over d-step reach of x] | 0.500 [0.500, 0.500] | 0.327 [0.325, 0.329] | 0.259 [0.257, 0.262] | 0.225 [0.222, 0.228] | 0.213 [0.210, 0.216] | 0.205 [0.202, 0.209] |
| E[best set prior: knows 1st hop or last letter] | 1.000 [1.000, 1.000] | 0.500 [0.500, 0.500] | 0.327 [0.325, 0.329] | 0.259 [0.257, 0.262] | 0.225 [0.222, 0.228] | 0.213 [0.210, 0.216] |
| oracle | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] |

#### aligned seed 1 (n = 2000 items per depth; Wilson 95% CIs; expected-accuracy rows E[.] use bootstrap CIs)

| arm | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| **fast weights (`fw_reset`)** | 1.000 [0.998, 1.000] | 0.999 [0.996, 0.999] | 0.919 [0.907, 0.931] | 0.530 [0.509, 0.552] | 0.179 [0.163, 0.197] | 0.172 [0.157, 0.190] |
| fast weights on WRONG lessons, scored vs the wrong lessons' own answer | 1.000 [0.998, 1.000] | 0.998 [0.994, 0.999] | 0.914 [0.901, 0.926] | 0.540 [0.518, 0.562] | 0.182 [0.166, 0.200] | 0.153 [0.138, 0.169] |
| content null (wrong lessons), scored vs TRUE answer | 0.229 [0.211, 0.248] | 0.349 [0.329, 0.371] | 0.297 [0.278, 0.318] | 0.290 [0.271, 0.311] | 0.160 [0.145, 0.177] | 0.154 [0.139, 0.170] |
| content null, only items where null answer != true answer | 0.000 [0.000, 0.002] | 0.000 [0.000, 0.003] | 0.015 [0.009, 0.022] | 0.087 [0.072, 0.104] | 0.138 [0.121, 0.158] | 0.139 [0.121, 0.159] |
| fast weights, same items (null answer != true answer) | 1.000 [0.998, 1.000] | 0.998 [0.993, 0.999] | 0.913 [0.897, 0.927] | 0.506 [0.478, 0.534] | 0.167 [0.148, 0.187] | 0.167 [0.147, 0.189] |
| foreign-lessons null (another episode's lessons) | 0.045 [0.037, 0.055] | 0.050 [0.041, 0.060] | 0.037 [0.030, 0.046] | 0.051 [0.042, 0.061] | 0.049 [0.040, 0.059] | 0.045 [0.037, 0.056] |
| no adaptation (beta=0) | 0.036 [0.029, 0.046] | 0.059 [0.050, 0.071] | 0.058 [0.049, 0.069] | 0.123 [0.109, 0.138] | 0.093 [0.081, 0.107] | 0.132 [0.117, 0.147] |
| template copier (last lesson answer) | 0.153 [0.138, 0.169] | 0.148 [0.133, 0.164] | 0.143 [0.129, 0.160] | 0.164 [0.148, 0.181] | 0.160 [0.144, 0.176] | 0.140 [0.125, 0.156] |
| one-hop lookup (x*w1) | 1.000 [0.998, 1.000] | 0.000 [0.000, 0.002] | 0.172 [0.156, 0.189] | 0.127 [0.113, 0.142] | 0.228 [0.210, 0.246] | 0.105 [0.092, 0.119] |
| E[set prior: uniform over d-step reach of x] | 0.500 [0.500, 0.500] | 0.328 [0.326, 0.330] | 0.262 [0.259, 0.265] | 0.230 [0.227, 0.233] | 0.214 [0.210, 0.217] | 0.204 [0.200, 0.208] |
| E[best set prior: knows 1st hop or last letter] | 1.000 [1.000, 1.000] | 0.500 [0.500, 0.500] | 0.328 [0.326, 0.330] | 0.262 [0.259, 0.265] | 0.230 [0.227, 0.233] | 0.214 [0.210, 0.217] |
| oracle | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] |

#### aligned seed 2 (n = 2000 items per depth; Wilson 95% CIs; expected-accuracy rows E[.] use bootstrap CIs)

| arm | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| **fast weights (`fw_reset`)** | 1.000 [0.998, 1.000] | 0.903 [0.889, 0.915] | 0.296 [0.276, 0.316] | 0.158 [0.143, 0.175] | 0.139 [0.125, 0.155] | 0.174 [0.158, 0.192] |
| fast weights on WRONG lessons, scored vs the wrong lessons' own answer | 1.000 [0.998, 1.000] | 0.894 [0.880, 0.907] | 0.323 [0.302, 0.343] | 0.161 [0.146, 0.178] | 0.136 [0.122, 0.152] | 0.179 [0.163, 0.196] |
| content null (wrong lessons), scored vs TRUE answer | 0.209 [0.192, 0.228] | 0.321 [0.301, 0.342] | 0.198 [0.181, 0.216] | 0.170 [0.154, 0.187] | 0.154 [0.139, 0.170] | 0.158 [0.142, 0.174] |
| content null, only items where null answer != true answer | 0.000 [0.000, 0.002] | 0.020 [0.014, 0.029] | 0.116 [0.100, 0.135] | 0.161 [0.141, 0.183] | 0.160 [0.142, 0.180] | 0.142 [0.124, 0.163] |
| fast weights, same items (null answer != true answer) | 1.000 [0.998, 1.000] | 0.914 [0.897, 0.928] | 0.278 [0.254, 0.302] | 0.142 [0.123, 0.163] | 0.135 [0.118, 0.154] | 0.171 [0.151, 0.193] |
| foreign-lessons null (another episode's lessons) | 0.036 [0.029, 0.046] | 0.052 [0.044, 0.063] | 0.043 [0.035, 0.053] | 0.039 [0.031, 0.048] | 0.044 [0.036, 0.054] | 0.042 [0.034, 0.051] |
| no adaptation (beta=0) | 0.038 [0.030, 0.047] | 0.041 [0.033, 0.051] | 0.042 [0.034, 0.052] | 0.051 [0.042, 0.062] | 0.032 [0.025, 0.040] | 0.043 [0.035, 0.053] |
| template copier (last lesson answer) | 0.161 [0.146, 0.178] | 0.167 [0.151, 0.183] | 0.161 [0.146, 0.178] | 0.163 [0.147, 0.180] | 0.160 [0.144, 0.176] | 0.167 [0.151, 0.184] |
| one-hop lookup (x*w1) | 1.000 [0.998, 1.000] | 0.000 [0.000, 0.002] | 0.165 [0.149, 0.182] | 0.141 [0.126, 0.157] | 0.224 [0.206, 0.242] | 0.118 [0.105, 0.133] |
| E[set prior: uniform over d-step reach of x] | 0.500 [0.500, 0.500] | 0.330 [0.328, 0.332] | 0.261 [0.259, 0.264] | 0.225 [0.222, 0.228] | 0.208 [0.205, 0.212] | 0.198 [0.194, 0.201] |
| E[best set prior: knows 1st hop or last letter] | 1.000 [1.000, 1.000] | 0.500 [0.500, 0.500] | 0.330 [0.328, 0.332] | 0.261 [0.259, 0.264] | 0.225 [0.222, 0.228] | 0.208 [0.205, 0.212] |
| oracle | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] |

#### aligned, 3 seeds pooled (n = 6000 items per depth; Wilson 95% CIs; expected-accuracy rows E[.] use bootstrap CIs)

| arm | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| **fast weights (`fw_reset`)** | 1.000 [0.999, 1.000] | 0.967 [0.962, 0.971] | 0.723 [0.711, 0.734] | 0.300 [0.289, 0.312] | 0.164 [0.155, 0.174] | 0.163 [0.154, 0.172] |
| fast weights on WRONG lessons, scored vs the wrong lessons' own answer | 1.000 [0.999, 1.000] | 0.964 [0.958, 0.968] | 0.730 [0.719, 0.741] | 0.311 [0.300, 0.323] | 0.160 [0.151, 0.169] | 0.161 [0.151, 0.170] |
| content null (wrong lessons), scored vs TRUE answer | 0.225 [0.215, 0.236] | 0.343 [0.331, 0.355] | 0.278 [0.266, 0.289] | 0.219 [0.208, 0.229] | 0.153 [0.145, 0.163] | 0.158 [0.149, 0.167] |
| content null, only items where null answer != true answer | 0.000 [0.000, 0.001] | 0.007 [0.005, 0.010] | 0.044 [0.038, 0.051] | 0.128 [0.117, 0.139] | 0.145 [0.135, 0.156] | 0.144 [0.132, 0.155] |
| fast weights, same items (null answer != true answer) | 1.000 [0.999, 1.000] | 0.970 [0.964, 0.975] | 0.714 [0.700, 0.728] | 0.277 [0.263, 0.292] | 0.158 [0.147, 0.169] | 0.151 [0.140, 0.163] |
| foreign-lessons null (another episode's lessons) | 0.041 [0.036, 0.046] | 0.050 [0.045, 0.056] | 0.041 [0.036, 0.046] | 0.044 [0.039, 0.050] | 0.046 [0.041, 0.052] | 0.042 [0.037, 0.047] |
| no adaptation (beta=0) | 0.039 [0.035, 0.045] | 0.053 [0.048, 0.059] | 0.056 [0.050, 0.062] | 0.101 [0.094, 0.109] | 0.057 [0.051, 0.063] | 0.084 [0.077, 0.091] |
| template copier (last lesson answer) | 0.159 [0.150, 0.168] | 0.157 [0.148, 0.167] | 0.152 [0.143, 0.161] | 0.161 [0.152, 0.171] | 0.159 [0.150, 0.168] | 0.153 [0.144, 0.162] |
| one-hop lookup (x*w1) | 1.000 [0.999, 1.000] | 0.000 [0.000, 0.001] | 0.173 [0.163, 0.182] | 0.137 [0.128, 0.145] | 0.233 [0.222, 0.243] | 0.104 [0.096, 0.112] |
| E[set prior: uniform over d-step reach of x] | 0.500 [0.500, 0.500] | 0.328 [0.327, 0.330] | 0.261 [0.259, 0.262] | 0.227 [0.225, 0.228] | 0.212 [0.210, 0.214] | 0.202 [0.200, 0.204] |
| E[best set prior: knows 1st hop or last letter] | 1.000 [1.000, 1.000] | 0.500 [0.500, 0.500] | 0.328 [0.327, 0.330] | 0.261 [0.259, 0.262] | 0.227 [0.225, 0.228] | 0.212 [0.210, 0.214] |
| oracle | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] |

#### old layout, seed 0 (n = 2000 items per depth; Wilson 95% CIs; expected-accuracy rows E[.] use bootstrap CIs)

| arm | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| **fast weights (`fw_reset`)** | 0.999 [0.996, 1.000] | 0.695 [0.674, 0.714] | 0.148 [0.134, 0.165] | 0.150 [0.135, 0.166] | 0.165 [0.149, 0.182] | 0.154 [0.139, 0.171] |
| fast weights on WRONG lessons, scored vs the wrong lessons' own answer | 0.999 [0.996, 0.999] | 0.683 [0.662, 0.703] | 0.146 [0.131, 0.162] | 0.153 [0.138, 0.169] | 0.146 [0.132, 0.163] | 0.156 [0.141, 0.173] |
| content null (wrong lessons), scored vs TRUE answer | 0.236 [0.218, 0.256] | 0.288 [0.269, 0.309] | 0.144 [0.129, 0.160] | 0.163 [0.147, 0.180] | 0.161 [0.145, 0.177] | 0.141 [0.126, 0.157] |
| content null, only items where null answer != true answer | 0.000 [0.000, 0.003] | 0.050 [0.039, 0.063] | 0.141 [0.123, 0.161] | 0.156 [0.137, 0.179] | 0.163 [0.144, 0.183] | 0.138 [0.119, 0.159] |
| fast weights, same items (null answer != true answer) | 0.999 [0.995, 1.000] | 0.688 [0.663, 0.713] | 0.153 [0.134, 0.173] | 0.142 [0.123, 0.163] | 0.160 [0.142, 0.181] | 0.157 [0.138, 0.179] |
| foreign-lessons null (another episode's lessons) | 0.045 [0.037, 0.055] | 0.058 [0.048, 0.069] | 0.040 [0.032, 0.050] | 0.043 [0.035, 0.053] | 0.042 [0.034, 0.051] | 0.038 [0.031, 0.048] |
| no adaptation (beta=0) | 0.040 [0.032, 0.049] | 0.154 [0.139, 0.170] | 0.052 [0.043, 0.063] | 0.135 [0.121, 0.151] | 0.046 [0.038, 0.056] | 0.108 [0.096, 0.123] |
| template copier (last lesson answer) | 0.163 [0.147, 0.179] | 0.158 [0.142, 0.174] | 0.152 [0.137, 0.168] | 0.156 [0.141, 0.173] | 0.157 [0.142, 0.174] | 0.151 [0.136, 0.168] |
| one-hop lookup (x*w1) | 1.000 [0.998, 1.000] | 0.000 [0.000, 0.002] | 0.181 [0.165, 0.198] | 0.141 [0.127, 0.157] | 0.246 [0.228, 0.266] | 0.087 [0.076, 0.101] |
| E[set prior: uniform over d-step reach of x] | 0.500 [0.500, 0.500] | 0.327 [0.325, 0.329] | 0.259 [0.257, 0.262] | 0.225 [0.222, 0.228] | 0.213 [0.210, 0.216] | 0.205 [0.202, 0.208] |
| E[best set prior: knows 1st hop or last letter] | 1.000 [1.000, 1.000] | 0.500 [0.500, 0.500] | 0.327 [0.325, 0.329] | 0.259 [0.257, 0.262] | 0.225 [0.222, 0.228] | 0.213 [0.210, 0.216] |
| oracle | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] | 1.000 [0.998, 1.000] |

### Paired differences, fast weights minus control (bootstrap 95% CI over items; positive = fast weights better)


#### aligned seed 0

| difference | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| fw minus content null, items where null answer != true answer | +1.000 [+1.000, +1.000] | +0.998 [+0.996, +1.000] | +0.944 [+0.932, +0.956] | +0.040 [+0.011, +0.069] | +0.033 [+0.008, +0.060] | -0.035 [-0.063, -0.007] |
| fw minus content null, all items | +0.763 [+0.745, +0.782] | +0.641 [+0.621, +0.663] | +0.615 [+0.592, +0.636] | +0.016 [-0.005, +0.037] | +0.027 [+0.006, +0.049] | -0.021 [-0.041, +0.000] |
| fw minus foreign-lessons null | +0.960 [+0.952, +0.969] | +0.952 [+0.942, +0.961] | +0.910 [+0.896, +0.922] | +0.169 [+0.148, +0.188] | +0.129 [+0.111, +0.148] | +0.103 [+0.086, +0.120] |
| fw minus no adaptation | +0.957 [+0.948, +0.966] | +0.940 [+0.929, +0.950] | +0.886 [+0.872, +0.899] | +0.083 [+0.060, +0.105] | +0.128 [+0.110, +0.147] | +0.064 [+0.045, +0.083] |
| fw minus template copier | +0.838 [+0.822, +0.853] | +0.842 [+0.825, +0.857] | +0.800 [+0.782, +0.819] | +0.056 [+0.033, +0.079] | +0.017 [-0.007, +0.040] | -0.011 [-0.033, +0.011] |
| fw minus one-hop lookup | +0.000 [+0.000, +0.000] | +0.999 [+0.998, +1.000] | +0.771 [+0.751, +0.791] | +0.070 [+0.047, +0.096] | -0.073 [-0.097, -0.050] | +0.053 [+0.034, +0.073] |
| fw minus E[best set prior] | +0.000 [+0.000, +0.000] | +0.499 [+0.497, +0.500] | +0.626 [+0.616, +0.635] | -0.047 [-0.065, -0.029] | -0.052 [-0.068, -0.035] | -0.072 [-0.088, -0.056] |

#### aligned seed 1

| difference | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| fw minus content null, items where null answer != true answer | +1.000 [+1.000, +1.000] | +0.998 [+0.995, +1.000] | +0.898 [+0.881, +0.914] | +0.419 [+0.386, +0.453] | +0.028 [+0.001, +0.054] | +0.028 [+0.001, +0.054] |
| fw minus content null, all items | +0.771 [+0.752, +0.789] | +0.649 [+0.628, +0.669] | +0.622 [+0.600, +0.645] | +0.240 [+0.213, +0.267] | +0.019 [-0.003, +0.042] | +0.018 [-0.004, +0.040] |
| fw minus foreign-lessons null | +0.955 [+0.946, +0.964] | +0.949 [+0.939, +0.958] | +0.882 [+0.868, +0.896] | +0.480 [+0.456, +0.503] | +0.131 [+0.112, +0.149] | +0.127 [+0.107, +0.146] |
| fw minus no adaptation | +0.964 [+0.955, +0.971] | +0.939 [+0.928, +0.950] | +0.862 [+0.846, +0.877] | +0.407 [+0.382, +0.434] | +0.086 [+0.065, +0.108] | +0.041 [+0.020, +0.062] |
| fw minus template copier | +0.847 [+0.831, +0.863] | +0.851 [+0.835, +0.866] | +0.776 [+0.756, +0.795] | +0.366 [+0.339, +0.394] | +0.020 [-0.004, +0.043] | +0.033 [+0.009, +0.056] |
| fw minus one-hop lookup | +0.000 [+0.000, +0.000] | +0.999 [+0.997, +1.000] | +0.748 [+0.727, +0.768] | +0.404 [+0.378, +0.429] | -0.048 [-0.072, -0.024] | +0.068 [+0.046, +0.089] |
| fw minus E[best set prior] | +0.000 [+0.000, +0.000] | +0.498 [+0.496, +0.500] | +0.591 [+0.580, +0.603] | +0.268 [+0.247, +0.290] | -0.050 [-0.067, -0.033] | -0.041 [-0.057, -0.025] |

#### aligned seed 2

| difference | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| fw minus content null, items where null answer != true answer | +1.000 [+1.000, +1.000] | +0.894 [+0.877, +0.911] | +0.161 [+0.134, +0.191] | -0.019 [-0.047, +0.009] | -0.025 [-0.048, -0.001] | +0.028 [+0.002, +0.055] |
| fw minus content null, all items | +0.790 [+0.772, +0.808] | +0.582 [+0.557, +0.606] | +0.099 [+0.073, +0.123] | -0.011 [-0.034, +0.010] | -0.015 [-0.035, +0.004] | +0.017 [-0.004, +0.037] |
| fw minus foreign-lessons null | +0.964 [+0.955, +0.972] | +0.851 [+0.834, +0.866] | +0.253 [+0.232, +0.275] | +0.119 [+0.102, +0.136] | +0.095 [+0.077, +0.113] | +0.133 [+0.116, +0.151] |
| fw minus no adaptation | +0.962 [+0.954, +0.971] | +0.862 [+0.847, +0.877] | +0.254 [+0.232, +0.277] | +0.107 [+0.088, +0.126] | +0.107 [+0.091, +0.124] | +0.131 [+0.112, +0.151] |
| fw minus template copier | +0.839 [+0.823, +0.856] | +0.737 [+0.716, +0.757] | +0.135 [+0.111, +0.160] | -0.005 [-0.028, +0.018] | -0.021 [-0.043, +0.001] | +0.007 [-0.017, +0.032] |
| fw minus one-hop lookup | +0.000 [+0.000, +0.000] | +0.903 [+0.890, +0.916] | +0.131 [+0.104, +0.157] | +0.017 [-0.006, +0.041] | -0.085 [-0.112, -0.059] | +0.056 [+0.033, +0.079] |
| fw minus E[best set prior] | +0.000 [+0.000, +0.000] | +0.403 [+0.390, +0.415] | -0.034 [-0.054, -0.014] | -0.103 [-0.119, -0.087] | -0.086 [-0.102, -0.070] | -0.034 [-0.051, -0.017] |

#### aligned, 3 seeds pooled

| difference | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| fw minus content null, items where null answer != true answer | +1.000 [+1.000, +1.000] | +0.963 [+0.957, +0.969] | +0.670 [+0.653, +0.686] | +0.150 [+0.131, +0.169] | +0.012 [-0.002, +0.027] | +0.008 [-0.008, +0.023] |
| fw minus content null, all items | +0.775 [+0.764, +0.786] | +0.624 [+0.611, +0.637] | +0.445 [+0.430, +0.461] | +0.082 [+0.068, +0.095] | +0.011 [-0.002, +0.023] | +0.005 [-0.007, +0.018] |
| fw minus foreign-lessons null | +0.960 [+0.954, +0.965] | +0.917 [+0.910, +0.924] | +0.682 [+0.670, +0.694] | +0.256 [+0.243, +0.268] | +0.118 [+0.107, +0.128] | +0.121 [+0.111, +0.131] |
| fw minus no adaptation | +0.961 [+0.956, +0.966] | +0.913 [+0.906, +0.921] | +0.667 [+0.655, +0.680] | +0.199 [+0.186, +0.213] | +0.107 [+0.096, +0.118] | +0.079 [+0.067, +0.090] |
| fw minus template copier | +0.841 [+0.832, +0.850] | +0.809 [+0.800, +0.820] | +0.571 [+0.556, +0.585] | +0.139 [+0.124, +0.154] | +0.005 [-0.008, +0.018] | +0.010 [-0.003, +0.023] |
| fw minus one-hop lookup | +0.000 [+0.000, +0.000] | +0.967 [+0.962, +0.971] | +0.550 [+0.535, +0.565] | +0.164 [+0.149, +0.178] | -0.069 [-0.083, -0.054] | +0.059 [+0.047, +0.071] |
| fw minus E[best set prior] | +0.000 [+0.000, +0.000] | +0.467 [+0.462, +0.471] | +0.394 [+0.383, +0.406] | +0.039 [+0.028, +0.051] | -0.063 [-0.072, -0.053] | -0.049 [-0.059, -0.040] |

#### old layout, seed 0

| difference | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| fw minus content null, items where null answer != true answer | +0.999 [+0.997, +1.000] | +0.639 [+0.610, +0.667] | +0.012 [-0.012, +0.034] | -0.015 [-0.040, +0.010] | -0.002 [-0.026, +0.021] | +0.020 [-0.005, +0.044] |
| fw minus content null, all items | +0.762 [+0.744, +0.781] | +0.406 [+0.379, +0.432] | +0.004 [-0.015, +0.022] | -0.013 [-0.032, +0.006] | +0.004 [-0.015, +0.025] | +0.013 [-0.006, +0.032] |
| fw minus foreign-lessons null | +0.954 [+0.945, +0.963] | +0.637 [+0.615, +0.659] | +0.108 [+0.092, +0.127] | +0.106 [+0.088, +0.124] | +0.123 [+0.106, +0.141] | +0.116 [+0.099, +0.135] |
| fw minus no adaptation | +0.960 [+0.951, +0.968] | +0.540 [+0.515, +0.566] | +0.097 [+0.079, +0.115] | +0.015 [-0.006, +0.036] | +0.119 [+0.101, +0.137] | +0.046 [+0.025, +0.067] |
| fw minus template copier | +0.837 [+0.820, +0.853] | +0.537 [+0.511, +0.563] | -0.004 [-0.026, +0.019] | -0.006 [-0.026, +0.015] | +0.008 [-0.014, +0.029] | +0.003 [-0.019, +0.025] |
| fw minus one-hop lookup | -0.001 [-0.003, +0.000] | +0.695 [+0.674, +0.714] | -0.033 [-0.056, -0.009] | +0.009 [-0.013, +0.030] | -0.082 [-0.107, -0.056] | +0.067 [+0.047, +0.088] |
| fw minus E[best set prior] | -0.001 [-0.003, +0.000] | +0.195 [+0.174, +0.214] | -0.178 [-0.194, -0.162] | -0.109 [-0.125, -0.093] | -0.060 [-0.077, -0.044] | -0.059 [-0.075, -0.043] |

#### Old vs aligned layout, seed 0, identical items (paired)

| difference | d1 | d2 | d3 | d4 | d5* | d6* |
|---|---|---|---|---|---|---|
| aligned fw minus old fw | +0.001 [+0.000, +0.003] | +0.304 [+0.285, +0.325] | +0.804 [+0.786, +0.822] | +0.062 [+0.037, +0.086] | +0.009 [-0.015, +0.032] | -0.013 [-0.036, +0.009] |

### Fast-weights accuracy by group class (`group_breakdown_4b.py`, `accuracy_by_group_class.csv`)

| run | group class | d2 | d3 | d4 |
|---|---|---|---|---|
| aligned_s0 | abelian (Z5-Z8) | 0.999 [0.996, 1.000] (n=1548) | 0.962 [0.951, 0.970] (n=1548) | 0.210 [0.190, 0.231] (n=1548) |
| aligned_s0 | non-abelian (S3, D4) | 0.998 [0.988, 1.000] (n=452) | 0.920 [0.892, 0.942] (n=452) | 0.219 [0.183, 0.259] (n=452) |
| aligned_s1 | abelian (Z5-Z8) | 1.000 [0.998, 1.000] (n=1533) | 0.937 [0.924, 0.948] (n=1533) | 0.560 [0.535, 0.585] (n=1533) |
| aligned_s1 | non-abelian (S3, D4) | 0.994 [0.981, 0.998] (n=467) | 0.861 [0.826, 0.889] (n=467) | 0.433 [0.388, 0.478] (n=467) |
| aligned_s2 | abelian (Z5-Z8) | 0.908 [0.893, 0.921] (n=1637) | 0.302 [0.280, 0.324] (n=1637) | 0.154 [0.137, 0.172] (n=1637) |
| aligned_s2 | non-abelian (S3, D4) | 0.882 [0.844, 0.911] (n=363) | 0.270 [0.227, 0.318] (n=363) | 0.176 [0.141, 0.219] (n=363) |
| old_s0 | abelian (Z5-Z8) | 0.742 [0.719, 0.763] (n=1548) | 0.156 [0.138, 0.175] (n=1548) | 0.165 [0.147, 0.184] (n=1548) |
| old_s0 | non-abelian (S3, D4) | 0.533 [0.487, 0.579] (n=452) | 0.124 [0.097, 0.157] (n=452) | 0.100 [0.075, 0.131] (n=452) |

Plots: `accuracy_by_depth_controls.png` (every arm per run with Wilson CIs), `paired_gaps.png` (fw minus content null and fw minus E[best set prior], per run), `training_curves.png`. Machine-readable: `accuracy_by_depth_arm.csv`, `paired_gaps.csv`, `analysis_4b.json` (training summary and the depth-1 two-candidate diagnostic).

## Interpretation (what is and is not supported)

1. **Depth 2 and 3 on aligned seeds 0 and 1: real composition, not format or structure exploitation.**
   - Fast weights vs content null on the items where the wrong lessons imply a different answer: depth 2 +0.998 [+0.996, +1.000] (s0), +0.998 [+0.995, +1.000] (s1); depth 3 +0.944 [+0.932, +0.956] (s0), +0.898 [+0.881, +0.914] (s1). Over all items (the null scores by luck when its answer coincides with the true one, 22-35% of depth 1-3 items) the gap is +0.64/+0.65 at depth 2 and +0.62/+0.62 at depth 3 with CIs of about +-0.02.
   - Stronger than a gap to a null: with the wrong lessons the same model answers *the wrong lessons' answer* (0.999 depth 2, 0.954 / 0.914 depth 3 on s0 / s1). Its output is a function of the lesson content, composed over the word. A format-only or template mechanism cannot do that, because names are random per episode and the null keeps the format. The foreign-lessons null (0.04-0.05 at every depth, below chance) and the no-adaptation arm (0.04-0.13) show the same dependence on lessons. (Foreign null 0.04-0.06, template copier 0.14-0.17 in every run and depth.)
   - These are well above every model-free shortcut: template copier 0.15-0.16, one-hop lookup 0.00 (d2) / 0.17-0.18 (d3), E[best set prior] 0.500 (d2) / 0.327-0.328 (d3), and above the 0.5 ceiling of a procedure that ignores one letter. fw minus E[best set prior]: depth 2 +0.499 / +0.498, depth 3 +0.626 / +0.591, all CIs excluding 0 by a wide margin.
   - Depth >= 2 composites are never in any lesson; the single-hop lookups are (depth 1 is exact lookup: 1.000 on every run, so depth 1 is recall, not composition).
2. **Seed 2 reaches depth 2 (0.903 [0.889, 0.915], +0.403 over the 0.5 shortcut ceiling, +0.894 over null on discriminating items) but not depth 3**: 0.296 [0.276, 0.316] is *below* E[best set prior] (0.330; -0.034 [-0.054, -0.014]) although above the null on discriminating items (+0.161 [+0.134, +0.191]) and the one-hop lookup (+0.131 [+0.104, +0.157]). It is partial: it depends on the lessons but does not beat what a set-prior guesser could do. Its depth-2 train accuracy was still rising at the end (0.896 in the final window; depth 2 first reached 0.9 at step 56.6k), so it is a slow seed, not a demonstrated ceiling.
3. **Depth 4: one seed.** s1 0.530 [0.509, 0.552]: vs null on discriminating items +0.419 [+0.386, +0.453], vs E[best set prior] +0.268 [+0.247, +0.290], and it composes the wrong lessons (0.540 vs the null answer). But it is only +0.03 above the 0.5 ceiling of "all letters but one", and it had about 3,600 steps at depth 4, so I would call it "composition at depth 4 begins, in one seed". s0 depth 4 = 0.212 [0.195, 0.230] is above the copier (0.156) and one-hop lookup (0.141) but *below* the set prior (0.259; -0.047 [-0.065, -0.029]), with fw minus null +0.016 [-0.005, +0.037] (all items) and +0.040 [+0.011, +0.069] (discriminating items): no usable evidence of composition at depth 4. s2 never trained depth 4 and is at chance (0.158; fw minus null on discriminating items -0.019 [-0.047, +0.009]).
4. **Depths 5-6 are chance for every run** (0.14-0.18, fw minus null within about +-0.04): length extrapolation was not achieved and, because depth 5 was never unlocked, was not trained.
5. **Old vs aligned layout (seed 0, identical items; the matched control).** Both models have exact depth 1 (0.999 vs 1.000). Aligned minus old: depth 2 +0.304 [+0.285, +0.325] (0.999 vs 0.695), depth 3 +0.804 [+0.786, +0.822], depth 4 +0.062 [+0.037, +0.086]. The old layout reaches depth 2 only partially: its depth-2 fw minus null on discriminating items is +0.639 [+0.610, +0.667] and fw minus E[best set prior] +0.195 [+0.174, +0.214], so even the old layout shows some composition at depth 2 (0.695 is above the 0.5 ceiling), but it never unlocked depth 3 (0.148, chance) in 60k steps. The old layout did **not** stall at depth 1: its depth-1 train accuracy reached 0.95 at step 10.7k, no later than the aligned runs (11.7-16.1k). So the exp 4 depth-1 plateau (5-8k steps) is mostly a step-budget effect, and the proposed key/query position mismatch is *not* what blocked depth 1; the aligned layout's measured benefit is at depth 2-3 (faster unlock and higher accuracy), not at depth 1. One seed for the old layout: the comparison is suggestive in size (a 0.30 depth-2 gap, 0.80 at depth 3) but is one seed against three.
6. **Do depth >= 2 results exceed a one-hop lookup or a task-structure shortcut?** Yes at depth 2 for all aligned seeds and for old s0; at depth 3 for s0 and s1 (not s2); at depth 4 for s1 only (and only just above the 0.5 ceiling). Not at depth 4 for s0/s2, and not at depth 5-6 anywhere. The set-prior numbers are expected accuracies of specific named shortcuts, not a proof that no shortcut exists; a shortcut that uses both the first hop and the last letter would be a stronger partial-composition model (0.5 at depth 3, 2 candidates). The depth 3 results on s0/s1 (0.92-0.95) exceed even that.

## Caveats
- **Three seeds, one old-layout seed, one model size (1.2M parameters by design: small so the whole study is cheap), 60k steps.** CIs are over evaluation items only (independent episodes within a depth); they say nothing about seed-to-seed spread, which is large: depth 3 is 0.953 / 0.919 / 0.296 and depth 4 is 0.212 / 0.530 / (not trained).
- **Compute-limited, not a flaw of PlasticCore**: each new depth took 12-37k steps after unlock to reach 0.9 train accuracy (s0: 12.3k for depth 2, 22.8k for depth 3; s1: 19.6k, 16.8k; s2: 36.6k for depth 2), depth 4 was unlocked at 55-56k steps (3.6-4.9k steps of depth-4 training), seed 2 unlocked depth 3 at 56.6k and was still climbing at depth 2. The depth-4 results (s0 0.21) and seed 2's depth-3 result are best read as "not yet trained", not "cannot be learned". Nothing here shows a limit of the architecture; nothing here shows it scales either.
- The pooled aligned rows mix seeds that reached different depths; per-seed rows are the primary evidence.
- The content null with deranged names is one construction of "wrong lessons, same format". On all-items scoring it is inflated by coincidences (null answer == true answer for 22-35% of depth 1-3 items); the discriminating-subset gap and the null-answer agreement address that. The foreign-lessons null changes the group and the names (not a pure content change) and is only a secondary control.
- The set-prior shortcuts are analytic expected accuracies for a guesser with specified partial information, not a trained baseline. A learned shortcut could exploit other group structure (identity words, commutativity in the cyclic groups: Z5-Z8 are abelian, so in those groups the order of the word does not matter and the answer depends only on the count of each generator; the model could be learning "count generators", still a composition but an easier one than non-abelian groups S3/D4). Per-group breakdown (below) shows depth 2-3 accuracy is high on the non-abelian groups S3 and D4 too, so "count the generators" is not the whole story, but n is small (363-467 items per cell).
- Each of the 2,000 episodes per run contributes one item at each of the six depths (sharing its lessons). Within a depth the items come from distinct episodes, so per-depth CIs treat them as independent; statistics pooled over depths would need an episode-cluster bootstrap and are not reported.
- The in-episode fast-weight path (meta-learned inner loop) is tested here. This is not a lasting W0 update, not the 6M published PlasticCore and not the 760M TTT chat model (exp 4 caveat unchanged).
- No weights are in the repo. Weights, GPU CSVs and logs are in the private HF dataset `dmontgomery40/plastic-gpu-results` (`results/main4b_60k`). Committed here: the 4 item-level control CSVs (about 150 KB each, gzip), train logs, tables, plots.

## Reproduce
```
# data: private HF dataset (needs HF_TOKEN), e.g.
python -c "from huggingface_hub import snapshot_download as s; s(repo_id='dmontgomery40/plastic-gpu-results', repo_type='dataset', allow_patterns=['results/main4b_60k/*'], local_dir='/workspace/plastic-exp/main4b_full')"
cd experiments/exp4_compose && export OMP_NUM_THREADS=2
F=/workspace/plastic-exp/main4b_full/results/main4b_60k; O=../results/exp4b
for r in aligned_s0:0 aligned_s1:1 aligned_s2:2 old_s0:0; do n=${r%:*}; s=${r#*:}
  python controls_4b.py --model $F/$n/model/model.pt --eval-seed $s --threads 2 --out $O/$n/items_controls.csv.gz --ref $F/$n/items_seed$s.csv.gz   # ~2 min each on 2 CPU threads
done
python analyze_controls_4b.py $O $O > $O/analysis_stdout.txt     # accuracy_by_depth_arm.csv, paired_gaps.csv, two PNGs
python analyze_4b.py $F                                          # training_curves.png, analysis_4b.json (written into $F)
python make_tables_4b.py $O > $O/tables.md                       # the tables above
python group_breakdown_4b.py $O > $O/group_table.md              # the group-class table
# GPU training (already run; do not rerun without budget): jobs/launch_4b.py main main4b_60k, see jobs/jobs_4b.log
```

## Recommendation for a further GPU run
Observed cost: the 60k-step job took about 2.6 h of training (9,270 s with four runs sharing one A10G at 95-99% utilisation) plus eval and upload, i.e. roughly $4 at $1.5/h; the 3 h cap would be $4.5. Remaining budget is about $7 (David's figure).
Suggested spend (estimates from that throughput; per-run speed with three runs in parallel was not measured and could be up to 1.3x faster, not slower):
1. **Warm-start continuation of the three aligned checkpoints for +60k steps** (needs a small `--init-from` option in `train.py` that loads the weights, restores the unlocked depths from the checkpoint's `unlock_steps`, uses a fresh AdamW and constant lr; test on CPU first). About 1.9-2.6 h, **roughly $3-4**. It tests directly whether depth 4 is learned by s0 and s2 and whether s2 reaches depth 3 (the open compute-limit question).
2. If the budget remains (~$3), two *additional* fresh aligned seeds with `--min-depth1-steps 12000` (depth 1 reaches 0.95 at 10.7-16.1k steps; the 20k floor wasted 4-8k steps) at 60k steps, about $2.5-3.5 for both, to see whether the seed-to-seed spread (depth 3: 0.30-0.95) is typical. Skip the old-layout run (already answered at 60k).
Do not spend on a bigger model first: the budget is better used on steps at the depths that are still climbing.
