# Plastic eigenvalue-range & rollback-leakage experiments — report

All numbers below are read from the result files in `experiments/results/` by the scripts named here.
No training was rerun for this write-up. Raw results and logs (`experiments/results/`, `experiments/logs/`) were not modified;
all derived files are under `report/`.

## 1. What was built

**Exp 1: eigenvalue range vs state tracking** (`experiments/exp1_eigen/`)
- `models.py`: single-mixer sequence models that reuse the repo's own block parts (CausalConv, RMSNorm, FastWeightMemory,
  selective-SSM scan). Within each family only the eigenvalue range of the state transition changes:
  - A: delta rule, β = sigmoid ∈ (0,1) → transition eigenvalues in [0,1] (the published/default setting)
  - B: identical module, β = 2·sigmoid ∈ (0,2) → eigenvalues in [−1,1] (Grazzi et al.)
  - C: diagonal selective SSM, decay a ∈ (0,1] (repo log-space scan)
  - D: same parameters, a_signed = 2a − 1 ∈ (−1,1], with a new sign-aware chunked scan (the repo scan clamps a > 0)
  - E: the repo's "chunk" rule (momentum + Newton–Schulz) with chunk = 1, β ∈ (0,1); parity only
  - L: 1-layer LSTM mixer (nonlinear control)
- `tasks.py`: parity, mod-3 counting, S3 composition (S5 also implemented); MQAR from `plastic.data.mqar`.
- `train.py`: one run (AdamW, warmup + cosine, grad clip 1.0); evaluates at fixed test lengths.
- `run_all.py`: lr search {1e-3, 3e-3} on seed 0 (selected by in-distribution val loss only), then seeds 1–2 at the chosen lr.
- `sanity.py` → `results/exp1/sanity.json`: chunked vs recurrent delta agree (max |Δ| ≤ 3.3e-6); transition eigenvalue range
  [9.8e-5, 0.998] at β_max = 1 and [−0.941, 0.999] at β_max = 2; signed scan matches the sequential scan (3.1e-6), whereas the
  repo's log-space scan gives an error of 4.96 with negative decays; B with β_max = 1 reproduces A bit for bit; S3/S5 group
  axioms and parity labels check out.
- `aggregate.py` → tables/plots.

**Exp 2: rollback leakage** (`experiments/exp2_rollback/`)
- `train_model.sh`: trains `lm_small_cpu`, a small PlasticCore LM (d_model 128, 4 layers, 4 heads, vocab 4096; WikiText, 6M train
  tokens; 2000 steps; AdamW). Held-out loss 4.52 (memory_value +0.445) per `logs/train_lm_small_cpu.log`.
- `run_exp2.sh`: calibrates the harness (`plastic calibrate`) and runs `probe.py` for seeds 0–2.
- `probe.py`: P (128 clean tokens) → X (1/2/4 chunks of benign/random/repeat_token/repeat_phrase/shuffled text) → C (256 clean
  tokens). X goes through the repo's `TransactionRunner` with the decision forced to rollback or commit (real code paths), or
  decided by the calibrated harness ("natural"). Extra conditions: rb_rewind (rollback + restore pre-X activation state) and
  rb_zero (rollback + zero the activation state). Measures KL(never‖cond) and NLL on C, with learning on ("direct") or memory frozen
  ("frozen"), and checks that S is bit-identical after rollback.
- `analyze.py` → tables/plots. `debug_nan.py`: diagnostic for the Muon NaN (see caveats).

## 2. Exact commands

Run from `experiments/` with the repo venv (`. ../.venv/bin/activate`). The exp1 training commands are reconstructed from
`run_all.py` and from the `args` stored in each result JSON (steps 4000, batch 64, d_model 64, 2 layers, 2 heads, wd 0,
1 thread per run). The original shell history was not saved, so worker counts are not known.

```bash
# Exp 1
python exp1_eigen/sanity.py
python exp1_eigen/run_all.py --tasks parity,mod3,s3 --variants A,B,C,D,L --steps 4000          # logs/exp1_main.log
python exp1_eigen/run_all.py --tasks mqar --variants A,B,C,D --steps 4000 \
    --extra "--train-min 4 --train-max 16 --test-lens 4,8,16,32,64"                             # logs/exp1_mqar.log
for s in 0 1 2; do python exp1_eigen/train.py --task parity --variant E --seed $s --lr 1e-3 --steps 4000 --threads 1 \
    --out results/exp1/runs_E/parity_E_s${s}_lr1e-3.json > logs/exp1_E_s$s.log; done
# Exp 2 (paths in these scripts are relative to the repo root)
bash exp2_rollback/train_model.sh
bash exp2_rollback/run_exp2.sh
# Write-up (this report; outputs go to report/, never to results/)
python exp1_eigen/aggregate.py ../report/exp1
python exp2_rollback/analyze.py ../report/exp2
cd .. && python report/exp2_extra.py
```

Changes to the analysis scripts in this write-up: `aggregate.py` and `analyze.py` originally wrote their outputs into
`results/exp1` and `results/exp2`. They now take an output directory (default `../report/expN`), so the raw-results folders
stay untouched. In `aggregate.py`, the MQAR panel's y-axis label now reads "answer accuracy". No computation was changed.

## 3. Exp 1 results

Mean ± sd over 3 seeds at the selected lr (from `report/exp1/summary.csv`; [min, max] and eigenvalue-usage stats are in
`report/exp1_tables.md`). Parity/mod3/s3: tested at L = 32 (longest training length) and 64–512 (extrapolation).
Chance: parity 0.5, mod3 0.333, s3 0.167.

**parity** (last-half per-token accuracy; trained on 8–32)

| variant | L=32 | L=64 | L=128 | L=256 | L=512 |
|---|---|---|---|---|---|
| A delta β∈(0,1) | 0.936 ± 0.019 | 0.535 ± 0.010 | 0.499 ± 0.004 | 0.500 ± 0.004 | 0.500 ± 0.000 |
| B delta β∈(0,2) | 1.000 ± 0.000 | 0.981 ± 0.028 | 0.649 ± 0.098 | 0.510 ± 0.008 | 0.500 ± 0.001 |
| C SSM a∈(0,1] | 0.962 ± 0.027 | 0.566 ± 0.054 | 0.499 ± 0.004 | 0.501 ± 0.004 | 0.500 ± 0.000 |
| D SSM a∈(−1,1] | 1.000 ± 0.000 | 0.859 ± 0.069 | 0.507 ± 0.005 | 0.500 ± 0.003 | 0.500 ± 0.000 |
| E NS-chunk β∈(0,1) | 0.900 ± 0.031 | 0.508 ± 0.007 | 0.498 ± 0.005 | 0.502 ± 0.003 | 0.499 ± 0.002 |
| L LSTM | 0.997 ± 0.005 | 0.877 ± 0.213 | 0.832 ± 0.291 | 0.835 ± 0.286 | 0.833 ± 0.289 |

**mod3** (last-half per-token accuracy; trained on 8–32)

| variant | L=32 | L=64 | L=128 | L=256 | L=512 |
|---|---|---|---|---|---|
| A delta β∈(0,1) | 0.985 ± 0.004 | 0.503 ± 0.037 | 0.334 ± 0.002 | 0.334 ± 0.004 | 0.333 ± 0.001 |
| B delta β∈(0,2) | 0.995 ± 0.005 | 0.679 ± 0.265 | 0.339 ± 0.011 | 0.331 ± 0.002 | 0.334 ± 0.002 |
| C SSM a∈(0,1] | 0.979 ± 0.018 | 0.470 ± 0.076 | 0.334 ± 0.003 | 0.336 ± 0.002 | 0.334 ± 0.000 |
| D SSM a∈(−1,1] | 0.987 ± 0.002 | 0.490 ± 0.013 | 0.333 ± 0.002 | 0.331 ± 0.003 | 0.333 ± 0.001 |
| L LSTM | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

**s3** (last-half per-token accuracy; trained on 8–32)

| variant | L=32 | L=64 | L=128 | L=256 | L=512 |
|---|---|---|---|---|---|
| A delta β∈(0,1) | 0.262 ± 0.027 | 0.170 ± 0.008 | 0.168 ± 0.002 | 0.166 ± 0.001 | 0.167 ± 0.001 |
| B delta β∈(0,2) | 0.762 ± 0.343 | 0.664 ± 0.341 | 0.580 ± 0.363 | 0.454 ± 0.335 | 0.262 ± 0.085 |
| C SSM a∈(0,1] | 0.286 ± 0.013 | 0.171 ± 0.004 | 0.167 ± 0.001 | 0.165 ± 0.002 | 0.167 ± 0.001 |
| D SSM a∈(−1,1] | 0.747 ± 0.334 | 0.413 ± 0.235 | 0.226 ± 0.068 | 0.184 ± 0.031 | 0.168 ± 0.006 |
| L LSTM | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

**mqar** (answer accuracy; L = # key-value pairs, trained 4–16; chance ≈0.016)

| variant | L=4 | L=8 | L=16 | L=32 | L=64 |
|---|---|---|---|---|---|
| A delta β∈(0,1) | 0.681 ± 0.273 | 0.553 ± 0.361 | 0.403 ± 0.347 | 0.175 ± 0.119 | 0.053 ± 0.009 |
| B delta β∈(0,2) | 0.884 ± 0.182 | 0.805 ± 0.287 | 0.685 ± 0.335 | 0.403 ± 0.221 | 0.120 ± 0.036 |
| C SSM a∈(0,1] | 0.515 ± 0.005 | 0.305 ± 0.012 | 0.167 ± 0.008 | 0.072 ± 0.006 | 0.023 ± 0.001 |
| D SSM a∈(−1,1] | 0.462 ± 0.076 | 0.284 ± 0.040 | 0.164 ± 0.029 | 0.069 ± 0.011 | 0.022 ± 0.001 |

Plot: `report/exp1_acc_vs_length.png` (training curves: `report/exp1/training_curves.png`).

Eigenvalue usage at the end of training (from `exp1_tables.md`): B's learned β exceeds 1 (negative eigenvalues) for 66%/46%
(L0/L1) of tokens on parity, 64%/75% on s3, and 34%/41% on mod3, but only 6%/24% on MQAR. D has negative decays on 42%/43%
of channels on parity. A, C and E never reach negative eigenvalues (frac = 0), which is guaranteed by their parameterization.

### Interpretation vs the hypothesis

- **Non-negative eigenvalues cannot do parity at length: supported.** A, C and E fit parity in distribution (0.90–0.96 at L = 32)
  but are at chance (0.50–0.57) from L = 64 on. Their eigenvalues stayed non-negative throughout training.
- **β up to 2 / negative decay fixes it: partly supported.** Allowing negative eigenvalues is what B and D use: they reach 1.000
  at L = 32 on all seeds and stay well above A and C at 2× the training length (parity L = 64: B 0.981, D 0.859 vs A 0.535, C 0.566).
  That also holds on S3, but not reliably: B has one seed at 1.00 through L = 128 and one seed that failed (0.37 at L = 32), and
  D has a similar spread. However, **neither fixes length generalization in this setup**. Parity B drops to 0.65 at L = 128 and is at
  or near chance at L = 256–512 (0.510, 0.500), and D is at chance from L = 128. On mod3 the extension gives essentially no gain (only one B seed reaches
  0.98 at L = 64; everything is at chance from L = 128). The LSTM control does generalize: mod3 and s3 are perfect to L = 512 on all
  seeds, and parity is perfect on 2 of 3 seeds (seed 1 fails beyond L = 64). So these tasks can be learned at this scale, and the
  linear variants' gap is in extrapolation, not in what they can represent at training length.
- **MQAR recall trade-off: not observed.** Extending the range did not cost recall. B has the best MQAR accuracy at every length
  (e.g. 0.685 vs A 0.403 at 16 pairs), and D ≈ C (0.164 vs 0.167 at 16 pairs). Seed variance is large, though: at 4 pairs, A has two seeds at
  about 0.52 and one at 1.00, and B has one seed at 0.67 and two at 0.99, and all models are far from saturating MQAR even in distribution. The honest reading is "no
  evidence of a cost", not "evidence of no cost".

## 4. Exp 2 results

From `report/exp2_tables.md` (360 session×type×size records, 3 seeds). Plot: `report/exp2_leakage_fraction.png`.

| check | result |
|---|---|
| S bit-identical after forced rollback | 360/360 |
| S bit-identical after the harness's own rollbacks | 238/238 |
| Leakage fraction KL(never‖rollback)/KL(never‖commit), C frozen | 0.04–0.07 per type/size (overall mean 0.050, max 0.160) |
| Leakage fraction, C direct (learning on) | 0.41–0.77 per type/size (overall mean 0.565) |
| Mean KL to never-seen over C (direct): rollback / rb_zero / rb_rewind / commit | 0.0179 / 0.0162 / 0 / 0.0328 |
| Mean KL to never-seen over C (frozen): rollback / rb_zero / rb_rewind / commit | 0.0176 / 0.0159 / 0 / 0.3960 |
| KL after rollback by C position (direct), tokens 0–16 / 16–64 / 64–256 | 0.2832 / 0.0008 / 0.0000 |
| KL after commit by C position (frozen), tokens 0–16 / 16–64 / 64–256 | 0.6720 / 0.3805 / 0.3769 |
| log p(poison token) uplift on C (frozen, repeat_token): rollback / commit / rb_rewind | +0.0198 / +0.3270 / 0.0000 |

Per type/size tables: `report/exp2_tables.md`.

### Interpretation

- Rollback works exactly on the committed memory: the fast-weight state S is restored bit for bit in every case.
- It is not a perfect "never happened", though. The rolled-back chunk still changed the transient activation state (recurrent h,
  conv buffers, last logits), so the next tokens differ. That residual effect is short-lived: large in C tokens 0–16, about 0 after
  64. A committed chunk, by contrast, keeps shifting predictions for the whole 256 tokens of C when memory is frozen.
- As a fraction of the commit effect, leakage is about 5% with frozen memory. With learning on during C it looks larger
  (about 0.4–0.8), but mainly because the commit effect is itself small there (commit KL 0.033 vs 0.396 frozen; at tokens 64–256
  the direct commit KL is 0.0026). Continued learning on clean C appears to wash out most of what was committed.
- **Does resetting activation state remove the leakage?** Restoring the pre-chunk activations (rb_rewind) removes it entirely
  (KL = 0 in all 360 records). Zeroing the activations (rb_zero) does not: it is only about 8% lower than plain rollback, because a
  zeroed state also differs from the P-conditioned state of the never-seen run. Full isolation requires snapshotting the
  activation state before each chunk, not just resetting it.

## 5. Caveats

- **Small scale, CPU only.** Exp 1 used about 109–151k-parameter models, 2 layers, 4000 steps, 256 test sequences per length.
  The 79 runs add up to about 14.1 CPU-hours (sum of `train_time_s`), with torch 2.14.1+cpu. Exp 2 used a 4-layer, d_model 128
  LM trained for 2000 steps (held-out loss 4.52), so it is a weak model. These results may not transfer to real-scale models.
- **Few seeds, high variance.** There are 3 seeds per cell. Several cells are bimodal: s3 B/D, mod3 B at L = 64, parity L,
  MQAR A/B. The means hide seed-level success or failure, so see [min, max] in `exp1_tables.md`.
- **lr search is minimal.** It covers 2 lrs, chosen on seed 0 by in-distribution val loss. For parity/B, parity/L, mod3/B and
  mod3/L, both lrs gave val loss 0.0, so the tie went to 1e-3.
- **Substitutions.**
  - Exp 2 uses a small locally trained PlasticCore LM instead of the gated HF checkpoint (`train_model.sh`).
  - That LM uses AdamW (`--no-muon`), because torch.optim.Muon caused NaN around step 100 (`debug_nan.py`).
  - Variant E uses chunk = 1, not the repo default of 64, which would never update within 8–32-token sequences. E got no lr
    search (lr 1e-3 only) and was run on parity only.
  - The LSTM control was not run on MQAR.
  - S5 is implemented, but the single pilot (`results/pilot/s5_B.json`) stayed at chance (about 0.008) even at L = 32, and S5 is
    not in the main grid. S3 was used as the non-abelian task.
- **Metric noise.** In exp2, the ΔNLL-based leakage ratio has near-zero denominators for several types (values such as −10.57 and
  2.69 appear), so it is not interpretable. Use the KL-based ratio.
- **Coverage.** MQAR "length" counts key-value pairs (sequence = 1 + 4·pairs tokens), and models are far from ceiling even in
  distribution. Parity/mod3/s3 L = 32 is the top of the training range, not held out.
- **Reconstructed commands.** Exp 1 commands are inferred from the scripts and stored args, not from a saved shell history.
- **Exploratory runs.** Pilot runs in `results/pilot/` were exploratory and are not used in any table.

## 6. Files

```
report/REPORT.md                       this file
report/exp1_tables.md                  exp1 full tables (mean ± sd [min,max]), eigen usage, lr selection
report/exp1_acc_vs_length.png          exp1 plot (accuracy vs length, 4 tasks)
report/exp1/{summary.md,summary.csv,runs.csv,acc_vs_length.png,training_curves.png,aggregate_stdout.txt}
report/exp2_tables.md                  exp2 headline checks + per type/size tables
report/exp2_leakage_fraction.png       exp2 plot (leakage fraction by type/size, direct and frozen)
report/exp2/{summary.md,summary.csv,summary.json,sessions.csv,leakage_fraction.png,kl_by_position.png,analyze_stdout.txt,extra_stats.txt}
report/exp2_extra.py                   one-off extra exp2 statistics (read-only)
experiments/exp1_eigen/*.py            models, tasks, train, run_all, sanity, aggregate
experiments/exp2_rollback/*            probe.py, analyze.py, run_exp2.sh, train_model.sh, debug_nan.py
experiments/results/{exp1,exp2,pilot}  raw results (unchanged)
experiments/logs/                      raw logs (unchanged)
/workspace/plastic-exp-results.tgz     archive of the above + repo source (excl. .venv, .git)
```
