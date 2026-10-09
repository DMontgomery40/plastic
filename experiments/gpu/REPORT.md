# GPU experiments (Hugging Face Jobs, 1x NVIDIA A10G)

Two experiments, both run from the repo at `dc55927` (`experiments/exp1_eigen` models/tasks unchanged) plus new code under `experiments/gpu/`:
- **Job A**: scaled eigenvalue-range ablation (the CPU exp1, redone with 8 seeds, a 4-value lr search, 3.3x more optimizer steps at 2x batch, test lengths up to 1024, and a train-to-64 curriculum for A vs B).
- **Job B**: memory-precision scaling for noiseless Gaussian regression (the openai/math #140 idea).

## Jobs, hardware, runtime, cost

All jobs: flavor `a10g-large` (1x NVIDIA A10G 24 GB, 12 vCPU, 46 GB RAM, listed at $0.025/min = $1.50/h), image `pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime` (torch 2.14.1+cu126). Cost = `durations.total_secs` x listed rate (from the Jobs API; I have not seen an invoice).

| job | id | purpose | status | total time | cost |
|---|---|---|---|---|---|
| bench1 | `6ac8ea36fee2c90070178ba1` | smoke test | ERROR (container setup, see below) | 58 s | $0.02 |
| bench2 | `6ac8ea84095c57808930640c` | smoke test + throughput bench | COMPLETED | 84 s | $0.04 |
| **Job A** | `6ac8eb4efee2c90070178bff` | scaled eigenvalue ablation, 231 runs | COMPLETED | 12,856 s = 214.3 min (3.57 h) | $5.36 |
| **Job B** | `6ac8eb52095c578089306442` | memory-precision scaling | COMPLETED | 132 s = 2.2 min | $0.05 |
| **total** | | | | | **about $5.47** of the $15 cap |

Every job had an HF timeout (30 min, 4.5 h, 40 min) and each step inside was wrapped in `timeout`. Nothing was relaunched; no job is running now.

**GPU use was verified**, not assumed. Each job's `env.json` shows `cuda: true`, device `NVIDIA A10G`; every run JSON records `gpu: NVIDIA A10G`. Job A's `gpu_util.csv` (30 s samples, 428 rows) has a mean GPU utilization of 99.4% (`results/exp1/raw/gpu_util.csv`). Job A ran 10 concurrent training processes on the one GPU (the models are tiny, so one process alone does not saturate it); the 34.1 h of summed per-run train time is therefore not wall-clock.

**The errored job (bench1, `6ac8ea36...`).** It failed 5 s after start with `ModuleNotFoundError: No module named 'huggingface_hub'`. The image's Python is PEP 668 "externally managed", so my `pip install huggingface_hub` was refused, and the next line (downloading the code archive) could not import it. No experiment code ran, no result depends on it, and it cost $0.02. I fixed it with `--break-system-packages` (in `jobs/launch.py`) and bench2 and both real jobs started cleanly. Apart from this job, there were no failures: all 231 Job A subprocesses returned rc=0 (driver log) and Job B completed all three d values; no Job A run had non-finite training steps.

## Reproducing

```
/workspace/hfjobs-venv/bin/python experiments/gpu/jobs/launch.py A jobA1 --timeout 4.5h --env WORKERS=10,STEPS=10000,JOB_A_SECS=15600
/workspace/hfjobs-venv/bin/python experiments/gpu/jobs/launch.py B jobB1 --timeout 40m --env JOB_B_SECS=2100
```
`launch.py` tars the code (`plastic/`, `exp1_eigen/models.py`, `tasks.py`, and the scripts in `experiments/gpu/`) and uploads it to the private dataset repo `dmontgomery40/plastic-gpu-results`. The job downloads it, runs `jobs/job_entry.sh`, and syncs outputs back to the same repo. The token is forwarded only as a job secret and is never written to a file. Raw outputs are also under `results/exp1/raw` and `results/memprec/raw` here. Aggregation: `exp1_scaled/aggregate_gpu.py`, `memprec/analyze_memprec.py`.

## Job A: scaled eigenvalue ablation

**Setup.** Variants A (delta rule, beta in (0,1)), B (beta in (0,2)), C (SSM decay in (0,1]), D (SSM decay in (-1,1)) and an LSTM control L, using the unchanged `exp1_eigen/models.py` (2 layers, d_model 64, about 109k params; LSTM 151k). Tasks: parity, mod3, s3. Training lengths are drawn uniformly from 8-32 (A-D, L) and, for the curriculum comparison, 8-64 (A and B only). Test lengths 32, 64, 128, 256, 512, 1024 with 512 fresh sequences each; the metric is per-token accuracy over the last half of the sequence (chance: parity 0.5, mod3 0.333, s3 0.167), identical to the CPU run. Versus the CPU run: 10,000 steps at batch 128 (CPU: 3,000 at 64), 8 seeds (CPU: 3), lr search over {3e-4, 1e-3, 3e-3, 1e-2} (CPU: {1e-3, 3e-3}). Per (task, variant, train-length group) the lr is chosen on seed 0 by **in-distribution validation loss** (lengths 8-32/8-64, never an extrapolation length); seeds 1-7 are then run at that lr. 231 runs = 15 groups x 11 + 6 groups x 11 (seed 0 at 4 lrs, plus 7 more seeds).

**Selection caveat that matters (read before the tables).** For several groups the in-distribution validation loss is essentially zero at every lr (for example parity/B and mod3/B: exactly 0 at three or four lrs). I resolved ties (loss within 1e-6) by higher validation accuracy and then the **smaller lr**, a rule fixed before I saw results. In those groups the choice is arbitrary with respect to length generalization, and for parity/B it picked 3e-4, which is the *worst* of the four lrs for extrapolation on seed 0 (see the sensitivity table in `results/exp1/summary.md`). So the headline B numbers below are conservative for B, not oracle-tuned.

### Accuracy vs test length, train lengths 8-32 (mean ± sd over 8 seeds; selected lr)

| task | variant | lr | L=32 | L=64 | L=128 | L=256 | L=512 | L=1024 |
|---|---|---|---|---|---|---|---|---|
| parity | A | 1e-3 | 0.994±0.004 | 0.647±0.030 | 0.501±0.004 | 0.501±0.004 | 0.500±0.000 | 0.500±0.001 |
| parity | B | 3e-4 | 1.000±0.000 | 0.975±0.047 | 0.639±0.085 | 0.504±0.002 | 0.500±0.001 | 0.500±0.001 |
| parity | C | 1e-2 | 0.989±0.008 | 0.620±0.030 | 0.499±0.004 | 0.499±0.005 | 0.500±0.001 | 0.500±0.001 |
| parity | D | 1e-3 | 0.999±0.003 | 0.859±0.116 | 0.515±0.018 | 0.498±0.003 | 0.500±0.000 | 0.501±0.001 |
| parity | L | 1e-3 | 1.000±0.001 | 0.941±0.122 | 0.872±0.228 | 0.858±0.227 | 0.841±0.231 | 0.820±0.249 |
| mod3 | A | 1e-3 | 0.999±0.001 | 0.601±0.045 | 0.334±0.003 | 0.334±0.000 | 0.333±0.001 | 0.333±0.001 |
| mod3 | B | 1e-3 | 1.000±0.000 | 0.788±0.162 | 0.356±0.050 | 0.333±0.002 | 0.334±0.001 | 0.334±0.000 |
| mod3 | C | 3e-3 | 1.000±0.000 | 0.613±0.024 | 0.335±0.002 | 0.334±0.000 | 0.333±0.001 | 0.333±0.000 |
| mod3 | D | 3e-3 | 1.000±0.000 | 0.625±0.034 | 0.335±0.002 | 0.334±0.000 | 0.333±0.001 | 0.333±0.000 |
| mod3 | L | 3e-3 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| s3 | A | 1e-2 | 0.324±0.014 | 0.187±0.012 | 0.166±0.003 | 0.167±0.001 | 0.167±0.001 | 0.167±0.000 |
| s3 | B | 1e-3 | 0.864±0.196 | 0.758±0.331 | 0.718±0.348 | 0.550±0.300 | 0.286±0.127 | 0.175±0.010 |
| s3 | C | 1e-2 | 0.359±0.043 | 0.201±0.016 | 0.167±0.001 | 0.167±0.000 | 0.166±0.000 | 0.167±0.000 |
| s3 | D | 1e-2 | 1.000±0.000 | 0.983±0.004 | 0.599±0.076 | 0.291±0.039 | 0.204±0.026 | 0.169±0.003 |
| s3 | L | 3e-4 | 0.876±0.239 | 0.590±0.298 | 0.485±0.339 | 0.449±0.359 | 0.427±0.368 | 0.412±0.373 |

Number of the 8 seeds with accuracy >= 0.99 at each length (train 8-32):

| task | variant | L=32 | L=64 | L=128 | L=256 | L=512 | L=1024 |
|---|---|---|---|---|---|---|---|
| parity | A | 7 | 0 | 0 | 0 | 0 | 0 |
| parity | B | 8 | 5 | 0 | 0 | 0 | 0 |
| parity | D | 8 | 0 | 0 | 0 | 0 | 0 |
| parity | L | 8 | 6 | 5 | 5 | 5 | 5 |
| mod3 | B | 8 | 1 | 0 | 0 | 0 | 0 |
| mod3 | L | 8 | 8 | 8 | 8 | 8 | 8 |
| s3 | A | 0 | 0 | 0 | 0 | 0 | 0 |
| s3 | B | 5 | 4 | 2 | 0 | 0 | 0 |
| s3 | D | 8 | 0 | 0 | 0 | 0 | 0 |
| s3 | L | 3 | 2 | 2 | 2 | 2 | 2 |

(All rows incl. C and the eigenvalue-usage diagnostics: `results/exp1/summary.md`; per-run data: `runs.csv`; plot `acc_vs_length_tmax32.png`.)

### Curriculum: train lengths 8-32 vs 8-64, A vs B (mean ± sd over 8 seeds)

| task | variant | train | L=64 | L=128 | L=256 | L=512 | L=1024 |
|---|---|---|---|---|---|---|---|
| parity | A | 8-32 | 0.647±0.030 | 0.501±0.004 | 0.501±0.004 | 0.500±0.000 | 0.500±0.001 |
| parity | A | 8-64 | 0.810±0.039 | 0.503±0.005 | 0.500±0.004 | 0.500±0.000 | 0.500±0.001 |
| parity | B | 8-32 | 0.975±0.047 | 0.639±0.085 | 0.504±0.002 | 0.500±0.001 | 0.500±0.001 |
| parity | B | 8-64 | 1.000±0.000 | 0.961±0.040 | 0.595±0.087 | 0.500±0.002 | 0.499±0.001 |
| mod3 | A | 8-32 | 0.601±0.045 | 0.334±0.003 | 0.334±0.000 | 0.333±0.001 | 0.333±0.001 |
| mod3 | A | 8-64 | 0.892±0.055 | 0.355±0.020 | 0.333±0.000 | 0.333±0.001 | 0.333±0.000 |
| mod3 | B | 8-32 | 0.788±0.162 | 0.356±0.050 | 0.333±0.002 | 0.334±0.001 | 0.334±0.000 |
| mod3 | B | 8-64 | 0.964±0.068 | 0.676±0.317 | 0.418±0.159 | 0.334±0.002 | 0.333±0.000 |
| s3 | A | 8-32 | 0.187±0.012 | 0.166±0.003 | 0.167±0.001 | 0.167±0.001 | 0.167±0.000 |
| s3 | A | 8-64 | 0.247±0.048 | 0.167±0.003 | 0.166±0.001 | 0.167±0.001 | 0.167±0.000 |
| s3 | B | 8-32 | 0.758±0.331 | 0.718±0.348 | 0.550±0.300 | 0.286±0.127 | 0.175±0.010 |
| s3 | B | 8-64 | 0.867±0.244 | 0.767±0.274 | 0.606±0.328 | 0.509±0.348 | 0.333±0.216 |

Seeds >= 0.99 for B, train 8-32 vs 8-64: parity L=64: 5 vs 8, L=128: 0 vs 3; mod3 L=64: 1 vs 4, L=128: 0 vs 2; s3 L=64: 4 vs 4, L=128: 2 vs 3, L=256: 0 vs 3, L=512: 0 vs 0. Plot: `curriculum_A_vs_B.png`.

### Comparison with the earlier CPU run (3 seeds, 3k steps x 64, train 8-32; mean accuracy)

| task | variant | L=64 CPU → GPU | L=128 CPU → GPU | L=256 CPU → GPU | L=512 CPU → GPU |
|---|---|---|---|---|---|
| parity | A | 0.535 → 0.647 | 0.499 → 0.501 | 0.500 → 0.501 | 0.500 → 0.500 |
| parity | B | 0.981 → 0.975 | 0.649 → 0.639 | 0.510 → 0.504 | 0.500 → 0.500 |
| parity | D | 0.859 → 0.859 | 0.507 → 0.515 | 0.500 → 0.498 | 0.500 → 0.500 |
| parity | L | 0.877 → 0.941 | 0.832 → 0.872 | 0.835 → 0.858 | 0.833 → 0.841 |
| mod3 | A | 0.503 → 0.601 | 0.334 → 0.334 | 0.334 → 0.334 | 0.333 → 0.333 |
| mod3 | B | 0.679 → 0.788 | 0.339 → 0.356 | 0.331 → 0.333 | 0.334 → 0.334 |
| s3 | A | 0.170 → 0.187 | 0.168 → 0.166 | 0.166 → 0.167 | 0.167 → 0.167 |
| s3 | B | 0.664 → 0.758 | 0.580 → 0.718 | 0.454 → 0.550 | 0.262 → 0.286 |
| s3 | D | 0.413 → 0.983 | 0.226 → 0.599 | 0.184 → 0.291 | 0.168 → 0.204 |
| s3 | L | 1.000 → 0.590 | 1.000 → 0.485 | 1.000 → 0.449 | 1.000 → 0.427 |

(All rows: `results/exp1/cpu_vs_gpu.md`.) The CPU numbers come from `report/exp1/summary.csv` at `dc55927`. Seeds and lr grids differ, so differences are small-sample comparisons, not controlled ablations.

### Interpretation (Job A)

1. **B's length generalization is better than A's, and it improves with more training, but it does not become length-unbounded.** Within the training distribution everything is solved. For extrapolation, B beats A on every task at every length where either is above chance: parity L=64 0.975 vs 0.647, mod3 L=64 0.788 vs 0.601, s3 L=64 0.758 vs 0.187 (train 8-32). A is at chance from L=128 on in all three tasks, with essentially no seed-to-seed spread. A never solves s3 at all (0/8 seeds >= 0.99 even at L=32), which is consistent with beta in (0,1) giving non-negative eigenvalues that cannot represent non-commuting reflections.
2. **More training steps alone (CPU 3k to GPU 10k, 3 to 8 seeds) changed B little on parity** (L=128 0.649 vs 0.639; the same 0.5 at L>=256). It helped B on s3 (L=128: 0.580 to 0.718; L=256: 0.454 to 0.550), (CPU had only 3 seeds, so that comparison is weak); the s3 spread is still huge (sd 0.3, seeds range 0.17 to 1.00).
3. **The curriculum (train to 64) moves B's frontier out by roughly one doubling, not to 1024.** Parity: B solves L=64 in 8/8 seeds (vs 5/8) and reaches 0.961 at L=128 (vs 0.639), but is at 0.595 at L=256 and chance at L=512. mod3 L=128: 0.356 to 0.676. s3 L=512: 0.286 to 0.509, L=1024: 0.175 to 0.333. A also improves with the curriculum (parity L=64 0.647 to 0.810, mod3 L=64 0.601 to 0.892) but is at chance by L=128 regardless. So the extrapolation gap between B and A is **not** just a training-length limit: A's failure beyond about 2x the training length persists with the longer curriculum, while B's reach grows with the training length. Both effects are real; the roughly-2x frontier suggests B's learned solution is approximate rather than the exact algorithmic one, and that errors compound (accuracy falls gradually rather than as a cliff, e.g. parity B 1.00, 0.96, 0.60, 0.50).
4. **The negative-eigenvalue range is being used.** At the end of training, B's beta exceeds 1 (hence has negative eigenvalues) for 66%/73% of positions in layers 0/1 on parity, 36%/38% on mod3, 60%/65% on s3; A's is exactly 0 by construction. This is the diagnostic that the extra range is exploited, not just available.
5. **The SSM comparison (C vs D) agrees: signed decay helps.** D beats C on s3 at every length (L=64: 0.983 vs 0.201), and on parity L=64 (0.859 vs 0.620). On mod3 C and D tie (0.61 vs 0.63 at L=64). D on s3 is, in fact, the best non-LSTM extrapolator at L=64 (0.983±0.004, all 8 seeds), better than B (0.758 with a huge spread) at that length, but decays faster beyond (L=256: 0.291 vs 0.550).
6. **The LSTM is not a clean upper bound here.** It solves mod3 perfectly at all lengths in 8/8 seeds and 5/8 seeds on parity at all lengths up to 1024, but on s3 only 2/8 seeds generalize, and it is worse than in the CPU run (which had 3/3 seeds at 1.00). Seed 0's lr search chose 3e-4 for s3/L because all four lrs tie at about zero validation loss on seed 0, and two of the other seven seeds at that lr only reach 0.93 and 0.70 in-distribution validation accuracy within 10k steps (under-training, not an extrapolation failure); the other four reach 0.99 or more in-distribution but still fail at long lengths (only seeds 0 and 2 generalize to 1024). Do not read the LSTM s3 row as evidence against LSTMs; read it as a hyperparameter-selection artefact.
7. **Selection caveat again.** For B on parity seed 0, a larger lr extrapolates much further than the selected 3e-4: lr 1e-2 reaches 0.997 at L=256, 0.964 at L=512 and 0.696 at L=1024 (train 8-32; seed 0 only, single run each, so anecdotal). For train 8-64, lr 3e-3 gives 1.000 at L=256 and 0.939 at L=512. This hints that B's ceiling is higher than the headline table shows if lr is tuned on extrapolation or with a better tie-break, but I did not run all 8 seeds at those lrs, and selecting on extrapolation accuracy would be oracle selection. It is the main follow-up (parity/mod3/s3 x A,B x 8 seeds at lr 1e-2 or 3e-3 costs roughly $1 to $2 on the same job setup, my estimate from per-run times).

### Caveats (Job A)
- Tiny models (about 109k params, 2 layers, d_model 64), 3 synthetic group tasks; this says nothing directly about language-model-scale behaviour.
- The lr is selected on seed 0 only, and the validation criterion saturates for several groups (above), so tie-breaking, not data, picked some lrs. Seeds differ strongly (large sd on s3 and on LSTM).
- 8 seeds give wide intervals on the s3 rows; I report mean, sd, min and max and the count of solved seeds, and have not run significance tests.
- The metric averages the last half of the sequence (as in the CPU run); the `acc_final` (last token only) is in `runs.csv`.
- In s3/L and several others the "lr search" differs from the CPU run's {1e-3, 3e-3}, so CPU vs GPU is not a like-for-like ablation of step count alone.
- E (NS-chunk rule) and MQAR were not rerun (not requested).

## Job B: how memory precision scales (noiseless Gaussian regression)

**Question.** This is family #140 in openai/math: a one-pass learner with about A·d² persistent bits needs Ω(d·log(1/ε)) noiseless Gaussian samples to reach angular error ε. Does the repo's delta rule (normalized keys) reach that order, which β is best, and what happens when the state is quantized?

**Setup** (`memprec/memprec.py`). The stream is x_t ~ N(0, I_d), y_t = ⟨x_t, s⟩ with s a random unit vector. We ran d ∈ {64, 128, 256} with 64 trials per d. All learners see the same x stream in each trial (common random numbers). The delta rule ran for 80·d samples with an fp64 state. Its update is k = x/|x|, v = y/|x|, s ← s + β_t (v − ⟨k, s⟩) k. Learners tested:
- β ∈ {0.25, 0.5, 0.75, 0.9, 1, 1.25, 1.5, 1.75}.
- A sigmoid gate β_t = σ(w·x_t + b) with fixed random w, for b = 0 (the repo's `beta_bias_init`, realised mean β = 0.500) and b = 3 (realised mean β = 0.93).
- β = 1 with an fp32 state.
- β ∈ {1, 1.5} with the state re-quantized after every update to 4, 8 or 16 bits. The grid is symmetric with a per-vector absmax scale; we tried both round-to-nearest and stochastic rounding.
- Comparator: exact chunk least squares in fp64. It accumulates G = XᵀX and r = Xᵀy and solves G⁺r every d/16 samples. We also ran it with G and r quantized to 8 and 16 bits.

The metric is samples-to-ε: the first n at which angle(ŝ, s) ≤ ε, for ε = 1e-1 … 1e-6. We report the median over trials. The job ran on one A10G (`torch.cuda.get_device_name` = NVIDIA A10G, recorded in every JSON).

**Fit of median samples-to-ε against d·ln(1/ε)** (18 cells: 3 d × 6 ε, using only cells that every trial reached). Theory for normalized keys is n ≈ (2/(2β−β²))·d·ln(1/ε), because E|e|² shrinks by a factor 1 − (2β−β²)/d per sample.

| learner | slope a (fit through origin) | R² | affine fit: a, c·d | theory a |
|---|---|---|---|---|
| β = 0.25 | 4.532 | 0.9994 | 4.667, −1.344 | 4.571 |
| β = 0.5 | 2.657 | 0.9999 | 2.671, −0.146 | 2.667 |
| β = 0.75 | 2.124 | 0.9999 | 2.121, 0.038 | 2.133 |
| β = 0.9 | 2.012 | 1.0000 | 2.008, 0.039 | 2.020 |
| **β = 1** | **1.990** | 1.0000 | 1.990, 0.007 | **2.000** |
| β = 1.25 | 2.118 | 0.9999 | 2.122, −0.040 | 2.133 |
| β = 1.5 | 2.654 | 1.0000 | 2.658, −0.035 | 2.667 |
| β = 1.75 | 4.558 | 1.0000 | 4.568, −0.100 | 4.571 |
| sigmoid gate, b = 0 (mean β 0.50) | 2.821 | 0.9999 | 2.838, −0.163 | – |
| sigmoid gate, b = 3 (mean β 0.93) | 2.012 | 0.9999 | 2.009, 0.028 | – |
| β = 1, fp32 state | 1.993 | 0.9999 | 1.996, −0.024 | – |
| exact chunk LS (fp64) | – | – | 0.000, **1.000** (n = d for every ε) | – |

**Median samples-to-ε, d = 256.** "–(x%)" means fewer than half the trials got there within 80·d samples.

| learner | 1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 | median final angle |
|---|---|---|---|---|---|---|---|
| β = 1 | 1186 | 2341 | 3528 | 4710 | 5873 | 7051 | 4.45e-16 |
| β = 0.5 | 1526 | 3127 | 4695 | 6293 | 7826 | 9413 | 8.63e-14 |
| β = 1.5 | 1560 | 3127 | 4686 | 6278 | 7840 | 9398 | 8.71e-14 |
| sigmoid gate, b = 0 | 1614 | 3324 | 4986 | 6676 | 8333 | 10020 | 5.11e-13 |
| β = 1, fp32 | 1186 | 2341 | 3528 | 4712 | 5874 | 7078 | 1.82e-07 |
| β = 1, 16-bit, nearest | 1187 | 2342 | 3572 | –(8%) | –(0%) | –(0%) | 1.23e-04 |
| β = 1, 16-bit, stochastic | 1186 | 2343 | 3621 | –(0%) | –(0%) | –(0%) | 4.90e-04 |
| β = 1, 8-bit, nearest | 1437 | –(0%) | –(0%) | –(0%) | –(0%) | –(0%) | 3.24e-02 |
| β = 1, 4-bit (both roundings) | –(0%) | –(0%) | –(0%) | –(0%) | –(0%) | –(0%) | ≈1.55 rad (no learning) |
| exact LS, fp64 | 256 | 256 | 256 | 256 | 256 | 256 | 4.71e-15 |
| LS, 16-bit stats | 272 | 304 | –(0%) | –(0%) | –(0%) | –(0%) | 1.27e-03 |
| LS, 8-bit stats | –(0%) | –(0%) | –(0%) | –(0%) | –(0%) | –(0%) | 4.08e-01 |

The tables for every d and every learner are in `results/memprec/summary.md` and `hitting_times.csv`; the fits are in `fits.csv`. Plots: `samples_vs_log_eps.png`, `error_curves.png`, `slope_vs_beta.png`.

**Reading.**
- **β = 1 is order-optimal for a d-word state, and it has the best constant.** Its samples-to-ε is linear in d·ln(1/ε) with slope 1.99 (theory 2), R² = 1.0000, and a negligible intercept. That matches the Ω(d·log(1/ε)) order in the #140 lower bound. The state is d numbers, far below A·d² bits.
- **The slope across β follows 2/(2β−β²) to within about 1–3%.** That function is minimized at β = 1. The (0, 2) range adds nothing in this setting: β = 1.5 is indistinguishable from β = 0.5 (2.654 vs 2.657), as the symmetric contraction factor predicts.
- **The sigmoid gate costs speed.** At the repo's initial bias (mean β = 0.5) it is about 42% slower than β = 1 (slope 2.82). It is slightly worse than a constant β = 0.5 because a varying β_t loses to its mean (Jensen). Pushed towards 1 (b = 3, mean 0.93) it nearly matches β = 1.
- **Exact LS needs exactly n = d samples for every ε**, so its sample count does not depend on ε. It gets there with d² high-precision statistics, outside the memory budget the bound covers. With those statistics quantized to 16 bits it stalls near 1e-3, and at 8 bits it never gets below about 0.2 rad. So LS's precision needs grow with log(1/ε).
- **Precision sets a floor on reachable ε.**
  - fp32: β = 1 keeps the fp64 sample counts down to 1e-6, then floors at about 1e-7.
  - 16 bits: the sample counts are unchanged down to the floor, which is about 1e-4 to 5e-4.
  - 8 bits: floor about 1e-2 to 3e-2.
  - 4 bits: re-quantizing after every update erases the writes (they are smaller than half a grid step), and the state never learns.
  
  Stochastic rounding did not help here; its floors were higher. As a rough rule, reaching ε needs about log2(1/ε) + O(log d) bits per coordinate. So the delta rule's total memory is about d·log(1/ε) bits, in line with the memory–sample trade-off the bound describes.

**Caveats (Job B).**
- 64 trials per d and d ≤ 256.
- The hitting time is the *first* crossing.
- The quantizer is one specific scheme: a per-vector absmax grid, re-applied after every update. Other schemes (error feedback, a fixed scale, block floating point) would give different floors.
- "Sigmoid β" is a fixed random gate, not a trained one.
- The theory constant is for E|e|²; the median of the log error decays slightly faster at small d, which is why the measured slope sits a little below theory.
