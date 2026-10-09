# GPU experiments (HF Jobs, NVIDIA A10G): scaled eigenvalue ablation (Job A) and memory-precision scaling (Job B)

Status: Job B done. Job A still running (interim).

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
