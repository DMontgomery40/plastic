# Exp 1 summary (selected lr, mean ± sd over seeds; [min, max])

## parity  (metric: per-token accuracy over the last half of the sequence)

| variant | lr | n | params | L=32 | L=64 | L=128 | L=256 | L=512 |
|---|---|---|---|---|---|---|---|---|
| A delta beta(0,1) | 0.001 | 3 | 109002 | 0.936±0.019 [0.92,0.96] | 0.535±0.010 [0.53,0.55] | 0.499±0.004 [0.50,0.50] | 0.500±0.004 [0.50,0.50] | 0.500±0.000 [0.50,0.50] |
| B delta beta(0,2) | 0.001 | 3 | 109002 | 1.000±0.000 [1.00,1.00] | 0.981±0.028 [0.95,1.00] | 0.649±0.098 [0.54,0.74] | 0.510±0.008 [0.50,0.52] | 0.500±0.001 [0.50,0.50] |
| C ssm a(0,1) | 0.003 | 3 | 108802 | 0.962±0.027 [0.94,0.99] | 0.566±0.054 [0.53,0.63] | 0.499±0.004 [0.50,0.50] | 0.501±0.004 [0.50,0.50] | 0.500±0.000 [0.50,0.50] |
| D ssm a(-1,1) | 0.003 | 3 | 108802 | 1.000±0.000 [1.00,1.00] | 0.859±0.069 [0.78,0.90] | 0.507±0.005 [0.50,0.51] | 0.500±0.003 [0.50,0.50] | 0.500±0.000 [0.50,0.50] |
| E NS-chunk rule | 0.001 | 3 | 109002 | 0.900±0.031 [0.87,0.93] | 0.508±0.007 [0.50,0.51] | 0.499±0.005 [0.50,0.50] | 0.502±0.003 [0.50,0.50] | 0.499±0.002 [0.50,0.50] |
| L LSTM (nonlinear) | 0.001 | 3 | 150402 | 0.997±0.005 [0.99,1.00] | 0.877±0.213 [0.63,1.00] | 0.832±0.291 [0.50,1.00] | 0.835±0.286 [0.50,1.00] | 0.833±0.289 [0.50,1.00] |

Eigenvalue-range usage at the end of training (last forward, mean over seeds):

- A: L0_beta_mean=0.357, L0_frac_beta_gt1=0.000, L0_frac_eig_neg=0.000, L0_min_eig=0.245, L1_beta_mean=0.629, L1_frac_beta_gt1=0.000, L1_frac_eig_neg=0.000, L1_min_eig=0.061
- B: L0_beta_mean=1.254, L0_frac_beta_gt1=0.657, L0_frac_eig_neg=0.657, L0_min_eig=-0.910, L1_beta_mean=1.058, L1_frac_beta_gt1=0.455, L1_frac_eig_neg=0.455, L1_min_eig=-0.637
- C: L0_frac_a_neg=0.000, L0_min_a=0.325, L1_frac_a_neg=0.000, L1_min_a=0.288
- D: L0_frac_a_neg=0.417, L0_min_a=-0.342, L1_frac_a_neg=0.428, L1_min_a=-0.656
- E: L0_beta_mean=0.414, L0_frac_beta_gt1=0.000, L0_frac_eig_neg=0.000, L0_min_eig=0.153, L1_beta_mean=0.315, L1_frac_beta_gt1=0.000, L1_frac_eig_neg=0.000, L1_min_eig=0.000

## mod3  (metric: per-token accuracy over the last half of the sequence)

| variant | lr | n | params | L=32 | L=64 | L=128 | L=256 | L=512 |
|---|---|---|---|---|---|---|---|---|
| A delta beta(0,1) | 0.003 | 3 | 109067 | 0.985±0.004 [0.98,0.99] | 0.503±0.037 [0.47,0.54] | 0.334±0.002 [0.33,0.34] | 0.334±0.004 [0.33,0.34] | 0.333±0.001 [0.33,0.33] |
| B delta beta(0,2) | 0.001 | 3 | 109067 | 0.995±0.005 [0.99,1.00] | 0.679±0.265 [0.51,0.98] | 0.339±0.012 [0.33,0.35] | 0.331±0.002 [0.33,0.33] | 0.334±0.002 [0.33,0.34] |
| C ssm a(0,1) | 0.001 | 3 | 108867 | 0.979±0.018 [0.96,1.00] | 0.470±0.076 [0.40,0.55] | 0.334±0.003 [0.33,0.34] | 0.336±0.002 [0.33,0.34] | 0.334±0.000 [0.33,0.33] |
| D ssm a(-1,1) | 0.001 | 3 | 108867 | 0.987±0.002 [0.99,0.99] | 0.490±0.013 [0.48,0.50] | 0.333±0.002 [0.33,0.33] | 0.331±0.002 [0.33,0.33] | 0.333±0.001 [0.33,0.33] |
| L LSTM (nonlinear) | 0.001 | 3 | 150467 | 1.000±0.000 [1.00,1.00] | 1.000±0.000 [1.00,1.00] | 1.000±0.000 [1.00,1.00] | 1.000±0.000 [1.00,1.00] | 1.000±0.000 [1.00,1.00] |

Eigenvalue-range usage at the end of training (last forward, mean over seeds):

- A: L0_beta_mean=0.375, L0_frac_beta_gt1=0.000, L0_frac_eig_neg=0.000, L0_min_eig=0.045, L1_beta_mean=0.691, L1_frac_beta_gt1=0.000, L1_frac_eig_neg=0.000, L1_min_eig=0.002
- B: L0_beta_mean=0.879, L0_frac_beta_gt1=0.337, L0_frac_eig_neg=0.337, L0_min_eig=-0.673, L1_beta_mean=0.891, L1_frac_beta_gt1=0.411, L1_frac_eig_neg=0.411, L1_min_eig=-0.908
- C: L0_frac_a_neg=0.000, L0_min_a=0.391, L1_frac_a_neg=0.000, L1_min_a=0.358
- D: L0_frac_a_neg=0.333, L0_min_a=-0.290, L1_frac_a_neg=0.142, L1_min_a=-0.177

## s3  (metric: per-token accuracy over the last half of the sequence)

| variant | lr | n | params | L=32 | L=64 | L=128 | L=256 | L=512 |
|---|---|---|---|---|---|---|---|---|
| A delta beta(0,1) | 0.003 | 3 | 109518 | 0.262±0.027 [0.23,0.28] | 0.170±0.008 [0.16,0.17] | 0.167±0.002 [0.17,0.17] | 0.166±0.001 [0.17,0.17] | 0.167±0.001 [0.17,0.17] |
| B delta beta(0,2) | 0.001 | 3 | 109518 | 0.762±0.342 [0.37,1.00] | 0.664±0.341 [0.32,1.00] | 0.580±0.363 [0.33,1.00] | 0.454±0.335 [0.20,0.83] | 0.262±0.085 [0.16,0.31] |
| C ssm a(0,1) | 0.003 | 3 | 109318 | 0.286±0.013 [0.27,0.30] | 0.171±0.004 [0.17,0.17] | 0.167±0.001 [0.17,0.17] | 0.165±0.002 [0.16,0.17] | 0.167±0.001 [0.17,0.17] |
| D ssm a(-1,1) | 0.003 | 3 | 109318 | 0.747±0.334 [0.37,1.00] | 0.413±0.235 [0.17,0.64] | 0.226±0.068 [0.17,0.30] | 0.184±0.031 [0.17,0.22] | 0.168±0.006 [0.16,0.17] |
| L LSTM (nonlinear) | 0.001 | 3 | 150918 | 1.000±0.000 [1.00,1.00] | 1.000±0.000 [1.00,1.00] | 1.000±0.000 [1.00,1.00] | 1.000±0.000 [1.00,1.00] | 1.000±0.000 [1.00,1.00] |

Eigenvalue-range usage at the end of training (last forward, mean over seeds):

- A: L0_beta_mean=0.366, L0_frac_beta_gt1=0.000, L0_frac_eig_neg=0.000, L0_min_eig=0.119, L1_beta_mean=0.406, L1_frac_beta_gt1=0.000, L1_frac_eig_neg=0.000, L1_min_eig=0.000
- B: L0_beta_mean=1.306, L0_frac_beta_gt1=0.635, L0_frac_eig_neg=0.635, L0_min_eig=-0.922, L1_beta_mean=1.544, L1_frac_beta_gt1=0.746, L1_frac_eig_neg=0.746, L1_min_eig=-1.000
- C: L0_frac_a_neg=0.000, L0_min_a=0.104, L1_frac_a_neg=0.000, L1_min_a=0.089
- D: L0_frac_a_neg=0.283, L0_min_a=-0.873, L1_frac_a_neg=0.360, L1_min_a=-0.898

## mqar  (metric: answer accuracy)

| variant | lr | n | params | L=4 | L=8 | L=16 | L=32 | L=64 |
|---|---|---|---|---|---|---|---|---|
| A delta beta(0,1) | 0.003 | 3 | 125450 | 0.681±0.273 [0.52,1.00] | 0.553±0.361 [0.34,0.97] | 0.403±0.347 [0.20,0.80] | 0.175±0.119 [0.10,0.31] | 0.053±0.009 [0.05,0.06] |
| B delta beta(0,2) | 0.003 | 3 | 125450 | 0.884±0.182 [0.67,0.99] | 0.805±0.287 [0.47,0.99] | 0.685±0.335 [0.30,0.92] | 0.403±0.221 [0.15,0.57] | 0.120±0.036 [0.08,0.14] |
| C ssm a(0,1) | 0.003 | 3 | 125250 | 0.515±0.005 [0.51,0.52] | 0.305±0.012 [0.29,0.31] | 0.167±0.008 [0.16,0.17] | 0.072±0.006 [0.07,0.08] | 0.023±0.001 [0.02,0.02] |
| D ssm a(-1,1) | 0.003 | 3 | 125250 | 0.462±0.076 [0.38,0.52] | 0.284±0.040 [0.24,0.31] | 0.164±0.029 [0.13,0.19] | 0.069±0.011 [0.06,0.08] | 0.022±0.001 [0.02,0.02] |

Eigenvalue-range usage at the end of training (last forward, mean over seeds):

- A: L0_beta_mean=0.257, L0_frac_beta_gt1=0.000, L0_frac_eig_neg=0.000, L0_min_eig=0.064, L1_beta_mean=0.661, L1_frac_beta_gt1=0.000, L1_frac_eig_neg=0.000, L1_min_eig=0.000
- B: L0_beta_mean=0.292, L0_frac_beta_gt1=0.056, L0_frac_eig_neg=0.056, L0_min_eig=-0.514, L1_beta_mean=0.613, L1_frac_beta_gt1=0.238, L1_frac_eig_neg=0.238, L1_min_eig=-0.937
- C: L0_frac_a_neg=0.000, L0_min_a=0.005, L1_frac_a_neg=0.000, L1_min_a=0.034
- D: L0_frac_a_neg=0.353, L0_min_a=-0.732, L1_frac_a_neg=0.506, L1_min_a=-0.836

## lr selection (seed 0, in-distribution val loss)

- mqar/A: chose 3e-3  (val loss by lr: 1e-3=1.73, 3e-3=1.67)
- mqar/B: chose 3e-3  (val loss by lr: 1e-3=1.69, 3e-3=1.43)
- mqar/C: chose 3e-3  (val loss by lr: 1e-3=2.46, 3e-3=2.07)
- mqar/D: chose 3e-3  (val loss by lr: 1e-3=3, 3e-3=2.05)
- parity/A: chose 1e-3  (val loss by lr: 1e-3=0.0161, 3e-3=0.105)
- parity/B: chose 1e-3  (val loss by lr: 1e-3=0, 3e-3=0)
- parity/C: chose 3e-3  (val loss by lr: 1e-3=0.0391, 3e-3=0.0189)
- parity/D: chose 3e-3  (val loss by lr: 1e-3=2.19e-08, 3e-3=2.23e-11)
- parity/L: chose 1e-3  (val loss by lr: 1e-3=0, 3e-3=0)
- mod3/A: chose 3e-3  (val loss by lr: 1e-3=0.016, 3e-3=0.00738)
- mod3/B: chose 1e-3  (val loss by lr: 1e-3=0, 3e-3=0)
- mod3/C: chose 1e-3  (val loss by lr: 1e-3=0.00277, 3e-3=0.0084)
- mod3/D: chose 1e-3  (val loss by lr: 1e-3=0.00538, 3e-3=0.0212)
- mod3/L: chose 1e-3  (val loss by lr: 1e-3=0, 3e-3=0)
- s3/A: chose 3e-3  (val loss by lr: 1e-3=0.923, 3e-3=0.788)
- s3/B: chose 1e-3  (val loss by lr: 1e-3=7.12e-08, 3e-3=0.00651)
- s3/C: chose 3e-3  (val loss by lr: 1e-3=0.873, 3e-3=0.68)
- s3/D: chose 3e-3  (val loss by lr: 1e-3=0.442, 3e-3=0.0767)
- s3/L: chose 1e-3  (val loss by lr: 1e-3=3.17e-10, 3e-3=9.16e-09)
