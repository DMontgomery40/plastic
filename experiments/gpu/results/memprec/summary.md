# Job B: memory-precision scaling, noiseless Gaussian regression

Trials per d: 64; devices: ['NVIDIA A10G']; delta stream length 80·d; LS stream 3·d, solved every d/16 samples.

Median samples to reach angular error ≤ eps (over trials). '(x%)' = fraction of trials that reached it within the stream; '–' = fewer than half reached it.

## d = 64

| method | eps=0.1 | eps=0.01 | eps=0.001 | eps=0.0001 | eps=1e-05 | eps=1e-06 | median final angle |
|---|---|---|---|---|---|---|---|
| delta_beta0.25 | 570 | 1294 | 1978 | 2663 | 3337 | 4022 | 2.36e-08 |
| delta_beta0.5 | 375 | 764 | 1154 | 1548 | 1950 | 2341 | 6.05e-14 |
| delta_beta0.75 | 310 | 617 | 924 | 1245 | 1543 | 1842 | 3.10e-16 |
| delta_beta0.9 | 296 | 589 | 872 | 1172 | 1460 | 1753 | 2.99e-16 |
| delta_beta1.0 | 292 | 584 | 868 | 1160 | 1448 | 1739 | 3.00e-16 |
| delta_beta1.25 | 316 | 624 | 924 | 1228 | 1538 | 1848 | 3.70e-16 |
| delta_beta1.5 | 388 | 778 | 1160 | 1560 | 1938 | 2328 | 5.60e-14 |
| delta_beta1.75 | 665 | 1338 | 1994 | 2666 | 3334 | 4012 | 2.12e-08 |
| delta_sigmoid_b0 | 406 | 816 | 1224 | 1628 | 2038 | 2444 | 2.93e-13 |
| delta_sigmoid_b3 | 296 | 592 | 878 | 1174 | 1458 | 1742 | 2.89e-16 |
| delta_beta1.0_q4_nearest | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 4.01e-01 |
| delta_beta1.0_q4_stochastic | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.22e+00 |
| delta_beta1.0_q8_nearest | 304 | – (17%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.36e-02 |
| delta_beta1.0_q8_stochastic | 322 | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 5.37e-02 |
| delta_beta1.0_q16_nearest | 292 | 584 | 872 | 1362 | – (0%) | – (0%) | 4.99e-05 |
| delta_beta1.0_q16_stochastic | 292 | 584 | 874 | – (0%) | – (0%) | – (0%) | 2.08e-04 |
| delta_beta1.5_q4_nearest | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 9.25e-01 |
| delta_beta1.5_q4_stochastic | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.56e+00 |
| delta_beta1.5_q8_nearest | 404 | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 3.49e-02 |
| delta_beta1.5_q8_stochastic | 448 | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 6.91e-02 |
| delta_beta1.5_q16_nearest | 388 | 779 | 1168 | 3696 (66%) | – (0%) | – (0%) | 1.32e-04 |
| delta_beta1.5_q16_stochastic | 387 | 776 | 1169 | – (0%) | – (0%) | – (0%) | 2.61e-04 |
| delta_beta1.0_fp32 | 292 | 584 | 868 | 1160 | 1448 | 1738 | 1.14e-07 |
| ls_exact | 64 | 64 | 64 | 64 | 64 | 64 | 2.97e-15 |
| ls_q16_nearest | 68 | 72 | 118 | – (0%) | – (0%) | – (0%) | 7.03e-04 |
| ls_q8_nearest | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.89e-01 |

## d = 128

| method | eps=0.1 | eps=0.01 | eps=0.001 | eps=0.0001 | eps=1e-05 | eps=1e-06 | median final angle |
|---|---|---|---|---|---|---|---|
| delta_beta0.25 | 1146 | 2590 | 3978 | 5338 | 6695 | 8040 | 2.21e-08 |
| delta_beta0.5 | 748 | 1558 | 2356 | 3129 | 3914 | 4682 | 6.93e-14 |
| delta_beta0.75 | 614 | 1248 | 1876 | 2504 | 3132 | 3740 | 3.88e-16 |
| delta_beta0.9 | 584 | 1174 | 1774 | 2364 | 2958 | 3548 | 3.30e-16 |
| delta_beta1.0 | 581 | 1164 | 1745 | 2340 | 2922 | 3504 | 3.42e-16 |
| delta_beta1.25 | 615 | 1230 | 1852 | 2476 | 3094 | 3724 | 4.28e-16 |
| delta_beta1.5 | 775 | 1558 | 2344 | 3118 | 3893 | 4681 | 7.56e-14 |
| delta_beta1.75 | 1339 | 2686 | 4035 | 5374 | 6724 | 8061 | 2.35e-08 |
| delta_sigmoid_b0 | 821 | 1646 | 2478 | 3313 | 4138 | 4968 | 4.06e-13 |
| delta_sigmoid_b3 | 584 | 1178 | 1774 | 2366 | 2963 | 3538 | 3.33e-16 |
| delta_beta1.0_q4_nearest | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.24e+00 |
| delta_beta1.0_q4_stochastic | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.56e+00 |
| delta_beta1.0_q8_nearest | 634 | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 2.08e-02 |
| delta_beta1.0_q8_stochastic | 759 | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 8.42e-02 |
| delta_beta1.0_q16_nearest | 581 | 1164 | 1748 | 3906 | – (0%) | – (0%) | 7.92e-05 |
| delta_beta1.0_q16_stochastic | 581 | 1164 | 1764 | – (0%) | – (0%) | – (0%) | 3.17e-04 |
| delta_beta1.5_q4_nearest | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.57e+00 |
| delta_beta1.5_q4_stochastic | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.57e+00 |
| delta_beta1.5_q8_nearest | 874 | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 5.26e-02 |
| delta_beta1.5_q8_stochastic | 1299 (91%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.02e-01 |
| delta_beta1.5_q16_nearest | 776 | 1560 | 2348 | – (0%) | – (0%) | – (0%) | 2.06e-04 |
| delta_beta1.5_q16_stochastic | 775 | 1558 | 2380 | – (0%) | – (0%) | – (0%) | 3.91e-04 |
| delta_beta1.0_fp32 | 581 | 1164 | 1745 | 2340 | 2922 | 3512 | 1.36e-07 |
| ls_exact | 128 | 128 | 128 | 128 | 128 | 128 | 4.23e-15 |
| ls_q16_nearest | 136 | 152 | 324 (89%) | – (0%) | – (0%) | – (0%) | 9.52e-04 |
| ls_q8_nearest | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 2.60e-01 |

## d = 256

| method | eps=0.1 | eps=0.01 | eps=0.001 | eps=0.0001 | eps=1e-05 | eps=1e-06 | median final angle |
|---|---|---|---|---|---|---|---|
| delta_beta0.25 | 2332 | 5216 | 7968 | 10707 | 13404 | 16128 | 2.36e-08 |
| delta_beta0.5 | 1526 | 3127 | 4695 | 6293 | 7826 | 9413 | 8.63e-14 |
| delta_beta0.75 | 1258 | 2527 | 3770 | 5003 | 6274 | 7520 | 5.43e-16 |
| delta_beta0.9 | 1202 | 2379 | 3571 | 4756 | 5947 | 7113 | 4.55e-16 |
| delta_beta1.0 | 1186 | 2341 | 3528 | 4710 | 5873 | 7051 | 4.45e-16 |
| delta_beta1.25 | 1250 | 2495 | 3756 | 4998 | 6273 | 7517 | 5.15e-16 |
| delta_beta1.5 | 1560 | 3127 | 4686 | 6278 | 7840 | 9398 | 8.71e-14 |
| delta_beta1.75 | 2669 | 5354 | 8043 | 10764 | 13429 | 16140 | 2.38e-08 |
| delta_sigmoid_b0 | 1614 | 3324 | 4986 | 6676 | 8333 | 10020 | 5.11e-13 |
| delta_sigmoid_b3 | 1198 | 2376 | 3570 | 4754 | 5938 | 7128 | 4.50e-16 |
| delta_beta1.0_q4_nearest | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.55e+00 |
| delta_beta1.0_q4_stochastic | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.57e+00 |
| delta_beta1.0_q8_nearest | 1437 | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 3.24e-02 |
| delta_beta1.0_q8_stochastic | – (30%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.29e-01 |
| delta_beta1.0_q16_nearest | 1187 | 2342 | 3572 | – (8%) | – (0%) | – (0%) | 1.23e-04 |
| delta_beta1.0_q16_stochastic | 1186 | 2343 | 3621 | – (0%) | – (0%) | – (0%) | 4.90e-04 |
| delta_beta1.5_q4_nearest | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.55e+00 |
| delta_beta1.5_q4_stochastic | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.59e+00 |
| delta_beta1.5_q8_nearest | 2234 | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 8.09e-02 |
| delta_beta1.5_q8_stochastic | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 1.64e-01 |
| delta_beta1.5_q16_nearest | 1560 | 3128 | 4782 | – (0%) | – (0%) | – (0%) | 3.17e-04 |
| delta_beta1.5_q16_stochastic | 1560 | 3125 | 4871 | – (0%) | – (0%) | – (0%) | 6.26e-04 |
| delta_beta1.0_fp32 | 1186 | 2341 | 3528 | 4712 | 5874 | 7078 | 1.82e-07 |
| ls_exact | 256 | 256 | 256 | 256 | 256 | 256 | 4.71e-15 |
| ls_q16_nearest | 272 | 304 | – (0%) | – (0%) | – (0%) | – (0%) | 1.27e-03 |
| ls_q8_nearest | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | – (0%) | 4.08e-01 |

## Fit: median samples-to-eps vs d·ln(1/eps) (only (d, eps) cells reached by every trial)

slope_origin: n ≈ a·d·ln(1/eps). affine: n ≈ a·d·ln(1/eps) + c·d. theory (normalized-key delta rule, E|e|² contracts by 1-(2β-β²)/d per sample): a = 2/(2β-β²).

| method | points | a (origin) | R² | a (affine) | c (per d) | R² | theory a |
|---|---|---|---|---|---|---|---|
| delta_beta0.25 | 18 | 4.532 | 0.9994 | 4.667 | -1.344 | 0.9999 | 4.571 |
| delta_beta0.5 | 18 | 2.657 | 0.9999 | 2.671 | -0.146 | 1.0000 | 2.667 |
| delta_beta0.75 | 18 | 2.124 | 0.9999 | 2.121 | 0.038 | 0.9999 | 2.133 |
| delta_beta0.9 | 18 | 2.012 | 1.0000 | 2.008 | 0.039 | 1.0000 | 2.020 |
| delta_beta1.0 | 18 | 1.990 | 1.0000 | 1.990 | 0.007 | 1.0000 | 2.000 |
| delta_beta1.25 | 18 | 2.118 | 0.9999 | 2.122 | -0.040 | 0.9999 | 2.133 |
| delta_beta1.5 | 18 | 2.654 | 1.0000 | 2.658 | -0.035 | 1.0000 | 2.667 |
| delta_beta1.75 | 18 | 4.558 | 1.0000 | 4.568 | -0.100 | 1.0000 | 4.571 |
| delta_sigmoid_b0 | 18 | 2.821 | 0.9999 | 2.838 | -0.163 | 0.9999 |  |
| delta_sigmoid_b3 | 18 | 2.012 | 0.9999 | 2.009 | 0.028 | 0.9999 |  |
| delta_beta1.0_q4_nearest | 0 | nan | nan | nan | nan | nan |  |
| delta_beta1.0_q4_stochastic | 0 | nan | nan | nan | nan | nan |  |
| delta_beta1.0_q8_nearest | 3 | 2.365 | 0.9885 | 1.990 | 0.864 | 0.9885 |  |
| delta_beta1.0_q8_stochastic | 2 | nan | nan | nan | nan | nan |  |
| delta_beta1.0_q16_nearest | 11 | 2.239 | 0.8643 | 2.690 | -2.656 | 0.8863 |  |
| delta_beta1.0_q16_stochastic | 9 | 2.018 | 0.9993 | 2.050 | -0.175 | 0.9995 |  |
| delta_beta1.5_q4_nearest | 0 | nan | nan | nan | nan | nan |  |
| delta_beta1.5_q4_stochastic | 0 | nan | nan | nan | nan | nan |  |
| delta_beta1.5_q8_nearest | 3 | 3.584 | 0.9649 | 3.015 | 1.309 | 0.9649 |  |
| delta_beta1.5_q8_stochastic | 1 | nan | nan | nan | nan | nan |  |
| delta_beta1.5_q16_nearest | 9 | 2.677 | 0.9997 | 2.717 | -0.212 | 0.9998 |  |
| delta_beta1.5_q16_stochastic | 9 | 2.705 | 0.9990 | 2.784 | -0.426 | 0.9994 |  |
| delta_beta1.0_fp32 | 18 | 1.993 | 0.9999 | 1.996 | -0.024 | 0.9999 |  |
| ls_exact | 18 | 0.100 | 0.1346 | -0.000 | 1.000 | 1.0000 |  |
| ls_q16_nearest | 7 | 0.296 | 0.7031 | 0.080 | 0.858 | 0.9813 |  |
| ls_q8_nearest | 0 | nan | nan | nan | nan | nan |  |

Sigmoid-gate realised mean beta: d=64: delta_sigmoid_b0=0.500, delta_sigmoid_b3=0.931; d=128: delta_sigmoid_b0=0.500, delta_sigmoid_b3=0.930; d=256: delta_sigmoid_b0=0.500, delta_sigmoid_b3=0.931
