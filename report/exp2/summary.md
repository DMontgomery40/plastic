# Exp 2 summary

Sessions x (type,size) combos: 360; seeds: [0, 1, 2]
Committed S after forced rollback bit-identical to pre-chunk S (torch.equal, all layers): 360/360
Same check on chunks the calibrated harness rolled back on its own (all X chunks rolled back): 238/238
Activation state changed by the rolled-back chunk (sum over layers ||h_rb - h_pre||): mean 22.636, min 10.054
Rewind (restore pre-chunk activations after rollback) KL to never-seen, max over all: direct 0, frozen 0

## C measured `direct` (learning on during C)

Leakage = KL(never||rollback) / KL(never||commit), per session then averaged; sd across the 3 per-seed means. ΔNLL = NLL(cond) − NLL(never) on C (nats/token).

| type | X chunks | natural decisions (rollback frac) | KL rollback | KL commit | **leakage KL** | KL rb_zero | ΔNLL rollback | ΔNLL commit | leakage ΔNLL | KL rb (tok 0-16) | KL rb (tok 64-256) | KL commit (tok 64-256) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 1 | 0.00 | 0.0191±0.0091 | 0.0359±0.0181 | **0.57±0.15** | 0.0176 | +0.0025 | +0.0035 | 0.73 | 0.3054 | 0.0000 | 0.0028 |
| benign | 2 | 0.00 | 0.0184±0.0128 | 0.0417±0.0169 | **0.42±0.07** | 0.0172 | -0.0007 | +0.0026 | -0.29 | 0.2930 | 0.0000 | 0.0044 |
| benign | 4 | 0.01 | 0.0178±0.0113 | 0.0428±0.0163 | **0.41±0.13** | 0.0153 | +0.0010 | +0.0076 | 0.13 | 0.2841 | 0.0000 | 0.0054 |
| random | 1 | 1.00 | 0.0163±0.0076 | 0.0262±0.0090 | **0.62±0.12** | 0.0150 | -0.0017 | -0.0074 | 0.23 | 0.2603 | 0.0000 | 0.0011 |
| random | 2 | 1.00 | 0.0169±0.0081 | 0.0313±0.0110 | **0.53±0.07** | 0.0165 | -0.0025 | -0.0082 | 0.31 | 0.2692 | 0.0000 | 0.0022 |
| random | 4 | 1.00 | 0.0174±0.0072 | 0.0356±0.0124 | **0.50±0.04** | 0.0165 | +0.0010 | -0.0029 | -0.36 | 0.2773 | 0.0000 | 0.0048 |
| repeat_phrase | 1 | 0.42 | 0.0187±0.0103 | 0.0319±0.0111 | **0.57±0.02** | 0.0176 | -0.0008 | +0.0019 | -0.42 | 0.2967 | 0.0000 | 0.0011 |
| repeat_phrase | 2 | 0.60 | 0.0198±0.0092 | 0.0330±0.0103 | **0.59±0.04** | 0.0175 | +0.0020 | -0.0002 | -10.57 | 0.3145 | 0.0000 | 0.0018 |
| repeat_phrase | 4 | 0.52 | 0.0184±0.0099 | 0.0334±0.0131 | **0.54±0.06** | 0.0173 | +0.0030 | +0.0032 | 0.93 | 0.2922 | 0.0000 | 0.0023 |
| repeat_token | 1 | 1.00 | 0.0177±0.0102 | 0.0223±0.0109 | **0.77±0.01** | 0.0148 | +0.0007 | -0.0022 | -0.32 | 0.2791 | 0.0001 | 0.0003 |
| repeat_token | 2 | 0.98 | 0.0175±0.0101 | 0.0224±0.0100 | **0.75±0.11** | 0.0146 | +0.0004 | -0.0040 | -0.09 | 0.2709 | 0.0001 | 0.0008 |
| repeat_token | 4 | 1.00 | 0.0199±0.0092 | 0.0262±0.0110 | **0.75±0.05** | 0.0159 | +0.0034 | -0.0006 | -5.84 | 0.3083 | 0.0002 | 0.0005 |
| shuffled | 1 | 1.00 | 0.0173±0.0094 | 0.0308±0.0131 | **0.56±0.06** | 0.0165 | +0.0021 | +0.0059 | 0.36 | 0.2758 | 0.0000 | 0.0015 |
| shuffled | 2 | 1.00 | 0.0158±0.0098 | 0.0379±0.0150 | **0.44±0.12** | 0.0146 | -0.0038 | +0.0043 | -0.90 | 0.2507 | 0.0000 | 0.0047 |
| shuffled | 4 | 1.00 | 0.0170±0.0092 | 0.0404±0.0191 | **0.44±0.03** | 0.0157 | -0.0002 | +0.0037 | -0.04 | 0.2703 | 0.0000 | 0.0048 |

- repeat_token: mean uplift of log p(poison tokens) on C vs never: rollback +0.0217, commit +0.0408, rb_rewind +0.0000, rb_zero +0.0010, natural +0.0223
- repeat_phrase: mean uplift of log p(poison tokens) on C vs never: rollback +0.0010, commit +0.0254, rb_rewind +0.0000, rb_zero -0.0058, natural +0.0170

## C measured `frozen` (memory frozen during C)

Leakage = KL(never||rollback) / KL(never||commit), per session then averaged; sd across the 3 per-seed means. ΔNLL = NLL(cond) − NLL(never) on C (nats/token).

| type | X chunks | natural decisions (rollback frac) | KL rollback | KL commit | **leakage KL** | KL rb_zero | ΔNLL rollback | ΔNLL commit | leakage ΔNLL | KL rb (tok 0-16) | KL rb (tok 64-256) | KL commit (tok 64-256) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benign | 1 | 0.00 | 0.0189±0.0089 | 0.4069±0.2079 | **0.06±0.02** | 0.0174 | +0.0027 | +0.0601 | 0.05 | 0.3012 | 0.0000 | 0.3828 |
| benign | 2 | 0.00 | 0.0183±0.0127 | 0.4778±0.2078 | **0.04±0.01** | 0.0170 | -0.0008 | +0.0598 | -0.01 | 0.2918 | 0.0000 | 0.4559 |
| benign | 4 | 0.01 | 0.0174±0.0109 | 0.4457±0.1921 | **0.04±0.01** | 0.0151 | +0.0013 | +0.0257 | 0.05 | 0.2781 | 0.0000 | 0.4241 |
| random | 1 | 1.00 | 0.0161±0.0076 | 0.3473±0.1155 | **0.05±0.01** | 0.0148 | -0.0019 | -0.0493 | 0.04 | 0.2566 | 0.0000 | 0.3274 |
| random | 2 | 1.00 | 0.0166±0.0081 | 0.3864±0.1263 | **0.05±0.01** | 0.0163 | -0.0024 | -0.0592 | 0.04 | 0.2655 | 0.0000 | 0.3659 |
| random | 4 | 1.00 | 0.0171±0.0074 | 0.4193±0.1544 | **0.04±0.01** | 0.0163 | +0.0006 | -0.0145 | -0.04 | 0.2733 | 0.0000 | 0.4043 |
| repeat_phrase | 1 | 0.42 | 0.0185±0.0103 | 0.4029±0.1475 | **0.05±0.01** | 0.0174 | -0.0010 | +0.0526 | -0.02 | 0.2939 | 0.0000 | 0.3833 |
| repeat_phrase | 2 | 0.60 | 0.0197±0.0093 | 0.4140±0.1252 | **0.05±0.01** | 0.0173 | +0.0026 | +0.0009 | 2.69 | 0.3125 | 0.0000 | 0.3943 |
| repeat_phrase | 4 | 0.52 | 0.0181±0.0097 | 0.4288±0.1841 | **0.05±0.01** | 0.0170 | +0.0030 | +0.0099 | 0.31 | 0.2886 | 0.0000 | 0.4126 |
| repeat_token | 1 | 1.00 | 0.0174±0.0103 | 0.2531±0.0785 | **0.07±0.00** | 0.0146 | +0.0006 | +0.0156 | 0.04 | 0.2747 | 0.0001 | 0.2384 |
| repeat_token | 2 | 0.98 | 0.0173±0.0103 | 0.3251±0.1180 | **0.06±0.01** | 0.0143 | -0.0005 | +0.0192 | -0.03 | 0.2679 | 0.0001 | 0.3105 |
| repeat_token | 4 | 1.00 | 0.0194±0.0089 | 0.3426±0.1382 | **0.06±0.01** | 0.0156 | +0.0019 | +0.0317 | 0.06 | 0.3017 | 0.0001 | 0.3218 |
| shuffled | 1 | 1.00 | 0.0170±0.0091 | 0.3703±0.1215 | **0.05±0.00** | 0.0163 | +0.0019 | +0.0044 | 0.42 | 0.2721 | 0.0000 | 0.3512 |
| shuffled | 2 | 1.00 | 0.0155±0.0097 | 0.4809±0.1842 | **0.04±0.02** | 0.0144 | -0.0036 | +0.0558 | -0.06 | 0.2477 | 0.0000 | 0.4646 |
| shuffled | 4 | 1.00 | 0.0167±0.0094 | 0.4396±0.1588 | **0.04±0.00** | 0.0155 | -0.0003 | +0.0431 | -0.01 | 0.2656 | 0.0000 | 0.4169 |

- repeat_token: mean uplift of log p(poison tokens) on C vs never: rollback +0.0198, commit +0.3270, rb_rewind +0.0000, rb_zero +0.0024, natural +0.0236
- repeat_phrase: mean uplift of log p(poison tokens) on C vs never: rollback -0.0001, commit +0.1455, rb_rewind +0.0000, rb_zero -0.0056, natural +0.1419

## Harness decisions on the clean continuation C

- never: commit=0.99, project=0.01
- rollback: commit=0.96, project=0.01, readonly=0.03, rollback=0.00
- commit: commit=0.70, project=0.01, readonly=0.28, rollback=0.00
- rb_rewind: commit=0.95, project=0.01, readonly=0.03, rollback=0.01
- rb_zero: commit=0.96, project=0.01, readonly=0.03, rollback=0.00
- natural: commit=0.87, project=0.01, readonly=0.11, rollback=0.01

Natural-decision reasons on X chunks (counts of reason keys): chunk_loss=555, cusum_alarm=38, session_read_only=31, canary_alignment=5, removed_ratio=5, dot=5, log_write_norm=5, log_delta_norm=1
