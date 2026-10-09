# Exp 4 summary (seeds [0, 1, 2]; 12000 first-answer items per seed per arm)

## all (n = 36000 pooled; chance 0.157)

| method | acc | Wilson 95% | per seed |
|---|---|---|---|
| fast weights only (lessons -> S, activations reset) | 0.163 | [0.159, 0.167] | 0.167, 0.160, 0.163 |
| in-context (S + activations carried) | 0.163 | [0.159, 0.167] | 0.166, 0.159, 0.164 |
| lessons in context, fast-weight writes frozen (beta=0) | 0.148 | [0.145, 0.152] | 0.150, 0.143, 0.151 |
| no adaptation (S = W0 = 0, no lessons) | 0.148 | [0.144, 0.152] | 0.150, 0.143, 0.151 |
| content null, fast weights only | 0.163 | [0.159, 0.167] | 0.166, 0.161, 0.162 |
| content null, in-context | 0.163 | [0.160, 0.167] | 0.165, 0.161, 0.164 |
| oracle composer (lookup + compose) | 1.000 | [1.000, 1.000] | 1.000, 1.000, 1.000 |
| lookup without composition (x*w1) | 0.274 | [0.270, 0.279] | 0.276, 0.272, 0.275 |
| lookup with the WRONG first generator (x*other) | 0.115 | [0.112, 0.118] | 0.118, 0.109, 0.117 |
| copy the start element | 0.146 | [0.142, 0.150] | 0.149, 0.141, 0.148 |
| copy the last lesson answer (template copier) | 0.157 | [0.153, 0.161] | 0.156, 0.151, 0.163 |
| content null, fast weights only: agrees with the null lessons' answer | 0.163 | [0.159, 0.167] | |
| content null, in-context: agrees with the null lessons' answer | 0.164 | [0.160, 0.167] | |

## depth1_recall (n = 6000 pooled; chance 0.157)

| method | acc | Wilson 95% | per seed |
|---|---|---|---|
| fast weights only (lessons -> S, activations reset) | 0.187 | [0.178, 0.197] | 0.197, 0.174, 0.192 |
| in-context (S + activations carried) | 0.187 | [0.177, 0.197] | 0.200, 0.172, 0.190 |
| lessons in context, fast-weight writes frozen (beta=0) | 0.013 | [0.011, 0.016] | 0.007, 0.014, 0.018 |
| no adaptation (S = W0 = 0, no lessons) | 0.011 | [0.009, 0.014] | 0.005, 0.013, 0.015 |
| content null, fast weights only | 0.188 | [0.179, 0.198] | 0.203, 0.181, 0.180 |
| content null, in-context | 0.189 | [0.179, 0.199] | 0.204, 0.184, 0.178 |
| oracle composer (lookup + compose) | 1.000 | [0.999, 1.000] | 1.000, 1.000, 1.000 |
| lookup without composition (x*w1) | 1.000 | [0.999, 1.000] | 1.000, 1.000, 1.000 |
| lookup with the WRONG first generator (x*other) | 0.000 | [0.000, 0.001] | 0.000, 0.000, 0.000 |
| copy the start element | 0.000 | [0.000, 0.001] | 0.000, 0.000, 0.000 |
| copy the last lesson answer (template copier) | 0.159 | [0.150, 0.168] | 0.163, 0.153, 0.161 |
| content null, fast weights only: agrees with the null lessons' answer | 0.183 | [0.174, 0.193] | |
| content null, in-context: agrees with the null lessons' answer | 0.184 | [0.174, 0.194] | |

## composition_depth_ge2 (n = 30000 pooled; chance 0.157)

| method | acc | Wilson 95% | per seed |
|---|---|---|---|
| fast weights only (lessons -> S, activations reset) | 0.158 | [0.154, 0.163] | 0.161, 0.158, 0.157 |
| in-context (S + activations carried) | 0.158 | [0.154, 0.162] | 0.159, 0.156, 0.159 |
| lessons in context, fast-weight writes frozen (beta=0) | 0.175 | [0.171, 0.180] | 0.179, 0.169, 0.178 |
| no adaptation (S = W0 = 0, no lessons) | 0.175 | [0.171, 0.180] | 0.179, 0.169, 0.178 |
| content null, fast weights only | 0.158 | [0.154, 0.162] | 0.159, 0.157, 0.158 |
| content null, in-context | 0.158 | [0.154, 0.163] | 0.158, 0.156, 0.162 |
| oracle composer (lookup + compose) | 1.000 | [1.000, 1.000] | 1.000, 1.000, 1.000 |
| lookup without composition (x*w1) | 0.129 | [0.125, 0.133] | 0.131, 0.126, 0.130 |
| lookup with the WRONG first generator (x*other) | 0.138 | [0.134, 0.142] | 0.142, 0.131, 0.140 |
| copy the start element | 0.175 | [0.171, 0.180] | 0.179, 0.169, 0.178 |
| copy the last lesson answer (template copier) | 0.156 | [0.152, 0.161] | 0.155, 0.151, 0.163 |
| content null, fast weights only: agrees with the null lessons' answer | 0.159 | [0.155, 0.163] | |
| content null, in-context: agrees with the null lessons' answer | 0.160 | [0.155, 0.164] | |

## composition_nontrivial (n = 17636 pooled; chance 0.155)

| method | acc | Wilson 95% | per seed |
|---|---|---|---|
| fast weights only (lessons -> S, activations reset) | 0.115 | [0.110, 0.120] | 0.114, 0.118, 0.113 |
| in-context (S + activations carried) | 0.107 | [0.103, 0.112] | 0.106, 0.109, 0.106 |
| lessons in context, fast-weight writes frozen (beta=0) | 0.000 | [0.000, 0.000] | 0.000, 0.000, 0.000 |
| no adaptation (S = W0 = 0, no lessons) | 0.000 | [0.000, 0.000] | 0.000, 0.000, 0.000 |
| content null, fast weights only | 0.116 | [0.111, 0.120] | 0.112, 0.120, 0.115 |
| content null, in-context | 0.115 | [0.110, 0.119] | 0.111, 0.117, 0.115 |
| oracle composer (lookup + compose) | 1.000 | [1.000, 1.000] | 1.000, 1.000, 1.000 |
| lookup without composition (x*w1) | 0.000 | [0.000, 0.000] | 0.000, 0.000, 0.000 |
| lookup with the WRONG first generator (x*other) | 0.195 | [0.189, 0.201] | 0.203, 0.184, 0.198 |
| copy the start element | 0.000 | [0.000, 0.000] | 0.000, 0.000, 0.000 |
| copy the last lesson answer (template copier) | 0.000 | [0.000, 0.000] | 0.000, 0.000, 0.000 |
| content null, fast weights only: agrees with the null lessons' answer | 0.119 | [0.114, 0.124] | |
| content null, in-context: agrees with the null lessons' answer | 0.118 | [0.113, 0.123] | |

## composition_null_discriminating (n = 19170 pooled; chance 0.154)

| method | acc | Wilson 95% | per seed |
|---|---|---|---|
| fast weights only (lessons -> S, activations reset) | 0.117 | [0.112, 0.122] | 0.115, 0.121, 0.114 |
| in-context (S + activations carried) | 0.116 | [0.112, 0.121] | 0.118, 0.118, 0.114 |
| lessons in context, fast-weight writes frozen (beta=0) | 0.000 | [0.000, 0.000] | 0.000, 0.000, 0.000 |
| no adaptation (S = W0 = 0, no lessons) | 0.000 | [0.000, 0.000] | 0.000, 0.000, 0.000 |
| content null, fast weights only | 0.117 | [0.112, 0.121] | 0.116, 0.121, 0.114 |
| content null, in-context | 0.117 | [0.112, 0.121] | 0.117, 0.119, 0.114 |
| oracle composer (lookup + compose) | 1.000 | [1.000, 1.000] | 1.000, 1.000, 1.000 |
| lookup without composition (x*w1) | 0.157 | [0.152, 0.163] | 0.160, 0.156, 0.156 |
| lookup with the WRONG first generator (x*other) | 0.162 | [0.157, 0.168] | 0.169, 0.154, 0.165 |
| copy the start element | 0.000 | [0.000, 0.000] | 0.000, 0.000, 0.000 |
| copy the last lesson answer (template copier) | 0.156 | [0.151, 0.162] | 0.152, 0.153, 0.164 |
| content null, fast weights only: agrees with the null lessons' answer | 0.119 | [0.115, 0.124] | |
| content null, in-context: agrees with the null lessons' answer | 0.119 | [0.114, 0.123] | |

## By depth (pooled over seeds; acc [Wilson 95%])

| depth | n | chance | fw_reset | continuous | ssm_only | no_adapt | null_fw_reset | null_continuous | oracle | one_hop | other_gen | copy_start | last_line |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6000 | 0.157 | 0.187 [0.178,0.197] | 0.187 [0.177,0.197] | 0.013 [0.011,0.016] | 0.011 [0.009,0.014] | 0.188 [0.179,0.198] | 0.189 [0.179,0.199] | 1.000 [0.999,1.000] | 1.000 [0.999,1.000] | 0.000 [0.000,0.001] | 0.000 [0.000,0.001] | 0.159 [0.150,0.168] |
| 2 | 6000 | 0.157 | 0.153 [0.144,0.163] | 0.153 [0.144,0.163] | 0.170 [0.161,0.180] | 0.170 [0.161,0.180] | 0.148 [0.140,0.158] | 0.150 [0.141,0.159] | 1.000 [0.999,1.000] | 0.000 [0.000,0.001] | 0.084 [0.077,0.091] | 0.170 [0.161,0.180] | 0.157 [0.148,0.167] |
| 3 | 6000 | 0.157 | 0.155 [0.146,0.165] | 0.154 [0.145,0.164] | 0.141 [0.132,0.150] | 0.141 [0.132,0.150] | 0.155 [0.146,0.165] | 0.156 [0.147,0.165] | 1.000 [0.999,1.000] | 0.173 [0.163,0.182] | 0.159 [0.149,0.168] | 0.141 [0.132,0.150] | 0.152 [0.143,0.161] |
| 4 | 6000 | 0.157 | 0.170 [0.161,0.180] | 0.170 [0.160,0.179] | 0.236 [0.225,0.247] | 0.236 [0.225,0.247] | 0.169 [0.159,0.178] | 0.167 [0.157,0.176] | 1.000 [0.999,1.000] | 0.137 [0.128,0.145] | 0.133 [0.124,0.141] | 0.236 [0.225,0.247] | 0.161 [0.152,0.171] |
| 5 | 6000 | 0.157 | 0.146 [0.137,0.155] | 0.145 [0.136,0.154] | 0.104 [0.097,0.112] | 0.104 [0.097,0.112] | 0.146 [0.138,0.156] | 0.146 [0.138,0.156] | 1.000 [0.999,1.000] | 0.233 [0.222,0.243] | 0.198 [0.188,0.208] | 0.104 [0.097,0.112] | 0.159 [0.150,0.168] |
| 6 | 6000 | 0.157 | 0.168 [0.159,0.178] | 0.169 [0.160,0.179] | 0.225 [0.215,0.236] | 0.225 [0.215,0.236] | 0.170 [0.160,0.179] | 0.173 [0.164,0.183] | 1.000 [0.999,1.000] | 0.104 [0.096,0.112] | 0.116 [0.108,0.125] | 0.225 [0.215,0.236] | 0.153 [0.144,0.162] |

## By group (fast weights only / in-context / no adaptation / content null)

| group/depth | n | chance | fw_reset | continuous | no_adapt | null_fw_reset |
|---|---|---|---|---|---|---|
| D4/depth1 | 760 | 0.125 | 0.132 | 0.129 | 0.009 | 0.124 |
| D4/depth_ge2 | 3800 | 0.125 | 0.125 | 0.127 | 0.174 | 0.123 |
| S3/depth1 | 522 | 0.167 | 0.193 | 0.201 | 0.013 | 0.207 |
| S3/depth_ge2 | 2610 | 0.167 | 0.172 | 0.173 | 0.207 | 0.179 |
| Z5/depth1 | 1450 | 0.200 | 0.242 | 0.246 | 0.009 | 0.244 |
| Z5/depth_ge2 | 7250 | 0.200 | 0.199 | 0.197 | 0.203 | 0.196 |
| Z6/depth1 | 578 | 0.167 | 0.192 | 0.190 | 0.014 | 0.189 |
| Z6/depth_ge2 | 2890 | 0.167 | 0.177 | 0.175 | 0.168 | 0.178 |
| Z7/depth1 | 1950 | 0.143 | 0.193 | 0.185 | 0.012 | 0.191 |
| Z7/depth_ge2 | 9750 | 0.143 | 0.145 | 0.145 | 0.165 | 0.143 |
| Z8/depth1 | 740 | 0.125 | 0.115 | 0.127 | 0.009 | 0.126 |
| Z8/depth_ge2 | 3700 | 0.125 | 0.125 | 0.125 | 0.132 | 0.125 |

## Paired differences on composition items (depth >= 2), episode-cluster bootstrap 95% CI

- fw_reset_minus_null_fw_reset_depth_ge2: +0.001 [-0.002, +0.003] (6000 clusters)
- fw_reset_minus_no_adapt_depth_ge2: -0.017 [-0.022, -0.012] (6000 clusters)
- continuous_minus_fw_reset_depth_ge2: -0.000 [-0.002, +0.002] (6000 clusters)
- fw_reset_minus_one_hop_depth_ge2: +0.029 [+0.024, +0.034] (6000 clusters)
