# Exp 4 summary (seeds [0]; 12000 first-answer items per seed per arm)

## all (n = 12000 pooled; chance 0.160)

| method | acc | Wilson 95% | per seed |
|---|---|---|---|
| fast weights only (lessons -> S, activations reset) | 0.198 | [0.191, 0.205] | 0.198 |
| in-context (S + activations carried) | 0.198 | [0.191, 0.205] | 0.198 |
| lessons in context, fast-weight writes frozen (beta=0) | 0.088 | [0.083, 0.093] | 0.088 |
| no adaptation (S = W0 = 0, no lessons) | 0.092 | [0.087, 0.097] | 0.092 |
| content null, fast weights only | 0.164 | [0.158, 0.171] | 0.164 |
| content null, in-context | 0.166 | [0.159, 0.172] | 0.166 |
| oracle composer (lookup + compose) | 1.000 | [1.000, 1.000] | 1.000 |
| lookup without composition (x*w1) | 0.276 | [0.268, 0.284] | 0.276 |
| lookup with the WRONG first generator (x*other) | 0.118 | [0.113, 0.124] | 0.118 |
| copy the start element | 0.149 | [0.143, 0.156] | 0.149 |
| copy the last lesson answer (template copier) | 0.156 | [0.150, 0.163] | 0.156 |
| content null, fast weights only: agrees with the null lessons' answer | 0.197 | [0.189, 0.204] | |
| content null, in-context: agrees with the null lessons' answer | 0.196 | [0.189, 0.203] | |

## depth1_recall (n = 2000 pooled; chance 0.160)

| method | acc | Wilson 95% | per seed |
|---|---|---|---|
| fast weights only (lessons -> S, activations reset) | 0.507 | [0.486, 0.529] | 0.507 |
| in-context (S + activations carried) | 0.512 | [0.490, 0.534] | 0.512 |
| lessons in context, fast-weight writes frozen (beta=0) | 0.026 | [0.020, 0.034] | 0.026 |
| no adaptation (S = W0 = 0, no lessons) | 0.024 | [0.018, 0.031] | 0.024 |
| content null, fast weights only | 0.198 | [0.181, 0.216] | 0.198 |
| content null, in-context | 0.205 | [0.188, 0.223] | 0.205 |
| oracle composer (lookup + compose) | 1.000 | [0.998, 1.000] | 1.000 |
| lookup without composition (x*w1) | 1.000 | [0.998, 1.000] | 1.000 |
| lookup with the WRONG first generator (x*other) | 0.000 | [0.000, 0.002] | 0.000 |
| copy the start element | 0.000 | [0.000, 0.002] | 0.000 |
| copy the last lesson answer (template copier) | 0.163 | [0.147, 0.179] | 0.163 |
| content null, fast weights only: agrees with the null lessons' answer | 0.499 | [0.477, 0.521] | |
| content null, in-context: agrees with the null lessons' answer | 0.499 | [0.477, 0.521] | |

## composition_depth_ge2 (n = 10000 pooled; chance 0.160)

| method | acc | Wilson 95% | per seed |
|---|---|---|---|
| fast weights only (lessons -> S, activations reset) | 0.136 | [0.129, 0.143] | 0.136 |
| in-context (S + activations carried) | 0.135 | [0.129, 0.142] | 0.135 |
| lessons in context, fast-weight writes frozen (beta=0) | 0.101 | [0.095, 0.107] | 0.101 |
| no adaptation (S = W0 = 0, no lessons) | 0.105 | [0.100, 0.112] | 0.105 |
| content null, fast weights only | 0.158 | [0.151, 0.165] | 0.158 |
| content null, in-context | 0.158 | [0.151, 0.165] | 0.158 |
| oracle composer (lookup + compose) | 1.000 | [1.000, 1.000] | 1.000 |
| lookup without composition (x*w1) | 0.131 | [0.125, 0.138] | 0.131 |
| lookup with the WRONG first generator (x*other) | 0.142 | [0.135, 0.149] | 0.142 |
| copy the start element | 0.179 | [0.172, 0.187] | 0.179 |
| copy the last lesson answer (template copier) | 0.155 | [0.148, 0.162] | 0.155 |
| content null, fast weights only: agrees with the null lessons' answer | 0.136 | [0.129, 0.143] | |
| content null, in-context: agrees with the null lessons' answer | 0.135 | [0.128, 0.142] | |

## composition_nontrivial (n = 5821 pooled; chance 0.158)

| method | acc | Wilson 95% | per seed |
|---|---|---|---|
| fast weights only (lessons -> S, activations reset) | 0.103 | [0.095, 0.111] | 0.103 |
| in-context (S + activations carried) | 0.102 | [0.094, 0.110] | 0.102 |
| lessons in context, fast-weight writes frozen (beta=0) | 0.022 | [0.019, 0.026] | 0.022 |
| no adaptation (S = W0 = 0, no lessons) | 0.023 | [0.020, 0.028] | 0.023 |
| content null, fast weights only | 0.188 | [0.179, 0.199] | 0.188 |
| content null, in-context | 0.189 | [0.179, 0.199] | 0.189 |
| oracle composer (lookup + compose) | 1.000 | [0.999, 1.000] | 1.000 |
| lookup without composition (x*w1) | 0.000 | [0.000, 0.001] | 0.000 |
| lookup with the WRONG first generator (x*other) | 0.203 | [0.193, 0.214] | 0.203 |
| copy the start element | 0.000 | [0.000, 0.001] | 0.000 |
| copy the last lesson answer (template copier) | 0.000 | [0.000, 0.001] | 0.000 |
| content null, fast weights only: agrees with the null lessons' answer | 0.105 | [0.098, 0.114] | |
| content null, in-context: agrees with the null lessons' answer | 0.104 | [0.097, 0.113] | |

## composition_null_discriminating (n = 6290 pooled; chance 0.157)

| method | acc | Wilson 95% | per seed |
|---|---|---|---|
| fast weights only (lessons -> S, activations reset) | 0.163 | [0.154, 0.172] | 0.163 |
| in-context (S + activations carried) | 0.162 | [0.153, 0.171] | 0.162 |
| lessons in context, fast-weight writes frozen (beta=0) | 0.023 | [0.019, 0.027] | 0.023 |
| no adaptation (S = W0 = 0, no lessons) | 0.023 | [0.019, 0.027] | 0.023 |
| content null, fast weights only | 0.197 | [0.188, 0.207] | 0.197 |
| content null, in-context | 0.197 | [0.188, 0.207] | 0.197 |
| oracle composer (lookup + compose) | 1.000 | [0.999, 1.000] | 1.000 |
| lookup without composition (x*w1) | 0.160 | [0.151, 0.169] | 0.160 |
| lookup with the WRONG first generator (x*other) | 0.169 | [0.160, 0.178] | 0.169 |
| copy the start element | 0.000 | [0.000, 0.001] | 0.000 |
| copy the last lesson answer (template copier) | 0.152 | [0.143, 0.161] | 0.152 |
| content null, fast weights only: agrees with the null lessons' answer | 0.163 | [0.154, 0.172] | |
| content null, in-context: agrees with the null lessons' answer | 0.161 | [0.152, 0.171] | |

## By depth (pooled over seeds; acc [Wilson 95%])

| depth | n | chance | fw_reset | continuous | ssm_only | no_adapt | null_fw_reset | null_continuous | oracle | one_hop | other_gen | copy_start | last_line |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2000 | 0.160 | 0.507 [0.486,0.529] | 0.512 [0.490,0.534] | 0.026 [0.020,0.034] | 0.024 [0.018,0.031] | 0.198 [0.181,0.216] | 0.205 [0.188,0.223] | 1.000 [0.998,1.000] | 1.000 [0.998,1.000] | 0.000 [0.000,0.002] | 0.000 [0.000,0.002] | 0.163 [0.147,0.179] |
| 2 | 2000 | 0.160 | 0.046 [0.038,0.056] | 0.050 [0.041,0.060] | 0.096 [0.084,0.110] | 0.103 [0.091,0.118] | 0.163 [0.147,0.179] | 0.161 [0.145,0.177] | 1.000 [0.998,1.000] | 0.000 [0.000,0.002] | 0.085 [0.074,0.098] | 0.171 [0.155,0.188] | 0.158 [0.142,0.174] |
| 3 | 2000 | 0.160 | 0.173 [0.158,0.191] | 0.172 [0.156,0.189] | 0.096 [0.083,0.109] | 0.101 [0.089,0.115] | 0.159 [0.144,0.176] | 0.158 [0.143,0.175] | 1.000 [0.998,1.000] | 0.181 [0.165,0.198] | 0.173 [0.158,0.191] | 0.151 [0.136,0.167] | 0.152 [0.137,0.168] |
| 4 | 2000 | 0.160 | 0.130 [0.116,0.145] | 0.129 [0.115,0.144] | 0.144 [0.129,0.160] | 0.148 [0.133,0.164] | 0.138 [0.124,0.154] | 0.139 [0.125,0.155] | 1.000 [0.998,1.000] | 0.141 [0.127,0.157] | 0.132 [0.117,0.147] | 0.257 [0.238,0.276] | 0.156 [0.141,0.173] |
| 5 | 2000 | 0.160 | 0.229 [0.211,0.248] | 0.224 [0.206,0.242] | 0.054 [0.045,0.065] | 0.057 [0.047,0.067] | 0.186 [0.170,0.204] | 0.190 [0.173,0.208] | 1.000 [0.998,1.000] | 0.246 [0.228,0.266] | 0.205 [0.188,0.224] | 0.084 [0.072,0.096] | 0.157 [0.142,0.174] |
| 6 | 2000 | 0.160 | 0.102 [0.089,0.116] | 0.103 [0.091,0.118] | 0.113 [0.100,0.128] | 0.118 [0.105,0.133] | 0.143 [0.128,0.159] | 0.141 [0.126,0.156] | 1.000 [0.998,1.000] | 0.087 [0.076,0.101] | 0.115 [0.101,0.129] | 0.234 [0.215,0.253] | 0.151 [0.136,0.168] |

## By group (fast weights only / in-context / no adaptation / content null)

| group/depth | n | chance | fw_reset | continuous | no_adapt | null_fw_reset |
|---|---|---|---|---|---|---|
| D4/depth1 | 300 | 0.125 | 0.547 | 0.547 | 0.027 | 0.137 |
| D4/depth_ge2 | 1500 | 0.125 | 0.127 | 0.125 | 0.101 | 0.129 |
| S3/depth1 | 152 | 0.167 | 0.520 | 0.526 | 0.013 | 0.178 |
| S3/depth_ge2 | 760 | 0.167 | 0.147 | 0.145 | 0.114 | 0.151 |
| Z5/depth1 | 650 | 0.200 | 0.485 | 0.492 | 0.018 | 0.260 |
| Z5/depth_ge2 | 3250 | 0.200 | 0.164 | 0.165 | 0.109 | 0.195 |
| Z6/depth1 | 148 | 0.167 | 0.520 | 0.534 | 0.047 | 0.182 |
| Z6/depth_ge2 | 740 | 0.167 | 0.123 | 0.127 | 0.103 | 0.176 |
| Z7/depth1 | 500 | 0.143 | 0.520 | 0.522 | 0.026 | 0.172 |
| Z7/depth_ge2 | 2500 | 0.143 | 0.125 | 0.122 | 0.106 | 0.136 |
| Z8/depth1 | 250 | 0.125 | 0.480 | 0.480 | 0.020 | 0.180 |
| Z8/depth_ge2 | 1250 | 0.125 | 0.096 | 0.097 | 0.097 | 0.133 |

## Paired differences on composition items (depth >= 2), episode-cluster bootstrap 95% CI

- fw_reset_minus_null_fw_reset_depth_ge2: -0.022 [-0.030, -0.013] (2000 clusters)
- fw_reset_minus_no_adapt_depth_ge2: +0.031 [+0.022, +0.040] (2000 clusters)
- continuous_minus_fw_reset_depth_ge2: -0.001 [-0.003, +0.002] (2000 clusters)
- fw_reset_minus_one_hop_depth_ge2: +0.005 [-0.002, +0.012] (2000 clusters)
