#!/bin/bash
s=$1
cd /workspace/r1/experiments/exp4_compose
OMP_NUM_THREADS=1 /workspace/plastic-exp4/.venv/bin/python ../r1_verified/gate_features.py --model /workspace/r1/runs/seg_s$s/model.pt --out /workspace/r1/runs/feat/ret_s$s.json --conds 0:struct:seg,64:struct:seg --n 600 --seed $s --threads 1
OMP_NUM_THREADS=1 /workspace/plastic-exp4/.venv/bin/python ../r1_verified/retention_eval.py --model /workspace/r1/runs/seg_s$s/model.pt --out /workspace/r1/runs/ret/extrap_s$s.json --Ks 256,512 --kinds struct,randiid --n 150 --batch 25 --seed $s --threads 1
