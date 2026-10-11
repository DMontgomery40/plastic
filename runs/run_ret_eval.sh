#!/bin/bash
cd /workspace/r1/experiments/exp4_compose
for s in "$@"; do
 OMP_NUM_THREADS=${THR:-1} /workspace/plastic-exp4/.venv/bin/python ../r1_verified/retention_eval.py --model /workspace/r1/runs/seg_s$s/model.pt --out /workspace/r1/runs/ret/seg_s$s.json --seed $s --n 300 --controls --marker-ablation --threads ${THR:-1}
done
