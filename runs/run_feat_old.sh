#!/bin/bash
# gate features on the ORIGINAL exp4c checkpoints (no retention training): K=0 and K=16 unmarked valid foreign lines
cd /workspace/r1/experiments/exp4_compose
for s in 0 1 2; do
  OMP_NUM_THREADS=1 /workspace/plastic-exp4/.venv/bin/python ../r1_verified/gate_features.py --model /workspace/exp4c/full/results/warm4c/aligned_s$s/model/model.pt --out /workspace/r1/runs/feat/old_s$s.json --conds 0:struct:none,16:struct:none --n 600 --seed $s --threads 1
done
