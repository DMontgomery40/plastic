#!/usr/bin/env bash
# Exp 2 pipeline (after experiments/exp2_rollback/train_model.sh has produced artifacts/models/lm_small_cpu)
set -e
cd "$(dirname "$0")/../.."
. .venv/bin/activate
OMP_NUM_THREADS=2 plastic calibrate lm_small_cpu --data artifacts/data/wikitext --chunks 512 --fisher-chunks 64 > experiments/logs/exp2_calibrate.log 2>&1
for s in 0 1 2; do
  python experiments/exp2_rollback/probe.py --seed $s --sessions 8 --threads 1 \
    --out experiments/results/exp2/probe_seed$s.json > experiments/logs/exp2_probe_seed$s.log 2>&1 &
done
wait
