#!/usr/bin/env bash
# Small PlasticCore LM (substitute for the gated HF checkpoint), CPU. AdamW (--no-muon): torch 2.14.1's
# torch.optim.Muon blew a W_r matrix up to ~5e7 after one step in this setup, causing NaN at step ~100.
set -e
cd "$(dirname "$0")/../.."
. .venv/bin/activate
plastic data prepare --corpus wikitext --out artifacts/data/wikitext --vocab 4096 --tokenizer-lines 100000 --max-train-tokens 6000000
OMP_NUM_THREADS=3 plastic train text --data artifacts/data/wikitext --steps 2000 --batch-size 16 --seq-len 256 \
  --d-model 128 --layers 4 --heads 4 --device cpu --log-every 25 --eval-every 500 --save-every 500 \
  --model-id lm_small_cpu --seed 0 --no-muon --lr-other 1e-3 --warmup-steps 200
