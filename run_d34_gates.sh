#!/bin/bash
# D3-D4 gate tests runner (CPU only, 3 seeds, limited compute)
set -eo pipefail

UVY=~/.local/bin/uv
CONDS="64:struct:seg"
N_REPLACE=50
N_ATTACKER=50
N_ROLLBACK=30
BATCH=10
THREADS=1

cd /workspace

for seed in 0 1 2; do
    for i_ckpt in 0 1 2; do
        model="runs/seg_s${i_ckpt}/model.pt"
        echo "[D3.1] seed=$seed ckpt=$i_ckpt: replace-rule test"
        $UVY run python experiments/r1_verified/gate_replace_rule.py \
            --model "$model" --out "runs/d3_gate_replace_s${i_ckpt}.json" \
            --conds "$CONDS" --n $N_REPLACE --batch $BATCH --seed $seed --threads $THREADS
        
        echo "[D3.2] seed=$seed ckpt=$i_ckpt: adaptive attacker test"
        $UVY run python experiments/r1_verified/gate_adaptive_attacker.py \
            --model "$model" --out "runs/d3_gate_attacker_s${i_ckpt}.json" \
            --conds "$CONDS" --n $N_ATTACKER --batch $BATCH --seed $seed --threads $THREADS
        
        echo "[D3.3] seed=$seed ckpt=$i_ckpt: rollback noninterference test"
        $UVY run python experiments/r1_verified/gate_rollback.py \
            --model "$model" --out "runs/d3_gate_rollback_s${i_ckpt}.json" \
            --conds "$CONDS" --n $N_ROLLBACK --batch $BATCH --seed $seed --threads $THREADS
    done
done

echo "D3-D4 gate tests complete. Outputs:"
ls -lah runs/d3_gate_*.json
