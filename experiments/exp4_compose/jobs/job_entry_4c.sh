#!/usr/bin/env bash
# Inside the HF GPU job.  bash job_entry_4c.sh warm TAG      (exp 4c: warm-start the three aligned 4b checkpoints)
# env: TLIM (per-run training wall-time cap, s; also sets the lr decay schedule), EVAL_BATCHES, SEEDS
set -u
MODE=$1; TAG=$2
OUT=/w/out/$TAG; mkdir -p $OUT /w/init
cd /w/experiments/exp4_compose
export PYTHONPATH=/w:/w/experiments/exp4_compose
echo "=== $(date -u) mode=$MODE tag=$TAG"
nvidia-smi | tee $OUT/nvidia-smi.txt
python -c "import torch,json; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'dev':torch.cuda.get_device_name(0)}))" | tee $OUT/env.json
( while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader >> $OUT/gpu_util.csv; sleep 30; done ) &
UTILPID=$!
SYNC="python /w/experiments/exp4_compose/jobs/sync.py $OUT results/$TAG"
( while true; do sleep 480; $SYNC; done ) &
SYNCPID=$!
run() {
  S=$1; NAME=aligned_s$S; D=$OUT/$NAME; mkdir -p $D
  python -c "from huggingface_hub import hf_hub_download as h; h('dmontgomery40/plastic-gpu-results','results/main4b_60k/$NAME/model/model.pt',repo_type='dataset',local_dir='/w/init')" > $D/download.log 2>&1
  python train.py --seed $S --layout aligned --device cuda --init-from /w/init/results/main4b_60k/$NAME/model/model.pt --rng-offset 7000 \
     --curriculum adaptive --train-depths 1,2,3,4,5 --lr-sched const_decay --time-decay --decay-frac 0.15 --warmup 300 --unlock-acc 0.9 \
     --steps 100000000 --time-limit-s ${TLIM:-1e9} --out $D/model > $D/train.log 2>&1
  python evaluate.py --model $D/model/model.pt --eval-seed $S --device cuda --batches ${EVAL_BATCHES:-40} --out $D/items_seed$S.csv.gz > $D/eval.log 2>&1
  echo "run $NAME finished $(date -u)" >> $OUT/runs_done.txt
}
PIDS=()
for s in ${SEEDS:-0 1 2}; do run $s & PIDS+=($!); done
wait "${PIDS[@]}"
echo "=== $(date -u) finished mode=$MODE"
kill $UTILPID $SYNCPID 2>/dev/null
$SYNC
