#!/usr/bin/env bash
# Inside the HF GPU job.  bash job_entry_4b.sh {smoke|main} TAG
# env: STEPS, MIN_D1, TLIM (per-run train wall-time cap, s), EVAL_BATCHES, ABORT_STEP, ABORT_ACC
set -u
MODE=$1; TAG=$2
OUT=/w/out/$TAG; mkdir -p $OUT
cd /w/experiments/exp4_compose
export PYTHONPATH=/w:/w/experiments/exp4_compose
echo "=== $(date -u) mode=$MODE tag=$TAG"
nvidia-smi | tee $OUT/nvidia-smi.txt
python -c "import torch,json; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'dev':torch.cuda.get_device_name(0)}))" | tee $OUT/env.json
nproc
( while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader >> $OUT/gpu_util.csv; sleep 30; done ) &
UTILPID=$!
SYNC="python /w/experiments/exp4_compose/jobs/sync.py $OUT results/$TAG"
( while true; do sleep 480; $SYNC; done ) &
SYNCPID=$!
python sanity_4b.py | tee $OUT/sanity_4b.txt
# run NAME LAYOUT SEED TRAIN_ARGS...  : train then evaluate (3 eval-seed streams are the 3 model seeds)
run() {
  NAME=$1; LAY=$2; SEED=$3; shift 3
  D=$OUT/$NAME; mkdir -p $D
  python train.py --seed $SEED --layout $LAY --device cuda --curriculum adaptive --train-depths 1,2,3,4,5 --lr-sched const_decay \
     --unlock-acc 0.9 --unlock-acc1 0.95 --time-limit-s ${TLIM:-1e9} --out $D/model "$@" > $D/train.log 2>&1
  python evaluate.py --model $D/model/model.pt --eval-seed $SEED --device cuda --batches ${EVAL_BATCHES:-40} --out $D/items_seed$SEED.csv.gz > $D/eval.log 2>&1
  echo "run $NAME finished $(date -u)" >> $OUT/runs_done.txt
}
case $MODE in
  smoke)
    PIDS=()
    for s in 0 1 2; do run aligned_s$s aligned $s --steps 300 --min-depth1-steps 200 --log-every 50 & PIDS+=($!); done
    run old_s0 old 0 --steps 300 --min-depth1-steps 200 --log-every 50 & PIDS+=($!)
    wait "${PIDS[@]}"
    ;;
  main)
    PIDS=()
    for s in 0 1 2; do
      run aligned_s$s aligned $s --steps ${STEPS:-60000} --min-depth1-steps ${MIN_D1:-20000} \
         --abort-depth1-step ${ABORT_STEP:-30000} --abort-depth1-acc ${ABORT_ACC:-0.6} & PIDS+=($!)
    done
    run old_s0 old 0 --steps ${STEPS:-60000} --min-depth1-steps ${MIN_D1:-20000} \
         --abort-depth1-step ${ABORT_STEP:-30000} --abort-depth1-acc ${ABORT_ACC:-0.6} & PIDS+=($!)   # matched old-layout control
    wait "${PIDS[@]}"
    ;;
esac
echo "=== $(date -u) finished mode=$MODE"
kill $UTILPID $SYNCPID 2>/dev/null
$SYNC
