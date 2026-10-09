#!/usr/bin/env bash
# Entry point inside the HF GPU job. Usage: bash experiments/gpu/jobs/job_entry.sh {bench|A|B} RUN_TAG
# Every long-running step is wrapped in `timeout` (in addition to the HF job timeout).
set -u
MODE=$1; TAG=$2
OUT=/w/out/$TAG; mkdir -p $OUT
echo "=== $(date -u) mode=$MODE tag=$TAG"
nvidia-smi | tee $OUT/nvidia-smi.txt
python -c "import torch,json; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'dev':torch.cuda.get_device_name(0)}))" | tee $OUT/env.json
nproc
# background GPU utilisation log (proves the GPU is used)
( while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader >> $OUT/gpu_util.csv; sleep 30; done ) &
UTILPID=$!
SYNC="python experiments/gpu/jobs/sync.py $OUT results/$TAG"
case $MODE in
  bench)
    B=$OUT/bench; mkdir -p $B
    for v in A B C D L; do
      timeout 600 python experiments/gpu/exp1_scaled/train_gpu.py --task s3 --variant $v --steps 300 --n-test 128 --out $B/c1_$v.json | tail -1
    done
    PIDS=()
    for i in 0 1; do for v in A B C D L; do
      timeout 900 python experiments/gpu/exp1_scaled/train_gpu.py --task s3 --variant $v --steps 300 --n-test 128 --seed $i --out $B/c10_${v}_$i.json > $B/c10_${v}_$i.log 2>&1 &
      PIDS+=($!)
    done; done
    sleep 20; nvidia-smi | tee $B/nvidia-smi-during-c10.txt
    wait "${PIDS[@]}"
    tail -n 1 $B/c10_*.log
    timeout 600 python experiments/gpu/memprec/memprec.py --d 64 --trials 64 --out $B/memprec
    ;;
  A)
    if [ -n "${RESUME_FROM:-}" ]; then  # pull finished runs of an earlier (timed-out) job so they are skipped
      python -c "from huggingface_hub import snapshot_download as s; s('dmontgomery40/plastic-gpu-results',repo_type='dataset',allow_patterns=['results/$RESUME_FROM/runs_tmax*/*.json'],local_dir='/tmp/resume')"
      cp -rn /tmp/resume/results/$RESUME_FROM/runs_tmax* $OUT/ 2>/dev/null; ls $OUT
    fi
    timeout ${JOB_A_SECS:-14400} python experiments/gpu/exp1_scaled/run_gpu.py --outdir $OUT --workers ${WORKERS:-10} \
      --steps ${STEPS:-10000} --sync-cmd "$SYNC" --sync-every 600 2>&1 | tee $OUT/driver.log
    ;;
  B)
    timeout ${JOB_B_SECS:-3600} python experiments/gpu/memprec/memprec.py --d 64,128,256 --trials 64 --out $OUT/memprec 2>&1 | tee $OUT/memprec.log
    ;;
esac
echo "=== $(date -u) finished mode=$MODE"
kill $UTILPID 2>/dev/null
$SYNC
