#!/usr/bin/env bash
# Exp 3 pipeline (exact commands used). Needs artifacts/models/lm_small_cpu (exp2) and artifacts/data/wikitext.
set -e
cd "$(dirname "$0")/../.."
. .venv/bin/activate
pip install hypothesis   # 6.168.5 was used
# (a) invariants
EXP3_MAX_EXAMPLES=200 EXP3_REAL_EXAMPLES=8 python -m pytest experiments/exp3_invariants/test_invariants.py -p no:cacheprovider -rA -v \
  --junitxml=experiments/results/exp3/pytest_junit.xml > experiments/logs/exp3_pytest.log 2>&1 || true
OMP_NUM_THREADS=1 python experiments/exp3_invariants/scale_real.py --out experiments/results/exp3/scale_real.json > experiments/logs/exp3_scale_real.log 2>&1
# (b) calibration records (re-run of `plastic calibrate`'s stream; raw records saved)
for s in shipped statements; do
  OMP_NUM_THREADS=1 python experiments/exp3_invariants/calib_records.py --suite $s \
    --out experiments/results/exp3/calib_records_$s.json > experiments/logs/exp3_calib_$s.log 2>&1 &
done; wait
# (b) policy runs: shipped calibration, threshold sweep, statement canaries
jobs=()
for seed in 0 1 2; do
  jobs+=("shipped shipped $seed")
  for t in 0.001 0.05 0.1 0.2; do jobs+=("shipped $t $seed"); done
  for t in 0.01 0.1; do jobs+=("statements $t $seed"); done
done
printf '%s\n' "${jobs[@]}" | xargs -P 7 -L 1 bash -c 'suite=$0; t=$1; seed=$2;
  rec=""; [ "$t" != shipped ] && rec="--records experiments/results/exp3/calib_records_$suite.json";
  OMP_NUM_THREADS=1 python experiments/exp3_invariants/fpr.py --suite $suite --target $t $rec --seed $seed --sessions 30 \
    --out experiments/results/exp3/fpr_${suite}_${t}_seed$seed.json > experiments/logs/exp3_fpr_${suite}_${t}_seed$seed.log 2>&1'
python experiments/exp3_invariants/analyze.py
