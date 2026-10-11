#!/bin/bash
# stop retention runs at step >=1500 (pre-decided)
while true; do
  alive=0
  for s in 0 1 2; do
    pid=$(pgrep -f "retention_train.py.*seg_s$s ")
    [ -z "$pid" ] && continue
    alive=1
    st=$(tail -n1 /workspace/r1/runs/seg_s$s.log | python3 -c "import sys,json
try: print(json.loads(sys.stdin.read())['step'])
except Exception: print(0)")
    if [ "$st" -ge 1500 ]; then kill $pid; echo "$(date) killed s$s at $st" >> /workspace/r1/runs/watch.log; fi
  done
  [ $alive -eq 0 ] && break
  sleep 20
done
