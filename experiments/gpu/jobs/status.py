"""Show status, duration and cost of jobs in jobs.log; optionally tail logs of one job.
/workspace/hfjobs-venv/bin/python experiments/gpu/jobs/status.py [JOB_ID [N_LINES]]"""
import sys, datetime as dt
from huggingface_hub import HfApi

PRICE = {"a10g-large": 1.50, "a10g-small": 1.00}
api = HfApi()
ids = [l.split()[3] for l in open("experiments/gpu/jobs/jobs.log") if l.strip()]
tot = 0.0
for i in ids:
    j = api.inspect_job(job_id=i)
    st = j.status.stage
    if j.finished_at is not None and j.durations is not None:
        secs = j.durations.total_secs
    else:
        secs = (dt.datetime.now(dt.timezone.utc) - j.created_at).total_seconds()
    hrs = secs / 3600
    cost = hrs * PRICE.get(str(j.flavor), 1.5)
    tot += cost
    print(i, j.flavor, st, f"{hrs*60:.1f} min", f"${cost:.2f}", j.status.message or "")
print(f"TOTAL (upper bound, created->finished/now): ${tot:.2f}")
if len(sys.argv) > 1:
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    lines = list(api.fetch_job_logs(job_id=sys.argv[1], follow=False))
    print("\n".join(str(x) for x in lines[-n:]))
