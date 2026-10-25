"""Package the code, upload it to the private HF dataset repo, and launch an HF GPU job.
Run from the repo root with the dedicated venv:  /workspace/hfjobs-venv/bin/python experiments/gpu/jobs/launch.py MODE TAG [--timeout 4h]
The token is read from the HF_TOKEN env var by huggingface_hub and forwarded as a job *secret*; it is never written to files."""
import argparse, io, os, subprocess, tarfile, time
from huggingface_hub import HfApi

REPO = "dmontgomery40/plastic-gpu-results"
IMAGE = "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime"
ap = argparse.ArgumentParser()
ap.add_argument("mode"); ap.add_argument("tag")
ap.add_argument("--flavor", default="a10g-large")
ap.add_argument("--timeout", default="1h")
ap.add_argument("--env", default="", help="k=v,k=v extra env (e.g. WORKERS=10,STEPS=10000,JOB_A_SECS=14000)")
a = ap.parse_args()
api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
files = ["plastic", "experiments/exp1_eigen/models.py", "experiments/exp1_eigen/tasks.py",
         "experiments/gpu/exp1_scaled/train_gpu.py", "experiments/gpu/exp1_scaled/run_gpu.py",
         "experiments/gpu/memprec/memprec.py", "experiments/gpu/jobs/job_entry.sh", "experiments/gpu/jobs/sync.py"]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    for f in files:
        tf.add(f, filter=lambda ti: None if "__pycache__" in ti.name else ti)
name = f"code/code-{a.tag}.tgz"
api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=name, repo_id=REPO, repo_type="dataset",
                commit_message=f"code for {a.tag}")
cmd = ("set -e; python -m pip install -q --break-system-packages 'huggingface_hub>=1.0' 2>&1 | tail -2; "
       f"python -c \"from huggingface_hub import hf_hub_download as h; h('{REPO}','{name}',repo_type='dataset',local_dir='/tmp/dl')\"; "
       f"mkdir -p /w && tar xzf /tmp/dl/{name} -C /w && cd /w && bash experiments/gpu/jobs/job_entry.sh {a.mode} {a.tag}")
env = dict(kv.split("=", 1) for kv in a.env.split(",") if kv)
env["PYTHONUNBUFFERED"] = "1"
job = api.run_job(image=IMAGE, command=["bash", "-c", cmd], env=env, secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
                  flavor=a.flavor, timeout=a.timeout, name=f"plastic-{a.tag}", labels={"project": "plastic-gpu"})
print("job", job.id, a.flavor, a.timeout, time.strftime("%H:%M:%S"))
with open("experiments/gpu/jobs/jobs.log", "a") as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} {job.id} mode={a.mode} tag={a.tag} flavor={a.flavor} timeout={a.timeout} env={env}\n")
