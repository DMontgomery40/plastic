"""Package exp4b code, upload to the private HF dataset repo, launch an HF GPU job.
Run from the repo root:  /workspace/hfjobs-venv/bin/python experiments/exp4_compose/jobs/launch_4b.py MODE TAG --timeout 40m --env K=V,K=V
HF_TOKEN comes from the env and is forwarded only as a job secret (never printed or written)."""
import argparse, io, os, tarfile, time
from huggingface_hub import HfApi

REPO = "dmontgomery40/plastic-gpu-results"
IMAGE = "pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime"
ap = argparse.ArgumentParser()
ap.add_argument("mode"); ap.add_argument("tag")
ap.add_argument("--flavor", default="a10g-large")
ap.add_argument("--timeout", default="30m")
ap.add_argument("--env", default="")
a = ap.parse_args()
api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
files = ["plastic"] + [f"experiments/exp4_compose/{f}" for f in ("task.py", "common.py", "train.py", "evaluate.py", "sanity_4b.py")] + \
        ["experiments/exp4_compose/jobs/job_entry_4b.sh", "experiments/exp4_compose/jobs/sync.py"]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    for f in files:
        tf.add(f, filter=lambda ti: None if "__pycache__" in ti.name else ti)
name = f"code/code-{a.tag}.tgz"
api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=name, repo_id=REPO, repo_type="dataset", commit_message=f"code for {a.tag}")
cmd = ("set -e; python -m pip install -q --break-system-packages 'huggingface_hub>=1.0' 2>&1 | tail -2; "
       f"python -c \"from huggingface_hub import hf_hub_download as h; h('{REPO}','{name}',repo_type='dataset',local_dir='/tmp/dl')\"; "
       f"mkdir -p /w && tar xzf /tmp/dl/{name} -C /w && cd /w && bash experiments/exp4_compose/jobs/job_entry_4b.sh {a.mode} {a.tag}")
env = dict(kv.split("=", 1) for kv in a.env.split(",") if kv)
env["PYTHONUNBUFFERED"] = "1"
job = api.run_job(image=IMAGE, command=["bash", "-c", cmd], env=env, secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
                  flavor=a.flavor, timeout=a.timeout, name=f"exp4b-{a.tag}", labels={"project": "plastic-exp4b"})
print("job", job.id, a.flavor, a.timeout, time.strftime("%H:%M:%S"))
with open("experiments/exp4_compose/jobs/jobs_4b.log", "a") as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} {job.id} mode={a.mode} tag={a.tag} flavor={a.flavor} timeout={a.timeout} env={env}\n")
