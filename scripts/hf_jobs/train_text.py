# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=1.32"]
# [tool.hf-jobs]
# flavor = "l4x1"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///
"""Train the plastic text model on Hugging Face Jobs.

Runs inside the job container: clones the repo at REPO_REF, installs it with uv,
prepares the corpus into the mounted bucket if missing, trains, and syncs the
model directory back to the bucket. Launch with scripts/hf_jobs/launch_text.sh.

Environment (set via -e / defaults): REPO_URL, REPO_REF, OUT_DIR (bucket mount),
CORPUS, VOCAB, STEPS, BATCH, SEQ_LEN, EVAL_EVERY, MODEL_JSON, EXTRA_ARGS.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time


def sh(cmd: str, **kw) -> None:
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def main() -> None:
    t0 = time.time()
    repo_url = os.environ.get("REPO_URL", "https://github.com/DMontgomery40/ttt_ssm_eval.git")
    repo_ref = os.environ.get("REPO_REF", "fuse")
    out_dir = os.environ.get("OUT_DIR", "/out")
    corpus = os.environ.get("CORPUS", "wikitext")
    vocab = os.environ.get("VOCAB", "8192")
    steps = os.environ.get("STEPS", "3000")
    batch = os.environ.get("BATCH", "32")
    seq_len = os.environ.get("SEQ_LEN", "1024")
    eval_every = os.environ.get("EVAL_EVERY", "250")
    model_json = os.environ.get("MODEL_JSON", "")
    extra = os.environ.get("EXTRA_ARGS", "")

    work = "/w"
    sh(f"git clone --depth 1 --branch {shlex.quote(repo_ref)} {shlex.quote(repo_url)} {work}")
    os.chdir(work)
    sh("pip install -q uv && uv sync --no-dev")
    print(f"[job] repo ready ({time.time() - t0:.0f}s)", flush=True)

    data_dir = os.path.join(out_dir, "data", corpus)
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        sh(f"uv run plastic data prepare --corpus {corpus} --out {shlex.quote(data_dir)} --vocab {vocab}")
    print(f"[job] data ready ({time.time() - t0:.0f}s)", flush=True)

    artifacts = os.path.join(out_dir, "artifacts")
    cmd = (
        f"uv run plastic train text --data {shlex.quote(data_dir)} --artifacts-root {shlex.quote(artifacts)} "
        f"--steps {steps} --batch-size {batch} --seq-len {seq_len} --eval-every {eval_every} --device cuda "
        f"--save-every {eval_every} --log-every 10"
    )
    if model_json:
        cmd += f" --model-json {shlex.quote(model_json)}"
    if extra:
        cmd += f" {extra}"
    sh(cmd)
    print(f"[job] training done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[job] failed: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)
