# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=1.32"]
# ///
"""Train the plastic text model on Hugging Face Jobs.

Runs inside the job container (a pytorch/pytorch CUDA image): takes the package
source from a mounted directory (SRC_DIR, synced from the local checkout) or clones
REPO_URL at REPO_REF, installs it into the image's Python (keeping the image's torch),
prepares the corpus into the mounted bucket if missing, and trains. Everything the
run produces lands under OUT_DIR on the bucket. Launch with scripts/hf_jobs/launch_text.sh.

Environment: SRC_DIR, REPO_URL, REPO_REF, OUT_DIR, CORPUS, VOCAB, STEPS, BATCH,
SEQ_LEN, EVAL_EVERY, MODEL_ID, MODEL_JSON, EXTRA_ARGS.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time


def sh(cmd: str) -> None:
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    t0 = time.time()
    src_dir = os.environ.get("SRC_DIR", "")
    repo_url = os.environ.get("REPO_URL", "https://github.com/DMontgomery40/ttt_ssm_eval.git")
    repo_ref = os.environ.get("REPO_REF", "fuse")
    out_dir = os.environ.get("OUT_DIR", "/out")
    corpus = os.environ.get("CORPUS", "wikitext")
    vocab = os.environ.get("VOCAB", "8192")
    steps = os.environ.get("STEPS", "3000")
    batch = os.environ.get("BATCH", "32")
    seq_len = os.environ.get("SEQ_LEN", "1024")
    eval_every = os.environ.get("EVAL_EVERY", "250")
    model_id = os.environ.get("MODEL_ID", "")
    model_json = os.environ.get("MODEL_JSON", "")
    extra = os.environ.get("EXTRA_ARGS", "")

    work = "/w"
    if src_dir:
        sh(f"cp -r {shlex.quote(src_dir)} {work}")
    else:
        sh(f"git clone --depth 1 --branch {shlex.quote(repo_ref)} {shlex.quote(repo_url)} {work}")
    os.chdir(work)
    # `hf jobs uv run` executes this script in an isolated uv environment; the image's
    # torch lives in its conda Python, so every model command uses that interpreter.
    py = os.environ.get("PYBIN") or next(
        (c for c in ("/opt/conda/bin/python", "/usr/local/bin/python", "/usr/bin/python3") if os.path.exists(c)),
        sys.executable,
    )
    sh(f"{py} -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'")
    sh(f"{py} -m pip install -q --no-deps -e . && {py} -m pip install -q 'numpy>=2.0' 'tokenizers>=0.21' "
       f"'datasets>=3.0' 'huggingface_hub>=1.32' 'fastapi>=0.128' 'uvicorn>=0.30' 'pydantic>=2.7'")
    print(f"[job] source ready ({time.time() - t0:.0f}s)", flush=True)

    data_dir = os.path.join(out_dir, "data", corpus)
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        sh(f"{py} -m plastic.cli data prepare --corpus {corpus} --out {shlex.quote(data_dir)} --vocab {vocab}")
    print(f"[job] data ready ({time.time() - t0:.0f}s)", flush=True)

    artifacts = os.path.join(out_dir, "artifacts")
    cmd = (
        f"{py} -m plastic.cli train text --data {shlex.quote(data_dir)} --artifacts-root {shlex.quote(artifacts)} "
        f"--steps {steps} --batch-size {batch} --seq-len {seq_len} --eval-every {eval_every} --device cuda "
        f"--save-every {eval_every} --log-every 10"
    )
    if model_id:
        cmd += f" --model-id {shlex.quote(model_id)}"
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
