# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=1.32"]
# ///
"""Chat SFT of a TTT-MLP base on Hugging Face Jobs (see scripts/hf_jobs/launch_sft.sh).

Inside the job container (pytorch CUDA image): copy the exported source from SRC_DIR, install the
runtime deps into the image's Python (keeping its torch), download the pinned base checkpoint from the
Hub, run scripts/train/sft_ttt_chat.py, and leave everything under OUT_DIR on the bucket.

Environment: SRC_DIR, OUT_DIR, BASE_REPO, MODEL_ID, MEASURE (steps; >0 = timing only), SEQ_LEN, BATCH,
GRAD_ACCUM, STEPS, LR, SUBSETS, MAX_ROWS, DTYPE, EXTRA_ARGS.
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
    os.environ.setdefault("PIP_BREAK_SYSTEM_PACKAGES", "1")
    src = os.environ["SRC_DIR"]
    out = os.environ.get("OUT_DIR", "/out")
    base = os.environ.get("BASE_REPO", "RetentionLabs/TTT-MLP-760M-Base-Pile-8k")
    model_id = os.environ.get("MODEL_ID", "ttt_mlp_760m_chat")
    measure = int(os.environ.get("MEASURE", "0"))
    work = "/w"
    sh(f"cp -r {shlex.quote(src)} {work}")
    os.chdir(work)
    py = os.environ.get("PYBIN") or next(
        (c for c in ("/opt/conda/bin/python", "/usr/local/bin/python", "/usr/bin/python3") if os.path.exists(c)), sys.executable)
    sh(f"{py} -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'")
    sh(f"{py} -m pip install -q 'transformers>=5.17,<6' 'datasets>=3' safetensors 'huggingface_hub>=1.32' numpy")
    ckpt = f"{work}/ckpt"
    sh(f"{py} -c \"from huggingface_hub import snapshot_download; snapshot_download('{base}', local_dir='{ckpt}')\"")
    dest = f"{out}/artifacts/models/{model_id}" if not measure else f"{out}/artifacts/measure/{model_id}-{int(t0)}"
    os.makedirs(dest, exist_ok=True)
    args = [
        "--checkpoint", ckpt, "--out", dest, "--device", "cuda",
        "--seq-len", os.environ.get("SEQ_LEN", "2048"), "--batch", os.environ.get("BATCH", "8"),
        "--grad-accum", os.environ.get("GRAD_ACCUM", "1"), "--steps", os.environ.get("STEPS", "2000"),
        "--lr", os.environ.get("LR", "2e-5"), "--dtype", os.environ.get("DTYPE", "bf16"),
        "--subsets", os.environ.get("SUBSETS", "everyday-conversations,smol-magpie-ultra,openhermes-100k,systemchats-30k,smol-constraints,smol-rewrite,smol-summarize"),
    ]
    if os.environ.get("MAX_ROWS"):
        args += ["--max-rows", os.environ["MAX_ROWS"]]
    if measure:
        args += ["--measure", str(measure)]
    extra = os.environ.get("EXTRA_ARGS", "")
    # the reference scan keeps every mini-batch's carried fast weights for backward; checkpoint groups (see
    # --grad-checkpoint-groups) and the expandable allocator keep a 760M/1.3B run inside 80 GB
    env = f"PYTHONPATH={work} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    sh(f"{env} {py} -m scripts.train.sft_ttt_chat {' '.join(shlex.quote(a) for a in args)} {extra}")
    sh(f"ls -la {shlex.quote(dest)}")
    print(f"[job] done in {round(time.time() - t0)} s -> {dest}", flush=True)


if __name__ == "__main__":
    main()
