#!/usr/bin/env bash
# fetch Job A partial/final outputs from the private HF dataset repo into experiments/gpu/results/exp1/raw
cd /workspace/plastic-gpu
/workspace/hfjobs-venv/bin/python -c "
from huggingface_hub import snapshot_download as s; s('dmontgomery40/plastic-gpu-results',repo_type='dataset',allow_patterns=['results/jobA1/*','results/jobA1/runs_tmax*/*'],local_dir='/workspace/gpu-dl')" >/dev/null 2>&1
mkdir -p experiments/gpu/results/exp1/raw
cp -r /workspace/gpu-dl/results/jobA1/. experiments/gpu/results/exp1/raw/ && rm -rf experiments/gpu/results/exp1/raw/logs
ls experiments/gpu/results/exp1/raw/runs_tmax32 2>/dev/null | wc -l; ls experiments/gpu/results/exp1/raw/runs_tmax64 2>/dev/null | wc -l; ls experiments/gpu/results/exp1/raw
