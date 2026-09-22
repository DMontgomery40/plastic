#!/usr/bin/env bash
# Launch the text training job on Hugging Face Jobs.
# Usage: scripts/hf_jobs/launch_text.sh [flavor] [steps] [extra env assignments...]
# Requires: hf >= 1.32 logged in, bucket DMontgomery40/plastic-runs (hf buckets create plastic-runs --private).
set -euo pipefail
FLAVOR="${1:-l4x1}"
STEPS="${2:-3000}"
shift $(( $# > 2 ? 2 : $# )) || true
ENVS=()
for kv in "$@"; do ENVS+=(-e "$kv"); done
hf jobs uv run --flavor "$FLAVOR" --timeout 3h --secrets HF_TOKEN --detach \
  -v hf://buckets/DMontgomery40/plastic-runs:/out \
  -e STEPS="$STEPS" -e OUT_DIR=/out "${ENVS[@]}" \
  --image pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel \
  scripts/hf_jobs/train_text.py
