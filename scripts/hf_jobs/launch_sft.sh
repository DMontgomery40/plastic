#!/usr/bin/env bash
# Launch the TTT-MLP chat SFT (or a throughput measurement) on Hugging Face Jobs from the local checkout.
#
# Usage: scripts/hf_jobs/launch_sft.sh <flavor> <timeout> [KEY=VALUE ...]
#   measurement: scripts/hf_jobs/launch_sft.sh a100-large 20m MEASURE=8 BATCH=8 SEQ_LEN=2048
#   real run:    scripts/hf_jobs/launch_sft.sh a100-large 4h STEPS=1500 BATCH=8 SEQ_LEN=2048 MODEL_ID=ttt_mlp_760m_chat
#
# The package source (plastic/, scripts/) is exported from git HEAD and mounted read-only; outputs go to the
# private bucket dmontgomery40/plastic-runs at /out. Paid compute: state flavor and estimate before launching.
set -euo pipefail
FLAVOR="${1:?flavor}"
TIMEOUT="${2:?timeout, e.g. 20m}"
shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPORT="${EXPORT_DIR:-/tmp/plastic-sft-export-$(date +%s)}"
mkdir -p "$EXPORT"
git -C "$ROOT" archive HEAD plastic scripts pyproject.toml README.md LICENSE | tar -x -C "$EXPORT"
echo "[launch] exported $(git -C "$ROOT" rev-parse --short HEAD) to $EXPORT"
ENVS=()
for kv in "$@"; do ENVS+=(-e "$kv"); done
hf jobs uv run --flavor "$FLAVOR" --timeout "$TIMEOUT" --secrets HF_TOKEN --detach \
  -v hf://buckets/dmontgomery40/plastic-runs:/out \
  -v "$EXPORT:/src:ro" \
  -e SRC_DIR=/src -e OUT_DIR=/out -e SRC_COMMIT="$(git -C "$ROOT" rev-parse HEAD)" "${ENVS[@]}" \
  --image pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel \
  "$ROOT/scripts/hf_jobs/sft_ttt.py"
