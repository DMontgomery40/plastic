#!/usr/bin/env bash
# Launch the text training job on Hugging Face Jobs from the local checkout.
#
# Usage: scripts/hf_jobs/launch_text.sh [flavor] [steps] [KEY=VALUE ...]
#   e.g. scripts/hf_jobs/launch_text.sh l4x1 3000 BATCH=32 MODEL_ID=lm_wikitext_l4
#
# The package source is exported from git HEAD into a temp directory and synced
# to the job as a read-only volume (no push required). Outputs go to the private
# bucket dmontgomery40/plastic-runs, mounted at /out. Requires hf >= 1.32, logged in.
set -euo pipefail
FLAVOR="${1:-l4x1}"
STEPS="${2:-3000}"
shift $(( $# > 2 ? 2 : $# )) || true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPORT="${EXPORT_DIR:-/tmp/plastic-src-export}"
rm -rf "$EXPORT" && mkdir -p "$EXPORT"
git -C "$ROOT" archive HEAD plastic pyproject.toml README.md LICENSE | tar -x -C "$EXPORT"
echo "[launch] exported $(git -C "$ROOT" rev-parse --short HEAD) to $EXPORT"
ENVS=()
for kv in "$@"; do ENVS+=(-e "$kv"); done
hf jobs uv run --flavor "$FLAVOR" --timeout 3h --secrets HF_TOKEN --detach \
  -v hf://buckets/dmontgomery40/plastic-runs:/out \
  -v "$EXPORT:/src:ro" \
  -e SRC_DIR=/src -e OUT_DIR=/out -e STEPS="$STEPS" "${ENVS[@]}" \
  --image pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel \
  "$ROOT/scripts/hf_jobs/train_text.py"
