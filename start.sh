#!/usr/bin/env bash
# Start the plastic API and the dashboard dev server.
#
#   ./start.sh
#   API_PORT=13580 DEVICE=mps ./start.sh
#
# No artifacts are generated here. With an empty store the dashboard shows the
# commands that fill it (`plastic data prepare`, `plastic train text ...`).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${ROOT_DIR}/artifacts}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-13579}"
DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-5173}"
DEVICE="${DEVICE:-cpu}"
API_URL="http://${API_HOST}:${API_PORT}"

API_PID=""
cleanup() {
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    echo "[start] stopping API (pid ${API_PID})"
    kill "${API_PID}" 2>/dev/null || true
    wait "${API_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[start] API      ${API_URL}  (artifacts: ${ARTIFACTS_ROOT}, device: ${DEVICE})"
uv run plastic serve \
  --artifacts-root "${ARTIFACTS_ROOT}" \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  --device "${DEVICE}" &
API_PID=$!

for _ in $(seq 1 60); do
  if curl -fsS "${API_URL}/api/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${API_PID}" 2>/dev/null; then
    echo "[start] the API exited before it became healthy" >&2
    exit 1
  fi
  sleep 1
done

if ! curl -fsS "${API_URL}/api/health" >/dev/null 2>&1; then
  echo "[start] the API did not answer ${API_URL}/api/health within 60s" >&2
  exit 1
fi
echo "[start] API is healthy"

echo "[start] dashboard http://${DASHBOARD_HOST}:${DASHBOARD_PORT}"
VITE_API_URL="${API_URL}" npm -C dashboard run dev -- --host "${DASHBOARD_HOST}" --port "${DASHBOARD_PORT}"
