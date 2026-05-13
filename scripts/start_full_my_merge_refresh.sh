#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-my_merge_full_refresh_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

LOG_PATH="${LOG_DIR}/launcher.log"
PID_PATH="${LOG_DIR}/runner.pid"

setsid nohup env RUN_TAG="${RUN_TAG}" bash "${ROOT_DIR}/scripts/run_full_my_merge_refresh.sh" > "${LOG_PATH}" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${PID_PATH}"

sleep 2
if ! ps -p "${PID}" >/dev/null 2>&1; then
  echo "failed to keep process alive"
  echo "run_tag=${RUN_TAG}"
  echo "log=${LOG_PATH}"
  exit 1
fi

echo "started pid=${PID}"
echo "run_tag=${RUN_TAG}"
echo "log=${LOG_PATH}"
echo "pid_file=${PID_PATH}"
