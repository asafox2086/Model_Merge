#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_TAG="${RUN_TAG:-20260719_followup}"
QUEUE_LOG_DIR="${ROOT_DIR}/logs/lamp_merge_followup_queue_${RUN_TAG}"
mkdir -p "${QUEUE_LOG_DIR}"

echo "[$(date '+%F %T')] follow-up queue started pid=$$"
while pgrep -f 'scripts/run_all_avg_eval.py.*lamp_merge_strict_baseline_probe_20260719' >/dev/null; do
  echo "[$(date '+%F %T')] waiting for existing strict baseline probe"
  sleep 60
done

echo "[$(date '+%F %T')] starting TIES/DARE baseline 2x2 run"
RUN_TAG="${RUN_TAG}_baseline_2x2" \
  bash "${SCRIPT_DIR}/run_lamp_merge_baseline_2x2_formal_parallel.sh" \
  > "${QUEUE_LOG_DIR}/baseline_2x2.driver.log" 2>&1

echo "[$(date '+%F %T')] starting DPR/LPC 3x10 scans"
RUN_TAG="${RUN_TAG}_hparam_3x10" \
  bash "${SCRIPT_DIR}/run_lamp_merge_hparam_3x10_formal_parallel.sh" \
  > "${QUEUE_LOG_DIR}/hparam_3x10.driver.log" 2>&1

touch "${QUEUE_LOG_DIR}/DONE"
echo "[$(date '+%F %T')] follow-up queue complete"
