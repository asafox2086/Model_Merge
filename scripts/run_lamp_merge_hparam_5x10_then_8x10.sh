#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${ROOT_DIR}/exp_analyze/run_lamp_merge_hparam_3x10_formal_parallel.sh"
RUN_TAG="${RUN_TAG:-20260720_full_queue}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${ROOT_DIR}/outputs/lamp_merge_hparam_${RUN_TAG}}"
BASE_LOG_DIR="${BASE_LOG_DIR:-${ROOT_DIR}/logs/lamp_merge_hparam_${RUN_TAG}}"

S_VALUES="12.5 13.75 15 16.25 17.5 18.75 20 21.25 22.5 23.75"
LAMBDA_VALUES="3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0 5.25"

mkdir -p "${BASE_OUTPUT_DIR}" "${BASE_LOG_DIR}"

run_phase() {
  local phase="$1"
  local gamma_values="$2"
  local tau_values="$3"
  local output_dir="${BASE_OUTPUT_DIR}/${phase}"
  local log_dir="${BASE_LOG_DIR}/${phase}"

  mkdir -p "${output_dir}" "${log_dir}"
  printf 'phase=%s\ngamma_values=%s\ns_values=%s\ntau_values=%s\nlambda_values=%s\n' \
    "${phase}" "${gamma_values}" "${S_VALUES}" "${tau_values}" "${LAMBDA_VALUES}" \
    > "${output_dir}.manifest"

  RUN_TAG="${RUN_TAG}_${phase}" \
  BASE_OUTPUT_ROOT="${output_dir}" \
  LOG_DIR="${log_dir}" \
  GAMMA_VALUES="${gamma_values}" \
  S_VALUES="${S_VALUES}" \
  TAU_VALUES="${tau_values}" \
  LAMBDA_VALUES="${LAMBDA_VALUES}" \
  RUN_FAMILIES="dpr lpc" \
  GPU_IDS="0 1 2 3" \
  NUM_WORKERS="3" \
  bash "${RUNNER}"
}

echo "[$(date '+%F %T')] starting full 5x10 hyperparameter queue"
run_phase "5x10" "0.40 0.45 0.50 0.55 0.60" "1.5 2.0 2.5 3.0 3.5"
echo "[$(date '+%F %T')] 5x10 complete; starting full 8x10 hyperparameter queue"
run_phase "8x10" "0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70" "1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5"
touch "${BASE_OUTPUT_DIR}/DONE"
echo "[$(date '+%F %T')] full 5x10 then 8x10 queue complete"
