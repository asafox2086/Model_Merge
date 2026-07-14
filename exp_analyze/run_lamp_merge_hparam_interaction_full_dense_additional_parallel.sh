#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${ROOT_DIR}/outputs/lamp_merge_hparam_interaction_full_20260714_hparam_interaction_local5x5_rerun}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${ROOT_DIR}/outputs/lamp_merge_hparam_interaction_full_${RUN_TAG}_dense_additional5x5}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/lamp_merge_hparam_interaction_full_${RUN_TAG}_dense_additional5x5}"

export GAMMA_VALUES="0.35 0.40 0.45 0.50 0.55"
export S_VALUES="13.75 16.25 18.75 21.25 23.75"
export TAU_VALUES="1.5 2.0 2.5 3.0 3.5"
export LAMBDA_VALUES="3.75 4.25 4.75 5.25 5.75"
export BASE_OUTPUT_ROOT
export LOG_DIR
export SUMMARY_ROOTS="${REFERENCE_ROOT} ${BASE_OUTPUT_ROOT}"
export EXPECTED_POINTS=100
export SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-32}"
export GPU_IDS="${GPU_IDS:-0 1 2 3}"

exec bash "${SCRIPT_DIR}/run_lamp_merge_hparam_interaction_full_parallel.sh"
