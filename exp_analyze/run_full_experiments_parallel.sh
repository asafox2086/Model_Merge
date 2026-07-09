#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data2/liyapeng_grp/.conda/envs/MM/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_full_analyze_parallel)}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-${SMALL_BATCH_SIZE}}"
CHECK_INTERVAL="${CHECK_INTERVAL:-600}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

STEPS=( ${STEPS:-main_table prototype_geometry internal_ablation prediction_diagnostics hparam_full} )

export PYTHON_BIN NUM_WORKERS SMALL_BATCH_SIZE BATCH_SIZE
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"
mkdir -p "${TMPDIR}" "${ROOT_DIR}/logs"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

validate_step() {
  local step="$1"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_full_outputs.py" --quiet --step "${step}"
}

show_validation() {
  local step="$1"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_full_outputs.py" --step "${step}" || true
}

running_lines() {
  local pattern="$1"
  pgrep -af "${pattern}" 2>/dev/null \
    | grep -v "run_full_experiments_parallel.sh" \
    | grep -v "run_full_experiments_sequential.sh" \
    || true
}

wait_existing() {
  local step="$1"
  local pattern="$2"
  [[ -n "${pattern}" ]] || return 0
  local lines
  lines="$(running_lines "${pattern}")"
  if [[ -z "${lines}" ]]; then
    return 0
  fi
  log "${step}: detected an existing full run; waiting instead of starting a duplicate"
  printf '%s\n' "${lines}"
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
    log "${step}: dry-run; would wait for existing process pattern=${pattern}"
    return 0
  fi
  while [[ -n "$(running_lines "${pattern}")" ]]; do
    sleep "${CHECK_INTERVAL}"
    log "${step}: still waiting for existing process pattern=${pattern}"
  done
  log "${step}: existing process finished"
}

run_or_wait_step() {
  local step="$1"
  local pattern="$2"
  shift 2

  log "${step}: validating existing outputs"
  if [[ "${FORCE}" != "1" && "${FORCE,,}" != "true" ]] && validate_step "${step}"; then
    log "${step}: complete; skip"
    return 0
  fi

  wait_existing "${step}" "${pattern}"
  if [[ "${FORCE}" != "1" && "${FORCE,,}" != "true" ]] && validate_step "${step}"; then
    log "${step}: complete after existing run; skip"
    return 0
  fi

  if [[ "$#" -eq 0 ]]; then
    log "${step}: no runner is defined for this validation-only step"
    show_validation "${step}"
    return 1
  fi

  log "${step}: command: $*"
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
    log "${step}: dry-run; skip execution"
    return 0
  fi

  "$@"
  log "${step}: validating generated outputs"
  if ! validate_step "${step}"; then
    show_validation "${step}"
    return 1
  fi
  log "${step}: complete"
}

join_gpus() {
  local sep=""
  local item
  for item in "$@"; do
    printf '%s%s' "${sep}" "${item}"
    sep=" "
  done
}

GPU_ARRAY=( ${GPU_IDS} )
GPU_COUNT="${#GPU_ARRAY[@]}"
if [[ "${GPU_COUNT}" -eq 0 ]]; then
  GPU_ARRAY=(0)
  GPU_COUNT=1
fi

first_gpu="${GPU_ARRAY[0]}"
last_gpu="${GPU_ARRAY[$((GPU_COUNT - 1))]}"
if [[ "${GPU_COUNT}" -ge 4 ]]; then
  default_internal_gpus="${GPU_ARRAY[0]}"
  default_prediction_gpus="$(join_gpus "${GPU_ARRAY[1]}" "${GPU_ARRAY[2]}")"
  default_hparam_gpus="${GPU_ARRAY[3]}"
elif [[ "${GPU_COUNT}" -eq 3 ]]; then
  default_internal_gpus="${GPU_ARRAY[0]}"
  default_prediction_gpus="${GPU_ARRAY[1]}"
  default_hparam_gpus="${GPU_ARRAY[2]}"
elif [[ "${GPU_COUNT}" -eq 2 ]]; then
  default_internal_gpus="${GPU_ARRAY[0]}"
  default_prediction_gpus="${GPU_ARRAY[0]}"
  default_hparam_gpus="${GPU_ARRAY[1]}"
else
  default_internal_gpus="${GPU_ARRAY[0]}"
  default_prediction_gpus="${GPU_ARRAY[0]}"
  default_hparam_gpus="${GPU_ARRAY[0]}"
fi

INTERNAL_GPU_IDS="${INTERNAL_GPU_IDS:-${default_internal_gpus}}"
PREDICTION_GPU_IDS="${PREDICTION_GPU_IDS:-${default_prediction_gpus}}"
HPARAM_GPU_IDS="${HPARAM_GPU_IDS:-${default_hparam_gpus}}"
GEOMETRY_GPU_ID="${GEOMETRY_GPU_ID:-${first_gpu}}"

run_main_table() {
  run_or_wait_step main_table ""
}

run_prototype_geometry() {
  run_or_wait_step \
    prototype_geometry \
    "analyze_lamp_merge_prototype_geometry.py" \
    env RUN_TAG="${RUN_TAG}_prototype_geometry" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_lamp_merge_prototype_geometry.py" \
        --device "cuda:${GEOMETRY_GPU_ID}" \
        --batch-size "${SMALL_BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}"
}

run_internal_ablation() {
  run_or_wait_step \
    internal_ablation \
    "run_lamp_merge_internal_ablation_full|lamp_merge_internal_ablation_full" \
    env RUN_TAG="${RUN_TAG}_internal_ablation" GPU_IDS="${INTERNAL_GPU_IDS}" NUM_WORKERS="${NUM_WORKERS}" SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE}" \
      "${SCRIPT_DIR}/run_lamp_merge_internal_ablation_full_parallel.sh"
}

run_prediction_diagnostics() {
  run_or_wait_step \
    prediction_diagnostics \
    "run_lamp_merge_prediction_diagnostics_full_parallel|collect_prediction_diagnostics.py.*prediction_diagnostics_full" \
    env RUN_TAG="${RUN_TAG}_prediction_diagnostics" GPU_IDS="${PREDICTION_GPU_IDS}" NUM_WORKERS="${NUM_WORKERS}" BATCH_SIZE="${BATCH_SIZE}" \
      "${SCRIPT_DIR}/run_lamp_merge_prediction_diagnostics_full_parallel.sh"
}

run_hparam_full() {
  run_or_wait_step \
    hparam_full \
    "run_lamp_merge_hparam_full_parallel|lamp_merge_hparam_full" \
    env RUN_TAG="${RUN_TAG}_hparam" GPU_IDS="${HPARAM_GPU_IDS}" NUM_WORKERS="${NUM_WORKERS}" SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE}" \
      "${SCRIPT_DIR}/run_lamp_merge_hparam_full_parallel.sh"
}

log "LAMP-Merge parallel full experiment queue"
log "root=${ROOT_DIR}"
log "steps=${STEPS[*]}"
log "gpu_ids=${GPU_IDS}"
log "internal_gpu_ids=${INTERNAL_GPU_IDS}"
log "prediction_gpu_ids=${PREDICTION_GPU_IDS}"
log "hparam_gpu_ids=${HPARAM_GPU_IDS}"
log "force=${FORCE} dry_run=${DRY_RUN}"

pids=()
for step in "${STEPS[@]}"; do
  case "${step}" in
    main_table) run_main_table ;;
    prototype_geometry) run_prototype_geometry & pids+=("$!") ;;
    internal_ablation) run_internal_ablation & pids+=("$!") ;;
    prediction_diagnostics) run_prediction_diagnostics & pids+=("$!") ;;
    hparam_full) run_hparam_full & pids+=("$!") ;;
    *)
      log "unknown step: ${step}"
      exit 2
      ;;
  esac
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  log "at least one requested full experiment failed"
  exit "${status}"
fi

log "all requested full experiments are complete"
