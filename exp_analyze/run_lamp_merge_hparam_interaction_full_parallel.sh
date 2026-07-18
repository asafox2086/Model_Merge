#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data2/liyapeng_grp/.conda/envs/MM/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${ROOT_DIR}/outputs/lamp_merge_hparam_interaction_full_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/lamp_merge_hparam_interaction_full_${RUN_TAG}}"
MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-${ROOT_DIR}/model_hub}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Med_data}"
PROTO_ROOT="${PROTO_ROOT:-${ROOT_DIR}/outputs/lamp_merge_client_local_proto_stats}"
GPU_IDS=( ${GPU_IDS:-0 1 2 3} )
NUM_WORKERS="${NUM_WORKERS:-0}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-64}"

DATASETS=( ${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224} )
SMALL_MODELS=( ${SMALL_MODELS:-resnet convnext vit_t swin_tiny} )
NUM_CLIENTS=( ${NUM_CLIENTS:-3 5 7} )
BETAS=( ${BETAS:-0 0.01 0.1} )

GAMMA_VALUES=( ${GAMMA_VALUES:-0.35 0.40 0.45 0.50 0.55} )
S_VALUES=( ${S_VALUES:-15 17.5 20 22.5 25} )
TAU_VALUES=( ${TAU_VALUES:-1.5 2.0 2.5 3.0 3.5} )
LAMBDA_VALUES=( ${LAMBDA_VALUES:-4.0 4.5 5.0 5.5 6.0} )
RUN_M1="${RUN_M1:-1}"
RUN_M2="${RUN_M2:-1}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"

mkdir -p "${BASE_OUTPUT_ROOT}" "${LOG_DIR}"
if [[ ! -d "${PROTO_ROOT}" ]]; then
  echo "Missing LAMP-Merge client statistics root: ${PROTO_ROOT}" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"
mkdir -p "${TMPDIR}"

safe_value() {
  printf "%s" "$1" | sed 's/\./p/g'
}

RUN_ITEMS=()
if [[ "${RUN_M1}" == "1" ]]; then
  for gamma in "${GAMMA_VALUES[@]}"; do
    for scale in "${S_VALUES[@]}"; do
      RUN_ITEMS+=("m1:${gamma}:${scale}")
    done
  done
fi
if [[ "${RUN_M2}" == "1" ]]; then
  for threshold in "${TAU_VALUES[@]}"; do
    for strength in "${LAMBDA_VALUES[@]}"; do
      RUN_ITEMS+=("m2:${threshold}:${strength}")
    done
  done
fi

if [[ ${#RUN_ITEMS[@]} -eq 0 ]]; then
  echo "No hyperparameter configurations selected." >&2
  exit 2
fi

run_item() {
  local item="$1"
  local gpu_id="$2"
  local family first_value second_value first_safe second_safe mode_root log_path
  IFS=':' read -r family first_value second_value <<< "${item}"
  first_safe="$(safe_value "${first_value}")"
  second_safe="$(safe_value "${second_value}")"

  local extra_args=()
  if [[ "${family}" == "m1" ]]; then
    mode_root="${BASE_OUTPUT_ROOT}/m1_gamma_${first_safe}_s_${second_safe}"
    log_path="${LOG_DIR}/m1_gamma_${first_safe}_s_${second_safe}.log"
    extra_args=(
      --lamp-merge-proto-count-power "${first_value}"
      --lamp-merge-reference-head-scale "${second_value}"
      --lamp-merge-reference-prior-threshold 2.5
      --lamp-merge-reference-prior-max-tau 5.0
    )
  else
    mode_root="${BASE_OUTPUT_ROOT}/m2_tau_${first_safe}_lambda_${second_safe}"
    log_path="${LOG_DIR}/m2_tau_${first_safe}_lambda_${second_safe}.log"
    extra_args=(
      --lamp-merge-proto-count-power 0.45
      --lamp-merge-reference-head-scale 20.0
      --lamp-merge-reference-prior-threshold "${first_value}"
      --lamp-merge-reference-prior-max-tau "${second_value}"
    )
  fi

  echo "[$(date '+%F %T')] start ${item} gpu=${gpu_id}" | tee "${log_path}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_all_avg_eval.py" \
    --model-hub-root "${MODEL_HUB_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${mode_root}" \
    --device "cuda:${gpu_id}" \
    --small-batch-size "${SMALL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --task-type small \
    --datasets "${DATASETS[@]}" \
    --small-models "${SMALL_MODELS[@]}" \
    --num-clients "${NUM_CLIENTS[@]}" \
    --betas "${BETAS[@]}" \
    --method lamp_merge \
    --merge-weight-mode equal \
    --lamp-merge-prototype-root "${PROTO_ROOT}" \
    --lamp-merge-ablation-mode full \
    "${extra_args[@]}" \
    --resume \
    --delete-merged 2>&1 | tee -a "${log_path}"
  echo "[$(date '+%F %T')] done ${item}" | tee -a "${log_path}"
}

run_worker() {
  local worker_index="$1"
  local gpu_id="$2"
  local item_index
  for item_index in "${!RUN_ITEMS[@]}"; do
    if [[ $((item_index % ${#GPU_IDS[@]})) -ne "${worker_index}" ]]; then
      continue
    fi
    run_item "${RUN_ITEMS[$item_index]}" "${gpu_id}"
  done
}

echo "[$(date '+%F %T')] LAMP-Merge full interaction sensitivity scan"
echo "root=${BASE_OUTPUT_ROOT}"
echo "log_dir=${LOG_DIR}"
echo "gpu_ids=${GPU_IDS[*]}"
echo "gamma_values=${GAMMA_VALUES[*]}"
echo "s_values=${S_VALUES[*]}"
echo "tau_values=${TAU_VALUES[*]}"
echo "lambda_values=${LAMBDA_VALUES[*]}"
echo "run_m1=${RUN_M1}"
echo "run_m2=${RUN_M2}"
echo "total_grid_points=${#RUN_ITEMS[@]}"

pids=()
for worker_index in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$worker_index]}"
  run_worker "${worker_index}" "${gpu_id}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[$(date '+%F %T')] at least one interaction scan failed; summary skipped" >&2
  exit "${status}"
fi

if [[ "${SKIP_SUMMARY}" == "1" ]]; then
  echo "[$(date '+%F %T')] complete full interaction sensitivity scan (summary skipped)"
  exit 0
fi

SUMMARY_ROOTS="${SUMMARY_ROOTS:-${BASE_OUTPUT_ROOT}}"
summary_args=()
for summary_root in ${SUMMARY_ROOTS}; do
  summary_args+=(--root "${summary_root}")
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_lamp_merge_hparam_interaction_full.py" \
  "${summary_args[@]}" \
  --expected-points "${EXPECTED_POINTS:-${#RUN_ITEMS[@]}}" \
  --output-csv "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_hparam_interaction_full.csv" \
  --client-average-csv "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_hparam_interaction_full_client_average.csv" \
  --summary-md "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_hparam_interaction_full_summary.md" \
  --figure-path "${ROOT_DIR}/My_merge_ret/figures/lamp_merge_hparam_interaction_full_sensitivity.png"

echo "[$(date '+%F %T')] complete full interaction sensitivity scan"
