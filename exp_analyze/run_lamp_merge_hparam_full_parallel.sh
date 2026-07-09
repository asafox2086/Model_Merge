#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/archive_outputs.sh"

PYTHON_BIN="${PYTHON_BIN:-/data2/liyapeng_grp/.conda/envs/MM/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${ROOT_DIR}/outputs/lamp_merge_hparam_full_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/lamp_merge_hparam_full_${RUN_TAG}}"
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
S_VALUES=( ${S_VALUES:-5 7 10 12 15 17 20 22 25 27 30 32 35 37 40} )
LAMBDA_VALUES=( ${LAMBDA_VALUES:-2 3 4 5 6 7 8 10} )

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

RUN_ITEMS=()
for value in "${S_VALUES[@]}"; do
  RUN_ITEMS+=("scale:${value}")
done
for value in "${LAMBDA_VALUES[@]}"; do
  RUN_ITEMS+=("lambda:${value}")
done

safe_value() {
  printf "%s" "$1" | sed 's/\./p/g'
}

run_item() {
  local item="$1"
  local gpu_id="$2"
  local family="${item%%:*}"
  local value="${item#*:}"
  local value_safe
  value_safe="$(safe_value "${value}")"
  local mode_root="${BASE_OUTPUT_ROOT}/${family}_${value_safe}"
  local log_path="${LOG_DIR}/${family}_${value_safe}.log"
  local extra_args=()
  if [[ "${family}" == "scale" ]]; then
    extra_args=(--lamp-merge-reference-head-scale "${value}" --lamp-merge-reference-prior-max-tau 5)
  else
    extra_args=(--lamp-merge-reference-head-scale 20 --lamp-merge-reference-prior-max-tau "${value}")
  fi

  echo "[$(date '+%F %T')] start hparam ${family}=${value} gpu=${gpu_id} output=${mode_root}" | tee "${log_path}"
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
  echo "[$(date '+%F %T')] done hparam ${family}=${value}" | tee -a "${log_path}"
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

echo "[$(date '+%F %T')] LAMP-Merge full hyperparameter scan"
echo "root=${BASE_OUTPUT_ROOT}"
echo "log_dir=${LOG_DIR}"
echo "gpu_ids=${GPU_IDS[*]}"
echo "s_values=${S_VALUES[*]}"
echo "lambda_values=${LAMBDA_VALUES[*]}"
echo "datasets=${DATASETS[*]}"
echo "small_models=${SMALL_MODELS[*]}"
echo "num_clients=${NUM_CLIENTS[*]}"
echo "betas=${BETAS[*]}"

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
  echo "[$(date '+%F %T')] at least one hparam scan failed; summary skipped" >&2
  exit "${status}"
fi

archive_existing_outputs \
  "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_hparam_full.csv" \
  "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_hparam_full_client_average.csv" \
  "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_hparam_full_by_dataset.csv" \
  "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_hparam_full_summary.md" \
  "${ROOT_DIR}/My_merge_ret/figures/lamp_merge_hparam_full_sensitivity.png" \
  "${ROOT_DIR}/My_merge_ret/figures/lamp_merge_hparam_full_sensitivity.pdf"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_lamp_merge_hparam_full.py" \
  --root "${BASE_OUTPUT_ROOT}" \
  --output-csv "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_hparam_full.csv" \
  --client-average-csv "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_hparam_full_client_average.csv" \
  --dataset-csv "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_hparam_full_by_dataset.csv" \
  --summary-md "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_hparam_full_summary.md" \
  --figure-path "${ROOT_DIR}/My_merge_ret/figures/lamp_merge_hparam_full_sensitivity.png"

echo "[$(date '+%F %T')] complete full hyperparameter scan"
