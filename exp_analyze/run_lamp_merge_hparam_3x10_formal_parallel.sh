#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data2/liyapeng_grp/.conda/envs/MM/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${ROOT_DIR}/outputs/lamp_merge_hparam_3x10_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/lamp_merge_hparam_3x10_${RUN_TAG}}"
MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-${ROOT_DIR}/model_hub}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Med_data}"
PROTO_ROOT="${PROTO_ROOT:-${ROOT_DIR}/outputs/lamp_merge_client_local_proto_stats}"
GPU_IDS=( ${GPU_IDS:-0 1 2 3} )
NUM_WORKERS="${NUM_WORKERS:-3}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-128}"

DATASETS=( ${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224} )
SMALL_MODELS=( ${SMALL_MODELS:-resnet convnext vit_t swin_tiny} )
NUM_CLIENTS=( ${NUM_CLIENTS:-3 5 7} )
BETAS=( ${BETAS:-0 0.01 0.1} )

GAMMA_VALUES=( ${GAMMA_VALUES:-0.50 0.55 0.60} )
S_VALUES=( ${S_VALUES:-13.75 15 16.25 17.5 18.75 20 21.25 22.5 23.75 25} )
TAU_VALUES=( ${TAU_VALUES:-2.0 2.5 3.0} )
LAMBDA_VALUES=( ${LAMBDA_VALUES:-3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0 5.25} )

mkdir -p "${BASE_OUTPUT_ROOT}" "${LOG_DIR}"

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
  printf '%s' "$1" | sed 's/\./p/g'
}

RUN_ITEMS=()
for gamma in "${GAMMA_VALUES[@]}"; do
  for scale in "${S_VALUES[@]}"; do
    RUN_ITEMS+=("dpr:${gamma}:${scale}")
  done
done
for threshold in "${TAU_VALUES[@]}"; do
  for strength in "${LAMBDA_VALUES[@]}"; do
    RUN_ITEMS+=("lpc:${threshold}:${strength}")
  done
done

run_item() {
  local item="$1"
  local gpu_id="$2"
  local family first_value second_value first_safe second_safe mode_root log_path ablation_mode
  local extra_args=()

  IFS=':' read -r family first_value second_value <<< "${item}"
  first_safe="$(safe_value "${first_value}")"
  second_safe="$(safe_value "${second_value}")"

  if [[ "${family}" == "dpr" ]]; then
    mode_root="${BASE_OUTPUT_ROOT}/dpr_gamma_${first_safe}_s_${second_safe}"
    log_path="${LOG_DIR}/dpr_gamma_${first_safe}_s_${second_safe}.log"
    ablation_mode="m1_only"
    extra_args=(
      --lamp-merge-proto-count-power "${first_value}"
      --lamp-merge-reference-head-scale "${second_value}"
      --lamp-merge-reference-prior-threshold 2.5
      --lamp-merge-reference-prior-max-tau 4.25
    )
  else
    mode_root="${BASE_OUTPUT_ROOT}/lpc_tau_${first_safe}_lambda_${second_safe}"
    log_path="${LOG_DIR}/lpc_tau_${first_safe}_lambda_${second_safe}.log"
    ablation_mode="full"
    extra_args=(
      --lamp-merge-proto-count-power 0.55
      --lamp-merge-reference-head-scale 18.75
      --lamp-merge-reference-prior-threshold "${first_value}"
      --lamp-merge-reference-prior-max-tau "${second_value}"
    )
  fi

  echo "[$(date '+%F %T')] start ${item} gpu=${gpu_id}" | tee "${log_path}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_all_avg_eval.py" \
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
    --lamp-merge-ablation-mode "${ablation_mode}" \
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

pids=()
for worker_index in "${!GPU_IDS[@]}"; do
  run_worker "${worker_index}" "${GPU_IDS[$worker_index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

if [[ "${status}" -ne 0 ]]; then
  echo "[$(date '+%F %T')] formal 3x10 scan failed" >&2
  exit "${status}"
fi

touch "${BASE_OUTPUT_ROOT}/DONE"
echo "[$(date '+%F %T')] formal 3x10 scan complete"
