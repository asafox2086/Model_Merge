#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data2/liyapeng_grp/.conda/envs/MM/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${ROOT_DIR}/outputs/lamp_merge_baseline_2x2_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/lamp_merge_baseline_2x2_${RUN_TAG}}"
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
MODES=( ${MODES:-reference_ties_head reference_ties_head_lpc reference_dare_head reference_dare_head_lpc} )

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

run_mode() {
  local mode="$1"
  local gpu_id="$2"
  local mode_root="${BASE_OUTPUT_ROOT}/${mode}"
  local log_path="${LOG_DIR}/${mode}.log"

  echo "[$(date '+%F %T')] start mode=${mode} gpu=${gpu_id}" | tee "${log_path}"
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
    --density 0.5 \
    --dare-seed 42 \
    --lamp-merge-prototype-root "${PROTO_ROOT}" \
    --lamp-merge-proto-count-power 0.55 \
    --lamp-merge-reference-head-scale 18.75 \
    --lamp-merge-reference-prior-threshold 2.5 \
    --lamp-merge-reference-prior-max-tau 4.25 \
    --lamp-merge-ablation-mode "${mode}" \
    --resume \
    --delete-merged 2>&1 | tee -a "${log_path}"
  echo "[$(date '+%F %T')] done mode=${mode}" | tee -a "${log_path}"
}

pids=()
for mode_index in "${!MODES[@]}"; do
  gpu_id="${GPU_IDS[$((mode_index % ${#GPU_IDS[@]}))]}"
  run_mode "${MODES[$mode_index]}" "${gpu_id}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

if [[ "${status}" -ne 0 ]]; then
  echo "[$(date '+%F %T')] baseline 2x2 run failed" >&2
  exit "${status}"
fi

touch "${BASE_OUTPUT_ROOT}/DONE"
echo "[$(date '+%F %T')] baseline 2x2 run complete"
