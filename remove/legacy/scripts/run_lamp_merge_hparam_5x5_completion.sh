#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data2/liyapeng_grp/.conda/envs/MM/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/lamp_merge_hparam_5x5_20260719_full}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/lamp_merge_hparam_5x5_20260719_full}"
MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-${ROOT_DIR}/model_hub}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Med_data}"
PROTO_ROOT="${PROTO_ROOT:-${ROOT_DIR}/outputs/lamp_merge_client_local_proto_stats}"
GPU_IDS=( ${GPU_IDS:-0 1 2 3} )

DATASETS=(bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224)
SMALL_MODELS=(resnet convnext vit_t swin_tiny)
NUM_CLIENTS=(3 5 7)
BETAS=(0 0.01 0.1)
DPR_GAMMAS=(0.45 0.55 0.60 0.65)
DPR_SCALES=(13.75 16.25 18.75 21.25 23.75)
LPC_TAUS=(1.5 3.5)
LPC_LAMBDAS=(3.25 3.75 4.25 4.75 5.25)

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
if [[ ! -d "${PROTO_ROOT}" ]]; then
  echo "Missing LAMP-Merge prototype statistics: ${PROTO_ROOT}" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM=false
export HF_LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

safe_value() {
  printf '%s' "$1" | sed 's/\./p/g'
}

RUN_ITEMS=()
for scale in "${DPR_SCALES[@]}"; do
  for gamma in "${DPR_GAMMAS[@]}"; do
    RUN_ITEMS+=("dpr:${gamma}:${scale}")
  done
done
for tau in "${LPC_TAUS[@]}"; do
  for lambda in "${LPC_LAMBDAS[@]}"; do
    RUN_ITEMS+=("lpc:${tau}:${lambda}")
  done
done

printf 'family,first_parameter,second_parameter\n' > "${OUTPUT_ROOT}/grid_manifest.csv"
for item in "${RUN_ITEMS[@]}"; do
  IFS=':' read -r family first second <<< "${item}"
  printf '%s,%s,%s\n' "${family}" "${first}" "${second}" >> "${OUTPUT_ROOT}/grid_manifest.csv"
done

run_item() {
  local item="$1"
  local gpu_id="$2"
  local family first second output_dir log_path
  IFS=':' read -r family first second <<< "${item}"
  if [[ "${family}" == "dpr" ]]; then
    output_dir="${OUTPUT_ROOT}/dpr_gamma_$(safe_value "${first}")_s_$(safe_value "${second}")"
  else
    output_dir="${OUTPUT_ROOT}/lpc_tau_$(safe_value "${first}")_lambda_$(safe_value "${second}")"
  fi
  log_path="${LOG_DIR}/$(basename "${output_dir}").log"

  echo "[$(date '+%F %T')] start ${item} on cuda:${gpu_id}" | tee -a "${log_path}"
  if [[ "${family}" == "dpr" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_all_avg_eval.py" \
      --lamp-merge-proto-count-power "${first}" \
      --lamp-merge-reference-head-scale "${second}" \
      --lamp-merge-reference-prior-threshold 2.5 \
      --lamp-merge-reference-prior-max-tau 4.25 \
      --output-root "${output_dir}" --device "cuda:${gpu_id}" \
      --model-hub-root "${MODEL_HUB_ROOT}" --data-root "${DATA_ROOT}" \
      --lamp-merge-prototype-root "${PROTO_ROOT}" --lamp-merge-ablation-mode full \
      --method lamp_merge --merge-weight-mode equal --task-type small \
      --datasets "${DATASETS[@]}" --small-models "${SMALL_MODELS[@]}" \
      --num-clients "${NUM_CLIENTS[@]}" --betas "${BETAS[@]}" \
      --small-batch-size 128 --num-workers 3 --resume --delete-merged 2>&1 | tee -a "${log_path}"
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_all_avg_eval.py" \
      --lamp-merge-proto-count-power 0.55 \
      --lamp-merge-reference-head-scale 18.75 \
      --lamp-merge-reference-prior-threshold "${first}" \
      --lamp-merge-reference-prior-max-tau "${second}" \
      --output-root "${output_dir}" --device "cuda:${gpu_id}" \
      --model-hub-root "${MODEL_HUB_ROOT}" --data-root "${DATA_ROOT}" \
      --lamp-merge-prototype-root "${PROTO_ROOT}" --lamp-merge-ablation-mode full \
      --method lamp_merge --merge-weight-mode equal --task-type small \
      --datasets "${DATASETS[@]}" --small-models "${SMALL_MODELS[@]}" \
      --num-clients "${NUM_CLIENTS[@]}" --betas "${BETAS[@]}" \
      --small-batch-size 128 --num-workers 3 --resume --delete-merged 2>&1 | tee -a "${log_path}"
  fi
  echo "[$(date '+%F %T')] done ${item}" | tee -a "${log_path}"
}

run_worker() {
  local worker_index="$1"
  local gpu_id="$2"
  local item_index
  for item_index in "${!RUN_ITEMS[@]}"; do
    if [[ $((item_index % ${#GPU_IDS[@]})) -eq "${worker_index}" ]]; then
      run_item "${RUN_ITEMS[$item_index]}" "${gpu_id}"
    fi
  done
}

echo "[$(date '+%F %T')] starting missing full-configuration 5x5 cells"
echo "output_root=${OUTPUT_ROOT}"
echo "DPR: gamma={0.45,0.50,0.55,0.60,0.65}, s={13.75,16.25,18.75,21.25,23.75}"
echo "LPC: tau={1.5,2.0,2.5,3.0,3.5}, lambda={3.25,3.75,4.25,4.75,5.25}"

pids=()
for worker_index in "${!GPU_IDS[@]}"; do
  run_worker "${worker_index}" "${GPU_IDS[$worker_index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
