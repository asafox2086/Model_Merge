#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/archive_outputs.sh"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data2/liyapeng_grp/.conda/envs/MM/bin/python" ]]; then
    PYTHON_BIN="/data2/liyapeng_grp/.conda/envs/MM/bin/python"
  elif command -v conda >/dev/null 2>&1; then
    PYTHON_BIN="conda run -n MM python"
  else
    PYTHON_BIN="python3"
  fi
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${ROOT_DIR}/outputs/lamp_merge_internal_ablation_full_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/lamp_merge_internal_ablation_full_${RUN_TAG}}"
MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-${ROOT_DIR}/model_hub}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Med_data}"
PROTO_ROOT="${PROTO_ROOT:-${ROOT_DIR}/outputs/lamp_merge_client_local_proto_stats}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-128}"

DATASETS=( ${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224} )
SMALL_MODELS=( ${SMALL_MODELS:-resnet convnext vit_t swin_tiny} )
NUM_CLIENTS=( ${NUM_CLIENTS:-3 5 7} )
BETAS=( ${BETAS:-0 0.01 0.1} )
MODES=( ${MODES:-full m1_only prototype_head_agg global_feature_mean support_only prototype_shuffle uniform_client_weight binary_support global_client_size_weight no_prevalence uniform_prevalence smoothed_prevalence avg_m2} )

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

echo "[$(date '+%F %T')] LAMP-Merge internal ablation full run"
echo "root=${BASE_OUTPUT_ROOT}"
echo "log_dir=${LOG_DIR}"
echo "device=${DEVICE}"
echo "datasets=${DATASETS[*]}"
echo "small_models=${SMALL_MODELS[*]}"
echo "num_clients=${NUM_CLIENTS[*]}"
echo "betas=${BETAS[*]}"
echo "modes=${MODES[*]}"

for mode in "${MODES[@]}"; do
  mode_root="${BASE_OUTPUT_ROOT}/${mode}"
  log_path="${LOG_DIR}/${mode}.log"
  echo "[$(date '+%F %T')] start mode=${mode} output=${mode_root}"
  # shellcheck disable=SC2086
  ${PYTHON_BIN} "${SCRIPT_DIR}/run_all_avg_eval.py" \
    --model-hub-root "${MODEL_HUB_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${mode_root}" \
    --device "${DEVICE}" \
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
    --lamp-merge-ablation-mode "${mode}" \
    --resume \
    --delete-merged 2>&1 | tee "${log_path}"
  echo "[$(date '+%F %T')] done mode=${mode}"
done

echo "[$(date '+%F %T')] summarizing ${BASE_OUTPUT_ROOT}"
archive_existing_outputs \
  "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full.csv" \
  "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full_summary.md"
# shellcheck disable=SC2086
${PYTHON_BIN} "${SCRIPT_DIR}/summarize_lamp_merge_internal_ablation_full.py" \
  --root "${BASE_OUTPUT_ROOT}" \
  --baseline-table "${ROOT_DIR}/My_merge_ret/汇总表.md" \
  --output-csv "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full.csv" \
  --summary-md "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full_summary.md"

echo "[$(date '+%F %T')] complete"
