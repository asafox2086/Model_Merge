#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/archive_outputs.sh"

PYTHON_BIN="${PYTHON_BIN:-/data2/liyapeng_grp/.conda/envs/MM/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-${ROOT_DIR}/model_hub}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Med_data}"
PROTO_ROOT="${PROTO_ROOT:-${ROOT_DIR}/outputs/lamp_merge_client_local_proto_stats}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/prediction_diagnostics_full_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/prediction_diagnostics_full_${RUN_TAG}}"
METRICS_CSV="${METRICS_CSV:-${ROOT_DIR}/My_merge_ret/reports/prediction_diagnostics_full.csv}"
SUMMARY_DIR="${SUMMARY_DIR:-${ROOT_DIR}/My_merge_ret/reports/prediction_diagnostics_full}"
FIGURE_DIR="${FIGURE_DIR:-${ROOT_DIR}/My_merge_ret/figures/prediction_diagnostics_full}"
GPU_IDS=( ${GPU_IDS:-0 1 2 3} )
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TOTAL_CASES="${TOTAL_CASES:-180}"
INCLUDE_CLIENTS="${INCLUDE_CLIENTS:-true}"

DATASETS=( ${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224} )
SMALL_MODELS=( ${SMALL_MODELS:-resnet convnext vit_t swin_tiny} )
NUM_CLIENTS=( ${NUM_CLIENTS:-3 5 7} )
BETAS=( ${BETAS:-0 0.01 0.1} )
METHODS=( ${METHODS:-avg avg_head ties dare_linear dare_ties regmean fisher breadcrumbs model_stock from iso_c free_merge robustmerge lamp_merge} )
MODES=( ${MODES:-full m1_only avg_m2 prototype_head_agg global_feature_mean support_only prototype_shuffle uniform_client_weight binary_support global_client_size_weight no_prevalence uniform_prevalence smoothed_prevalence} )

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "$(dirname "${METRICS_CSV}")" "${SUMMARY_DIR}" "${FIGURE_DIR}"
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

client_flag="--include-clients"
if [[ "${INCLUDE_CLIENTS}" == "false" || "${INCLUDE_CLIENTS}" == "0" || "${INCLUDE_CLIENTS}" == "no" ]]; then
  client_flag="--no-include-clients"
fi

run_shard() {
  local shard_index="$1"
  local gpu_id="$2"
  local shard_count="${#GPU_IDS[@]}"
  local shard_size=$(( (TOTAL_CASES + shard_count - 1) / shard_count ))
  local skip=$(( shard_index * shard_size ))
  local limit="${shard_size}"
  local shard_csv="${LOG_DIR}/metrics_shard_${shard_index}.csv"
  local shard_summary="${LOG_DIR}/summary_shard_${shard_index}"
  local shard_figures="${LOG_DIR}/figures_shard_${shard_index}"
  local shard_output="${OUTPUT_ROOT}/shard_${shard_index}"
  local log_path="${LOG_DIR}/shard_${shard_index}.log"

  if [[ "${skip}" -ge "${TOTAL_CASES}" ]]; then
    echo "skip shard=${shard_index}; start offset ${skip} exceeds ${TOTAL_CASES}" | tee "${log_path}"
    return 0
  fi

  echo "[$(date '+%F %T')] start prediction diagnostics shard=${shard_index} gpu=${gpu_id} skip=${skip} limit=${limit}" | tee "${log_path}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/collect_prediction_diagnostics.py" \
    --model-hub-root "${MODEL_HUB_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${shard_output}" \
    --metrics-csv "${shard_csv}" \
    --summary-dir "${shard_summary}" \
    --figure-dir "${shard_figures}" \
    --task-type small \
    --datasets "${DATASETS[@]}" \
    --small-models "${SMALL_MODELS[@]}" \
    --num-clients "${NUM_CLIENTS[@]}" \
    --betas "${BETAS[@]}" \
    --methods "${METHODS[@]}" \
    --lamp-merge-ablation-modes "${MODES[@]}" \
    --merge-weight-mode equal \
    --split test \
    --device "cuda:${gpu_id}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --skip "${skip}" \
    --limit "${limit}" \
    --lamp-merge-prototype-root "${PROTO_ROOT}" \
    ${client_flag} \
    --resume \
    --delete-merged 2>&1 | tee -a "${log_path}"
  echo "[$(date '+%F %T')] done prediction diagnostics shard=${shard_index}" | tee -a "${log_path}"
}

echo "[$(date '+%F %T')] LAMP-Merge full prediction diagnostics"
echo "output_root=${OUTPUT_ROOT}"
echo "metrics_csv=${METRICS_CSV}"
echo "summary_dir=${SUMMARY_DIR}"
echo "figure_dir=${FIGURE_DIR}"
echo "gpu_ids=${GPU_IDS[*]}"
echo "methods=${METHODS[*]}"
echo "modes=${MODES[*]}"
echo "datasets=${DATASETS[*]}"
echo "small_models=${SMALL_MODELS[*]}"
echo "num_clients=${NUM_CLIENTS[*]}"
echo "betas=${BETAS[*]}"

pids=()
for shard_index in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$shard_index]}"
  run_shard "${shard_index}" "${gpu_id}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[$(date '+%F %T')] at least one prediction-diagnostics shard failed" >&2
  exit "${status}"
fi

tmp_csv="${METRICS_CSV}.tmp"
rm -f "${tmp_csv}"
first=1
for shard_csv in "${LOG_DIR}"/metrics_shard_*.csv; do
  [[ -s "${shard_csv}" ]] || continue
  if [[ "${first}" -eq 1 ]]; then
    cat "${shard_csv}" >> "${tmp_csv}"
    first=0
  else
    tail -n +2 "${shard_csv}" >> "${tmp_csv}"
  fi
done
archive_existing_outputs "${METRICS_CSV}" "${SUMMARY_DIR}" "${FIGURE_DIR}"
mv "${tmp_csv}" "${METRICS_CSV}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/collect_prediction_diagnostics.py" \
  --metrics-csv "${METRICS_CSV}" \
  --summary-dir "${SUMMARY_DIR}" \
  --figure-dir "${FIGURE_DIR}" \
  --summarize-only

echo "[$(date '+%F %T')] complete full prediction diagnostics"
