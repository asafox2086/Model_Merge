#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/archive_outputs.sh"

PYTHON_BIN="${PYTHON_BIN:-/data2/liyapeng_grp/.conda/envs/MM/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${ROOT_DIR}/outputs/lamp_merge_m2_internal_ablation_full_${RUN_TAG}}"
FORMAL_ROOT="${FORMAL_ROOT:-${ROOT_DIR}/outputs/lamp_merge_internal_ablation_full_20260718_formal_gamma055_s18p75_tau2p5_lambda4p25/full}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/lamp_merge_m2_internal_ablation_full_${RUN_TAG}}"
MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-${ROOT_DIR}/model_hub}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Med_data}"
PROTO_ROOT="${PROTO_ROOT:-${ROOT_DIR}/outputs/lamp_merge_client_local_proto_stats}"
GPU_IDS=( ${GPU_IDS:-0 1 2 3} )
NUM_WORKERS="${NUM_WORKERS:-3}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-128}"

DATASETS=( bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224 )
SHARD_A_DATASETS=( bloodmnist_224 organcmnist_224 )
SHARD_B_DATASETS=( dermamnist_224 organsmnist_224 chaoshengmnist_224 )
SMALL_MODELS=( resnet convnext vit_t swin_tiny )
NUM_CLIENTS=( 3 5 7 )
BETAS=( 0 0.01 0.1 )
MODES=( always_on_calibration client_balanced_prevalence )

if [[ "${#GPU_IDS[@]}" -lt 4 ]]; then
  echo "This runner requires four GPU IDs, got: ${GPU_IDS[*]}" >&2
  exit 2
fi

mkdir -p "${BASE_OUTPUT_ROOT}" "${LOG_DIR}"
if [[ ! -d "${PROTO_ROOT}" ]]; then
  echo "Missing LAMP-Merge client statistics root: ${PROTO_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${FORMAL_ROOT}" ]]; then
  echo "Missing formal LAMP-Merge output root: ${FORMAL_ROOT}" >&2
  exit 2
fi
if [[ ! -e "${BASE_OUTPUT_ROOT}/full" ]]; then
  ln -s "${FORMAL_ROOT}" "${BASE_OUTPUT_ROOT}/full"
fi

export TOKENIZERS_PARALLELISM="false"
export HF_LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"
mkdir -p "${TMPDIR}"

run_shard() {
  local mode="$1"
  local gpu_id="$2"
  local shard_name="$3"
  shift 3
  local shard_datasets=( "$@" )
  local mode_root="${BASE_OUTPUT_ROOT}/shards/${mode}_${shard_name}"
  local log_path="${LOG_DIR}/${mode}_${shard_name}.log"
  echo "[$(date '+%F %T')] start mode=${mode} shard=${shard_name} gpu=${gpu_id} datasets=${shard_datasets[*]} output=${mode_root}" | tee "${log_path}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_all_avg_eval.py" \
    --model-hub-root "${MODEL_HUB_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${mode_root}" \
    --device "cuda:${gpu_id}" \
    --small-batch-size "${SMALL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --task-type small \
    --datasets "${shard_datasets[@]}" \
    --small-models "${SMALL_MODELS[@]}" \
    --num-clients "${NUM_CLIENTS[@]}" \
    --betas "${BETAS[@]}" \
    --method lamp_merge \
    --merge-weight-mode equal \
    --lamp-merge-prototype-root "${PROTO_ROOT}" \
    --lamp-merge-ablation-mode "${mode}" \
    --resume \
    --delete-merged 2>&1 | tee -a "${log_path}"
  echo "[$(date '+%F %T')] done mode=${mode} shard=${shard_name} gpu=${gpu_id}" | tee -a "${log_path}"
}

run_shard always_on_calibration "${GPU_IDS[0]}" a "${SHARD_A_DATASETS[@]}" &
pids=( "$!" )
run_shard always_on_calibration "${GPU_IDS[1]}" b "${SHARD_B_DATASETS[@]}" &
pids+=( "$!" )
run_shard client_balanced_prevalence "${GPU_IDS[2]}" a "${SHARD_A_DATASETS[@]}" &
pids+=( "$!" )
run_shard client_balanced_prevalence "${GPU_IDS[3]}" b "${SHARD_B_DATASETS[@]}" &
pids+=( "$!" )

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "[$(date '+%F %T')] at least one M2 internal ablation failed" >&2
  exit "${status}"
fi

for mode in "${MODES[@]}"; do
  mode_root="${BASE_OUTPUT_ROOT}/${mode}"
  mkdir -p "${mode_root}/eval"
  cp -a "${BASE_OUTPUT_ROOT}/shards/${mode}_a/eval/." "${mode_root}/eval/"
  cp -a "${BASE_OUTPUT_ROOT}/shards/${mode}_b/eval/." "${mode_root}/eval/"
done

report_prefix="${ROOT_DIR}/My_merge_ret/reports/lamp_merge_m2_internal_ablation_full"
archive_existing_outputs \
  "${report_prefix}.csv" \
  "${report_prefix}_raw.csv" \
  "${report_prefix}_client_average.csv" \
  "${report_prefix}_by_dataset.csv" \
  "${report_prefix}_summary.md"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_lamp_merge_internal_ablation_full.py" \
  --root "${BASE_OUTPUT_ROOT}" \
  --baseline-table "${ROOT_DIR}/My_merge_ret/汇总表.md" \
  --output-csv "${report_prefix}.csv" \
  --raw-csv "${report_prefix}_raw.csv" \
  --client-average-csv "${report_prefix}_client_average.csv" \
  --dataset-csv "${report_prefix}_by_dataset.csv" \
  --summary-md "${report_prefix}_summary.md" \
  --require-modes "${MODES[@]}"

echo "[$(date '+%F %T')] complete output=${BASE_OUTPUT_ROOT}"
