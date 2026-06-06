#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.gpuenv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.gpuenv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-my_merge_two_module_validated_${STAMP}}"
GPU_IDS="${GPU_IDS:-0}"
DATASETS="${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224}"
SMALL_MODELS="${SMALL_MODELS:-resnet convnext vit_t swin_tiny}"
CLIP_MODELS="${CLIP_MODELS:-openai/clip-vit-base-patch32}"
TASK_TYPES="${TASK_TYPES:-small vlm}"

REPRO_METHODS="${REPRO_METHODS:-avg ties}"
REPRO_TASK_TYPES="${REPRO_TASK_TYPES:-${TASK_TYPES}}"
REPRO_TOLERANCE="${REPRO_TOLERANCE:-5e-5}"
REPRO_SAMPLE_LIMIT="${REPRO_SAMPLE_LIMIT:-2}"
REPRO_EXTRA_ARGS="${REPRO_EXTRA_ARGS:---limit ${REPRO_SAMPLE_LIMIT}}"
RUN_REPRO="${RUN_REPRO:-true}"

ABLATIONS="${ABLATIONS:-full no_client_information no_fusion_selection no_adaptive_candidates}"
MY_MERGE_STATS_MAX_BATCHES="${MY_MERGE_STATS_MAX_BATCHES:-16}"
MY_MERGE_EVAL_MAX_BATCHES="${MY_MERGE_EVAL_MAX_BATCHES:-1}"
MY_MERGE_BN_BATCHES="${MY_MERGE_BN_BATCHES:-4}"
MY_MERGE_EXPORT_DIAGNOSTICS="${MY_MERGE_EXPORT_DIAGNOSTICS:-true}"
MY_MERGE_VIZ_MAX_BATCHES="${MY_MERGE_VIZ_MAX_BATCHES:-1}"
MY_MERGE_VIZ_MAX_PLOTS="${MY_MERGE_VIZ_MAX_PLOTS:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STATS_SPLIT="${STATS_SPLIT:-val}"
STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-32}"
STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-0}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-64}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-32}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"
DELETE_FLAG="${DELETE_FLAG:---delete-merged}"
RESUME_FLAG="${RESUME_FLAG:---resume}"
REPRO_MODE="${REPRO_MODE:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/${RUN_TAG}}"
REPRO_OUTPUT_ROOT="${REPRO_OUTPUT_ROOT:-${OUTPUT_ROOT}/reproduction}"
MY_MERGE_OUTPUT_ROOT="${MY_MERGE_OUTPUT_ROOT:-${OUTPUT_ROOT}/my_merge_ablation_grid}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_TAG}}"
PUBLISH_RESULTS="${PUBLISH_RESULTS:-true}"
PUBLISH_ROOT="${PUBLISH_ROOT:-${ROOT_DIR}/My_merge_ret}"

mkdir -p "${LOG_ROOT}" "${REPRO_OUTPUT_ROOT}/reports" "${MY_MERGE_OUTPUT_ROOT}/reports"

read -r -a REPRO_METHOD_ARRAY <<< "${REPRO_METHODS}"
read -r -a ABLATION_ARRAY <<< "${ABLATIONS}"

{
  echo "run_tag=${RUN_TAG}"
  echo "gpu_ids=${GPU_IDS}"
  echo "datasets=${DATASETS}"
  echo "small_models=${SMALL_MODELS}"
  echo "clip_models=${CLIP_MODELS}"
  echo "task_types=${TASK_TYPES}"
  echo "repro_methods=${REPRO_METHODS}"
  echo "repro_task_types=${REPRO_TASK_TYPES}"
  echo "repro_tolerance=${REPRO_TOLERANCE}"
  echo "ablations=${ABLATIONS}"
  echo "my_merge_stats_max_batches=${MY_MERGE_STATS_MAX_BATCHES}"
  echo "my_merge_eval_max_batches=${MY_MERGE_EVAL_MAX_BATCHES}"
  echo "my_merge_bn_batches=${MY_MERGE_BN_BATCHES}"
  echo "my_merge_export_diagnostics=${MY_MERGE_EXPORT_DIAGNOSTICS}"
  echo "my_merge_diagnostics_plot=false"
  echo "my_merge_viz_max_batches=${MY_MERGE_VIZ_MAX_BATCHES}"
  echo "my_merge_viz_max_plots=${MY_MERGE_VIZ_MAX_PLOTS}"
  echo "num_workers=${NUM_WORKERS}"
  echo "stats_split=${STATS_SPLIT}"
  echo "stats_batch_size=${STATS_BATCH_SIZE}"
  echo "stats_num_workers=${STATS_NUM_WORKERS}"
  echo "small_batch_size=${SMALL_BATCH_SIZE}"
  echo "vlm_batch_size=${VLM_BATCH_SIZE}"
  echo "max_parallel_jobs=${MAX_PARALLEL_JOBS}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "repro_output_root=${REPRO_OUTPUT_ROOT}"
  echo "my_merge_output_root=${MY_MERGE_OUTPUT_ROOT}"
  echo "log_root=${LOG_ROOT}"
} | tee "${OUTPUT_ROOT}/run_config.txt" > "${MY_MERGE_OUTPUT_ROOT}/reports/validated_run_config.txt"

if [[ "${RUN_REPRO}" == "true" ]]; then
  echo "[$(date '+%F %T')] reproduction start"
  env \
    RUN_TAG="${RUN_TAG}_repro" \
    OUTPUT_ROOT="${REPRO_OUTPUT_ROOT}" \
    LOG_ROOT="${LOG_ROOT}/reproduction" \
    GPU_IDS="${GPU_IDS}" \
    MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS}" \
    DATASETS="${DATASETS}" \
    SMALL_MODELS="${SMALL_MODELS}" \
    CLIP_MODELS="${CLIP_MODELS}" \
    FORMAL_METHODS="${REPRO_METHODS}" \
    FORMAL_TASK_TYPES="${REPRO_TASK_TYPES}" \
    FORMAL_EXTRA_ARGS="${REPRO_EXTRA_ARGS}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    STATS_SPLIT="${STATS_SPLIT}" \
    STATS_BATCH_SIZE="${STATS_BATCH_SIZE}" \
    STATS_NUM_WORKERS="${STATS_NUM_WORKERS}" \
    SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE}" \
    VLM_BATCH_SIZE="${VLM_BATCH_SIZE}" \
    DELETE_FLAG="${DELETE_FLAG}" \
    RESUME_FLAG="${RESUME_FLAG}" \
    REPRO_MODE="${REPRO_MODE}" \
    bash run_compare_multi_gpu.sh

  "${PYTHON_BIN}" scripts/compare_eval_to_baseline.py \
    --base result/all_results.md \
    --output-root "${REPRO_OUTPUT_ROOT}" \
    --methods "${REPRO_METHOD_ARRAY[@]}" \
    --tolerance "${REPRO_TOLERANCE}" \
    --dest-md "${REPRO_OUTPUT_ROOT}/reports/reproduction_check.md" \
    --dest-csv "${REPRO_OUTPUT_ROOT}/reports/reproduction_check.csv"
  echo "[$(date '+%F %T')] reproduction check passed"
else
  echo "[$(date '+%F %T')] reproduction skipped"
fi

for ablation in "${ABLATION_ARRAY[@]}"; do
  echo "[$(date '+%F %T')] my_merge ablation start | ${ablation}"
  env \
    ABLATION="${ablation}" \
    OUTPUT_ROOT="${MY_MERGE_OUTPUT_ROOT}/${ablation}" \
    LOG_ROOT="${LOG_ROOT}/my_merge/${ablation}" \
    GPU_IDS="${GPU_IDS}" \
    MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS}" \
    DATASETS="${DATASETS}" \
    SMALL_MODELS="${SMALL_MODELS}" \
    CLIP_MODELS="${CLIP_MODELS}" \
    TASK_TYPES="${TASK_TYPES}" \
    MY_MERGE_STATS_MAX_BATCHES="${MY_MERGE_STATS_MAX_BATCHES}" \
    MY_MERGE_EVAL_MAX_BATCHES="${MY_MERGE_EVAL_MAX_BATCHES}" \
    MY_MERGE_BN_BATCHES="${MY_MERGE_BN_BATCHES}" \
    MY_MERGE_EXPORT_DIAGNOSTICS="${MY_MERGE_EXPORT_DIAGNOSTICS}" \
    MY_MERGE_DIAGNOSTICS_PLOT=false \
    MY_MERGE_VIZ_MAX_BATCHES="${MY_MERGE_VIZ_MAX_BATCHES}" \
    MY_MERGE_VIZ_MAX_PLOTS="${MY_MERGE_VIZ_MAX_PLOTS}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    STATS_BATCH_SIZE="${STATS_BATCH_SIZE}" \
    SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE}" \
    VLM_BATCH_SIZE="${VLM_BATCH_SIZE}" \
    DELETE_FLAG="${DELETE_FLAG}" \
    RESUME_FLAG="${RESUME_FLAG}" \
    REPRO_MODE="${REPRO_MODE}" \
    bash scripts/run_my_merge_ablation_split_grid.sh
done

"${PYTHON_BIN}" scripts/summarize_my_merge_ablations.py \
  --grid-root "${MY_MERGE_OUTPUT_ROOT}" \
  --baseline result/all_results.md \
  --dest "${MY_MERGE_OUTPUT_ROOT}/reports/ablation_summary.md"

"${PYTHON_BIN}" scripts/generate_ablation_combined_results_table.py \
  --base result/all_results.md \
  --grid-root "${MY_MERGE_OUTPUT_ROOT}" \
  --dest "${MY_MERGE_OUTPUT_ROOT}/reports/all_results_ablation_combined.md"

if [[ "${PUBLISH_RESULTS}" == "true" ]]; then
  mkdir -p "${PUBLISH_ROOT}/reports"
  cp "${MY_MERGE_OUTPUT_ROOT}/reports/all_results_ablation_combined.md" "${PUBLISH_ROOT}/汇总表.md"
  cp "${MY_MERGE_OUTPUT_ROOT}/reports/ablation_summary.md" "${PUBLISH_ROOT}/reports/ablation_summary.md"
  cp "${MY_MERGE_OUTPUT_ROOT}/reports/validated_run_config.txt" "${PUBLISH_ROOT}/reports/validated_run_config.txt"
  if [[ -f "${REPRO_OUTPUT_ROOT}/reports/reproduction_check.md" ]]; then
    cp "${REPRO_OUTPUT_ROOT}/reports/reproduction_check.md" "${PUBLISH_ROOT}/reports/reproduction_check.md"
    cp "${REPRO_OUTPUT_ROOT}/reports/reproduction_check.csv" "${PUBLISH_ROOT}/reports/reproduction_check.csv"
  fi
fi

echo "[$(date '+%F %T')] validated my_merge run finished"
echo "my_merge_output_root=${MY_MERGE_OUTPUT_ROOT}"
echo "summary=${MY_MERGE_OUTPUT_ROOT}/reports/all_results_ablation_combined.md"
if [[ "${PUBLISH_RESULTS}" == "true" ]]; then
  echo "published_summary=${PUBLISH_ROOT}/汇总表.md"
fi
