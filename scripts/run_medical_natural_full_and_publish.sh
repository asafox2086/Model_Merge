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
RUN_TAG="${RUN_TAG:-codex_medical_natural_full_${STAMP}}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-2}"

MEDICAL_DATASETS="${MEDICAL_DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224}"
MEDICAL_SMALL_MODELS="${MEDICAL_SMALL_MODELS:-resnet convnext vit_t swin_tiny}"
MEDICAL_CLIP_MODELS="${MEDICAL_CLIP_MODELS:-openai/clip-vit-base-patch32}"

NATURAL_DATASETS="${NATURAL_DATASETS:-cifar10_32 cifar100_32 svhn_32 tinyimagenet_64}"
NATURAL_SMALL_MODELS="${NATURAL_SMALL_MODELS:-resnet convnext vit_t swin_tiny resnet34 densenet mobilenet efficientnet}"
NATURAL_METHODS="${NATURAL_METHODS:-avg ties dare_linear dare_ties regmean fisher breadcrumbs model_stock from iso_c free_merge robustmerge}"

ABLATIONS="${ABLATIONS:-full no_client_information no_fusion_selection no_adaptive_candidates}"
MY_MERGE_STATS_MAX_BATCHES="${MY_MERGE_STATS_MAX_BATCHES:-16}"
MY_MERGE_EVAL_MAX_BATCHES="${MY_MERGE_EVAL_MAX_BATCHES:-1}"
MY_MERGE_BN_BATCHES="${MY_MERGE_BN_BATCHES:-4}"
MY_MERGE_EXPORT_DIAGNOSTICS="${MY_MERGE_EXPORT_DIAGNOSTICS:-true}"
MY_MERGE_VIZ_MAX_BATCHES="${MY_MERGE_VIZ_MAX_BATCHES:-1}"
MY_MERGE_VIZ_MAX_PLOTS="${MY_MERGE_VIZ_MAX_PLOTS:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_TAG}}"
MEDICAL_ROOT="${MEDICAL_ROOT:-${OUTPUT_ROOT}/medical}"
NATURAL_BASE_ROOT="${NATURAL_BASE_ROOT:-${OUTPUT_ROOT}/natural_formal}"
NATURAL_MY_ROOT="${NATURAL_MY_ROOT:-${OUTPUT_ROOT}/natural_my_merge_ablation_grid}"
PUBLISH_ROOT="${PUBLISH_ROOT:-${ROOT_DIR}/My_merge_ret}"

mkdir -p "${OUTPUT_ROOT}/reports" "${LOG_ROOT}" "${PUBLISH_ROOT}/reports"

{
  echo "run_tag=${RUN_TAG}"
  echo "gpu_ids=${GPU_IDS}"
  echo "max_parallel_jobs=${MAX_PARALLEL_JOBS}"
  echo "medical_datasets=${MEDICAL_DATASETS}"
  echo "medical_small_models=${MEDICAL_SMALL_MODELS}"
  echo "medical_clip_models=${MEDICAL_CLIP_MODELS}"
  echo "natural_datasets=${NATURAL_DATASETS}"
  echo "natural_small_models=${NATURAL_SMALL_MODELS}"
  echo "natural_methods=${NATURAL_METHODS}"
  echo "ablations=${ABLATIONS}"
  echo "my_merge_stats_max_batches=${MY_MERGE_STATS_MAX_BATCHES}"
  echo "my_merge_eval_max_batches=${MY_MERGE_EVAL_MAX_BATCHES}"
  echo "my_merge_bn_batches=${MY_MERGE_BN_BATCHES}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "medical_root=${MEDICAL_ROOT}"
  echo "natural_base_root=${NATURAL_BASE_ROOT}"
  echo "natural_my_root=${NATURAL_MY_ROOT}"
} | tee "${OUTPUT_ROOT}/reports/domain_full_config.txt"

echo "[$(date '+%F %T')] medical my_merge full start"
env \
  RUN_TAG="${RUN_TAG}_medical" \
  OUTPUT_ROOT="${MEDICAL_ROOT}" \
  LOG_ROOT="${LOG_ROOT}/medical" \
  MODEL_HUB_ROOT="${ROOT_DIR}/model_hub" \
  DATA_ROOT="${ROOT_DIR}/Med_data" \
  GPU_IDS="${GPU_IDS}" \
  MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS}" \
  DATASETS="${MEDICAL_DATASETS}" \
  SMALL_MODELS="${MEDICAL_SMALL_MODELS}" \
  CLIP_MODELS="${MEDICAL_CLIP_MODELS}" \
  TASK_TYPES="small vlm" \
  RUN_REPRO=false \
  ABLATIONS="${ABLATIONS}" \
  MY_MERGE_STATS_MAX_BATCHES="${MY_MERGE_STATS_MAX_BATCHES}" \
  MY_MERGE_EVAL_MAX_BATCHES="${MY_MERGE_EVAL_MAX_BATCHES}" \
  MY_MERGE_BN_BATCHES="${MY_MERGE_BN_BATCHES}" \
  MY_MERGE_EXPORT_DIAGNOSTICS="${MY_MERGE_EXPORT_DIAGNOSTICS}" \
  MY_MERGE_VIZ_MAX_BATCHES="${MY_MERGE_VIZ_MAX_BATCHES}" \
  MY_MERGE_VIZ_MAX_PLOTS="${MY_MERGE_VIZ_MAX_PLOTS}" \
  PUBLISH_RESULTS=false \
  bash scripts/run_validated_my_merge_full.sh

echo "[$(date '+%F %T')] natural formal baseline start"
env \
  RUN_TAG="${RUN_TAG}_natural_formal" \
  OUTPUT_ROOT="${NATURAL_BASE_ROOT}" \
  LOG_ROOT="${LOG_ROOT}/natural_formal" \
  MODEL_HUB_ROOT="${ROOT_DIR}/natural_model_hub" \
  DATA_ROOT="${ROOT_DIR}/Med_data" \
  GPU_IDS="${GPU_IDS}" \
  MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS}" \
  DATASETS="${NATURAL_DATASETS}" \
  SMALL_MODELS="${NATURAL_SMALL_MODELS}" \
  FORMAL_TASK_TYPES="small" \
  FORMAL_METHODS="${NATURAL_METHODS}" \
  NUM_WORKERS=0 \
  STATS_NUM_WORKERS=0 \
  SMALL_BATCH_SIZE=64 \
  DELETE_FLAG=--delete-merged \
  RESUME_FLAG=--resume \
  bash run_compare_multi_gpu.sh

echo "[$(date '+%F %T')] natural my_merge ablations start"
read -r -a ABLATION_ARRAY <<< "${ABLATIONS}"
for ablation in "${ABLATION_ARRAY[@]}"; do
  env \
    ABLATION="${ablation}" \
    OUTPUT_ROOT="${NATURAL_MY_ROOT}/${ablation}" \
    LOG_ROOT="${LOG_ROOT}/natural_my_merge/${ablation}" \
    MODEL_HUB_ROOT="${ROOT_DIR}/natural_model_hub" \
    DATA_ROOT="${ROOT_DIR}/Med_data" \
    GPU_IDS="${GPU_IDS}" \
    MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS}" \
    DATASETS="${NATURAL_DATASETS}" \
    SMALL_MODELS="${NATURAL_SMALL_MODELS}" \
    TASK_TYPES="small" \
    MY_MERGE_DOMAIN_CONTROL=true \
    MY_MERGE_STATS_MAX_BATCHES="${MY_MERGE_STATS_MAX_BATCHES}" \
    MY_MERGE_EVAL_MAX_BATCHES="${MY_MERGE_EVAL_MAX_BATCHES}" \
    MY_MERGE_BN_BATCHES="${MY_MERGE_BN_BATCHES}" \
    MY_MERGE_EXPORT_DIAGNOSTICS="${MY_MERGE_EXPORT_DIAGNOSTICS}" \
    MY_MERGE_DIAGNOSTICS_PLOT=false \
    MY_MERGE_VIZ_MAX_BATCHES="${MY_MERGE_VIZ_MAX_BATCHES}" \
    MY_MERGE_VIZ_MAX_PLOTS="${MY_MERGE_VIZ_MAX_PLOTS}" \
    NUM_WORKERS=0 \
    STATS_NUM_WORKERS=0 \
    SMALL_BATCH_SIZE=64 \
    DELETE_FLAG=--delete-merged \
    RESUME_FLAG=--resume \
    bash scripts/run_my_merge_ablation_split_grid.sh
done

echo "[$(date '+%F %T')] generating combined tables"
MEDICAL_COMBINED="${MEDICAL_ROOT}/my_merge_ablation_grid/reports/all_results_ablation_combined.md"
NATURAL_BASE_TABLE="${OUTPUT_ROOT}/reports/natural_formal_results.md"
NATURAL_COMBINED="${OUTPUT_ROOT}/reports/natural_all_results_ablation_combined.md"

read -r -a NATURAL_DATASET_ARRAY <<< "${NATURAL_DATASETS}"
read -r -a NATURAL_MODEL_ARRAY <<< "${NATURAL_SMALL_MODELS}"
read -r -a NATURAL_METHOD_ARRAY <<< "${NATURAL_METHODS}"

"${PYTHON_BIN}" scripts/generate_formal_results_table.py \
  --output-root "${NATURAL_BASE_ROOT}" \
  --dest "${NATURAL_BASE_TABLE}" \
  --section-title "Natural Small" \
  --task-type small \
  --datasets "${NATURAL_DATASET_ARRAY[@]}" \
  --small-models "${NATURAL_MODEL_ARRAY[@]}" \
  --methods "${NATURAL_METHOD_ARRAY[@]}"

"${PYTHON_BIN}" scripts/generate_ablation_combined_results_table.py \
  --base "${NATURAL_BASE_TABLE}" \
  --grid-root "${NATURAL_MY_ROOT}" \
  --dest "${NATURAL_COMBINED}"

"${PYTHON_BIN}" scripts/publish_domain_summary_tables.py \
  --medical "${MEDICAL_COMBINED}" \
  --natural "${NATURAL_COMBINED}" \
  --dest "${PUBLISH_ROOT}/汇总表.md"

cp "${MEDICAL_ROOT}/my_merge_ablation_grid/reports/ablation_summary.md" "${PUBLISH_ROOT}/reports/ablation_summary.md"
cp "${OUTPUT_ROOT}/reports/domain_full_config.txt" "${PUBLISH_ROOT}/reports/domain_full_config.txt"
cp "${NATURAL_BASE_TABLE}" "${PUBLISH_ROOT}/reports/natural_formal_results.md"
cp "${NATURAL_COMBINED}" "${PUBLISH_ROOT}/reports/natural_all_results_ablation_combined.md"

echo "[$(date '+%F %T')] domain full run finished"
echo "output_root=${OUTPUT_ROOT}"
echo "published_summary=${PUBLISH_ROOT}/汇总表.md"
