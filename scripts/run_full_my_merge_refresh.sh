#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.gpuenv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

RUN_TAG="${RUN_TAG:-my_merge_full_refresh_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0}"
DATASETS="${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224}"
SMALL_MODELS="${SMALL_MODELS:-resnet convnext vit_t swin_tiny}"
CLIP_MODELS="${CLIP_MODELS:-openai/clip-vit-base-patch32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/custom_methods_${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_TAG}}"
NUM_WORKERS_SPLIT="${NUM_WORKERS_SPLIT:-0}"
STATS_SPLIT="${STATS_SPLIT:-val}"
STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-32}"
STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-${NUM_WORKERS_SPLIT}}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-64}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-32}"
MY_MERGE_STATS_MAX_BATCHES="${MY_MERGE_STATS_MAX_BATCHES:-1}"
MY_MERGE_EVAL_MAX_BATCHES="${MY_MERGE_EVAL_MAX_BATCHES:-1}"
MY_MERGE_BN_BATCHES="${MY_MERGE_BN_BATCHES:-0}"
MY_MERGE_FEATURE_SUMMARY_ROOT="${MY_MERGE_FEATURE_SUMMARY_ROOT:-}"
MY_MERGE_REQUIRE_FEATURE_SUMMARY="${MY_MERGE_REQUIRE_FEATURE_SUMMARY:-false}"
PREPARE_REFERENCE_CACHE="${PREPARE_REFERENCE_CACHE:-false}"
MY_MERGE_TABLE_EXTRA_ROOTS="${MY_MERGE_TABLE_EXTRA_ROOTS:-}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

echo "[$(date +%F\ %T)] my_merge full refresh start"
echo "[$(date +%F\ %T)] RUN_TAG=${RUN_TAG}"
echo "[$(date +%F\ %T)] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[$(date +%F\ %T)] LOG_ROOT=${LOG_ROOT}"
echo "[$(date +%F\ %T)] GPU_IDS=${GPU_IDS}"
echo "[$(date +%F\ %T)] DATASETS=${DATASETS}"
echo "[$(date +%F\ %T)] SMALL_MODELS=${SMALL_MODELS}"
echo "[$(date +%F\ %T)] CLIP_MODELS=${CLIP_MODELS}"
echo "[$(date +%F\ %T)] NUM_WORKERS_SPLIT=${NUM_WORKERS_SPLIT}"
echo "[$(date +%F\ %T)] STATS_SPLIT=${STATS_SPLIT}"
echo "[$(date +%F\ %T)] STATS_BATCH_SIZE=${STATS_BATCH_SIZE}"
echo "[$(date +%F\ %T)] STATS_NUM_WORKERS=${STATS_NUM_WORKERS}"
echo "[$(date +%F\ %T)] MY_MERGE_STATS_MAX_BATCHES=${MY_MERGE_STATS_MAX_BATCHES}"
echo "[$(date +%F\ %T)] MY_MERGE_EVAL_MAX_BATCHES=${MY_MERGE_EVAL_MAX_BATCHES}"
echo "[$(date +%F\ %T)] MY_MERGE_BN_BATCHES=${MY_MERGE_BN_BATCHES}"
echo "[$(date +%F\ %T)] MY_MERGE_FEATURE_SUMMARY_ROOT=${MY_MERGE_FEATURE_SUMMARY_ROOT}"
echo "[$(date +%F\ %T)] MY_MERGE_REQUIRE_FEATURE_SUMMARY=${MY_MERGE_REQUIRE_FEATURE_SUMMARY}"
echo "[$(date +%F\ %T)] PREPARE_REFERENCE_CACHE=${PREPARE_REFERENCE_CACHE}"
echo "[$(date +%F\ %T)] MY_MERGE_TABLE_EXTRA_ROOTS=${MY_MERGE_TABLE_EXTRA_ROOTS}"

read -r -a GPU_ARRAY <<< "${GPU_IDS}"
read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a SMALL_MODEL_ARRAY <<< "${SMALL_MODELS}"
read -r -a CLIP_MODEL_ARRAY <<< "${CLIP_MODELS}"

FEATURE_SUMMARY_ARGS=()
if [[ -n "${MY_MERGE_FEATURE_SUMMARY_ROOT}" ]]; then
  FEATURE_SUMMARY_ARGS+=(--my-merge-feature-summary-root "${MY_MERGE_FEATURE_SUMMARY_ROOT}")
fi
if [[ "${MY_MERGE_REQUIRE_FEATURE_SUMMARY}" == "true" ]]; then
  FEATURE_SUMMARY_ARGS+=(--my-merge-require-feature-summary)
else
  FEATURE_SUMMARY_ARGS+=(--no-my-merge-require-feature-summary)
fi

if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "GPU_IDS is empty" >&2
  exit 1
fi

if [[ "${PREPARE_REFERENCE_CACHE}" == "true" ]]; then
  echo "[$(date +%F\ %T)] preparing local reference cache | task_type=small"
  env HF_LOCAL_FILES_ONLY="${CACHE_HF_LOCAL_FILES_ONLY:-1}" \
    HF_HUB_OFFLINE="${CACHE_HF_HUB_OFFLINE:-1}" \
    TRANSFORMERS_OFFLINE="${CACHE_TRANSFORMERS_OFFLINE:-1}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/cache_reference_models.py" \
    --model-hub-root "${ROOT_DIR}/model_hub" \
    --task-type small \
    --datasets "${DATASET_ARRAY[@]}" \
    --small-models "${SMALL_MODEL_ARRAY[@]}" \
    --clip-models "${CLIP_MODEL_ARRAY[@]}"

  echo "[$(date +%F\ %T)] preparing local reference cache | task_type=vlm"
  env HF_LOCAL_FILES_ONLY="${CACHE_HF_LOCAL_FILES_ONLY:-1}" \
    HF_HUB_OFFLINE="${CACHE_HF_HUB_OFFLINE:-1}" \
    TRANSFORMERS_OFFLINE="${CACHE_TRANSFORMERS_OFFLINE:-1}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/cache_reference_models.py" \
    --model-hub-root "${ROOT_DIR}/model_hub" \
    --task-type vlm \
    --datasets "${DATASET_ARRAY[@]}" \
    --small-models "${SMALL_MODEL_ARRAY[@]}" \
    --clip-models "${CLIP_MODEL_ARRAY[@]}"
else
  echo "[$(date +%F\ %T)] skipping reference cache preparation"
fi

export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

run_small_job() {
  local model_name="$1"
  local gpu_id="$2"
  local job_name="small_${model_name}__my_merge"
  local job_output="${OUTPUT_ROOT}/${job_name}"
  local job_log="${LOG_ROOT}/${job_name}__gpu${gpu_id}.log"
  mkdir -p "${job_output}" "$(dirname "${job_log}")"
  {
    echo "[$(date +%F\ %T)] start ${job_name} gpu=${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_all_avg_eval.py" \
      --model-hub-root "${ROOT_DIR}/model_hub" \
      --data-root "${ROOT_DIR}/Med_data" \
      --output-root "${job_output}" \
      --device cuda:0 \
      --num-workers "${NUM_WORKERS_SPLIT}" \
      --stats-split "${STATS_SPLIT}" \
      --stats-batch-size "${STATS_BATCH_SIZE}" \
      --stats-num-workers "${STATS_NUM_WORKERS}" \
      --task-type small \
      --method my_merge \
      --resume \
      --delete-merged \
      --merge-weight-mode equal \
      --small-batch-size "${SMALL_BATCH_SIZE}" \
      --vlm-batch-size 1 \
      --my-merge-stats-max-batches "${MY_MERGE_STATS_MAX_BATCHES}" \
      --my-merge-eval-max-batches "${MY_MERGE_EVAL_MAX_BATCHES}" \
      --my-merge-bn-batches "${MY_MERGE_BN_BATCHES}" \
      "${FEATURE_SUMMARY_ARGS[@]}" \
      --datasets "${DATASET_ARRAY[@]}" \
      --small-models "${model_name}"
    echo "[$(date +%F\ %T)] done ${job_name} gpu=${gpu_id}"
  } >> "${job_log}" 2>&1
}

run_vlm_job() {
  local clip_model="$1"
  local gpu_id="$2"
  local safe_name="${clip_model##*/}"
  local job_name="vlm_${safe_name}__my_merge"
  local job_output="${OUTPUT_ROOT}/${job_name}"
  local job_log="${LOG_ROOT}/${job_name}__gpu${gpu_id}.log"
  mkdir -p "${job_output}" "$(dirname "${job_log}")"
  {
    echo "[$(date +%F\ %T)] start ${job_name} gpu=${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_all_avg_eval.py" \
      --model-hub-root "${ROOT_DIR}/model_hub" \
      --data-root "${ROOT_DIR}/Med_data" \
      --output-root "${job_output}" \
      --device cuda:0 \
      --num-workers "${NUM_WORKERS_SPLIT}" \
      --stats-split "${STATS_SPLIT}" \
      --stats-batch-size "${STATS_BATCH_SIZE}" \
      --stats-num-workers "${STATS_NUM_WORKERS}" \
      --task-type vlm \
      --method my_merge \
      --resume \
      --delete-merged \
      --merge-weight-mode equal \
      --small-batch-size 1 \
      --vlm-batch-size "${VLM_BATCH_SIZE}" \
      --my-merge-stats-max-batches "${MY_MERGE_STATS_MAX_BATCHES}" \
      --my-merge-eval-max-batches "${MY_MERGE_EVAL_MAX_BATCHES}" \
      --my-merge-bn-batches "${MY_MERGE_BN_BATCHES}" \
      "${FEATURE_SUMMARY_ARGS[@]}" \
      --datasets "${DATASET_ARRAY[@]}" \
      --clip-models "${clip_model}"
    echo "[$(date +%F\ %T)] done ${job_name} gpu=${gpu_id}"
  } >> "${job_log}" 2>&1
}

job_idx=0
fail=0
for model_name in "${SMALL_MODEL_ARRAY[@]}"; do
  gpu_id="${GPU_ARRAY[$((job_idx % ${#GPU_ARRAY[@]}))]}"
  if ! run_small_job "${model_name}" "${gpu_id}"; then
    fail=1
  fi
  job_idx=$((job_idx + 1))
done
for clip_model in "${CLIP_MODEL_ARRAY[@]}"; do
  gpu_id="${GPU_ARRAY[$((job_idx % ${#GPU_ARRAY[@]}))]}"
  if ! run_vlm_job "${clip_model}" "${gpu_id}"; then
    fail=1
  fi
  job_idx=$((job_idx + 1))
done

if (( fail != 0 )); then
  echo "[$(date +%F\ %T)] one or more my_merge jobs failed" >&2
  exit 1
fi

TABLE_OUTPUT_ROOTS=("${OUTPUT_ROOT}")
if [[ -n "${MY_MERGE_TABLE_EXTRA_ROOTS}" ]]; then
  read -r -a EXTRA_ROOT_ARRAY <<< "${MY_MERGE_TABLE_EXTRA_ROOTS}"
  for extra_root in "${EXTRA_ROOT_ARRAY[@]}"; do
    TABLE_OUTPUT_ROOTS+=("${extra_root}")
  done
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/generate_my_merge_master_table.py" \
  --output-root "${TABLE_OUTPUT_ROOTS[@]}" \
  --dest "${ROOT_DIR}/My_merge_ret/reports/all_results_my_merge.md"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/generate_combined_results_table.py" \
  --base "${ROOT_DIR}/result/all_results.md" \
  --mine "${ROOT_DIR}/My_merge_ret/reports/all_results_my_merge.md" \
  --dest "${ROOT_DIR}/My_merge_ret/汇总表.md"

echo "[$(date +%F\ %T)] my_merge full refresh done"
