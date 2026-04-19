#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/scripts/multi_gpu_common.sh"

FORMAL_METHODS=( ${FORMAL_METHODS:-${FORMAL_METHODS_DEFAULT[*]}} )

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/${RUN_TAG}/formal_compare}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/formal_compare_${RUN_TAG}}"

mkdir -p "${TMPDIR}" "${LOG_ROOT}" "${OUTPUT_ROOT}"

read -r -a GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
if (( GPU_COUNT == 0 )); then
  echo "GPU_IDS is empty" >&2
  exit 1
fi

RESUME_ARGS=()
DELETE_ARGS=()
build_boolean_flag_args "${RESUME_FLAG}" "--resume" "--no-resume" RESUME_ARGS
build_boolean_flag_args "${DELETE_FLAG}" "--delete-merged" "--no-delete-merged" DELETE_ARGS
validate_methods "${FORMAL_METHODS[@]}"

prepare_reference_cache "all"

run_job() {
  local task_type="$1"
  local method="$2"
  local gpu_id="$3"
  local job_name="${task_type}__${method}"
  local output_dir="${OUTPUT_ROOT}/${job_name}"
  local log_path="${LOG_ROOT}/${task_type}__${method}__gpu${gpu_id}.log"
  local -a method_args=()
  local -a cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_all_avg_eval.py"
    --model-hub-root "${MODEL_HUB_ROOT}"
    --data-root "${DATA_ROOT}"
    --output-root "${output_dir}"
    --device "${DEVICE}"
    --num-workers "${NUM_WORKERS}"
    --task-type "${task_type}"
    --method "${method}"
    "${RESUME_ARGS[@]}"
    "${DELETE_ARGS[@]}"
  )

  build_method_args "${method}" method_args
  cmd+=( "${method_args[@]}" )

  if [[ "${task_type}" == "small" ]]; then
    cmd+=(--small-batch-size "${SMALL_BATCH_SIZE}" --vlm-batch-size 1 --datasets "${DATASETS[@]}" --small-models "${SMALL_MODELS[@]}")
  else
    cmd+=(--small-batch-size 1 --vlm-batch-size "${VLM_BATCH_SIZE}" --datasets "${DATASETS[@]}" --clip-models "${CLIP_MODELS[@]}")
  fi

  {
    echo "[$(date +%F\ %T)] start | task_type=${task_type} | method=${method} | gpu=${gpu_id}"
    echo "[$(date +%F\ %T)] job_name=${job_name}"
    echo "[$(date +%F\ %T)] output_root=${output_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}"
    echo "[$(date +%F\ %T)] done | task_type=${task_type} | method=${method} | gpu=${gpu_id}"
  } >> "${log_path}" 2>&1
}

JOBS=()
for method in "${FORMAL_METHODS[@]}"; do
  JOBS+=( "small|${method}" )
done
for method in "${FORMAL_METHODS[@]}"; do
  JOBS+=( "vlm|${method}" )
done

echo "[$(date +%F\ %T)] RUN_TAG=${RUN_TAG}"
echo "[$(date +%F\ %T)] GPUs=${GPU_IDS}"
echo "[$(date +%F\ %T)] device=${DEVICE}"
echo "[$(date +%F\ %T)] datasets=${DATASETS[*]}"
echo "[$(date +%F\ %T)] small_models=${SMALL_MODELS[*]}"
echo "[$(date +%F\ %T)] clip_models=${CLIP_MODELS[*]}"
echo "[$(date +%F\ %T)] methods=${FORMAL_METHODS[*]}"
echo "[$(date +%F\ %T)] logs=${LOG_ROOT}"
echo "[$(date +%F\ %T)] outputs=${OUTPUT_ROOT}"
echo "[$(date +%F\ %T)] HF_ENDPOINT=${HF_ENDPOINT:-<unset>}"
echo "[$(date +%F\ %T)] HF_LOCAL_FILES_ONLY=${HF_LOCAL_FILES_ONLY}"
echo "[$(date +%F\ %T)] HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}"
echo "[$(date +%F\ %T)] TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-<unset>}"

for worker_idx in "${!GPU_ARRAY[@]}"; do
  gpu_id="${GPU_ARRAY[worker_idx]}"
  (
    for job_idx in "${!JOBS[@]}"; do
      if (( job_idx % GPU_COUNT != worker_idx )); then
        continue
      fi
      IFS='|' read -r task_type method <<< "${JOBS[job_idx]}"
      run_job "${task_type}" "${method}" "${gpu_id}"
    done
  ) &
done

wait

echo "[$(date +%F\ %T)] formal compare finished"
