#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/multi_gpu_common.sh"

TASK_TYPES=( ${TASK_TYPES:-small vlm} )
ABLATION="${ABLATION:-full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/my_merge_split_${ABLATION}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/my_merge_split_${ABLATION}_$(date +%Y%m%d_%H%M%S)}"
MY_MERGE_STATS_MAX_BATCHES="${MY_MERGE_STATS_MAX_BATCHES:-16}"
MY_MERGE_EVAL_MAX_BATCHES="${MY_MERGE_EVAL_MAX_BATCHES:-1}"
MY_MERGE_BN_BATCHES="${MY_MERGE_BN_BATCHES:-4}"
MY_MERGE_EXPORT_DIAGNOSTICS="${MY_MERGE_EXPORT_DIAGNOSTICS:-true}"
MY_MERGE_DIAGNOSTICS_PLOT="${MY_MERGE_DIAGNOSTICS_PLOT:-false}"
MY_MERGE_DOMAIN_CONTROL="${MY_MERGE_DOMAIN_CONTROL:-false}"
MY_MERGE_VIZ_MAX_BATCHES="${MY_MERGE_VIZ_MAX_BATCHES:-1}"
MY_MERGE_VIZ_MAX_PLOTS="${MY_MERGE_VIZ_MAX_PLOTS:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"

mkdir -p "${OUTPUT_ROOT}/reports" "${LOG_ROOT}"

read -r -a GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
if (( GPU_COUNT == 0 )); then
  echo "GPU_IDS is empty" >&2
  exit 1
fi
if (( MAX_PARALLEL_JOBS < 1 )); then
  MAX_PARALLEL_JOBS=1
fi
if (( GPU_COUNT > MAX_PARALLEL_JOBS )); then
  GPU_ARRAY=("${GPU_ARRAY[@]:0:${MAX_PARALLEL_JOBS}}")
  GPU_COUNT="${#GPU_ARRAY[@]}"
fi

RESUME_ARGS=()
DELETE_ARGS=()
build_boolean_flag_args "${RESUME_FLAG}" "--resume" "--no-resume" RESUME_ARGS
build_boolean_flag_args "${DELETE_FLAG}" "--delete-merged" "--no-delete-merged" DELETE_ARGS

case " ${TASK_TYPES[*]} " in
  *" small "*)
    prepare_reference_cache "small"
    ;;
esac
case " ${TASK_TYPES[*]} " in
  *" vlm "*)
    prepare_reference_cache "vlm"
    ;;
esac

diagnostic_args=(
  --stats-split "${STATS_SPLIT}"
  --stats-batch-size "${STATS_BATCH_SIZE}"
  --stats-num-workers "${STATS_NUM_WORKERS}"
  --my-merge-ablation "${ABLATION}"
  --my-merge-stats-max-batches "${MY_MERGE_STATS_MAX_BATCHES}"
  --my-merge-eval-max-batches "${MY_MERGE_EVAL_MAX_BATCHES}"
  --my-merge-bn-batches "${MY_MERGE_BN_BATCHES}"
  --my-merge-viz-max-batches "${MY_MERGE_VIZ_MAX_BATCHES}"
  --my-merge-viz-max-plots "${MY_MERGE_VIZ_MAX_PLOTS}"
)
if [[ "${MY_MERGE_EXPORT_DIAGNOSTICS}" == "true" ]]; then
  diagnostic_args+=(--my-merge-export-diagnostics)
else
  diagnostic_args+=(--no-my-merge-export-diagnostics)
fi
if [[ "${MY_MERGE_DIAGNOSTICS_PLOT}" == "true" ]]; then
  diagnostic_args+=(--my-merge-diagnostics-plot)
else
  diagnostic_args+=(--no-my-merge-diagnostics-plot)
fi
if [[ "${MY_MERGE_DOMAIN_CONTROL}" == "true" ]]; then
  diagnostic_args+=(--my-merge-domain-control)
fi

run_small_model_job() {
  local model_name="$1"
  local gpu_id="$2"
  local job_name="small_${model_name}__my_merge"
  local output_dir="${OUTPUT_ROOT}/${job_name}"
  local log_path="${LOG_ROOT}/${job_name}__gpu${gpu_id}.log"
  mkdir -p "${output_dir}" "$(dirname "${log_path}")"
  {
    echo "[$(date +%F\ %T)] start | ablation=${ABLATION} | task_type=small | model=${model_name} | gpu=${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_all_avg_eval.py" \
      --model-hub-root "${MODEL_HUB_ROOT}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${output_dir}" \
      --device "${DEVICE}" \
      --num-workers "${NUM_WORKERS}" \
      --task-type small \
      --method my_merge \
      "${RESUME_ARGS[@]}" \
      "${DELETE_ARGS[@]}" \
      --merge-weight-mode equal \
      "${diagnostic_args[@]}" \
      --small-batch-size "${SMALL_BATCH_SIZE}" \
      --vlm-batch-size 1 \
      --datasets "${DATASETS[@]}" \
      --small-models "${model_name}"
    echo "[$(date +%F\ %T)] done | ablation=${ABLATION} | task_type=small | model=${model_name} | gpu=${gpu_id}"
  } >> "${log_path}" 2>&1
}

run_vlm_job() {
  local clip_model="$1"
  local gpu_id="$2"
  local safe_name="${clip_model##*/}"
  local job_name="vlm_${safe_name}__my_merge"
  local output_dir="${OUTPUT_ROOT}/${job_name}"
  local log_path="${LOG_ROOT}/${job_name}__gpu${gpu_id}.log"
  mkdir -p "${output_dir}" "$(dirname "${log_path}")"
  {
    echo "[$(date +%F\ %T)] start | ablation=${ABLATION} | task_type=vlm | model=${clip_model} | gpu=${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_all_avg_eval.py" \
      --model-hub-root "${MODEL_HUB_ROOT}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${output_dir}" \
      --device "${DEVICE}" \
      --num-workers "${NUM_WORKERS}" \
      --task-type vlm \
      --method my_merge \
      "${RESUME_ARGS[@]}" \
      "${DELETE_ARGS[@]}" \
      --merge-weight-mode equal \
      "${diagnostic_args[@]}" \
      --small-batch-size 1 \
      --vlm-batch-size "${VLM_BATCH_SIZE}" \
      --datasets "${DATASETS[@]}" \
      --clip-models "${clip_model}"
    echo "[$(date +%F\ %T)] done | ablation=${ABLATION} | task_type=vlm | model=${clip_model} | gpu=${gpu_id}"
  } >> "${log_path}" 2>&1
}

JOBS=()
case " ${TASK_TYPES[*]} " in
  *" small "*)
    for model_name in "${SMALL_MODELS[@]}"; do
      JOBS+=("small|${model_name}")
    done
    ;;
esac
case " ${TASK_TYPES[*]} " in
  *" vlm "*)
    for clip_model in "${CLIP_MODELS[@]}"; do
      JOBS+=("vlm|${clip_model}")
    done
    ;;
esac

{
  echo "output_root=${OUTPUT_ROOT}"
  echo "log_root=${LOG_ROOT}"
  echo "ablation=${ABLATION}"
  echo "gpu_ids=${GPU_IDS}"
  echo "active_gpu_ids=${GPU_ARRAY[*]}"
  echo "max_parallel_jobs=${MAX_PARALLEL_JOBS}"
  echo "task_types=${TASK_TYPES[*]}"
  echo "datasets=${DATASETS[*]}"
  echo "small_models=${SMALL_MODELS[*]}"
  echo "clip_models=${CLIP_MODELS[*]}"
  echo "num_workers=${NUM_WORKERS}"
  echo "stats_split=${STATS_SPLIT}"
  echo "stats_batch_size=${STATS_BATCH_SIZE}"
  echo "stats_num_workers=${STATS_NUM_WORKERS}"
  echo "small_batch_size=${SMALL_BATCH_SIZE}"
  echo "vlm_batch_size=${VLM_BATCH_SIZE}"
  echo "my_merge_stats_max_batches=${MY_MERGE_STATS_MAX_BATCHES}"
  echo "my_merge_eval_max_batches=${MY_MERGE_EVAL_MAX_BATCHES}"
  echo "my_merge_bn_batches=${MY_MERGE_BN_BATCHES}"
  echo "my_merge_diagnostics_plot=${MY_MERGE_DIAGNOSTICS_PLOT}"
  echo "my_merge_domain_control=${MY_MERGE_DOMAIN_CONTROL}"
} > "${OUTPUT_ROOT}/reports/ablation_split_config.txt"

for worker_idx in "${!GPU_ARRAY[@]}"; do
  gpu_id="${GPU_ARRAY[worker_idx]}"
  (
    for job_idx in "${!JOBS[@]}"; do
      if (( job_idx % GPU_COUNT != worker_idx )); then
        continue
      fi
      IFS='|' read -r task_type model_name <<< "${JOBS[job_idx]}"
      if [[ "${task_type}" == "small" ]]; then
        run_small_model_job "${model_name}" "${gpu_id}"
      else
        run_vlm_job "${model_name}" "${gpu_id}"
      fi
    done
  ) &
done

wait

echo "[$(date +%F\ %T)] split ablation finished | ablation=${ABLATION} | output=${OUTPUT_ROOT}"
