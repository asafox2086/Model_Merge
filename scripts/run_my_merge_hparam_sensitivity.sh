#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.gpuenv/bin/python" ]]; then
    PYTHON="$ROOT/.gpuenv/bin/python"
  else
    PYTHON="python3"
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/my_merge_hparam_sensitivity_$STAMP}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-$ROOT/model_hub}"
DATA_ROOT="${DATA_ROOT:-$ROOT/Med_data}"
DATASETS="${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 chaoshengmnist_224}"
SMALL_MODELS="${SMALL_MODELS:-resnet}"
TASK_TYPE="${TASK_TYPE:-small}"
LIMIT="${LIMIT:-0}"
SETTINGS="${SETTINGS:-stats1_eval1_bn0 stats2_eval1_bn0 stats4_eval2_bn1 stats4_eval4_bn2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STATS_SPLIT="${STATS_SPLIT:-val}"
STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-32}"
STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-0}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-64}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-32}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

read -r -a DATASET_ARGS <<< "$DATASETS"
read -r -a SMALL_MODEL_ARGS <<< "$SMALL_MODELS"
read -r -a SETTING_ARGS <<< "$SETTINGS"

mkdir -p "$OUTPUT_ROOT/reports"
{
  echo "output_root=$OUTPUT_ROOT"
  echo "device=$DEVICE"
  echo "datasets=$DATASETS"
  echo "small_models=$SMALL_MODELS"
  echo "task_type=$TASK_TYPE"
  echo "settings=$SETTINGS"
  echo "limit=$LIMIT"
  echo "stats_split=$STATS_SPLIT"
  echo "stats_batch_size=$STATS_BATCH_SIZE"
  echo "stats_num_workers=$STATS_NUM_WORKERS"
} > "$OUTPUT_ROOT/reports/hparam_config.txt"

for setting in "${SETTING_ARGS[@]}"; do
  stats="$(sed -n 's/.*stats\([0-9][0-9]*\\|all\).*/\1/p' <<< "$setting")"
  eval_batches="$(sed -n 's/.*eval\([0-9][0-9]*\\|all\).*/\1/p' <<< "$setting")"
  bn="$(sed -n 's/.*bn\([0-9][0-9]*\\|all\).*/\1/p' <<< "$setting")"
  stats="${stats:-1}"
  eval_batches="${eval_batches:-1}"
  bn="${bn:-0}"
  [[ "$stats" == "all" ]] && stats=0
  [[ "$eval_batches" == "all" ]] && eval_batches=0
  [[ "$bn" == "all" ]] && bn=0

  out_dir="$OUTPUT_ROOT/$setting"
  args=(
    --model-hub-root "$MODEL_HUB_ROOT"
    --data-root "$DATA_ROOT"
    --output-root "$out_dir"
    --device "$DEVICE"
    --num-workers "$NUM_WORKERS"
    --stats-split "$STATS_SPLIT"
    --stats-batch-size "$STATS_BATCH_SIZE"
    --stats-num-workers "$STATS_NUM_WORKERS"
    --small-batch-size "$SMALL_BATCH_SIZE"
    --vlm-batch-size "$VLM_BATCH_SIZE"
    --task-type "$TASK_TYPE"
    --datasets "${DATASET_ARGS[@]}"
    --method my_merge
    --merge-weight-mode equal
    --my-merge-ablation full
    --my-merge-stats-max-batches "$stats"
    --my-merge-eval-max-batches "$eval_batches"
    --my-merge-bn-batches "$bn"
    --delete-merged
    --my-merge-export-diagnostics
    --no-my-merge-diagnostics-plot
  )
  if [[ "$TASK_TYPE" == "small" ]]; then
    args+=(--small-models "${SMALL_MODEL_ARGS[@]}")
  fi
  if [[ "$LIMIT" != "0" ]]; then
    args+=(--limit "$LIMIT")
  fi
  echo "[$(date '+%H:%M:%S')] hparam setting=$setting output=$out_dir"
  "$PYTHON" scripts/run_all_avg_eval.py "${args[@]}" 2>&1 | tee -a "$OUTPUT_ROOT/reports/${setting}.log"
done

"$PYTHON" scripts/plot_my_merge_hparam_sensitivity.py \
  --root "$OUTPUT_ROOT" \
  --dest-dir "$OUTPUT_ROOT/reports/hparam_plots"

echo "Hyperparameter sensitivity finished: $OUTPUT_ROOT"
