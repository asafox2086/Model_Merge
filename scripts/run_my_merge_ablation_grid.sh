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
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/my_merge_ablation_grid_$STAMP}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-$ROOT/model_hub}"
DATA_ROOT="${DATA_ROOT:-$ROOT/Med_data}"
DATASETS="${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224}"
SMALL_MODELS="${SMALL_MODELS:-resnet convnext vit_t swin_tiny}"
CLIP_MODELS="${CLIP_MODELS:-openai/clip-vit-base-patch32}"
TASK_TYPES="${TASK_TYPES:-small vlm}"
ABLATIONS="${ABLATIONS:-full no_domain_preprocess no_rarity no_focal no_domain_focus no_layerwise no_residual no_candidate_bank no_calibration avg_only}"
LIMIT="${LIMIT:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-32}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-128}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-64}"
MY_MERGE_STATS_MAX_BATCHES="${MY_MERGE_STATS_MAX_BATCHES:-4}"
MY_MERGE_EVAL_MAX_BATCHES="${MY_MERGE_EVAL_MAX_BATCHES:-2}"
MY_MERGE_BN_BATCHES="${MY_MERGE_BN_BATCHES:-4}"

read -r -a DATASET_ARGS <<< "$DATASETS"
read -r -a SMALL_MODEL_ARGS <<< "$SMALL_MODELS"
read -r -a CLIP_MODEL_ARGS <<< "$CLIP_MODELS"
read -r -a TASK_TYPE_ARGS <<< "$TASK_TYPES"
read -r -a ABLATION_ARGS <<< "$ABLATIONS"

mkdir -p "$OUTPUT_ROOT/reports"
{
  echo "output_root=$OUTPUT_ROOT"
  echo "device=$DEVICE"
  echo "datasets=$DATASETS"
  echo "small_models=$SMALL_MODELS"
  echo "clip_models=$CLIP_MODELS"
  echo "task_types=$TASK_TYPES"
  echo "ablations=$ABLATIONS"
  echo "limit=$LIMIT"
  echo "my_merge_stats_max_batches=$MY_MERGE_STATS_MAX_BATCHES"
  echo "my_merge_eval_max_batches=$MY_MERGE_EVAL_MAX_BATCHES"
  echo "my_merge_bn_batches=$MY_MERGE_BN_BATCHES"
} > "$OUTPUT_ROOT/reports/ablation_grid_config.txt"

for ablation in "${ABLATION_ARGS[@]}"; do
  for task_type in "${TASK_TYPE_ARGS[@]}"; do
    out_dir="$OUTPUT_ROOT/$ablation"
    common_args=(
      --model-hub-root "$MODEL_HUB_ROOT"
      --data-root "$DATA_ROOT"
      --output-root "$out_dir"
      --device "$DEVICE"
      --num-workers "$NUM_WORKERS"
      --stats-batch-size "$STATS_BATCH_SIZE"
      --small-batch-size "$SMALL_BATCH_SIZE"
      --vlm-batch-size "$VLM_BATCH_SIZE"
      --task-type "$task_type"
      --datasets "${DATASET_ARGS[@]}"
      --method my_merge
      --merge-weight-mode equal
      --my-merge-ablation "$ablation"
      --my-merge-stats-max-batches "$MY_MERGE_STATS_MAX_BATCHES"
      --my-merge-eval-max-batches "$MY_MERGE_EVAL_MAX_BATCHES"
      --my-merge-bn-batches "$MY_MERGE_BN_BATCHES"
    )
    if [[ "$LIMIT" != "0" ]]; then
      common_args+=(--limit "$LIMIT")
    fi
    if [[ "$task_type" == "small" ]]; then
      common_args+=(--small-models "${SMALL_MODEL_ARGS[@]}")
    else
      common_args+=(--clip-models "${CLIP_MODEL_ARGS[@]}")
    fi
    echo "[$(date '+%H:%M:%S')] ablation=$ablation task_type=$task_type output=$out_dir"
    "$PYTHON" scripts/run_all_avg_eval.py "${common_args[@]}" 2>&1 | tee -a "$OUTPUT_ROOT/reports/${ablation}_${task_type}.log"
  done
done

"$PYTHON" scripts/summarize_my_merge_ablations.py \
  --grid-root "$OUTPUT_ROOT" \
  --baseline result/all_results.md \
  --dest "$OUTPUT_ROOT/reports/ablation_summary.md"

echo "Ablation grid finished: $OUTPUT_ROOT"
