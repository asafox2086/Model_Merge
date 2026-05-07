#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-codex_full_my_merge_20260507}"
REPRO_TAG="${REPRO_TAG:-codex_full_repro_20260507}"
DATASETS="${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224}"
SMALL_MODELS="${SMALL_MODELS:-resnet convnext vit_t swin_tiny}"
CLIP_MODELS="${CLIP_MODELS:-openai/clip-vit-base-patch32}"
GPU_IDS="${GPU_IDS:-0 1}"

while pgrep -f "${REPRO_TAG}|run_compare_multi_gpu.sh" >/dev/null; do
  sleep 60
done

env \
  RUN_TAG="${RUN_TAG}" \
  GPU_IDS="${GPU_IDS}" \
  DATASETS="${DATASETS}" \
  SMALL_MODELS="${SMALL_MODELS}" \
  CLIP_MODELS="${CLIP_MODELS}" \
  TASK_TYPES="small vlm" \
  CUSTOM_METHODS="my_merge" \
  bash run_custom_methods_multi_gpu.sh

python3 scripts/generate_my_merge_master_table.py \
  --output-root "outputs/custom_methods_${RUN_TAG}" \
  --dest "My_merge_ret/all_results_my_merge.md"
