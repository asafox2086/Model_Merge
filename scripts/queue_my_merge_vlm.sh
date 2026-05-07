#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

while pgrep -f "codex_my_merge_small_b5|outputs/custom_methods_codex_my_merge_small_b5" >/dev/null; do
  sleep 60
done

RUN_TAG="${RUN_TAG:-codex_my_merge_vlm}"
GPU_IDS="${GPU_IDS:-1}"
DATASETS="${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224}"
CLIP_MODELS="${CLIP_MODELS:-openai/clip-vit-base-patch32}"

env \
  RUN_TAG="${RUN_TAG}" \
  GPU_IDS="${GPU_IDS}" \
  DATASETS="${DATASETS}" \
  CLIP_MODELS="${CLIP_MODELS}" \
  TASK_TYPES="vlm" \
  CUSTOM_METHODS="my_merge" \
  bash run_custom_methods_multi_gpu.sh
