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

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
AB_ROOT="${AB_ROOT:-$ROOT/outputs/my_merge_ablation_three_module_full_$STAMP}"
HP_ROOT="${HP_ROOT:-$ROOT/outputs/my_merge_hparam_sensitivity_$STAMP}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/my_merge_full_autorun_$STAMP}"
mkdir -p "$LOG_ROOT"

AB_DEVICE="${AB_DEVICE:-cuda:0}"
HP_DEVICE="${HP_DEVICE:-cuda:1}"
MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-$ROOT/model_hub}"
DATA_ROOT="${DATA_ROOT:-$ROOT/Med_data}"

echo "[$(date '+%F %T')] autorun start"
echo "ablation_root=$AB_ROOT"
echo "hparam_root=$HP_ROOT"
echo "log_root=$LOG_ROOT"

(
  OUTPUT_ROOT="$AB_ROOT" \
  DEVICE="$AB_DEVICE" \
  MODEL_HUB_ROOT="$MODEL_HUB_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  ABLATIONS="${ABLATIONS:-full no_medical_prior no_client_information no_fusion_selection avg_only}" \
  TASK_TYPES="${TASK_TYPES:-small vlm}" \
  DATASETS="${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224}" \
  SMALL_MODELS="${SMALL_MODELS:-resnet convnext vit_t swin_tiny}" \
  CLIP_MODELS="${CLIP_MODELS:-openai/clip-vit-base-patch32}" \
  MY_MERGE_STATS_MAX_BATCHES="${MY_MERGE_STATS_MAX_BATCHES:-1}" \
  MY_MERGE_EVAL_MAX_BATCHES="${MY_MERGE_EVAL_MAX_BATCHES:-1}" \
  MY_MERGE_BN_BATCHES="${MY_MERGE_BN_BATCHES:-0}" \
  DELETE_MERGED="${DELETE_MERGED:-true}" \
  MY_MERGE_EXPORT_DIAGNOSTICS="${MY_MERGE_EXPORT_DIAGNOSTICS:-true}" \
  MY_MERGE_DIAGNOSTICS_PLOT="${MY_MERGE_DIAGNOSTICS_PLOT:-true}" \
  MY_MERGE_VIZ_MAX_BATCHES="${MY_MERGE_VIZ_MAX_BATCHES:-1}" \
  MY_MERGE_VIZ_MAX_PLOTS="${MY_MERGE_VIZ_MAX_PLOTS:-60}" \
  NUM_WORKERS="${NUM_WORKERS:-2}" \
  STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-16}" \
  SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-128}" \
  VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-64}" \
  bash scripts/run_my_merge_ablation_grid.sh
) > "$LOG_ROOT/ablation.log" 2>&1 &
ab_pid=$!

(
  OUTPUT_ROOT="$HP_ROOT" \
  DEVICE="$HP_DEVICE" \
  MODEL_HUB_ROOT="$MODEL_HUB_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  DATASETS="${HP_DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 chaoshengmnist_224}" \
  SMALL_MODELS="${HP_SMALL_MODELS:-resnet}" \
  TASK_TYPE="${HP_TASK_TYPE:-small}" \
  SETTINGS="${HP_SETTINGS:-stats1_eval1_bn0 stats2_eval1_bn0 stats4_eval2_bn1 stats4_eval4_bn2}" \
  MY_MERGE_EXPORT_DIAGNOSTICS=true \
  NUM_WORKERS="${HP_NUM_WORKERS:-2}" \
  STATS_BATCH_SIZE="${HP_STATS_BATCH_SIZE:-32}" \
  SMALL_BATCH_SIZE="${HP_SMALL_BATCH_SIZE:-128}" \
  VLM_BATCH_SIZE="${HP_VLM_BATCH_SIZE:-64}" \
  bash scripts/run_my_merge_hparam_sensitivity.sh
) > "$LOG_ROOT/hparam.log" 2>&1 &
hp_pid=$!

wait "$ab_pid"
wait "$hp_pid"

"$PYTHON" scripts/finalize_my_merge_results.py \
  --ablation-root "$AB_ROOT" \
  --hparam-root "$HP_ROOT" \
  --run-name "my_merge_completed_$STAMP" \
  --commit \
  --push \
  --remote MM \
  --branch main \
  > "$LOG_ROOT/finalize.log" 2>&1

echo "[$(date '+%F %T')] autorun finished"
