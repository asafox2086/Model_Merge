#!/usr/bin/env bash
set -euo pipefail
cd /data2/liyapeng_grp/program/MedMNISTMerge
PY=/data2/liyapeng_grp/.conda/envs/MM/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
STATS_ROOT=outputs/my_merge_reference_proto_stats_recall_full_table_20260705
OUT_ROOT=outputs/my_merge_reference_proto_recall_full_table_20260705
MINE=My_merge_ret/all_results_my_merge_full_table_20260705.md
DEST=My_merge_ret/汇总表.md
DATASETS=(bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224)
SMALL_MODELS=(resnet convnext vit_t swin_tiny)
CLIENTS=(3 5 7)
BETAS=(0 0.01 0.1)

echo "[$(date '+%F %T')] full-table run start"
echo "[$(date '+%F %T')] exporting small reference prototype stats"
$PY scripts/export_my_merge_prototypes.py \
  --output-root "$STATS_ROOT" \
  --datasets "${DATASETS[@]}" \
  --small-models "${SMALL_MODELS[@]}" \
  --num-clients "${CLIENTS[@]}" \
  --betas "${BETAS[@]}" \
  --split train \
  --device cuda:0 \
  --batch-size 128 \
  --num-workers 4 \
  --max-samples-per-client 4096 \
  --max-samples-per-class 1024

echo "[$(date '+%F %T')] evaluating small table"
$PY scripts/run_all_avg_eval.py \
  --method my_merge \
  --task-type small \
  --datasets "${DATASETS[@]}" \
  --small-models "${SMALL_MODELS[@]}" \
  --num-clients "${CLIENTS[@]}" \
  --betas "${BETAS[@]}" \
  --num-workers 4 \
  --small-batch-size 128 \
  --device cuda:0 \
  --output-root "$OUT_ROOT" \
  --my-merge-prototype-root "$STATS_ROOT" \
  --my-merge-reference-head-mode cosine \
  --my-merge-reference-head-scale 20 \
  --my-merge-reference-prior-threshold 2.5 \
  --my-merge-reference-prior-max-tau 6 \
  --my-merge-reference-prior-saturation 3

echo "[$(date '+%F %T')] evaluating vlm table"
$PY scripts/run_all_avg_eval.py \
  --method my_merge \
  --task-type vlm \
  --datasets "${DATASETS[@]}" \
  --clip-models openai/clip-vit-base-patch32 \
  --num-clients "${CLIENTS[@]}" \
  --betas "${BETAS[@]}" \
  --num-workers 4 \
  --vlm-batch-size 64 \
  --device cuda:0 \
  --output-root "$OUT_ROOT"

echo "[$(date '+%F %T')] generating my_merge table"
$PY scripts/generate_my_merge_master_table.py --output-root "$OUT_ROOT" --dest "$MINE"

echo "[$(date '+%F %T')] updating combined summary"
$PY scripts/generate_combined_results_table.py --base result/all_results.md --mine "$MINE" --dest "$DEST"

echo "[$(date '+%F %T')] analyzing summary"
$PY scripts/analyze_my_merge_results.py --table "$DEST" > My_merge_ret/full_table_analysis_20260705.txt

echo "[$(date '+%F %T')] full-table run done"
