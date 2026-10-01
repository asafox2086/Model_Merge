#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/archive_outputs.sh"

PYTHON_BIN="${PYTHON_BIN:-/data2/liyapeng_grp/.conda/envs/MM/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/lamp_merge_full_evidence_queue_${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

ABLATION_TAG="${ABLATION_TAG:-$(cat "${ROOT_DIR}/logs/lamp_merge_internal_ablation_full_current_tag.txt" 2>/dev/null || echo 20260707_full_internal_ablation_v5_tmux_fullscope)}"
ABLATION_ROOT="${ABLATION_ROOT:-${ROOT_DIR}/outputs/lamp_merge_internal_ablation_full_${ABLATION_TAG}}"
ABLATION_LOG="${ABLATION_LOG:-${ROOT_DIR}/logs/lamp_merge_internal_ablation_full_${ABLATION_TAG}.tmux.log}"

DIAG_TAG="${DIAG_TAG:-${RUN_TAG}}"
HPARAM_TAG="${HPARAM_TAG:-${RUN_TAG}}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-64}"

wait_for_ablation() {
  echo "[$(date '+%F %T')] waiting for full internal ablation root=${ABLATION_ROOT}"
  while tmux has-session -t lamp_internal_full 2>/dev/null; do
    local count
    count="$(find "${ABLATION_ROOT}" -path '*/eval.json' -print 2>/dev/null | wc -l | tr -d ' ')"
    echo "[$(date '+%F %T')] lamp_internal_full still running; eval_json=${count}"
    sleep 600
  done
  echo "[$(date '+%F %T')] lamp_internal_full session ended"
  if [[ -f "${ABLATION_LOG}" ]]; then
    tail -80 "${ABLATION_LOG}"
  fi
}

summarize_ablation() {
  echo "[$(date '+%F %T')] summarizing full internal ablation"
  archive_existing_outputs \
    "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full.csv" \
    "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full_client_average.csv" \
    "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full_by_dataset.csv" \
    "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full_summary.md"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_lamp_merge_internal_ablation_full.py" \
    --root "${ABLATION_ROOT}" \
    --baseline-table "${ROOT_DIR}/My_merge_ret/汇总表.md" \
    --output-csv "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full.csv" \
    --client-average-csv "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full_client_average.csv" \
    --dataset-csv "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full_by_dataset.csv" \
    --summary-md "${ROOT_DIR}/My_merge_ret/reports/lamp_merge_internal_ablation_full_summary.md"
}

run_prediction_diagnostics() {
  echo "[$(date '+%F %T')] running full prediction diagnostics"
  RUN_TAG="${DIAG_TAG}" GPU_IDS="${GPU_IDS}" NUM_WORKERS="${NUM_WORKERS}" BATCH_SIZE="${SMALL_BATCH_SIZE}" \
    "${SCRIPT_DIR}/run_lamp_merge_prediction_diagnostics_full_parallel.sh"
}

run_hparam_scan() {
  echo "[$(date '+%F %T')] running full hyperparameter scan"
  RUN_TAG="${HPARAM_TAG}" GPU_IDS="${GPU_IDS}" NUM_WORKERS="${NUM_WORKERS}" SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE}" \
    "${SCRIPT_DIR}/run_lamp_merge_hparam_full_parallel.sh"
}

run_prototype_geometry() {
  echo "[$(date '+%F %T')] running full prototype geometry and t-SNE analysis"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_lamp_merge_prototype_geometry.py" \
    --device "cuda:${GPU_IDS%% *}" \
    --batch-size "${SMALL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}"
}

echo "[$(date '+%F %T')] LAMP-Merge full evidence queue"
echo "run_tag=${RUN_TAG}"
echo "ablation_root=${ABLATION_ROOT}"
echo "gpu_ids=${GPU_IDS}"

wait_for_ablation
summarize_ablation
run_prediction_diagnostics
run_hparam_scan
run_prototype_geometry

echo "[$(date '+%F %T')] full evidence queue complete"
