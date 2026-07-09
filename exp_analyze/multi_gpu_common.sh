#!/usr/bin/env bash

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ROOT_DIR="$(cd "${COMMON_DIR}/.." && pwd)"

ROOT_DIR="${ROOT_DIR:-${COMMON_ROOT_DIR}}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${HOME}/.conda/envs/MM/bin/python" ]]; then
    PYTHON_BIN="${HOME}/.conda/envs/MM/bin/python"
  elif [[ -x "/data2/liyapeng_grp/.conda/envs/MM/bin/python" ]]; then
    PYTHON_BIN="/data2/liyapeng_grp/.conda/envs/MM/bin/python"
  elif [[ -x "${ROOT_DIR}/.gpuenv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.gpuenv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-${ROOT_DIR}/model_hub}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Med_data}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0}"
DEVICE="${DEVICE:-cuda:0}"

NUM_WORKERS="${NUM_WORKERS:-0}"
STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-0}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-64}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-32}"

DENSITY="${DENSITY:-0.5}"
DARE_SEED="${DARE_SEED:-42}"
FISHER_EPS="${FISHER_EPS:-1e-8}"
FISHER_MAX_BATCHES="${FISHER_MAX_BATCHES:-1}"
REGMEAN_EPS="${REGMEAN_EPS:-1e-6}"
REGMEAN_MAX_BATCHES="${REGMEAN_MAX_BATCHES:-1}"
REGMEAN_MAX_DIM="${REGMEAN_MAX_DIM:-1024}"
STATS_SPLIT="${STATS_SPLIT:-val}"
STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-32}"
ISO_COMMON_SPACE_FRACTION="${ISO_COMMON_SPACE_FRACTION:-0.8}"
FREE_FILTER_RATIO="${FREE_FILTER_RATIO:-0.7}"
FREE_SCALING="${FREE_SCALING:-1.0}"
ROBUSTMERGE_MASK_RATIO="${ROBUSTMERGE_MASK_RATIO:-0.2}"
ROBUSTMERGE_ATT_RATIO="${ROBUSTMERGE_ATT_RATIO:-0.2}"
ROBUSTMERGE_FUSE_WEIGHT="${ROBUSTMERGE_FUSE_WEIGHT:-2.0}"

RESUME_FLAG="${RESUME_FLAG:-}"
DELETE_FLAG="${DELETE_FLAG:-}"

DATASETS=( ${DATASETS:-bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224} )
SMALL_MODELS=( ${SMALL_MODELS:-resnet convnext vit_t swin_tiny} )
CLIP_MODELS=( ${CLIP_MODELS:-openai/clip-vit-base-patch32} )

FORMAL_METHODS_DEFAULT=( avg ties dare_linear dare_ties regmean fisher breadcrumbs model_stock from iso_c free_merge robustmerge )
ALL_SUPPORTED_METHODS=( avg ties dare_linear dare_ties regmean fisher breadcrumbs model_stock from iso_c iso_cts free_merge robustmerge adamerging lamp_merge )

export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

REPRO_MODE="${REPRO_MODE:-1}"
PREPARE_REFERENCE_CACHE="${PREPARE_REFERENCE_CACHE:-false}"
case "${REPRO_MODE,,}" in
  0|false|no|off)
    HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-0}"
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
    ;;
  *)
    HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    ;;
esac
export HF_LOCAL_FILES_ONLY HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"
mkdir -p "${TMPDIR}"
export TMPDIR
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

build_method_args() {
  local method="$1"
  local -n out_ref="$2"
  out_ref=(--merge-weight-mode equal)
  case "${method}" in
    avg|ties|breadcrumbs|model_stock|from|iso_c|free_merge|robustmerge)
      ;;
    lamp_merge)
      out_ref+=(
        --stats-split "${STATS_SPLIT}"
        --stats-batch-size "${STATS_BATCH_SIZE}"
        --stats-num-workers "${STATS_NUM_WORKERS}"
      )
      ;;
    dare_linear|dare_ties)
      out_ref+=(--density "${DENSITY}" --dare-seed "${DARE_SEED}")
      ;;
    regmean)
      out_ref+=(--stats-split "${STATS_SPLIT}" --stats-batch-size "${STATS_BATCH_SIZE}" --stats-num-workers "${STATS_NUM_WORKERS}" --regmean-max-batches "${REGMEAN_MAX_BATCHES}" --regmean-max-dim "${REGMEAN_MAX_DIM}" --regmean-eps "${REGMEAN_EPS}")
      ;;
    fisher)
      out_ref+=(--stats-split "${STATS_SPLIT}" --stats-batch-size "${STATS_BATCH_SIZE}" --stats-num-workers "${STATS_NUM_WORKERS}" --fisher-max-batches "${FISHER_MAX_BATCHES}" --fisher-eps "${FISHER_EPS}")
      ;;
    *)
      echo "Unsupported method: ${method}" >&2
      return 1
      ;;
  esac

  [[ "${method}" == "breadcrumbs" ]] && out_ref+=(--breadcrumbs-top-k-keep 0.2 --breadcrumbs-top-k-remove 0.1 --breadcrumbs-alpha 1.0)
  [[ "${method}" == "model_stock" ]] && out_ref+=(--model-stock-k 2.0)
  [[ "${method}" == "from" ]] && out_ref+=(--from-k 1.0)
  [[ "${method}" == "iso_c" ]] && out_ref+=(--iso-common-space-fraction "${ISO_COMMON_SPACE_FRACTION}")
  [[ "${method}" == "free_merge" ]] && out_ref+=(--free-filter-ratio "${FREE_FILTER_RATIO}" --free-scaling "${FREE_SCALING}")
  [[ "${method}" == "robustmerge" ]] && out_ref+=(--robustmerge-mask-ratio "${ROBUSTMERGE_MASK_RATIO}" --robustmerge-att-ratio "${ROBUSTMERGE_ATT_RATIO}" --robustmerge-fuse-weight "${ROBUSTMERGE_FUSE_WEIGHT}")
  return 0
}

build_boolean_flag_args() {
  local value="${1-}"
  local true_flag="$2"
  local false_flag="$3"
  local -n out_ref="$4"
  case "${value,,}" in
    0|false|no|off|--no-resume|--no-delete-merged)
      out_ref=("${false_flag}")
      ;;
    1|true|yes|on|--resume|--delete-merged|"")
      out_ref=("${true_flag}")
      ;;
    *)
      out_ref=("${value}")
      ;;
  esac
}

validate_methods() {
  local method
  for method in "$@"; do
    case "${method}" in
      avg|ties|dare_linear|dare_ties|regmean|fisher|breadcrumbs|model_stock|from|iso_c|iso_cts|free_merge|robustmerge|adamerging|lamp_merge)
        ;;
      *)
        echo "Unsupported method: ${method}" >&2
        return 1
        ;;
    esac
  done
}

prepare_reference_cache() {
  local task_type="$1"
  if [[ "${PREPARE_REFERENCE_CACHE}" != "true" ]]; then
    echo "[$(date +%F\ %T)] skipping reference cache preparation | task_type=${task_type}"
    return 0
  fi
  local -a cmd=(
    "${PYTHON_BIN}" "${COMMON_DIR}/cache_reference_models.py"
    --model-hub-root "${MODEL_HUB_ROOT}"
    --task-type "${task_type}"
    --datasets "${DATASETS[@]}"
    --small-models "${SMALL_MODELS[@]}"
    --clip-models "${CLIP_MODELS[@]}"
  )
  echo "[$(date +%F\ %T)] preparing local reference cache | task_type=${task_type}"
  local old_hf_local="${HF_LOCAL_FILES_ONLY}"
  local old_hf_offline="${HF_HUB_OFFLINE}"
  local old_tf_offline="${TRANSFORMERS_OFFLINE}"
  export HF_LOCAL_FILES_ONLY=0
  export HF_HUB_OFFLINE=0
  export TRANSFORMERS_OFFLINE=0
  "${cmd[@]}"
  export HF_LOCAL_FILES_ONLY="${old_hf_local}"
  export HF_HUB_OFFLINE="${old_hf_offline}"
  export TRANSFORMERS_OFFLINE="${old_tf_offline}"
}
