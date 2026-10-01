#!/usr/bin/env bash

archive_existing_outputs() {
  local tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
  local archive_root="${ARCHIVE_ROOT:-${ROOT_DIR}/My_merge_ret/archive/full_outputs/${tag}}"
  local src rel dest suffix

  for src in "$@"; do
    [[ -e "${src}" ]] || continue
    rel="$(realpath --relative-to="${ROOT_DIR}" "${src}" 2>/dev/null || basename "${src}")"
    dest="${archive_root}/${rel}"
    mkdir -p "$(dirname "${dest}")"
    if [[ -e "${dest}" ]]; then
      suffix="$(date +%H%M%S)"
      dest="${dest}.${suffix}"
    fi
    cp -a "${src}" "${dest}"
    echo "[$(date '+%F %T')] archived existing output: ${rel} -> ${dest}"
  done
}
