#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

while pgrep -f "codex_my_merge_small_a4|codex_my_merge_vlm|outputs/custom_methods_codex_my_merge_small_a4|outputs/custom_methods_codex_my_merge_vlm" >/dev/null; do
  sleep 60
done

python3 scripts/generate_my_merge_master_table.py \
  --output-root \
  outputs/custom_methods_codex_my_merge_small_a4 \
  outputs/custom_methods_codex_my_merge_small_b5 \
  outputs/custom_methods_codex_my_merge_vlm \
  --dest My_merge_ret/all_results_my_merge.md
