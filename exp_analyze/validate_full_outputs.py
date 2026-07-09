#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - the MM environment provides torch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "My_merge_ret" / "reports"
FIGURE_DIR = ROOT / "My_merge_ret" / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lamp_merge_stats import default_prototype_root, validate_lamp_merge_stats_payload

DEFAULT_PROTOTYPE_ROOT = default_prototype_root(ROOT)

EXPECTED_DATASETS = 5
EXPECTED_MODELS = 4
EXPECTED_CLIENT_COUNTS = 3
EXPECTED_BETAS = 3
EXPECTED_RAW_CELLS = EXPECTED_DATASETS * EXPECTED_MODELS * EXPECTED_CLIENT_COUNTS * EXPECTED_BETAS
EXPECTED_CLIENT_AVG_CELLS = EXPECTED_DATASETS * EXPECTED_MODELS * EXPECTED_CLIENT_COUNTS
EXPECTED_INTERNAL_MODES = 13
EXPECTED_HPARAM_VALUES = 23
EXPECTED_DIAGNOSTIC_METHODS = 25
EXPECTED_DIAGNOSTIC_CLIENT_ROWS = EXPECTED_DATASETS * EXPECTED_MODELS * EXPECTED_BETAS * (3 + 5 + 7)
EXPECTED_DIAGNOSTIC_ROWS = EXPECTED_RAW_CELLS * EXPECTED_DIAGNOSTIC_METHODS + EXPECTED_DIAGNOSTIC_CLIENT_ROWS


class CheckResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def require_file(result: CheckResult, path: Path) -> bool:
    result.require(path.exists(), f"missing file: {path.relative_to(ROOT)}")
    result.require(not path.exists() or path.stat().st_size > 0, f"empty file: {path.relative_to(ROOT)}")
    return path.exists() and path.stat().st_size > 0


def require_m2_prevalence_uploads(result: CheckResult) -> None:
    """Require the client-side class-prevalence upload used by M2.

    Older prototype-stat exports only contain M1 prototype evidence. They can
    reproduce M1, but they cannot validate the final M1+M2 LAMP-Merge method.
    """

    result.require(DEFAULT_PROTOTYPE_ROOT.exists(), f"missing prototype stats root: {DEFAULT_PROTOTYPE_ROOT.relative_to(ROOT)}")
    if not DEFAULT_PROTOTYPE_ROOT.exists():
        return
    paths = sorted(DEFAULT_PROTOTYPE_ROOT.glob("small/*/*/clients_*/*/seed_*/prototype_stats.pt"))
    result.require(len(paths) >= EXPECTED_RAW_CELLS, f"prototype stats files={len(paths)}, expected at least {EXPECTED_RAW_CELLS}")
    if torch is None:
        result.require(False, "torch is required to validate prototype_stats.pt files")
        return

    for path in paths:
        try:
            stats = torch.load(path, map_location="cpu")
            validate_lamp_merge_stats_payload(stats, path)
        except Exception as exc:  # pragma: no cover - defensive validation.
            result.require(False, f"invalid prototype stats: {path.relative_to(ROOT)} ({exc})")
            return


def check_main_table() -> CheckResult:
    result = CheckResult("main_table")
    path = ROOT / "My_merge_ret" / "汇总表.md"
    if require_file(result, path):
        text = path.read_text(encoding="utf-8", errors="replace")
        result.require("LAMP-Merge" in text, "main table does not contain LAMP-Merge row")
        result.require(text.count("#### Client Average") >= EXPECTED_MODELS, "main table is missing client-average sections")
    return result


def check_internal_ablation() -> CheckResult:
    result = CheckResult("internal_ablation")
    require_m2_prevalence_uploads(result)
    summary = REPORT_DIR / "lamp_merge_internal_ablation_full_summary.md"
    overall = REPORT_DIR / "lamp_merge_internal_ablation_full.csv"
    client_avg = REPORT_DIR / "lamp_merge_internal_ablation_full_client_average.csv"
    by_dataset = REPORT_DIR / "lamp_merge_internal_ablation_full_by_dataset.csv"
    for path in (summary, overall, client_avg, by_dataset):
        require_file(result, path)

    fields, rows = read_csv(overall)
    result.require(len(rows) == EXPECTED_INTERNAL_MODES, f"internal summary rows={len(rows)}, expected={EXPECTED_INTERNAL_MODES}")
    for field in ("mode", "label", "raw_cells", "client_average_cells", "client_average_mean_acc"):
        result.require(field in fields, f"internal summary missing column: {field}")
    if rows:
        incomplete = [row.get("label") or row.get("mode") for row in rows if int(float(row.get("raw_cells") or 0)) != EXPECTED_RAW_CELLS]
        result.require(not incomplete, f"internal ablation has incomplete raw cells: {incomplete[:5]}")

    fields, rows = read_csv(client_avg)
    result.require(len(rows) == EXPECTED_CLIENT_AVG_CELLS, f"internal client-average rows={len(rows)}, expected={EXPECTED_CLIENT_AVG_CELLS}")
    for field in ("dataset", "model", "num_clients", "LAMP-Merge", "M1 only", "avg+M2"):
        result.require(field in fields, f"internal client-average missing column: {field}")
    return result


def check_prototype_geometry() -> CheckResult:
    result = CheckResult("prototype_geometry")
    full_csv = REPORT_DIR / "lamp_merge_prototype_geometry_full.csv"
    dataset_csv = REPORT_DIR / "lamp_merge_prototype_geometry_by_dataset.csv"
    summary = REPORT_DIR / "lamp_merge_prototype_geometry_summary.md"
    for path in (full_csv, dataset_csv, summary):
        require_file(result, path)

    fields, rows = read_csv(full_csv)
    result.require(len(rows) >= EXPECTED_RAW_CELLS, f"prototype geometry rows={len(rows)}, expected at least {EXPECTED_RAW_CELLS}")
    for field in ("dataset", "model", "num_clients", "beta", "label", "mean_pairwise_cosine"):
        result.require(field in fields, f"prototype geometry missing column: {field}")

    figure_root = FIGURE_DIR / "lamp_merge_prototype_geometry"
    result.require(figure_root.exists(), f"missing figure dir: {figure_root.relative_to(ROOT)}")
    figures = list(figure_root.glob("*.png")) if figure_root.exists() else []
    result.require(len(figures) >= EXPECTED_DATASETS, f"prototype geometry figures={len(figures)}, expected at least {EXPECTED_DATASETS}")
    return result


def check_prediction_diagnostics() -> CheckResult:
    result = CheckResult("prediction_diagnostics")
    require_m2_prevalence_uploads(result)
    metrics = REPORT_DIR / "prediction_diagnostics_full.csv"
    summary_dir = REPORT_DIR / "prediction_diagnostics_full"
    figure_dir = FIGURE_DIR / "prediction_diagnostics_full"
    require_file(result, metrics)
    result.require(summary_dir.exists(), f"missing summary dir: {summary_dir.relative_to(ROOT)}")
    result.require(figure_dir.exists(), f"missing figure dir: {figure_dir.relative_to(ROOT)}")

    fields, rows = read_csv(metrics)
    result.require(len(rows) >= EXPECTED_DIAGNOSTIC_ROWS, f"prediction diagnostic rows={len(rows)}, expected at least {EXPECTED_DIAGNOSTIC_ROWS}")
    for field in ("dataset", "model", "num_clients", "beta", "method", "accuracy", "balanced_accuracy", "macro_f1", "collapse_ratio"):
        result.require(field in fields, f"prediction diagnostics missing column: {field}")
    return result


def check_hparam_full() -> CheckResult:
    result = CheckResult("hparam_full")
    require_m2_prevalence_uploads(result)
    overall = REPORT_DIR / "lamp_merge_hparam_full.csv"
    client_avg = REPORT_DIR / "lamp_merge_hparam_full_client_average.csv"
    by_dataset = REPORT_DIR / "lamp_merge_hparam_full_by_dataset.csv"
    summary = REPORT_DIR / "lamp_merge_hparam_full_summary.md"
    figure = FIGURE_DIR / "lamp_merge_hparam_full_sensitivity.png"
    for path in (overall, client_avg, by_dataset, summary, figure):
        require_file(result, path)

    fields, rows = read_csv(overall)
    result.require(len(rows) == EXPECTED_HPARAM_VALUES, f"hparam summary rows={len(rows)}, expected={EXPECTED_HPARAM_VALUES}")
    for field in ("module", "parameter", "symbol", "value", "raw_cells", "client_average_cells"):
        result.require(field in fields, f"hparam summary missing column: {field}")
    if rows:
        incomplete = [f"{row.get('symbol')}={row.get('value')}" for row in rows if int(float(row.get("raw_cells") or 0)) != EXPECTED_RAW_CELLS]
        result.require(not incomplete, f"hparam has incomplete raw cells: {incomplete[:8]}")

    fields, rows = read_csv(client_avg)
    expected_rows = EXPECTED_HPARAM_VALUES * EXPECTED_CLIENT_AVG_CELLS
    result.require(len(rows) == expected_rows, f"hparam client-average rows={len(rows)}, expected={expected_rows}")
    return result


CHECKS = {
    "main_table": check_main_table,
    "internal_ablation": check_internal_ablation,
    "prototype_geometry": check_prototype_geometry,
    "prediction_diagnostics": check_prediction_diagnostics,
    "hparam_full": check_hparam_full,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Validate full-scope LAMP-Merge experiment outputs.")
    parser.add_argument("--step", choices=sorted(CHECKS), action="append", default=[])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = args.step or list(CHECKS)
    results = [CHECKS[name]() for name in names]
    for result in results:
        status = "OK" if result.ok else "MISSING"
        if not args.quiet:
            print(f"[{status}] {result.name}")
            for message in result.errors:
                print(f"  error: {message}")
            for message in result.warnings:
                print(f"  warning: {message}")
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
