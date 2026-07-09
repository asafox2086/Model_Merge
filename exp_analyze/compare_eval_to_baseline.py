#!/usr/bin/env python3
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_combined_results_table import parse_tables, try_parse_number
from generate_lamp_merge_master_table import SETTINGS, SMALL_DATASETS


FIELDS = [
    "status",
    "method",
    "task_type",
    "dataset",
    "model",
    "num_clients",
    "beta",
    "actual",
    "expected",
    "delta",
    "csv_path",
]


def parse_args():
    p = argparse.ArgumentParser("Compare reproduced eval_summary.csv rows against result/all_results.md")
    p.add_argument("--base", default="result/all_results.md", help="Baseline master table")
    p.add_argument("--output-root", nargs="+", required=True, help="One or more output roots to scan")
    p.add_argument("--methods", nargs="*", default=None, help="Methods to compare; default compares all rows found")
    p.add_argument("--tolerance", type=float, default=5e-5)
    p.add_argument("--dest-md", default="")
    p.add_argument("--dest-csv", default="")
    p.add_argument("--max-report-rows", type=int, default=80)
    p.add_argument("--no-strict", action="store_true", help="Do not return non-zero on mismatches")
    return p.parse_args()


def setting_index(num_clients, beta):
    for idx, (clients, item_beta, _label) in enumerate(SETTINGS):
        if int(num_clients) == int(clients) and abs(float(beta) - float(item_beta)) <= 1e-12:
            return idx
    return None


def baseline_lookup(path):
    _intro, tables = parse_tables(Path(path))
    lookup = {}
    for (section, model_name, kind), table in tables.items():
        if kind != "Raw":
            continue
        task_type = "small" if section == "Small" else "vlm"
        for row in table["rows"]:
            method = row["method"]
            for dataset_idx, dataset in enumerate(SMALL_DATASETS):
                for setting_idx, (num_clients, beta, _label) in enumerate(SETTINGS):
                    col_idx = dataset_idx * len(SETTINGS) + setting_idx
                    if col_idx >= len(row["values"]):
                        continue
                    value = try_parse_number(row["values"][col_idx])
                    if value is None:
                        continue
                    key = (method, task_type, dataset, model_name, int(num_clients), float(beta))
                    lookup[key] = value
    return lookup


def discover_eval_csvs(output_roots):
    paths = set()
    for root in output_roots:
        root = Path(root)
        if root.is_file() and root.name == "eval_summary.csv":
            paths.add(root)
        elif root.exists():
            paths.update(root.rglob("reports/eval_summary.csv"))
    return sorted(path for path in paths if path.is_file())


def model_candidates(row):
    if row["task_type"] == "small":
        return [row.get("model", "")]
    candidates = []
    for value in (row.get("clip_model", ""), row.get("model", "")):
        if value and value not in candidates:
            candidates.append(value)
        if value and "/" not in value:
            prefixed = f"openai/{value}"
            if prefixed not in candidates:
                candidates.append(prefixed)
    return candidates


def compare_row(row, csv_path, lookup, methods, tolerance):
    method = row.get("method", "")
    if methods and method not in methods:
        return None
    try:
        task_type = row["task_type"]
        dataset = row["dataset"]
        num_clients = int(float(row["num_clients"]))
        beta = float(row["beta"])
        actual = float(row["test_acc"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "invalid_row",
            "method": method,
            "task_type": row.get("task_type", ""),
            "dataset": row.get("dataset", ""),
            "model": row.get("model") or row.get("clip_model", ""),
            "num_clients": row.get("num_clients", ""),
            "beta": row.get("beta", ""),
            "actual": row.get("test_acc", ""),
            "expected": "",
            "delta": "",
            "csv_path": str(csv_path),
        }

    idx = setting_index(num_clients, beta)
    model_name = ""
    expected = None
    if idx is not None:
        for candidate in model_candidates(row):
            key = (method, task_type, dataset, candidate, num_clients, beta)
            if key in lookup:
                model_name = candidate
                expected = lookup[key]
                break
    if not model_name:
        model_name = row.get("model") if task_type == "small" else row.get("clip_model") or row.get("model", "")

    if idx is None:
        status = "unknown_setting"
        delta = ""
    elif expected is None:
        status = "missing_baseline"
        delta = ""
    else:
        delta_value = actual - expected
        status = "ok" if abs(delta_value) <= tolerance else "mismatch"
        delta = f"{delta_value:.12g}"

    return {
        "status": status,
        "method": method,
        "task_type": task_type,
        "dataset": dataset,
        "model": model_name,
        "num_clients": str(num_clients),
        "beta": f"{beta:g}",
        "actual": f"{actual:.12g}",
        "expected": "" if expected is None else f"{expected:.12g}",
        "delta": delta,
        "csv_path": str(csv_path),
    }


def load_comparisons(output_roots, lookup, methods, tolerance):
    rows = []
    for csv_path in discover_eval_csvs(output_roots):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                item = compare_row(row, csv_path, lookup, methods, tolerance)
                if item is not None:
                    rows.append(item)
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def render_markdown(rows, output_roots, base, tolerance, max_rows):
    counts = Counter(row["status"] for row in rows)
    by_method = Counter((row["method"], row["status"]) for row in rows)
    lines = [
        "# Reproduction Check",
        "",
        f"- Baseline: `{base}`",
        f"- Output roots: `{', '.join(str(root) for root in output_roots)}`",
        f"- Tolerance: `{tolerance:g}`",
        "",
        "## Status",
        "",
        "| status | rows |",
        "| --- | --- |",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(["", "## By Method", "", "| method | status | rows |", "| --- | --- | --- |"])
    for (method, status), count in sorted(by_method.items()):
        lines.append(f"| {method} | {status} | {count} |")

    bad_rows = [row for row in rows if row["status"] != "ok"]
    if bad_rows:
        lines.extend(
            [
                "",
                f"## Non-OK Rows (first {max_rows})",
                "",
                "| status | method | task | dataset | model | c | beta | actual | expected | delta |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in bad_rows[:max_rows]:
            lines.append(
                "| {status} | {method} | {task_type} | {dataset} | {model} | {num_clients} | {beta} | {actual} | {expected} | {delta} |".format(
                    **row
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    output_roots = [Path(item) for item in args.output_root]
    lookup = baseline_lookup(args.base)
    rows = load_comparisons(output_roots, lookup, set(args.methods or []), args.tolerance)
    if args.dest_csv:
        write_csv(args.dest_csv, rows)
    if args.dest_md:
        path = Path(args.dest_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_markdown(rows, output_roots, args.base, args.tolerance, args.max_report_rows),
            encoding="utf-8",
        )

    counts = Counter(row["status"] for row in rows)
    print(f"checked={len(rows)} " + " ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    has_bad = any(row["status"] != "ok" for row in rows)
    if has_bad and not args.no_strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
