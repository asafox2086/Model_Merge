#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_combined_results_table import parse_tables, try_parse_number
from generate_my_merge_master_table import SETTINGS, SMALL_DATASETS


RAW_KIND = "Raw"


def parse_args():
    p = argparse.ArgumentParser("Summarize my_merge ablation grids")
    p.add_argument("--grid-root", required=True, help="Root containing one subdirectory per ablation preset")
    p.add_argument("--baseline", default="result/all_results.md", help="Original result table used for best-method comparison")
    p.add_argument("--dest", required=True, help="Destination markdown path")
    p.add_argument("--method", default="my_merge", help="Method name to include from eval_summary.csv")
    return p.parse_args()


def fmt(value):
    return "-" if value is None else f"{value:.4f}"


def key_for_row(row):
    task_type = row["task_type"]
    model_name = row["model"] if task_type == "small" else row["clip_model"]
    return (
        task_type,
        row["dataset"],
        model_name,
        int(float(row["num_clients"])),
        float(row["beta"]),
    )


def beta_token(beta):
    return str(float(beta)).replace(".", "p")


def recover_eval_row(csv_path, row):
    if str(row.get("test_acc", "")).strip():
        return row
    task_type = row.get("task_type", "")
    dataset = row.get("dataset", "")
    model_name = row.get("model") if task_type == "small" else row.get("clip_model")
    if not all([task_type, dataset, model_name, row.get("num_clients"), row.get("beta"), row.get("seed"), row.get("method")]):
        return row
    output_root = csv_path.parent.parent
    eval_json = (
        output_root
        / "eval"
        / task_type
        / dataset
        / model_name
        / f"clients_{int(float(row['num_clients']))}"
        / f"beta_{beta_token(row['beta'])}"
        / f"seed_{int(float(row['seed']))}"
        / row["method"]
        / "eval.json"
    )
    if not eval_json.exists():
        return row
    with eval_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    recovered = dict(row)
    for field in ("test_acc", "test_loss", "num_samples", "source_image_size", "eval_image_size", "image_resize"):
        if field in payload:
            recovered[field] = payload[field]
    return recovered


def discover_eval_csvs(grid_root):
    grid_root = Path(grid_root)
    paths = sorted(set(grid_root.glob("*/reports/eval_summary.csv")) | set(grid_root.glob("*/*/reports/eval_summary.csv")))
    return [path for path in paths if path.is_file()]


def ablation_name(grid_root, csv_path):
    rel = csv_path.parent.parent.relative_to(grid_root)
    return rel.parts[0] if rel.parts else "unknown"


def load_ablation_rows(grid_root, method):
    rows_by_ablation = defaultdict(list)
    for csv_path in discover_eval_csvs(grid_root):
        name = ablation_name(Path(grid_root), csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("method") == method:
                    row = recover_eval_row(csv_path, row)
                    if str(row.get("test_acc", "")).strip():
                        rows_by_ablation[name].append(row)
    return rows_by_ablation


def baseline_best_lookup(path):
    _intro, tables = parse_tables(Path(path))
    lookup = {}
    raw_labels = [label for _clients, _beta, label in SETTINGS]
    for (section, model_name, kind), table in tables.items():
        if kind != RAW_KIND:
            continue
        task_type = "small" if section == "Small" else "vlm"
        for dataset_idx, dataset in enumerate(SMALL_DATASETS):
            for setting_idx, (num_clients, beta, _label) in enumerate(SETTINGS):
                col_idx = dataset_idx * len(raw_labels) + setting_idx
                values = []
                for row in table["rows"]:
                    if col_idx >= len(row["values"]):
                        continue
                    value = try_parse_number(row["values"][col_idx])
                    if value is not None:
                        values.append(value)
                if values:
                    lookup[(task_type, dataset, model_name, num_clients, float(beta))] = max(values)
    return lookup


def build_full_lookup(rows_by_ablation):
    full_rows = rows_by_ablation.get("full", [])
    return {key_for_row(row): float(row["test_acc"]) for row in full_rows if row.get("test_acc") not in (None, "")}


def summarize_rows(rows, baseline_lookup, full_lookup):
    acc_values = []
    delta_baseline = []
    delta_full = []
    wins = ties = losses = 0
    by_dataset = defaultdict(lambda: {"n": 0, "delta_full": [], "wins": 0, "ties": 0, "losses": 0})

    for row in rows:
        key = key_for_row(row)
        acc = float(row["test_acc"])
        acc_values.append(acc)
        base = baseline_lookup.get(key)
        if base is not None:
            delta = acc - base
            delta_baseline.append(delta)
            if delta > 5e-5:
                wins += 1
                by_dataset[row["dataset"]]["wins"] += 1
            elif abs(delta) <= 5e-5:
                ties += 1
                by_dataset[row["dataset"]]["ties"] += 1
            else:
                losses += 1
                by_dataset[row["dataset"]]["losses"] += 1
        full = full_lookup.get(key)
        if full is not None:
            item_delta = acc - full
            delta_full.append(item_delta)
            by_dataset[row["dataset"]]["delta_full"].append(item_delta)
        by_dataset[row["dataset"]]["n"] += 1

    mean_acc = sum(acc_values) / len(acc_values) if acc_values else None
    mean_delta_baseline = sum(delta_baseline) / len(delta_baseline) if delta_baseline else None
    mean_delta_full = sum(delta_full) / len(delta_full) if delta_full else None
    return {
        "n": len(rows),
        "mean_acc": mean_acc,
        "mean_delta_baseline": mean_delta_baseline,
        "mean_delta_full": mean_delta_full,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "by_dataset": by_dataset,
    }


def render_table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")


def render_summary(rows_by_ablation, baseline_lookup):
    full_lookup = build_full_lookup(rows_by_ablation)
    summaries = {name: summarize_rows(rows, baseline_lookup, full_lookup) for name, rows in sorted(rows_by_ablation.items())}

    lines = [
        "# my_merge Ablation Summary",
        "",
        "- `delta_vs_full` compares each ablation against the `full` my_merge run on exactly matched task keys.",
        "- `W/T/L` compares each ablation against the best original method in `result/all_results.md`.",
        "- A useful medical component should usually lower performance when disabled, especially on its matched modality.",
        "",
        "## Overall",
        "",
    ]
    overall_rows = []
    for name, summary in summaries.items():
        overall_rows.append(
            [
                name,
                str(summary["n"]),
                fmt(summary["mean_acc"]),
                fmt(summary["mean_delta_full"]),
                fmt(summary["mean_delta_baseline"]),
                f'{summary["wins"]}/{summary["ties"]}/{summary["losses"]}',
            ]
        )
    render_table(
        lines,
        ["ablation", "rows", "mean_acc", "delta_vs_full", "delta_vs_best_original", "W/T/L"],
        overall_rows,
    )

    lines.extend(["", "## Dataset Breakdown", ""])
    dataset_rows = []
    for name, summary in summaries.items():
        for dataset in SMALL_DATASETS:
            item = summary["by_dataset"].get(dataset)
            if not item:
                continue
            deltas = item["delta_full"]
            mean_delta = sum(deltas) / len(deltas) if deltas else None
            short_name = re.sub(r"_224$", "", dataset)
            dataset_rows.append(
                [
                    name,
                    short_name,
                    str(item["n"]),
                    fmt(mean_delta),
                    f'{item["wins"]}/{item["ties"]}/{item["losses"]}',
                ]
            )
    render_table(lines, ["ablation", "dataset", "rows", "delta_vs_full", "W/T/L"], dataset_rows)
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    grid_root = Path(args.grid_root)
    rows_by_ablation = load_ablation_rows(grid_root, args.method)
    baseline_lookup = baseline_best_lookup(args.baseline)
    content = render_summary(rows_by_ablation, baseline_lookup)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
