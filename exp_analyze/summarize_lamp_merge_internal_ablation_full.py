#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_combined_results_table import parse_tables, try_parse_number


FORMAL_SMALL_DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
]
FORMAL_SMALL_MODELS = ["resnet", "convnext", "vit_t", "swin_tiny"]
SETTINGS = [(3, 0.0), (3, 0.01), (3, 0.1), (5, 0.0), (5, 0.01), (5, 0.1), (7, 0.0), (7, 0.01), (7, 0.1)]

MODE_LABELS = {
    "full": "LAMP-Merge",
    "m1_only": "M1 only",
    "avg_m2": "avg+M2",
    "prototype_head_agg": "Classifier-head aggregation",
    "global_feature_mean": "Global-feature mean",
    "support_only": "Support-only synthetic head",
    "prototype_shuffle": "Shuffled-label prototype",
    "uniform_client_weight": "Uniform client weight",
    "binary_support": "Binary support only",
    "global_client_size_weight": "Global client-size weight",
    "no_prevalence": "No prevalence calibration",
    "uniform_prevalence": "Uniform prevalence prior",
    "smoothed_prevalence": "Smoothed prevalence prior",
}

MODE_GROUPS = {
    "full": "Formal method",
    "m1_only": "Module-level",
    "avg_m2": "Module-level",
    "prototype_head_agg": "Prototype information",
    "global_feature_mean": "Prototype information",
    "support_only": "Prototype information",
    "prototype_shuffle": "Prototype information",
    "uniform_client_weight": "Statistical information",
    "binary_support": "Statistical information",
    "global_client_size_weight": "Statistical information",
    "no_prevalence": "Statistical information",
    "uniform_prevalence": "Statistical information",
    "smoothed_prevalence": "Statistical information",
}

TAG_RE = re.compile(r"<.*?>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize full-scale LAMP-Merge internal ablations.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-table", type=Path, default=ROOT / "My_merge_ret" / "汇总表.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_internal_ablation_full.csv")
    parser.add_argument("--client-average-csv", type=Path, default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_internal_ablation_full_client_average.csv")
    parser.add_argument("--dataset-csv", type=Path, default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_internal_ablation_full_by_dataset.csv")
    parser.add_argument("--summary-md", type=Path, default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_internal_ablation_full_summary.md")
    return parser.parse_args()


def clean_method(value: object) -> str:
    return TAG_RE.sub("", str(value)).strip()


def raw_key(row: dict[str, str]) -> tuple[str, str, str, int, float]:
    model = row["model"] if row["task_type"] == "small" else row["clip_model"]
    return (row["task_type"], row["dataset"], model, int(float(row["num_clients"])), float(row["beta"]))


def client_average_key(key: tuple[str, str, str, int, float]) -> tuple[str, str, str, int]:
    task_type, dataset, model, num_clients, _beta = key
    return (task_type, dataset, model, num_clients)


def read_eval_lookup(root: Path) -> dict[tuple[str, str, str, int, float], float]:
    lookup: dict[tuple[str, str, str, int, float], float] = {}
    json_paths = sorted((root / "eval").glob("**/eval.json"))
    if json_paths:
        for path in json_paths:
            try:
                import json

                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if row.get("test_acc") is None:
                continue
            lookup[raw_key(row)] = float(row["test_acc"])
        return lookup

    path = root / "reports" / "eval_summary.csv"
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("test_acc"):
                    continue
                lookup[raw_key(row)] = float(row["test_acc"])
    return lookup


def load_avg_from_table(path: Path) -> dict[tuple[str, str, str, int, float], float]:
    _intro, tables = parse_tables(path)
    lookup: dict[tuple[str, str, str, int, float], float] = {}
    for (section, model, kind), table in tables.items():
        if section != "Small" or kind != "Raw" or model not in FORMAL_SMALL_MODELS:
            continue
        avg_row = None
        for row in table["rows"]:
            if clean_method(row["method"]) == "avg":
                avg_row = row
                break
        if avg_row is None:
            continue
        for ds_idx, dataset in enumerate(FORMAL_SMALL_DATASETS):
            for setting_idx, (num_clients, beta) in enumerate(SETTINGS):
                value_idx = ds_idx * len(SETTINGS) + setting_idx
                if value_idx >= len(avg_row["values"]):
                    continue
                value = try_parse_number(avg_row["values"][value_idx])
                if value is not None:
                    lookup[("small", dataset, model, num_clients, beta)] = float(value)
    return lookup


def build_client_average_lookup(raw_lookup: dict[tuple[str, str, str, int, float], float]) -> dict[tuple[str, str, str, int], float]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for key, value in raw_lookup.items():
        if key[0] == "small" and key[1] in FORMAL_SMALL_DATASETS and key[2] in FORMAL_SMALL_MODELS:
            grouped[client_average_key(key)].append(float(value))
    return {key: mean(values) for key, values in grouped.items() if len(values) == 3}


def fmt(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[dict[str, object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize_mode(mode: str, lookup: dict[tuple[str, str, str, int, float], float], avg_lookup, formal_lookup) -> dict[str, object]:
    formal_keys = {
        ("small", dataset, model, num_clients, beta)
        for dataset in FORMAL_SMALL_DATASETS
        for model in FORMAL_SMALL_MODELS
        for num_clients, beta in SETTINGS
    }
    values = [lookup[key] for key in formal_keys if key in lookup]
    comparable_avg = [key for key in formal_keys if key in lookup and key in avg_lookup]
    comparable_formal = [key for key in formal_keys if key in lookup and key in formal_lookup]
    client_lookup = build_client_average_lookup(lookup)
    avg_client = build_client_average_lookup(avg_lookup)
    formal_client = build_client_average_lookup(formal_lookup)
    client_keys = {
        ("small", dataset, model, num_clients)
        for dataset in FORMAL_SMALL_DATASETS
        for model in FORMAL_SMALL_MODELS
        for num_clients in [3, 5, 7]
    }
    client_values = [client_lookup[key] for key in client_keys if key in client_lookup]
    client_avg_comparable = [key for key in client_keys if key in client_lookup and key in avg_client]
    client_formal_comparable = [key for key in client_keys if key in client_lookup and key in formal_client]
    return {
        "mode": mode,
        "label": MODE_LABELS.get(mode, mode),
        "group": MODE_GROUPS.get(mode, ""),
        "raw_cells": len(values),
        "raw_mean_acc": mean(values) if values else "",
        "raw_min_acc": min(values) if values else "",
        "raw_max_acc": max(values) if values else "",
        "raw_ge_avg": sum(1 for key in comparable_avg if lookup[key] >= avg_lookup[key]),
        "raw_ge_avg_total": len(comparable_avg),
        "raw_ge_lamp": sum(1 for key in comparable_formal if lookup[key] >= formal_lookup[key]),
        "raw_ge_lamp_total": len(comparable_formal),
        "client_average_cells": len(client_values),
        "client_average_mean_acc": mean(client_values) if client_values else "",
        "client_average_ge_avg": sum(1 for key in client_avg_comparable if client_lookup[key] >= avg_client[key]),
        "client_average_ge_avg_total": len(client_avg_comparable),
        "client_average_ge_lamp": sum(1 for key in client_formal_comparable if client_lookup[key] >= formal_client[key]),
        "client_average_ge_lamp_total": len(client_formal_comparable),
        "mean_margin_vs_avg": (
            mean([client_lookup[key] - avg_client[key] for key in client_avg_comparable])
            if client_avg_comparable
            else ""
        ),
        "mean_margin_vs_lamp": (
            mean([client_lookup[key] - formal_client[key] for key in client_formal_comparable])
            if client_formal_comparable
            else ""
        ),
    }


def build_client_average_rows(mode_lookups, avg_lookup, formal_lookup):
    avg_client = build_client_average_lookup(avg_lookup)
    formal_client = build_client_average_lookup(formal_lookup)
    mode_client = {mode: build_client_average_lookup(lookup) for mode, lookup in mode_lookups.items()}
    all_keys = sorted({
        ("small", dataset, model, num_clients)
        for dataset in FORMAL_SMALL_DATASETS
        for model in FORMAL_SMALL_MODELS
        for num_clients in [3, 5, 7]
    })
    rows = []
    for key in all_keys:
        values = {"avg": avg_client.get(key), "LAMP-Merge": formal_client.get(key)}
        for mode, lookup in mode_client.items():
            values[MODE_LABELS.get(mode, mode)] = lookup.get(key)
        present = {name: value for name, value in values.items() if value is not None}
        if not present:
            continue
        best = max(present.values())
        best_methods = [name for name, value in present.items() if value >= best - 1e-12]
        task_type, dataset, model, num_clients = key
        row = {
            "task_type": task_type,
            "dataset": dataset,
            "model": model,
            "num_clients": num_clients,
            "best_value": best,
            "best_methods": ";".join(best_methods),
        }
        for name, value in values.items():
            row[name] = value if value is not None else ""
        rows.append(row)
    return rows


def build_dataset_rows(client_rows):
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in client_rows:
        grouped[str(row["dataset"])].append(row)
    method_names = sorted({key for row in client_rows for key in row.keys() if key not in {
        "task_type", "dataset", "model", "num_clients", "best_value", "best_methods"
    }})
    out = []
    for dataset, rows in sorted(grouped.items()):
        item = {"dataset": dataset, "client_average_cells": len(rows)}
        for method in method_names:
            values = [float(row[method]) for row in rows if row.get(method) not in {"", None}]
            item[f"{method}_mean_acc"] = mean(values) if values else ""
            item[f"{method}_best_or_tied"] = sum(1 for row in rows if method in str(row.get("best_methods", "")).split(";"))
        out.append(item)
    return out


def render_summary(root: Path, summary_rows, client_rows, dataset_rows) -> str:
    lines = [
        "# LAMP-Merge Full-Scale Internal Ablation Summary",
        "",
        f"Experiment root: `{root}`.",
        "",
        "Coverage: five medical image datasets, four small-model backbones, three client counts, and three Dirichlet beta values. Each ablation therefore contains 180 raw cells and 60 client-average cells when complete. The client-average metric first averages the three beta values for each fixed dataset, backbone, and client count.",
        "",
        "This table is a module-internal ablation. Therefore, each replacement is evaluated against the final LAMP-Merge implementation rather than against `avg`. The `avg` baseline is used in the main comparison table, not as the decision criterion for internal component validity.",
        "",
        "## Overall Summary",
        "",
    ]
    md_rows = []
    for row in summary_rows:
        md_rows.append({
            "Setting": row["label"],
            "Group": row["group"],
            "Raw cells": row["raw_cells"],
            "Raw mean Acc": fmt(row["raw_mean_acc"]),
            "Raw >= LAMP": f"{row['raw_ge_lamp']}/{row['raw_ge_lamp_total']}",
            "Client-average cells": row["client_average_cells"],
            "Client-average mean Acc": fmt(row["client_average_mean_acc"]),
            "Client-average >= LAMP": f"{row['client_average_ge_lamp']}/{row['client_average_ge_lamp_total']}",
            "Mean margin vs LAMP": fmt(row["mean_margin_vs_lamp"], digits=4),
        })
    lines.append(markdown_table(
        [
            "Setting",
            "Group",
            "Raw cells",
            "Raw mean Acc",
            "Raw >= LAMP",
            "Client-average cells",
            "Client-average mean Acc",
            "Client-average >= LAMP",
            "Mean margin vs LAMP",
        ],
        md_rows,
    ))
    lines.extend(["", "## Dataset-Level Client Average", ""])
    dataset_md = []
    key_methods = [
        "LAMP-Merge",
        "Classifier-head aggregation",
        "Shuffled-label prototype",
        "Global-feature mean",
        "Support-only synthetic head",
        "Uniform client weight",
        "Global client-size weight",
        "No prevalence calibration",
        "Uniform prevalence prior",
    ]
    for row in dataset_rows:
        item = {"Dataset": row["dataset"], "Cells": row["client_average_cells"]}
        lamp_value = row.get("LAMP-Merge_mean_acc", "")
        lamp_float = float(lamp_value) if lamp_value != "" else None
        item["LAMP-Merge"] = fmt(lamp_value) if lamp_value != "" else "-"
        for method in key_methods:
            if method == "LAMP-Merge":
                continue
            value = row.get(f"{method}_mean_acc", "")
            if value == "" or lamp_float is None:
                item[f"{method} margin"] = "-"
            else:
                item[f"{method} margin"] = fmt(float(value) - lamp_float)
        dataset_md.append(item)
    dataset_headers = [
        "Dataset",
        "Cells",
        "LAMP-Merge",
        "Classifier-head aggregation margin",
        "Shuffled-label prototype margin",
        "Global-feature mean margin",
        "Support-only synthetic head margin",
        "Uniform client weight margin",
        "Global client-size weight margin",
        "No prevalence calibration margin",
        "Uniform prevalence prior margin",
    ]
    lines.append(markdown_table(dataset_headers, dataset_md))
    lines.extend([
        "",
        "## Client-Average Cell File",
        "",
        "The complete per-cell table is written to `lamp_merge_internal_ablation_full_client_average.csv`. It includes all methods, the best method set for each `(dataset, backbone, K)` cell, and can be used directly for module-internal ablation tables in the paper.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    avg_lookup = load_avg_from_table(args.baseline_table)
    mode_lookups = {
        path.name: read_eval_lookup(path)
        for path in sorted(args.root.iterdir())
        if path.is_dir() and ((path / "reports" / "eval_summary.csv").exists() or (path / "eval").exists())
    }
    if "full" not in mode_lookups:
        raise SystemExit(f"Missing full mode under {args.root}")
    formal_lookup = mode_lookups["full"]
    summary_rows = [summarize_mode(mode, lookup, avg_lookup, formal_lookup) for mode, lookup in sorted(mode_lookups.items())]
    summary_rows.sort(key=lambda row: (row["group"], row["mode"]))

    client_rows = build_client_average_rows(mode_lookups, avg_lookup, formal_lookup)
    dataset_rows = build_dataset_rows(client_rows)

    write_csv(args.output_csv, summary_rows, [
        "mode",
        "label",
        "group",
        "raw_cells",
        "raw_mean_acc",
        "raw_min_acc",
        "raw_max_acc",
        "raw_ge_avg",
        "raw_ge_avg_total",
        "raw_ge_lamp",
        "raw_ge_lamp_total",
        "client_average_cells",
        "client_average_mean_acc",
        "client_average_ge_avg",
        "client_average_ge_avg_total",
        "client_average_ge_lamp",
        "client_average_ge_lamp_total",
        "mean_margin_vs_avg",
        "mean_margin_vs_lamp",
    ])
    method_fields = sorted({key for row in client_rows for key in row.keys() if key not in {
        "task_type", "dataset", "model", "num_clients", "best_value", "best_methods"
    }})
    write_csv(args.client_average_csv, client_rows, [
        "task_type",
        "dataset",
        "model",
        "num_clients",
        *method_fields,
        "best_value",
        "best_methods",
    ])
    dataset_fields = sorted({key for row in dataset_rows for key in row.keys()})
    write_csv(args.dataset_csv, dataset_rows, dataset_fields)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(render_summary(args.root, summary_rows, client_rows, dataset_rows), encoding="utf-8")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.client_average_csv}")
    print(f"wrote {args.dataset_csv}")
    print(f"wrote {args.summary_md}")


if __name__ == "__main__":
    main()
