#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FORMAL_SMALL_DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
]
FORMAL_SMALL_MODELS = ["resnet", "convnext", "vit_t", "swin_tiny"]
SETTINGS = [(3, 0.0), (3, 0.01), (3, 0.1), (5, 0.0), (5, 0.01), (5, 0.1), (7, 0.0), (7, 0.01), (7, 0.1)]

PARAM_INFO = {
    "scale": ("M1", "prototype head scale", "s"),
    "lambda": ("M2", "long-tail calibration strength", "lambda"),
    "tau": ("M2", "long-tail calibration strength", "lambda"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize full-scope LAMP-Merge hyperparameter scans.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_hparam_full.csv")
    parser.add_argument("--client-average-csv", type=Path, default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_hparam_full_client_average.csv")
    parser.add_argument("--dataset-csv", type=Path, default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_hparam_full_by_dataset.csv")
    parser.add_argument("--summary-md", type=Path, default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_hparam_full_summary.md")
    parser.add_argument("--figure-path", type=Path, default=ROOT / "My_merge_ret" / "figures" / "lamp_merge_hparam_full_sensitivity.png")
    return parser.parse_args()


def beta_key(value: object) -> float:
    return float(value)


def raw_key(row: dict[str, object]) -> tuple[str, str, str, int, float]:
    model = str(row["model"]) if row["task_type"] == "small" else str(row.get("clip_model", ""))
    return (str(row["task_type"]), str(row["dataset"]), model, int(row["num_clients"]), beta_key(row["beta"]))


def client_key(key: tuple[str, str, str, int, float]) -> tuple[str, str, str, int]:
    task_type, dataset, model, num_clients, _beta = key
    return (task_type, dataset, model, num_clients)


def parse_scan_dir(path: Path) -> tuple[str, float] | None:
    match = re.match(r"^(scale|lambda|tau)_([0-9]+(?:p[0-9]+)?)$", path.name)
    if not match:
        return None
    family = match.group(1)
    value = float(match.group(2).replace("p", "."))
    if family == "tau":
        family = "lambda"
    return family, value


def read_eval_lookup(root: Path) -> dict[tuple[str, str, str, int, float], float]:
    lookup: dict[tuple[str, str, str, int, float], float] = {}
    for path in sorted((root / "eval").glob("**/eval.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("test_acc") is None:
            continue
        lookup[raw_key(row)] = float(row["test_acc"])
    if lookup:
        return lookup

    summary_path = root / "reports" / "eval_summary.csv"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("test_acc"):
                    continue
                lookup[raw_key(row)] = float(row["test_acc"])
    return lookup


def formal_keys() -> set[tuple[str, str, str, int, float]]:
    return {
        ("small", dataset, model, num_clients, beta)
        for dataset in FORMAL_SMALL_DATASETS
        for model in FORMAL_SMALL_MODELS
        for num_clients, beta in SETTINGS
    }


def build_client_average_lookup(raw_lookup: dict[tuple[str, str, str, int, float], float]) -> dict[tuple[str, str, str, int], float]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for key, value in raw_lookup.items():
        if key in formal_keys():
            grouped[client_key(key)].append(float(value))
    return {key: mean(values) for key, values in grouped.items() if len(values) == 3}


def fmt(value: object, digits: int = 4) -> str:
    if value in {"", None}:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "-"
    return f"{number:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def markdown_table(headers: list[str], rows: list[dict[str, object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def collect_scan_rows(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    raw_rows: list[dict[str, object]] = []
    client_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    keys = formal_keys()

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_scan_dir(child)
        if parsed is None:
            continue
        family, value = parsed
        module, parameter, symbol = PARAM_INFO[family]
        lookup = read_eval_lookup(child)
        raw_values = [lookup[key] for key in keys if key in lookup]
        client_lookup = build_client_average_lookup(lookup)
        client_values = list(client_lookup.values())
        raw_rows.append(
            {
                "module": module,
                "parameter": parameter,
                "symbol": symbol,
                "value": value,
                "root": str(child),
                "raw_cells": len(raw_values),
                "raw_mean_acc": mean(raw_values) if raw_values else "",
                "raw_min_acc": min(raw_values) if raw_values else "",
                "raw_max_acc": max(raw_values) if raw_values else "",
                "client_average_cells": len(client_values),
                "client_average_mean_acc": mean(client_values) if client_values else "",
                "client_average_min_acc": min(client_values) if client_values else "",
                "client_average_max_acc": max(client_values) if client_values else "",
            }
        )
        for key, acc in sorted(client_lookup.items()):
            task_type, dataset, model, num_clients = key
            client_rows.append(
                {
                    "module": module,
                    "parameter": parameter,
                    "symbol": symbol,
                    "value": value,
                    "task_type": task_type,
                    "dataset": dataset,
                    "model": model,
                    "num_clients": num_clients,
                    "client_average_acc": acc,
                }
            )
        by_dataset: dict[str, list[float]] = defaultdict(list)
        for key, acc in client_lookup.items():
            by_dataset[key[1]].append(acc)
        for dataset, values in sorted(by_dataset.items()):
            dataset_rows.append(
                {
                    "module": module,
                    "parameter": parameter,
                    "symbol": symbol,
                    "value": value,
                    "dataset": dataset,
                    "client_average_cells": len(values),
                    "client_average_mean_acc": mean(values),
                }
            )
    raw_rows.sort(key=lambda row: (str(row["module"]), str(row["parameter"]), float(row["value"])))
    client_rows.sort(key=lambda row: (str(row["module"]), str(row["parameter"]), float(row["value"]), str(row["dataset"]), str(row["model"]), int(row["num_clients"])))
    dataset_rows.sort(key=lambda row: (str(row["module"]), str(row["parameter"]), float(row["value"]), str(row["dataset"])))
    return raw_rows, client_rows, dataset_rows


def build_summary_md(root: Path, rows: list[dict[str, object]], dataset_rows: list[dict[str, object]]) -> str:
    lines = [
        "# LAMP-Merge Full-Scope Hyperparameter Sensitivity",
        "",
        f"Experiment root: `{root}`.",
        "",
        "The sensitivity analysis is evaluated on the full medical-image grid: five datasets, four backbones, three client counts, and three beta values. Each value therefore has 180 raw cells and 60 client-average cells when complete.",
        "",
        "## Overall",
        "",
    ]
    md_rows = []
    for row in rows:
        md_rows.append(
            {
                "Module": row["module"],
                "Parameter": row["parameter"],
                "Symbol": row["symbol"],
                "Value": fmt(row["value"], 2),
                "Raw cells": row["raw_cells"],
                "Raw mean Acc": fmt(row["raw_mean_acc"]),
                "Client-average cells": row["client_average_cells"],
                "Client-average mean Acc": fmt(row["client_average_mean_acc"]),
            }
        )
    lines.append(markdown_table(["Module", "Parameter", "Symbol", "Value", "Raw cells", "Raw mean Acc", "Client-average cells", "Client-average mean Acc"], md_rows))
    lines.extend(["", "## Dataset-Level Client Average", ""])
    ds_md = []
    for row in dataset_rows:
        ds_md.append(
            {
                "Module": row["module"],
                "Symbol": row["symbol"],
                "Value": fmt(row["value"], 2),
                "Dataset": row["dataset"],
                "Cells": row["client_average_cells"],
                "Mean Acc": fmt(row["client_average_mean_acc"]),
            }
        )
    lines.append(markdown_table(["Module", "Symbol", "Value", "Dataset", "Cells", "Mean Acc"], ds_md))
    return "\n".join(lines) + "\n"


def plot(rows: list[dict[str, object]], figure_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["module"]), str(row["parameter"]), str(row["symbol"]))].append(row)
    order = sorted(grouped)
    if not order:
        return
    fig, axes = plt.subplots(1, len(order), figsize=(5.2 * len(order), 3.3), dpi=220, squeeze=False)
    for ax, key in zip(axes[0], order):
        items = sorted(grouped[key], key=lambda row: float(row["value"]))
        xs = [float(row["value"]) for row in items]
        ys = [float(row["client_average_mean_acc"]) for row in items]
        display_symbol = r"$s$" if key[2] == "s" else r"$\lambda$"
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4.0, color="#1f77b4")
        best_idx = max(range(len(ys)), key=lambda idx: ys[idx])
        ax.scatter(
            [xs[best_idx]],
            [ys[best_idx]],
            marker="*",
            s=95,
            color="#d62728",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label="Grid maximum",
        )
        default_value = 20.0 if key[2] == "s" else 5.0
        if default_value in xs:
            default_idx = xs.index(default_value)
            ax.axvline(
                default_value,
                color="#2ca02c",
                linestyle="--",
                linewidth=1.1,
                alpha=0.85,
                label="Selected setting",
            )
            ax.scatter([default_value], [ys[default_idx]], marker="s", s=34, color="#2ca02c", edgecolor="white", linewidth=0.5, zorder=5)
        ax.set_title(f"{key[0]}: {display_symbol}", fontsize=9)
        ax.set_xlabel(f"Value of {display_symbol}", fontsize=8)
        ax.set_ylabel("client-average accuracy", fontsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("Full-scope LAMP-Merge hyperparameter sensitivity", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    args = parse_args()
    raw_rows, client_rows, dataset_rows = collect_scan_rows(args.root)
    write_csv(args.output_csv, raw_rows, [
        "module",
        "parameter",
        "symbol",
        "value",
        "root",
        "raw_cells",
        "raw_mean_acc",
        "raw_min_acc",
        "raw_max_acc",
        "client_average_cells",
        "client_average_mean_acc",
        "client_average_min_acc",
        "client_average_max_acc",
    ])
    write_csv(args.client_average_csv, client_rows, [
        "module",
        "parameter",
        "symbol",
        "value",
        "task_type",
        "dataset",
        "model",
        "num_clients",
        "client_average_acc",
    ])
    write_csv(args.dataset_csv, dataset_rows, [
        "module",
        "parameter",
        "symbol",
        "value",
        "dataset",
        "client_average_cells",
        "client_average_mean_acc",
    ])
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(build_summary_md(args.root, raw_rows, dataset_rows), encoding="utf-8")
    plot(raw_rows, args.figure_path)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.client_average_csv}")
    print(f"wrote {args.dataset_csv}")
    print(f"wrote {args.summary_md}")
    print(f"wrote {args.figure_path}")


if __name__ == "__main__":
    main()
