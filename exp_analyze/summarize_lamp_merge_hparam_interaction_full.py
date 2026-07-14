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

FORMAL_DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
]
FORMAL_MODELS = ["resnet", "convnext", "vit_t", "swin_tiny"]
FORMAL_SETTINGS = [
    (3, 0.0),
    (3, 0.01),
    (3, 0.1),
    (5, 0.0),
    (5, 0.01),
    (5, 0.1),
    (7, 0.0),
    (7, 0.01),
    (7, 0.1),
]

SCAN_PATTERN = re.compile(
    r"^(?P<module>m1)_gamma_(?P<gamma>[0-9]+(?:p[0-9]+)?)_s_(?P<scale>[0-9]+(?:p[0-9]+)?)$"
    r"|^(?P<m2>m2)_tau_(?P<tau>[0-9]+(?:p[0-9]+)?)_lambda_(?P<lambda>[0-9]+(?:p[0-9]+)?)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize full-scope LAMP-Merge interaction scans.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--client-average-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--figure-path", type=Path, required=True)
    return parser.parse_args()


def parse_value(value: str) -> float:
    return float(value.replace("p", "."))


def parse_scan_directory(path: Path) -> tuple[str, str, float, str, float] | None:
    match = SCAN_PATTERN.match(path.name)
    if match is None:
        return None
    if match.group("module") == "m1":
        return ("diagnostic prototype reconstruction", "s", parse_value(match.group("scale")), "gamma", parse_value(match.group("gamma")))
    return ("long-tail prevalence calibration", "lambda", parse_value(match.group("lambda")), "tau", parse_value(match.group("tau")))


def raw_key(row: dict[str, object]) -> tuple[str, str, str, int, float]:
    return (
        str(row["task_type"]),
        str(row["dataset"]),
        str(row["model"]),
        int(row["num_clients"]),
        float(row["beta"]),
    )


def formal_keys() -> set[tuple[str, str, str, int, float]]:
    return {
        ("small", dataset, model, num_clients, beta)
        for dataset in FORMAL_DATASETS
        for model in FORMAL_MODELS
        for num_clients, beta in FORMAL_SETTINGS
    }


def read_eval_lookup(root: Path) -> dict[tuple[str, str, str, int, float], float]:
    lookup: dict[tuple[str, str, str, int, float], float] = {}
    for path in sorted((root / "eval").glob("**/eval.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("test_acc") is not None:
            lookup[raw_key(row)] = float(row["test_acc"])
    if lookup:
        return lookup

    summary_path = root / "reports" / "eval_summary.csv"
    if not summary_path.exists():
        return lookup
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("test_acc"):
                lookup[raw_key(row)] = float(row["test_acc"])
    return lookup


def build_client_averages(raw_lookup: dict[tuple[str, str, str, int, float], float]) -> dict[tuple[str, str, str, int], float]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    allowed = formal_keys()
    for raw_case, accuracy in raw_lookup.items():
        if raw_case in allowed:
            task_type, dataset, model, num_clients, _ = raw_case
            grouped[(task_type, dataset, model, num_clients)].append(accuracy)
    return {case: mean(accuracies) for case, accuracies in grouped.items() if len(accuracies) == 3}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect_rows(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    client_rows: list[dict[str, object]] = []
    allowed = formal_keys()
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        parsed = parse_scan_directory(directory)
        if parsed is None:
            continue
        module, x_symbol, x_value, curve_symbol, curve_value = parsed
        raw_lookup = read_eval_lookup(directory)
        raw_accuracies = [raw_lookup[case] for case in allowed if case in raw_lookup]
        client_averages = build_client_averages(raw_lookup)
        summary_rows.append(
            {
                "module": module,
                "x_symbol": x_symbol,
                "x_value": x_value,
                "curve_symbol": curve_symbol,
                "curve_value": curve_value,
                "root": str(directory),
                "raw_cells": len(raw_accuracies),
                "raw_mean_acc": mean(raw_accuracies) if raw_accuracies else "",
                "client_average_cells": len(client_averages),
                "client_average_mean_acc": mean(client_averages.values()) if client_averages else "",
            }
        )
        for case, accuracy in sorted(client_averages.items()):
            task_type, dataset, model, num_clients = case
            client_rows.append(
                {
                    "module": module,
                    "x_symbol": x_symbol,
                    "x_value": x_value,
                    "curve_symbol": curve_symbol,
                    "curve_value": curve_value,
                    "task_type": task_type,
                    "dataset": dataset,
                    "model": model,
                    "num_clients": num_clients,
                    "client_average_acc": accuracy,
                }
            )
    summary_rows.sort(key=lambda row: (str(row["module"]), float(row["curve_value"]), float(row["x_value"])))
    client_rows.sort(key=lambda row: (str(row["module"]), float(row["curve_value"]), float(row["x_value"]), str(row["dataset"]), str(row["model"]), int(row["num_clients"])))
    return summary_rows, client_rows


def format_value(value: object, digits: int = 4) -> str:
    if value in {None, ""}:
        return "-"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric_value):
        return "-"
    return f"{numeric_value:.{digits}f}"


def write_summary(path: Path, root: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# LAMP-Merge Full-Scope Interaction Hyperparameter Analysis",
        "",
        f"Experiment root: `{root}`.",
        "",
        "Every grid point evaluates the full formal medical benchmark: five datasets, four vision backbones, three client counts, and three Dirichlet skew levels. A complete point therefore contains 180 raw cells and 60 client-average cells.",
        "",
        "The diagnostic-prototype grid fixes a value of $\\gamma$ for each curve and scans $s$. The prevalence-calibration grid fixes a value of $\\tau$ for each curve and scans $\\lambda$.",
        "",
        "| Module | Curve | X value | Raw cells | Raw mean Acc | Client-average cells | Client-average mean Acc |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        curve_symbol = {"gamma": r"\gamma", "tau": r"\tau"}.get(str(row["curve_symbol"]), str(row["curve_symbol"]))
        curve = f"${curve_symbol}={format_value(row['curve_value'], 2)}$"
        x_value = format_value(row["x_value"], 2)
        lines.append(
            f"| {row['module']} | {curve} | {x_value} | {row['raw_cells']} | {format_value(row['raw_mean_acc'])} | {row['client_average_cells']} | {format_value(row['client_average_mean_acc'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict[str, object]], figure_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    module_order = ["diagnostic prototype reconstruction", "long-tail prevalence calibration"]
    module_labels = {
        "diagnostic prototype reconstruction": (r"Diagnostic prototype reconstruction: scan $s$ at fixed $\gamma$", r"$s$", r"\gamma", 0.45),
        "long-tail prevalence calibration": (r"Long-tail prevalence calibration: scan $\lambda$ at fixed $\tau$", r"$\lambda$", r"\tau", 2.5),
    }
    colors = ["#4E79A7", "#59A14F", "#F2CF5B", "#E15759", "#9C755F"]
    markers = ["o", "s", "D", "^", "P"]
    figure, axes = pyplot.subplots(1, 2, figsize=(11.5, 3.6), dpi=260)
    for axis, module in zip(axes, module_order):
        module_rows = [row for row in rows if row["module"] == module]
        grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
        for row in module_rows:
            grouped[float(row["curve_value"])].append(row)
        title, x_label, curve_label, selected_curve = module_labels[module]
        for index, curve_value in enumerate(sorted(grouped)):
            curve_rows = sorted(grouped[curve_value], key=lambda row: float(row["x_value"]))
            x_values = [float(row["x_value"]) for row in curve_rows]
            y_values = [float(row["client_average_mean_acc"]) for row in curve_rows]
            selected = math.isclose(curve_value, selected_curve, rel_tol=0.0, abs_tol=1e-9)
            axis.plot(
                x_values,
                y_values,
                color=colors[index],
                marker=markers[index],
                markersize=4.4,
                linewidth=2.3 if selected else 1.4,
                alpha=1.0 if selected else 0.88,
                label=rf"${curve_label}={curve_value:g}$",
            )
        axis.axvline(20 if module == module_order[0] else 5, color="#333333", linestyle="--", linewidth=0.9, alpha=0.75)
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.set_xlabel(x_label, fontsize=10)
        axis.set_ylabel("Client-average accuracy", fontsize=10)
        axis.grid(axis="y", color="#D7D7D7", linewidth=0.7)
        axis.tick_params(axis="both", labelsize=9)
        axis.legend(title="Fixed value", fontsize=8, title_fontsize=8, frameon=False, ncol=1, loc="best")
    figure.tight_layout(pad=0.8, w_pad=1.6)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    args = parse_args()
    summary_rows, client_rows = collect_rows(args.root)
    write_csv(
        args.output_csv,
        summary_rows,
        [
            "module",
            "x_symbol",
            "x_value",
            "curve_symbol",
            "curve_value",
            "root",
            "raw_cells",
            "raw_mean_acc",
            "client_average_cells",
            "client_average_mean_acc",
        ],
    )
    write_csv(
        args.client_average_csv,
        client_rows,
        [
            "module",
            "x_symbol",
            "x_value",
            "curve_symbol",
            "curve_value",
            "task_type",
            "dataset",
            "model",
            "num_clients",
            "client_average_acc",
        ],
    )
    write_summary(args.summary_md, args.root, summary_rows)
    plot(summary_rows, args.figure_path)
    expected_points = 50
    completed_points = sum(1 for row in summary_rows if row["raw_cells"] == 180 and row["client_average_cells"] == 60)
    print(f"completed_points={completed_points}/{expected_points}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.client_average_csv}")
    print(f"wrote {args.summary_md}")
    print(f"wrote {args.figure_path}")


if __name__ == "__main__":
    main()
