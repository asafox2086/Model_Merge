#!/usr/bin/env python3
"""Collect strict asynchronous reports and plot macro-F1 over client arrivals."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from style import METHOD_COLORS, darken_color, polish_axes, save_png_pdf, setup_style


DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
]
MODELS = ["resnet", "convnext", "vit_t", "swin_tiny"]
DATASET_LABELS = {
    "bloodmnist_224": "BloodMNIST",
    "dermamnist_224": "DermaMNIST",
    "organcmnist_224": "OrganCMNIST",
    "organsmnist_224": "OrganSMNIST",
    "chaoshengmnist_224": "UltrasoundMNIST",
}
MODEL_LABELS = {
    "resnet": "ResNet",
    "convnext": "ConvNeXt",
    "vit_t": "ViT-T",
    "swin_tiny": "Swin-T",
}
MARKERS = {"resnet": "o", "convnext": "D", "vit_t": "^", "swin_tiny": "s"}
LINESTYLES = {"resnet": "-", "convnext": "--", "vit_t": "-.", "swin_tiny": ":"}


def report_path(outputs_root, dataset, model):
    short_dataset = dataset.removesuffix("_224")
    return (
        outputs_root
        / f"new_lamp_merge_async_{short_dataset}_{model}_k7"
        / "reports"
        / "new_lamp_merge_async_k1_to_k7.json"
    )


def collect_rows(outputs_root):
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            path = report_path(outputs_root, dataset, model)
            if not path.exists():
                raise FileNotFoundError(f"Missing asynchronous report: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("delivery_mode") != "strict_per_client_upload":
                raise ValueError(f"Unexpected delivery mode in {path}")
            report_rows = report.get("rows", [])
            if [row.get("k") for row in report_rows] != list(range(1, 8)):
                raise ValueError(f"Expected k=1..7 in {path}")
            amp_enabled = bool(report.get("amp_enabled", report_rows[0].get("amp_enabled", False)))
            for row in report_rows:
                acc = float(row["acc"])
                macro_f1 = float(row["macro_f1"])
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in (acc, macro_f1)):
                    raise ValueError(f"Invalid metrics in {path}")
                rows.append(
                    {
                        "dataset": dataset,
                        "backbone": model,
                        "k": int(row["k"]),
                        "acc": f"{acc:.12f}",
                        "macro_f1": f"{macro_f1:.12f}",
                        "amp_enabled": str(amp_enabled).lower(),
                    }
                )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metric_values(rows, model, metric):
    values = [
        float(row[metric])
        for row in rows
        if row["dataset"] == "bloodmnist_224" and row["backbone"] == model
    ]
    if len(values) != 7:
        raise ValueError(f"Expected seven values for bloodmnist_224/{model}")
    return values


def draw_metric(axis, rows, metric, title, ylabel):
    for model in MODELS:
        values = _metric_values(rows, model, metric)
        color = METHOD_COLORS[model]
        axis.plot(
            range(1, 8),
            values,
            label=MODEL_LABELS[model],
            color=color,
            linestyle=LINESTYLES[model],
            marker=MARKERS[model],
            markerfacecolor=color,
            markeredgecolor=darken_color(color, 0.65),
            markeredgewidth=1.3,
            zorder=3,
        )
    axis.set_title(title, pad=8, fontweight="bold")
    axis.set_xlabel("Received clients (k)")
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(1, 8))
    axis.set_ylim(0, 1.02)
    polish_axes(axis, y_grid=True, x_grid=False)


def plot_metric(rows, metric, title, ylabel, output_base):
    setup_style("line")
    figure, axis = plt.subplots(figsize=(7.4, 4.6), dpi=300)
    draw_metric(axis, rows, metric, title, ylabel)
    axis.legend(loc="lower right", frameon=False)
    figure.tight_layout()
    save_png_pdf(figure, str(output_base))
    plt.close(figure)


def plot_combined(rows, output_base):
    setup_style("line")
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 4.6), dpi=300, sharex=True, sharey=True)
    draw_metric(axes[0], rows, "acc", "BloodMNIST: Accuracy", "Test accuracy")
    draw_metric(axes[1], rows, "macro_f1", "BloodMNIST: Macro-F1", "Test macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    save_png_pdf(figure, str(output_base))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    args.result_root.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.outputs_root)
    write_csv(args.result_root / "async_grid_k1_to_k7.csv", rows)
    final_rows = [row for row in rows if row["k"] == 7]
    write_csv(args.result_root / "async_grid_k7.csv", final_rows)
    plot_metric(
        rows,
        "macro_f1",
        "BloodMNIST: Macro-F1",
        "Test macro-F1",
        args.result_root / "async_macro_f1_k1_to_k7",
    )
    plot_metric(
        rows,
        "acc",
        "BloodMNIST: Accuracy",
        "Test accuracy",
        args.result_root / "async_blood_acc_k1_to_k7",
    )
    plot_combined(rows, args.result_root / "async_blood_acc_macro_f1_k1_to_k7")


if __name__ == "__main__":
    main()
