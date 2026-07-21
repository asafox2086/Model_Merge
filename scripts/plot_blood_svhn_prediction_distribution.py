#!/usr/bin/env python3
"""Plot separate matched BloodMNIST and SVHN AVG prediction distributions."""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "scripts" / "plot_templates"
if str(TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_DIR))

from style import darken_color, polish_axes, save_png_pdf, setup_style


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blood-metrics", type=Path, required=True)
    parser.add_argument("--svhn-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_avg_row(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    avg_rows = [row for row in rows if row.get("model") == "avg" and row.get("split") == "test"]
    if len(avg_rows) != 1:
        raise ValueError(f"Expected exactly one AVG test row in {path}, found {len(avg_rows)}")
    row = avg_rows[0]
    prediction_counts = np.asarray(ast.literal_eval(row["pred_counts"]), dtype=np.int64)
    support = np.asarray(ast.literal_eval(row["support"]), dtype=np.int64)
    if prediction_counts.ndim != 1 or support.shape != prediction_counts.shape:
        raise ValueError(f"Invalid class counts in {path}")
    if (prediction_counts < 0).any() or (support < 0).any():
        raise ValueError(f"Negative class count in {path}")
    if int(prediction_counts.sum()) != int(support.sum()):
        raise ValueError(f"Prediction count and support differ in {path}")
    return prediction_counts, support


def write_source_data(path, counts):
    total = int(counts.sum())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("class", "prediction_count", "prediction_percent"),
            lineterminator="\n",
        )
        writer.writeheader()
        for class_index in range(len(counts)):
            writer.writerow(
                {
                    "class": class_index,
                    "prediction_count": int(counts[class_index]),
                    "prediction_percent": f"{100 * counts[class_index] / total:.6f}",
                }
            )


def plot_distribution(output_base, counts, domain_name, color):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Prediction counts must have a positive total")
    percentages = counts * 100.0 / total
    setup_style("bar")
    figure, axis = plt.subplots(figsize=(7.6, 5.1), dpi=350)
    positions = np.arange(len(counts))
    bars = axis.bar(positions, percentages, width=0.62, color=color, edgecolor=darken_color(color, 0.65), linewidth=1.4, zorder=3)
    for bar, count in zip(bars, counts):
        if count:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.1, f"{count}\n({bar.get_height():.1f}%)", ha="center", va="bottom", fontsize=9)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"Class {index}" for index in positions])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Test predictions (%)")
    axis.set_ylim(0, 108)
    axis.set_title(f"AVG prediction distribution: {domain_name}\nResNet / K=3 / test set (n={total:,})", pad=10, fontweight="bold")
    polish_axes(axis, y_grid=True, x_grid=False)
    figure.tight_layout()
    save_png_pdf(figure, str(output_base), dpi=350)
    plt.close(figure)


def main():
    args = parse_args()
    blood_counts, blood_support = load_avg_row(args.blood_metrics)
    svhn_counts, svhn_support = load_avg_row(args.svhn_metrics)
    if not np.array_equal(blood_support, svhn_support):
        raise ValueError("Blood and SVHN test class supports are not matched")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_distribution(args.output_dir / "blood_avg_prediction_distribution", blood_counts, "Medical BloodMNIST", "#4F81BD")
    plot_distribution(args.output_dir / "svhn_avg_prediction_distribution", svhn_counts, "Natural SVHN", "#C0504D")
    write_source_data(args.output_dir / "blood_avg_prediction_distribution.csv", blood_counts)
    write_source_data(args.output_dir / "svhn_avg_prediction_distribution.csv", svhn_counts)
    print(f"Wrote prediction distribution plot to {args.output_dir}")


if __name__ == "__main__":
    main()
