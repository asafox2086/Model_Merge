#!/usr/bin/env python3
"""Plot matched BloodMNIST/SVHN AVG prediction distributions."""

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


def write_source_data(path, blood_counts, svhn_counts):
    total = int(blood_counts.sum())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("class", "blood_prediction_count", "blood_prediction_percent", "svhn_prediction_count", "svhn_prediction_percent"),
            lineterminator="\n",
        )
        writer.writeheader()
        for class_index in range(len(blood_counts)):
            writer.writerow(
                {
                    "class": class_index,
                    "blood_prediction_count": int(blood_counts[class_index]),
                    "blood_prediction_percent": f"{100 * blood_counts[class_index] / total:.6f}",
                    "svhn_prediction_count": int(svhn_counts[class_index]),
                    "svhn_prediction_percent": f"{100 * svhn_counts[class_index] / total:.6f}",
                }
            )


def plot_distribution(output_base, blood_counts, svhn_counts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if blood_counts.shape != svhn_counts.shape:
        raise ValueError("Blood and SVHN must have equal class counts")
    total = int(blood_counts.sum())
    if total <= 0 or int(svhn_counts.sum()) != total:
        raise ValueError("Domains must have the same positive test-set size")
    blood_percent = blood_counts * 100.0 / total
    svhn_percent = svhn_counts * 100.0 / total
    setup_style("bar")
    figure, axis = plt.subplots(figsize=(9.4, 5.1), dpi=350)
    positions = np.arange(len(blood_counts))
    width = 0.36
    colors = {"BloodMNIST": "#4F81BD", "SVHN": "#C0504D"}
    bars = (
        (axis.bar(positions - width / 2, blood_percent, width, label="Medical BloodMNIST", color=colors["BloodMNIST"], edgecolor=darken_color(colors["BloodMNIST"], 0.65), linewidth=1.4, zorder=3), blood_counts),
        (axis.bar(positions + width / 2, svhn_percent, width, label="Natural SVHN", color=colors["SVHN"], edgecolor=darken_color(colors["SVHN"], 0.65), linewidth=1.4, zorder=3), svhn_counts),
    )
    for bar_group, counts in bars:
        for bar, count in zip(bar_group, counts):
            if count:
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.1, f"{count}\n({bar.get_height():.1f}%)", ha="center", va="bottom", fontsize=9)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"Class {index}" for index in positions])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Test predictions (%)")
    axis.set_ylim(0, 108)
    axis.set_title("AVG prediction distribution: BloodMNIST vs SVHN\nResNet / K=3 / matched test set (n=3,421 per domain)", pad=10, fontweight="bold")
    axis.legend(loc="upper right", frameon=False)
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
    output_base = args.output_dir / "blood_svhn_avg_prediction_distribution"
    plot_distribution(output_base, blood_counts, svhn_counts)
    write_source_data(args.output_dir / "blood_svhn_avg_prediction_distribution.csv", blood_counts, svhn_counts)
    print(f"Wrote prediction distribution plot to {args.output_dir}")


if __name__ == "__main__":
    main()
