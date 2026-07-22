#!/usr/bin/env python3
"""Plot matched DermaMNIST/CIFAR-100 RegMean prediction distributions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "scripts" / "plot_templates"
if str(TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_DIR))

from style import darken_color, polish_axes, save_png_pdf, setup_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derma-csv", type=Path, required=True)
    parser.add_argument("--natural-csv", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--layout", choices=("panels", "overlay"), default="panels")
    return parser.parse_args()


def read_distribution(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    classes = np.asarray([int(row["class"]) for row in rows], dtype=int)
    predicted = np.asarray([float(row["prediction_percent"]) for row in rows], dtype=float)
    true = np.asarray([float(row["true_percent"]) for row in rows], dtype=float)
    if not (
        classes.ndim == predicted.ndim == true.ndim == 1
        and len(classes) == len(predicted) == len(true)
        and np.array_equal(classes, np.arange(len(classes)))
        and np.isfinite(predicted).all()
        and np.isfinite(true).all()
        and np.isclose(predicted.sum(), 100.0, atol=1e-4)
        and np.isclose(true.sum(), 100.0, atol=1e-4)
    ):
        raise ValueError(f"Invalid class distribution in {path}")
    return classes, predicted, true


def plot_panel(axis, classes, predicted, true, title, prediction_color):
    axis.plot(
        classes,
        true,
        color="#202020",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#202020",
        markeredgewidth=1.5,
        linewidth=2.6,
        label="True test distribution",
        zorder=4,
    )
    axis.plot(
        classes,
        predicted,
        color=prediction_color,
        marker="D",
        markerfacecolor=prediction_color,
        markeredgecolor=darken_color(prediction_color, 0.64),
        markeredgewidth=1.2,
        linewidth=2.8,
        label="RegMean predictions",
        zorder=5,
    )
    axis.set_title(title, fontweight="bold", pad=8)
    axis.set_xticks(classes)
    axis.set_xticklabels([f"{class_index}" for class_index in classes])
    axis.set_xlabel("Class index")
    axis.set_ylim(-2, 106)
    polish_axes(axis, y_grid=True, x_grid=False)


def plot_overlay(axis, classes, derma_predicted, natural_predicted, true):
    axis.plot(
        classes,
        true,
        color="#202020",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#202020",
        markeredgewidth=1.5,
        linewidth=2.6,
        label="True test distribution",
        zorder=3,
    )
    for values, color, label, marker in (
        (derma_predicted, "#4F81BD", "DermaMNIST RegMean", "D"),
        (natural_predicted, "#C0504D", "CIFAR-100 RegMean", "s"),
    ):
        axis.plot(
            classes,
            values,
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor=darken_color(color, 0.64),
            markeredgewidth=1.2,
            linewidth=2.8,
            label=label,
            zorder=4,
        )
    axis.set_xticks(classes)
    axis.set_xticklabels([f"{class_index}" for class_index in classes])
    axis.set_xlabel("Class index")
    axis.set_ylabel("Test distribution (%)")
    axis.set_ylim(-2, 106)
    axis.set_title("Matched prediction distributions", fontweight="bold", pad=8)
    axis.legend(loc="upper left", frameon=False)
    polish_axes(axis, y_grid=True, x_grid=False)


def main() -> None:
    args = parse_args()
    derma_classes, derma_predicted, derma_true = read_distribution(args.derma_csv)
    natural_classes, natural_predicted, natural_true = read_distribution(args.natural_csv)
    if not np.array_equal(derma_classes, natural_classes):
        raise ValueError("The two domains must have the same ordered classes")
    if not np.allclose(derma_true, natural_true, atol=1e-4):
        raise ValueError("The strict control must have identical true test distributions")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_style("line")
    if args.layout == "overlay":
        figure, axis = plt.subplots(figsize=(7.4, 4.6), dpi=350)
        plot_overlay(axis, derma_classes, derma_predicted, natural_predicted, derma_true)
        figure.tight_layout()
    else:
        figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), dpi=350, sharey=True)
        plot_panel(axes[0], derma_classes, derma_predicted, derma_true, "Medical DermaMNIST", "#4F81BD")
        plot_panel(
            axes[1],
            natural_classes,
            natural_predicted,
            natural_true,
            "Natural CIFAR-100 semantic control",
            "#C0504D",
        )
        axes[0].set_ylabel("Test distribution (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
        figure.subplots_adjust(left=0.08, right=0.995, top=0.86, bottom=0.24, wspace=0.12)
    save_png_pdf(figure, str(args.output_base), dpi=350)
    plt.close(figure)


if __name__ == "__main__":
    main()
