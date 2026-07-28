#!/usr/bin/env python3
"""Plot the lowest non-Fisher baseline against LAMP-Merge on BloodMNIST."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import darken_color, polish_axes, save_png_pdf, setup_style


ROOT = Path(__file__).resolve().parent


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = {row["method"]: float(row["test_acc"]) for row in rows}
    if set(values) != {
        "avg", "avg_head", "ties", "dare_linear", "dare_ties", "regmean",
        "breadcrumbs", "model_stock", "from", "iso_c", "free_merge", "robustmerge",
        "lamp_merge",
    }:
        raise ValueError("Unexpected baseline-result methods")
    return values


def plot(values, output_base):
    methods = ["ties", "lamp_merge"]
    labels = ["TIES\n(worst baseline)", "LAMP-Merge\n(ours)"]
    scores = np.asarray([values[method] for method in methods])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Accuracy values must be finite probabilities")

    setup_style()
    figure, axis = plt.subplots(figsize=(7.8, 5.2), dpi=300)
    colors = ["#C0504D", "#4F81BD"]
    bars = axis.bar(
        np.arange(len(methods)),
        scores,
        width=0.58,
        color=colors,
        edgecolor=[darken_color(color, 0.65) for color in colors],
        linewidth=1.7,
        zorder=3,
    )
    for bar, score in zip(bars, scores):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.025,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
        )
    axis.set_title("BloodMNIST / ResNet (K=7)", pad=5, fontweight="bold")
    axis.set_ylabel("Test Acc", labelpad=3)
    axis.set_xticks(np.arange(len(methods)), labels)
    axis.set_ylim(0, 0.9)
    axis.tick_params(pad=3)
    polish_axes(axis)
    figure.tight_layout(pad=0.45)
    save_png_pdf(figure, str(output_base))
    plt.close(figure)


if __name__ == "__main__":
    plot(load_rows(ROOT / "blood_nonfisher_baseline_acc.csv"), ROOT / "ties_vs_lamp_merge_acc")
