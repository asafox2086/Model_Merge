#!/usr/bin/env python3
"""Plot the true DermaMNIST class distribution used in the introduction."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "scripts" / "plot_templates"
if str(TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_DIR))

from style import polish_axes, setup_style  # noqa: E402


DATA_PATH = ROOT / "Med_data" / "dermamnist_224.npz"
FIGURE_BASE = ROOT / "figures" / "derma_true_class_distribution"
CSV_PATH = ROOT / "论文实验数据" / "Derma真实类别分布.csv"
BAR_COLOR = "#6E9FC8"
BAR_EDGE = "#355C7D"


def save_png_pdf(figure: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_base) + ".png", dpi=350, bbox_inches="tight")
    figure.savefig(str(output_base) + ".pdf", bbox_inches="tight")


def load_distribution() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(DATA_PATH) as data:
        labels = np.asarray(data["test_labels"]).reshape(-1).astype(int)
    classes = np.arange(int(labels.max()) + 1)
    counts = np.bincount(labels, minlength=len(classes))
    percentages = counts / counts.sum() * 100.0
    if not np.isclose(percentages.sum(), 100.0):
        raise ValueError("Class percentages do not sum to 100")
    return classes, counts, percentages


def write_csv(classes: np.ndarray, counts: np.ndarray, percentages: np.ndarray) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "说明",
                "本表对应 introduction 受控对比图中 DermaMNIST 数据集的真实测试集类别分布；用于支撑第二个观测：医学图像存在长尾患病率分布。数值直接由 Med_data/dermamnist_224.npz 的 test_labels 统计得到。",
            ]
        )
        writer.writerow(["Class", "Test count", "True test distribution (%)"])
        for class_id, count, percentage in zip(classes, counts, percentages):
            writer.writerow([int(class_id), int(count), f"{percentage:.4f}"])


def plot_distribution(classes: np.ndarray, counts: np.ndarray, percentages: np.ndarray) -> None:
    setup_style("bar")
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )
    figure, axis = plt.subplots(figsize=(4.7, 2.75), dpi=350)
    bars = axis.bar(
        classes,
        percentages,
        width=0.64,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=1.35,
        zorder=3,
    )
    peak_index = int(np.argmax(percentages))
    bars[peak_index].set_color("#3F73A3")
    bars[peak_index].set_edgecolor("#274A68")
    bars[peak_index].set_linewidth(1.9)

    for class_id, percentage in zip(classes, percentages):
        offset = 1.25 if percentage < 15 else 1.9
        axis.text(
            class_id,
            percentage + offset,
            f"{percentage:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.7,
            fontweight="bold" if class_id == peak_index else "normal",
            color="#274A68" if class_id == peak_index else "#222222",
        )

    axis.set_xticks(classes)
    axis.set_xticklabels([f"C{int(class_id)}" for class_id in classes])
    axis.set_xlabel("DermaMNIST class")
    axis.set_ylabel("True distribution (%)")
    axis.set_ylim(0, 74)
    axis.set_title("DermaMNIST True Class Distribution", fontweight="bold", pad=6)
    polish_axes(axis, y_grid=True, x_grid=False)
    figure.tight_layout()
    save_png_pdf(figure, FIGURE_BASE)
    plt.close(figure)


def main() -> None:
    classes, counts, percentages = load_distribution()
    write_csv(classes, counts, percentages)
    plot_distribution(classes, counts, percentages)


if __name__ == "__main__":
    main()
