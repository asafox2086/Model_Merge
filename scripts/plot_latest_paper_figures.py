#!/usr/bin/env python3
"""Plot manuscript figures from paper-aligned CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "scripts" / "plot_templates"
if str(TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_DIR))

from style import darken_color, polish_axes, save_png_pdf, setup_style  # noqa: E402


PAPER_COLORS = {
    "Weight-Averaging Baseline": "#7F7F7F",
    "Baseline + DPR": "#9BBB59",
    "LAMP-Merge": "#C0504D",
    "Weight Averaging": "#7F7F7F",
    "TIES-Merging": "#9BBB59",
    "DARE-Linear": "#4F81BD",
    "True distribution": "#8064A2",
}

LINE_STYLES = [
    ("o", "-"),
    ("s", "--"),
    ("^", "-."),
    ("D", ":"),
    ("v", "--"),
]


def read_annotated_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first = next(reader)
        header = next(reader) if first and first[0] == "说明" else first
        return [dict(zip(header, row)) for row in reader if row]


def finite_array(values: list[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def plot_module_ablation(csv_dir: Path, figure_dir: Path) -> None:
    rows = read_annotated_csv(csv_dir / "消融.csv")
    datasets = ["Blood", "Derma", "Organ-C", "Organ-S", "Ultrasound"]
    labels = ["Weight-Averaging Baseline", "Baseline + DPR", "LAMP-Merge"]
    values = {}
    for row, label in zip(rows, labels):
        values[label] = finite_array(
            [float(row[f"{dataset} ACC (%)"]) for dataset in datasets],
            (len(datasets),),
            label,
        )

    setup_style("bar")
    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=300)
    x = np.arange(len(datasets))
    width = 0.24
    for index, label in enumerate(labels):
        offset = (index - 1.0) * width
        bars = ax.bar(
            x + offset,
            values[label],
            width=width,
            label=label,
            color=PAPER_COLORS[label],
            edgecolor=darken_color(PAPER_COLORS[label], 0.65),
            linewidth=1.25,
            zorder=3,
        )
        for bar, value in zip(bars, values[label]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Client-average ACC (%)")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, max(max(item) for item in values.values()) + 10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False)
    polish_axes(ax, y_grid=True, x_grid=False)
    fig.tight_layout()
    save_png_pdf(fig, str(figure_dir / "01_module_ablation_accuracy"), dpi=350)
    plt.close(fig)


def plot_baseline_2x2_ablation(csv_dir: Path, figure_dir: Path) -> None:
    rows = read_annotated_csv(csv_dir / "基线2x2消融.csv")
    settings = ["TIES-Merging", "TIES-Merging + LPC", "DARE-Linear", "DARE-Linear + LPC"]
    dataset_fields = ["Blood", "Derma", "Organ-C", "Organ-S", "Ultrasound", "Avg"]
    row_by_setting = {row["Setting"]: row for row in rows}
    if set(row_by_setting) != set(settings):
        raise ValueError(f"Unexpected baseline 2x2 settings: {sorted(row_by_setting)}")

    values = {
        setting: finite_array(
            [
                float(row_by_setting[setting]["Avg ACC (%)"])
                if dataset == "Avg"
                else float(row_by_setting[setting][f"{dataset} ACC (%)"])
                for dataset in dataset_fields
            ],
            (len(dataset_fields),),
            setting,
        )
        for setting in settings
    }
    if any((series < 0).any() or (series > 100).any() for series in values.values()):
        raise ValueError("Baseline 2x2 accuracy is outside [0, 100]")

    style_specs = {
        "TIES-Merging": ("TIES", "#C6E0B4", ""),
        "TIES-Merging + LPC": ("TIES + LPC", PAPER_COLORS["TIES-Merging"], "///"),
        "DARE-Linear": ("DARE", "#BDD7EE", ""),
        "DARE-Linear + LPC": ("DARE + LPC", PAPER_COLORS["DARE-Linear"], "///"),
    }

    setup_style("bar")
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 6.8,
            "axes.linewidth": 1.0,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 3.15), dpi=300)
    y = np.arange(len(dataset_fields))
    height = 0.19
    for index, setting in enumerate(settings):
        label, color, hatch = style_specs[setting]
        offset = (index - (len(settings) - 1) / 2) * height
        bars = ax.barh(
            y + offset,
            values[setting],
            height=height,
            label=label,
            color=color,
            edgecolor=darken_color(color, 0.58),
            linewidth=0.75,
            hatch=hatch,
            zorder=3,
        )
        average_bar = bars[-1]
        average_value = values[setting][-1]
        ax.text(
            average_value + 0.8,
            average_bar.get_y() + average_bar.get_height() / 2,
            f"{average_value:.1f}",
            va="center",
            fontsize=5.8,
            color="#30343B",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(["Blood", "Derma", "Organ-C", "Organ-S", "US", "Avg"])
    ax.invert_yaxis()
    ax.axhline(4.5, color="#8C8C8C", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xlim(0, 75)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("Client-average ACC (%)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    polish_axes(ax, y_grid=False, x_grid=True)
    fig.tight_layout(pad=0.45)
    save_png_pdf(fig, str(figure_dir / "03_baseline_2x2_ablation"), dpi=350)
    plt.close(fig)


def plot_internal_ablations(csv_dir: Path, figure_dir: Path) -> None:
    diagnostic = read_annotated_csv(csv_dir / "诊断原型重建内部消融.csv")
    prevalence = read_annotated_csv(csv_dir / "长尾患病率校准内部消融.csv")
    for rows, name in [(diagnostic, "diagnostic"), (prevalence, "prevalence")]:
        values = finite_array([float(row["Avg ACC (%)"]) for row in rows], (len(rows),), name)
        if (values < 0).any() or (values > 100).any():
            raise ValueError(f"{name} accuracy is outside [0, 100]")

    setup_style("dashboard")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.7), dpi=300, gridspec_kw={"width_ratios": [1.55, 1.0]})
    panels = [
        (axes[0], diagnostic, "Diagnostic Prototype Reconstruction\n(DPR)"),
        (axes[1], prevalence, "Long-tail Prevalence Calibration\n(LPC)"),
    ]
    for axis, rows, title in panels:
        labels = [row["Setting"] for row in rows]
        values = np.asarray([float(row["Avg ACC (%)"]) for row in rows])
        colors = []
        for index, label in enumerate(labels):
            if label == "LAMP-Merge":
                colors.append(PAPER_COLORS["LAMP-Merge"])
            elif "prevalence" in label.lower():
                colors.append("#4F81BD" if index % 2 else "#9BBB59")
            elif index <= 4:
                colors.append(["#4F81BD", "#5B9BD5", "#8FAADC", "#4472C4"][max(0, index - 1) % 4])
            else:
                colors.append(["#70AD47", "#A9D18E", "#548235"][index % 3])
        y = np.arange(len(labels))
        bars = axis.barh(
            y,
            values,
            color=colors,
            edgecolor=[darken_color(color, 0.65) for color in colors],
            linewidth=1.2,
            zorder=3,
        )
        axis.set_yticks(y)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 70)
        axis.set_xlabel("Mean ACC (%)")
        axis.set_title(title, fontweight="bold", pad=10, fontsize=16)
        for bar, value in zip(bars, values):
            axis.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=10)
        polish_axes(axis, y_grid=False, x_grid=True)

    fig.tight_layout(w_pad=3.0)
    save_png_pdf(fig, str(figure_dir / "02_internal_module_ablations"), dpi=350)
    plt.close(fig)


def plot_ultrasound_distribution(csv_dir: Path, figure_dir: Path) -> None:
    rows = read_annotated_csv(csv_dir / "超声预测类别分布.csv")
    methods = ["LAMP-Merge", "Weight Averaging", "TIES-Merging", "DARE-Linear", "True distribution"]
    by_series = {row["Series"]: row for row in rows}
    x = np.arange(8)
    setup_style("line")
    fig, ax = plt.subplots(figsize=(10.4, 5.6), dpi=300)
    for index, method in enumerate(methods):
        row = by_series[method]
        values = finite_array(
            [float(row[f"Class {class_index} proportion (%)"]) for class_index in range(8)],
            (8,),
            method,
        )
        marker, linestyle = LINE_STYLES[index]
        label = method if method == "True distribution" else f"{method} (TV={float(row['TV']):.3f})"
        color = PAPER_COLORS[method]
        ax.plot(
            x,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor=color,
            markeredgecolor=darken_color(color, 0.65),
            markeredgewidth=1.4,
            linewidth=2.5,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {index}" for index in x])
    ax.set_xlabel("Ultrasound class")
    ax.set_ylabel("Class proportion (%)")
    ax.set_ylim(0, 35)
    ax.set_title("Predicted Class Distribution on Ultrasound", fontweight="bold", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    polish_axes(ax, y_grid=True, x_grid=False)
    fig.tight_layout()
    save_png_pdf(fig, str(figure_dir / "04_ultrasound_class_distribution"), dpi=350)
    plt.close(fig)


def plot_hparam_curves(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    curve_field: str,
    x_field: str,
    selected_curve: float,
    selected_x: float,
    title: str,
    x_label: str,
    curve_symbol: str,
) -> None:
    grouped: dict[float, dict[float, float]] = defaultdict(dict)
    for row in rows:
        curve_value = float(row[curve_field])
        x_value = float(row[x_field])
        accuracy = float(row["Mean ACC (%)"])
        if not np.isfinite([curve_value, x_value, accuracy]).all():
            raise ValueError(f"Non-finite hyperparameter row: {row}")
        grouped[curve_value][x_value] = accuracy
    curves = sorted(grouped)
    x_values = sorted({value for curve in grouped.values() for value in curve})
    matrix = np.asarray([[grouped[curve][x_value] for x_value in x_values] for curve in curves])
    if matrix.shape != (len(curves), len(x_values)) or not np.isfinite(matrix).all():
        raise ValueError("Hyperparameter grid is incomplete")

    palette = ["#4F81BD", "#F79646", "#9BBB59", "#C0504D", "#8064A2"]
    for index, curve in enumerate(curves):
        marker, linestyle = LINE_STYLES[index % len(LINE_STYLES)]
        axis.plot(
            x_values,
            matrix[index],
            label=rf"${curve_symbol}={curve:g}$",
            color=palette[index % len(palette)],
            marker=marker,
            linestyle=linestyle,
            markeredgecolor=darken_color(palette[index % len(palette)], 0.65),
            markeredgewidth=1.3,
            linewidth=2.2,
            zorder=3,
        )
    selected_curve_index = curves.index(selected_curve)
    selected_x_index = x_values.index(selected_x)
    axis.axvline(selected_x, color="black", linestyle="--", linewidth=1.3, alpha=0.65)
    axis.scatter(
        [selected_x],
        [matrix[selected_curve_index, selected_x_index]],
        marker="*",
        s=180,
        color="#FFD966",
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
        label="Fixed method",
    )
    data_min = float(matrix.min())
    data_max = float(matrix.max())
    data_span = max(data_max - data_min, 0.1)
    expanded_span = 2.5 * data_span
    center = (data_min + data_max) / 2
    axis.set_ylim(center - expanded_span / 2, center + expanded_span / 2)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Client-average ACC (%)")
    axis.set_title(title, fontweight="bold", pad=10)
    axis.legend(loc="best", frameon=False, ncol=2)
    polish_axes(axis, y_grid=True, x_grid=False)


def plot_hparams(csv_dir: Path, figure_dir: Path) -> None:
    diagnostic = read_annotated_csv(csv_dir / "超参数分析_诊断原型重建.csv")
    prevalence = read_annotated_csv(csv_dir / "超参数分析_长尾患病率校准.csv")
    setup_style("line")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), dpi=300)
    plot_hparam_curves(
        axes[0],
        diagnostic,
        "Evidence exponent gamma",
        "Prototype-head scale s",
        0.55,
        18.75,
        "Diagnostic Prototype Reconstruction (DPR)",
        "Prototype-head scale s",
        r"\gamma",
    )
    plot_hparam_curves(
        axes[1],
        prevalence,
        "Activation threshold tau",
        "Calibration strength lambda",
        2.5,
        4.25,
        "Long-tail Prevalence Calibration (LPC)",
        "Calibration strength lambda",
        r"\tau",
    )
    fig.tight_layout(w_pad=2.8)
    save_png_pdf(fig, str(figure_dir / "05_hyperparameter_sensitivity"), dpi=350)
    plt.close(fig)


def plot_tsne(csv_dir: Path, figure_dir: Path) -> None:
    path = csv_dir / "tSNE输出概率坐标.csv"
    if not path.exists():
        return
    rows = read_annotated_csv(path)
    methods = ["LAMP-Merge", "Weight Averaging", "TIES-Merging", "DARE-Linear"]
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["Method"]].append(row)
    if any(len(by_method[method]) == 0 for method in methods):
        raise ValueError("t-SNE CSV does not contain all paper methods")

    setup_style("scatter")
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.0), dpi=300)
    class_colors = plt.get_cmap("tab10")(np.arange(8))
    for axis, method in zip(axes, methods):
        method_rows = sorted(by_method[method], key=lambda row: int(row["Sample index"]))
        coordinates = np.asarray([[float(row["t-SNE x"]), float(row["t-SNE y"])] for row in method_rows])
        labels = np.asarray([int(row["True label"]) for row in method_rows])
        probabilities = np.asarray(
            [[float(row[f"Probability class {index}"]) for index in range(8)] for row in method_rows]
        )
        if coordinates.shape != (len(method_rows), 2) or not np.isfinite(coordinates).all():
            raise ValueError(f"Invalid t-SNE coordinates for {method}")
        if probabilities.shape != (len(method_rows), 8) or not np.isfinite(probabilities).all():
            raise ValueError(f"Invalid probabilities for {method}")
        accuracy = 100 * float(np.mean(np.argmax(probabilities, axis=1) == labels))
        for class_index in range(8):
            mask = labels == class_index
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=9,
                alpha=0.68,
                color=class_colors[class_index],
                edgecolors="none",
                label=f"Class {class_index}",
            )
        axis.set_title(f"{method}\nACC={accuracy:.2f}%", fontweight="bold", fontsize=14)
        axis.set_xticks([])
        axis.set_yticks([])
        polish_axes(axis, y_grid=False, x_grid=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_png_pdf(fig, str(figure_dir / "07_tsne_output_probability"), dpi=350)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, default=ROOT / "论文实验数据")
    args = parser.parse_args()
    figure_dir = args.paper_dir.resolve() / "figures"
    csv_dir = args.csv_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_module_ablation(csv_dir, figure_dir)
    plot_baseline_2x2_ablation(csv_dir, figure_dir)
    plot_internal_ablations(csv_dir, figure_dir)
    plot_ultrasound_distribution(csv_dir, figure_dir)
    plot_hparams(csv_dir, figure_dir)
    plot_tsne(csv_dir, figure_dir)
    print(f"Wrote paper figures to {figure_dir}")


if __name__ == "__main__":
    main()
