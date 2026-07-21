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

from style import ABLATION_COLORS, darken_color, polish_axes, save_png, setup_style  # noqa: E402


PAPER_COLORS = {
    "Weight-Averaging Baseline": "#7F7F7F",
    "Baseline + DPR": "#9BBB59",
    "LAMP-Merge": "#FFC000",
    "Weight Averaging": "#7F7F7F",
    "TIES-Merging": "#9BBB59",
    "DARE-Linear": "#4F81BD",
    "Breadcrumbs": "#8064A2",
    "True distribution": "#8064A2",
}

LINE_STYLES = [
    ("o", "-"),
    ("s", "--"),
    ("^", "-."),
    ("D", ":"),
    ("v", "--"),
]

INTERNAL_ABLATION_ABBREVIATIONS = {
    "Classifier-head aggregation": "CHA",
    "Global-feature mean": "GFM",
    "Support-only synthetic head": "SSH",
    "Shuffled-label prototype": "SLP",
    "Uniform client weight": "UCW",
    "Binary support only": "BSO",
    "Global client-size weight": "GCSW",
    "No prevalence calibration": "NPC",
    "Uniform prevalence prior": "UPP",
    "Always-on calibration": "AOC",
    "Client-balanced prior": "CBP",
    "LAMP-Merge": "LAMP",
}


def expand_interval(limits: tuple[float, float], factor: float) -> tuple[float, float]:
    lower, upper = limits
    center = (lower + upper) / 2
    half_span = (upper - lower) * factor / 2
    return center - half_span, center + half_span


def read_annotated_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first = next(reader)
        header = next(reader) if first and first[0] == "说明" else first
        return [dict(zip(header, row)) for row in reader if row]


def load_completed_dpr_points() -> list[dict[str, str]]:
    roots = [
        ROOT / "outputs" / "lamp_merge_hparam_5x5_20260719_full",
        ROOT / "outputs" / "lamp_merge_hparam_3x10_20260719_full_dpr",
    ]
    rows: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for run_dir in sorted(root.glob("dpr_gamma_*_s_*")):
            status_path = run_dir / "reports" / "batch_status.csv"
            if not status_path.exists():
                continue
            with status_path.open(newline="", encoding="utf-8") as handle:
                status_rows = list(csv.DictReader(handle))
            if len(status_rows) != 180 or any(item["status"] != "OK" for item in status_rows):
                continue
            grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
            for item in status_rows:
                grouped[(item["dataset"], item["model"], item["num_clients"])].append(float(item["test_acc"]))
            if len(grouped) != 60 or any(len(values) != 3 for values in grouped.values()):
                raise ValueError(f"Unexpected DPR coverage in {run_dir}")
            parts = run_dir.name.removeprefix("dpr_gamma_").split("_s_")
            gamma = float(parts[0].replace("p", "."))
            scale = float(parts[1].replace("p", "."))
            rows.append(
                {
                    "Source": "Full configuration grid",
                    "Evidence exponent gamma": f"{gamma:.2f}",
                    "Prototype-head scale s": f"{scale:.2f}",
                    "Mean ACC (%)": f"{100 * np.mean([np.mean(values) for values in grouped.values()]):.4f}",
                }
            )
    unique = {(row["Evidence exponent gamma"], row["Prototype-head scale s"]): row for row in rows}
    return list(unique.values())


def load_completed_lpc_points() -> list[dict[str, str]]:
    root = ROOT / "outputs" / "lamp_merge_hparam_5x5_20260719_full"
    rows: list[dict[str, str]] = []
    if not root.exists():
        return rows
    for run_dir in sorted(root.glob("lpc_tau_*_lambda_*")):
        status_path = run_dir / "reports" / "batch_status.csv"
        if not status_path.exists():
            continue
        with status_path.open(newline="", encoding="utf-8") as handle:
            status_rows = list(csv.DictReader(handle))
        if len(status_rows) != 180 or any(item["status"] != "OK" for item in status_rows):
            continue
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for item in status_rows:
            grouped[(item["dataset"], item["model"], item["num_clients"])].append(float(item["test_acc"]))
        if len(grouped) != 60 or any(len(values) != 3 for values in grouped.values()):
            raise ValueError(f"Unexpected LPC coverage in {run_dir}")
        parts = run_dir.name.removeprefix("lpc_tau_").split("_lambda_")
        tau = float(parts[0].replace("p", "."))
        strength = float(parts[1].replace("p", "."))
        rows.append(
            {
                "Source": "Full configuration grid",
                "Activation threshold tau": f"{tau:.2f}",
                "Calibration strength lambda": f"{strength:.2f}",
                "Mean ACC (%)": f"{100 * np.mean([np.mean(values) for values in grouped.values()]):.4f}",
            }
        )
    return rows


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
    save_png(fig, str(figure_dir / "01_module_ablation_accuracy"), dpi=350)
    plt.close(fig)


def plot_baseline_2x2_ablation(csv_dir: Path, figure_dir: Path) -> None:
    rows = read_annotated_csv(csv_dir / "基线2x2消融.csv")
    baselines = ["TIES-Merging", "DARE-Linear"]
    settings = ["Baseline", "Baseline + DPR", "Baseline + LPC", "Baseline + DPR + LPC"]
    row_by_configuration = {(row["Baseline"], row["Setting"]): row for row in rows}
    expected = {(baseline, setting) for baseline in baselines for setting in settings}
    if set(row_by_configuration) != expected:
        raise ValueError(
            "Unexpected baseline 2x2 configurations: "
            f"{sorted(set(row_by_configuration).symmetric_difference(expected))}"
        )

    values = {
        baseline: finite_array(
            [float(row_by_configuration[(baseline, setting)]["Overall ACC (%)"]) for setting in settings],
            (len(settings),),
            baseline,
        )
        for baseline in baselines
    }
    if any((series < 0).any() or (series > 100).any() for series in values.values()):
        raise ValueError("Baseline 2x2 accuracy is outside [0, 100]")

    setup_style("bar")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.4,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.8,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.78), dpi=300, sharey=True)
    x = np.arange(len(settings))
    for axis, baseline in zip(axes, baselines):
        bars = axis.bar(
            x,
            values[baseline],
            width=0.64,
            color=[*ABLATION_COLORS[:-1], PAPER_COLORS["LAMP-Merge"]],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        for bar, value in zip(bars, values[baseline]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )
        axis.set_title(baseline, pad=3)
        axis.set_xticks([])
        axis.set_ylim(0, 70)
        axis.set_yticks(np.arange(0, 71, 10))
        axis.tick_params(direction="in", top=True, right=True, width=0.7, length=2.5)
        polish_axes(axis, y_grid=True, x_grid=False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)

    fig.text(0.012, 0.56, "Overall client-average ACC (%)", va="center", rotation="vertical", fontsize=7.8)
    fig.legend(
        bars,
        settings,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=5.3,
        handlelength=1.25,
        columnspacing=0.45,
        handletextpad=0.25,
    )
    fig.subplots_adjust(left=0.16, right=0.99, top=0.84, bottom=0.28, wspace=0.24)
    save_png(fig, str(figure_dir / "03_baseline_2x2_ablation"), dpi=350)
    plt.close(fig)


def plot_internal_ablations(csv_dir: Path, figure_dir: Path) -> None:
    diagnostic = read_annotated_csv(csv_dir / "诊断原型重建内部消融.csv")
    prevalence = read_annotated_csv(csv_dir / "长尾患病率校准内部消融.csv")
    for rows, name in [(diagnostic, "diagnostic"), (prevalence, "prevalence")]:
        values = finite_array([float(row["Avg ACC (%)"]) for row in rows], (len(rows),), name)
        if (values < 0).any() or (values > 100).any():
            raise ValueError(f"{name} accuracy is outside [0, 100]")

    setup_style("dashboard")
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.55,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(4.15, 1.95),
        dpi=300,
        gridspec_kw={"width_ratios": [1.2, 1.45]},
    )
    panels = [
        (axes[0], diagnostic, "DPR"),
        (axes[1], prevalence, "LPC"),
    ]
    for axis, rows, title in panels:
        settings = [row["Setting"] for row in rows]
        labels = [INTERNAL_ABLATION_ABBREVIATIONS.get(setting, setting) for setting in settings]
        values = np.asarray([float(row["Avg ACC (%)"]) for row in rows])
        colors = []
        for index, setting in enumerate(settings):
            if setting == "LAMP-Merge":
                colors.append(PAPER_COLORS["LAMP-Merge"])
            elif "prevalence" in setting.lower():
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
            linewidth=0.7,
            zorder=3,
        )
        axis.set_yticks(y)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        if title == "DPR":
            axis.set_xlim(0, 85)
        else:
            axis.set_xlim(55, 65)
        axis.set_xlabel("Mean ACC (%)")
        axis.set_title(title, fontweight="bold", pad=3)
        for bar, value in zip(bars, values):
            axis.text(
                value + 0.7,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontsize=6.3,
            )
        polish_axes(axis, y_grid=False, x_grid=True)
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)

    fig.tight_layout(w_pad=0.7, pad=0.25)
    save_png(fig, str(figure_dir / "02_internal_module_ablations"), dpi=350)
    plt.close(fig)


def plot_ultrasound_distribution(csv_dir: Path, figure_dir: Path) -> None:
    rows = read_annotated_csv(csv_dir / "超声预测类别分布.csv")
    methods = ["LAMP-Merge", "TIES-Merging", "DARE-Linear", "Breadcrumbs"]
    by_series = {row["Series"]: row for row in rows}
    x = np.arange(8)
    setup_style("line")
    fig, ax = plt.subplots(figsize=(10.4, 5.6), dpi=300)
    plotted_values = []
    for index, method in enumerate(methods):
        row = by_series[method]
        values = finite_array(
            [float(row[f"Class {class_index} absolute difference vs. true (pp)"]) for class_index in range(8)],
            (8,),
            method,
        )
        if np.any(values < 0):
            raise ValueError(f"{method} has negative absolute prediction-truth gaps")
        plotted_values.append(values)
        marker, linestyle = LINE_STYLES[index]
        label = method
        color = PAPER_COLORS[method]
        linewidth = 3.8 if method == "LAMP-Merge" else 2.5
        marker_edge_color = "#7F6000" if method == "LAMP-Merge" else darken_color(color, 0.65)
        ax.plot(
            x,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor=color,
            markeredgecolor=marker_edge_color,
            markeredgewidth=1.7 if method == "LAMP-Merge" else 1.4,
            linewidth=linewidth,
            zorder=4 if method == "LAMP-Merge" else 3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {index}" for index in x])
    ax.set_xlabel("Ultrasound class")
    ax.set_ylabel("Absolute predicted--true gap (pp)")
    limit = max(5.0, 5.0 * np.ceil(max(float(values.max()) for values in plotted_values) / 5.0))
    ax.set_ylim(0, limit)
    ax.set_title("Absolute Prediction Gaps on Ultrasound", fontweight="bold", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    polish_axes(ax, y_grid=True, x_grid=False)
    fig.tight_layout()
    save_png(fig, str(figure_dir / "04_ultrasound_class_distribution"), dpi=350)
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
    displayed_curves: list[tuple[str, float]],
    y_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] | None = None,
    legend_location: str = "best",
    legend_anchor: tuple[float, float] | None = None,
) -> None:
    grouped: dict[tuple[str, float], dict[float, float]] = defaultdict(dict)
    for row in rows:
        source = row["Source"]
        curve_value = float(row[curve_field])
        x_value = float(row[x_field])
        accuracy = float(row["Mean ACC (%)"])
        if not np.isfinite([curve_value, x_value, accuracy]).all():
            raise ValueError(f"Non-finite hyperparameter row: {row}")
        grouped[(source, curve_value)][x_value] = accuracy
    missing = [item for item in displayed_curves if item not in grouped]
    if missing:
        raise ValueError(f"Missing requested hyperparameter curves: {missing}")

    palette = ["#4F81BD", "#F79646", "#9BBB59", "#C0504D", "#8064A2", "#70AD47"]
    source_styles = {"Full configuration grid": "-"}
    plotted_values = []
    for index, (source, curve) in enumerate(displayed_curves):
        marker, linestyle = LINE_STYLES[index % len(LINE_STYLES)]
        linestyle = source_styles[source]
        x_values = np.asarray(sorted(grouped[(source, curve)]), dtype=float)
        values = np.asarray([grouped[(source, curve)][value] for value in x_values], dtype=float)
        if len(x_values) < 3 or not np.isfinite(values).all():
            raise ValueError(f"Incomplete hyperparameter curve: {(source, curve)}")
        plotted_values.extend(values)
        axis.plot(
            x_values,
            values,
            label=rf"${curve_symbol}={curve:g}$",
            color=palette[index % len(palette)],
            marker=marker,
            linestyle=linestyle,
            markeredgecolor=darken_color(palette[index % len(palette)], 0.65),
            markeredgewidth=1.3,
            linewidth=2.2,
            zorder=3,
        )
    selected_row = next(
        row
        for row in rows
        if row["Source"] == "Full configuration grid"
        and float(row[curve_field]) == selected_curve
        and float(row[x_field]) == selected_x
    )
    axis.axvline(selected_x, color="black", linestyle="--", linewidth=1.3, alpha=0.65)
    axis.scatter(
        [selected_x],
        [float(selected_row["Mean ACC (%)"])],
        marker="*",
        s=180,
        color=PAPER_COLORS["LAMP-Merge"],
        edgecolor="#7F6000",
        linewidth=1.0,
        zorder=5,
        label="_nolegend_",
    )
    if y_limits is None:
        data_min = min(plotted_values)
        data_max = max(plotted_values)
        data_span = max(data_max - data_min, 0.1)
        expanded_span = 1.35 * data_span
        center = (data_min + data_max) / 2
        axis.set_ylim(center - expanded_span / 2, center + expanded_span / 2)
    else:
        axis.set_ylim(*y_limits)
    if x_limits is not None:
        axis.set_xlim(*x_limits)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Mean ACC (%)")
    axis.set_title(title, fontweight="bold", pad=4)
    legend_kwargs = {"loc": legend_location, "frameon": False, "ncol": 3, "fontsize": 5.2, "columnspacing": 0.6, "handletextpad": 0.3}
    if legend_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = legend_anchor
    axis.legend(**legend_kwargs)
    polish_axes(axis, y_grid=True, x_grid=False)


def plot_hparams(csv_dir: Path, figure_dir: Path) -> None:
    diagnostic = read_annotated_csv(csv_dir / "超参数分析_诊断原型重建.csv")
    if not diagnostic:
        diagnostic = load_completed_dpr_points()
    prevalence = read_annotated_csv(csv_dir / "超参数分析_长尾患病率校准.csv")
    prevalence_by_point = {
        (row["Activation threshold tau"], row["Calibration strength lambda"]): row
        for row in prevalence
    }
    prevalence_by_point.update(
        {
            (row["Activation threshold tau"], row["Calibration strength lambda"]): row
            for row in load_completed_lpc_points()
        }
    )
    prevalence = list(prevalence_by_point.values())
    setup_style("line")
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(4.45, 2.15), dpi=300)
    if diagnostic:
        plot_hparam_curves(
            axes[0],
            diagnostic,
            "Support exponent gamma",
            "Prototype head scale s",
            0.55,
            18.75,
            "DPR",
            r"Prototype-head scale $s$",
            r"\gamma",
            [("Full configuration grid", value) for value in (0.4, 0.45, 0.5, 0.55, 0.6)],
            legend_location="lower left",
            legend_anchor=(0.01, 0.02),
            x_limits=(13.25, 24.25),
            y_limits=expand_interval((61.3, 62.7), 3.0),
        )
    else:
        dpr_axis = axes[0]
        dpr_axis.set_xlim(16.25, 21.25)
        dpr_axis.set_ylim(55.0, 65.0)
        dpr_axis.set_xlabel(r"Prototype-head scale $s$")
        dpr_axis.set_ylabel("Mean ACC (%)")
        dpr_axis.set_title("DPR", fontweight="bold", pad=4)
        polish_axes(dpr_axis, y_grid=True, x_grid=False)
    lpc_axis = axes[1]
    plot_hparam_curves(
        lpc_axis,
        prevalence,
        "Activation threshold tau",
        "Calibration strength lambda",
        2.5,
        4.25,
        "LPC",
        r"Calibration strength $\lambda$",
        r"\tau",
        [
            ("Full configuration grid", 1.5),
            ("Full configuration grid", 2.0),
            ("Full configuration grid", 2.5),
            ("Full configuration grid", 3.0),
            ("Full configuration grid", 3.5),
        ],
        expand_interval((60.6, 63.4), 3.0),
        (3.75, 5.25),
        legend_location="lower center",
        legend_anchor=(0.5, 0.05),
    )
    for axis in axes:
        lower, upper = axis.get_ylim()
        axis.set_yticks(np.linspace(lower, upper, 5))
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
    fig.tight_layout(w_pad=0.65, pad=0.35)
    save_png(fig, str(figure_dir / "05_hyperparameter_sensitivity"), dpi=350)
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
    save_png(fig, str(figure_dir / "07_tsne_output_probability"), dpi=350)
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
