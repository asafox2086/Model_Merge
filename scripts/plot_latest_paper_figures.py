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


def parse_metric_pair(value: str, name: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split("/")]
    if len(parts) != 2:
        raise ValueError(f"{name} is not an ACC / F1 pair: {value}")
    acc, macro_f1 = (float(part) for part in parts)
    if not np.isfinite([acc, macro_f1]).all() or not (0 <= acc <= 100) or not (0 <= macro_f1 <= 100):
        raise ValueError(f"{name} has invalid metric values: {value}")
    return acc, macro_f1


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

    values = {}
    for baseline in baselines:
        metric_pairs = finite_array(
            [
                metric
                for setting in settings
                for metric in parse_metric_pair(
                    row_by_configuration[(baseline, setting)]["Overall ACC / F1 (%)"],
                    f"{baseline}:{setting}",
                )
            ],
            (2 * len(settings),),
            baseline,
        ).reshape(len(settings), 2)
        values[baseline] = metric_pairs
    if any((series < 0).any() or (series > 100).any() for series in values.values()):
        raise ValueError("Baseline 2x2 metrics are outside [0, 100]")

    setup_style("bar")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.3,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(4.45, 1.82), dpi=300, sharey=True)
    x = np.arange(len(settings))
    width = 0.33
    setting_colors = [*ABLATION_COLORS[:-1], PAPER_COLORS["LAMP-Merge"]]
    legend_handles = []
    for axis, baseline in zip(axes, baselines):
        acc_values = values[baseline][:, 0]
        f1_values = values[baseline][:, 1]
        acc_bars = axis.bar(
            x - width / 2,
            acc_values,
            width=width,
            color=setting_colors,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
            label="ACC",
        )
        f1_bars = axis.bar(
            x + width / 2,
            f1_values,
            width=width,
            color=[darken_color(color, 0.86) for color in setting_colors],
            edgecolor="black",
            linewidth=0.7,
            hatch="///",
            zorder=3,
            label="Macro-F1",
        )
        if not legend_handles:
            legend_handles = [acc_bars[0], f1_bars[0]]
        for bars, series in [(acc_bars, acc_values), (f1_bars, f1_values)]:
            for bar, value in zip(bars, series):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.0,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.2,
                    rotation=90,
                )
        axis.set_title(baseline, pad=2)
        axis.set_xticks([])
        axis.set_ylim(0, 70)
        axis.set_yticks(np.arange(0, 71, 10))
        axis.tick_params(direction="in", top=True, right=True, width=0.7, length=2.5)
        polish_axes(axis, y_grid=True, x_grid=False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)

    axes[0].set_ylabel("Overall client-average score (%)", labelpad=3)
    fig.legend(
        legend_handles,
        ["ACC", "Macro-F1"],
        loc="upper center",
        bbox_to_anchor=(0.56, 0.995),
        ncol=2,
        frameon=False,
        fontsize=5.6,
        handlelength=1.35,
        columnspacing=0.85,
        handletextpad=0.35,
    )
    fig.legend(
        acc_bars,
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
    fig.subplots_adjust(left=0.115, right=0.995, top=0.79, bottom=0.27, wspace=0.18)
    save_png(fig, str(figure_dir / "03_baseline_2x2_ablation"), dpi=350)
    plt.close(fig)


def plot_internal_ablations(csv_dir: Path, figure_dir: Path) -> None:
    diagnostic = read_annotated_csv(csv_dir / "诊断原型重建内部消融.csv")
    prevalence = read_annotated_csv(csv_dir / "长尾患病率校准内部消融.csv")
    for rows, name in [(diagnostic, "diagnostic"), (prevalence, "prevalence")]:
        values = finite_array(
            [metric for row in rows for metric in parse_metric_pair(row["Avg ACC / F1 (%)"], f"{name}:{row['Setting']}")],
            (2 * len(rows),),
            name,
        )
        if (values < 0).any() or (values > 100).any():
            raise ValueError(f"{name} metrics are outside [0, 100]")

    setup_style("dashboard")
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.55,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(4.45, 2.22),
        dpi=300,
        gridspec_kw={"width_ratios": [1.2, 1.05]},
    )
    panels = [
        (axes[0], diagnostic, "DPR"),
        (axes[1], prevalence, "LPC"),
    ]
    legend_handles = None
    for axis, rows, title in panels:
        settings = [row["Setting"] for row in rows]
        labels = [INTERNAL_ABLATION_ABBREVIATIONS.get(setting, setting) for setting in settings]
        metric_pairs = np.asarray(
            [parse_metric_pair(row["Avg ACC / F1 (%)"], f"{title}:{row['Setting']}") for row in rows],
            dtype=float,
        )
        acc_values = metric_pairs[:, 0]
        f1_values = metric_pairs[:, 1]
        y = np.arange(len(labels))
        height = 0.34
        acc_bars = axis.barh(
            y - height / 2,
            acc_values,
            height=height,
            color="#4F81BD",
            edgecolor=darken_color("#4F81BD", 0.65),
            linewidth=0.7,
            zorder=3,
            label="ACC",
        )
        f1_bars = axis.barh(
            y + height / 2,
            f1_values,
            height=height,
            color="#F79646",
            edgecolor=darken_color("#F79646", 0.65),
            linewidth=0.7,
            zorder=3,
            label="Macro-F1",
        )
        legend_handles = (acc_bars[0], f1_bars[0])
        axis.set_yticks(y)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 85)
        axis.set_xlabel("Mean score (%)")
        axis.set_title(title, fontweight="bold", pad=3)
        for bars, values in [(acc_bars, acc_values), (f1_bars, f1_values)]:
            for bar, value in zip(bars, values):
                axis.text(
                    value + 0.75,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}",
                    va="center",
                    fontsize=5.8,
                )
        polish_axes(axis, y_grid=False, x_grid=True)
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)

    if legend_handles is None:
        raise ValueError("No internal ablation bars were plotted")
    fig.legend(
        legend_handles,
        ["ACC", "Macro-F1"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.tight_layout(w_pad=0.7, pad=0.25)
    fig.subplots_adjust(bottom=0.2)
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
            x_limits=(12.0, 24.25),
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
        (2.9, 5.35),
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


RADAR_DATASET_LABELS = {
    "bloodmnist_224": "Blood",
    "dermamnist_224": "Derma",
    "organcmnist_224": "Organ-C",
    "organsmnist_224": "Organ-S",
    "chaoshengmnist_224": "Ultrasound",
}

RADAR_BACKBONE_LABELS = {
    "resnet": "ResNet",
    "convnext": "ConvNeXt",
    "vit_t": "ViT-Tiny",
    "swin_tiny": "Swin-Tiny",
}

RADAR_METHOD_LABELS = {
    "avg": "Weight Avg",
    "ties": "TIES",
    "dare_linear": "DARE-Linear",
    "dare_ties": "DARE-TIES",
    "regmean": "RegMean",
    "fisher": "Fisher",
    "breadcrumbs": "Breadcrumbs",
    "model_stock": "Model Stock",
    "from": "FROM",
    "iso_c": "Iso-C",
    "free_merge": "FreeMerge",
    "robustmerge": "RobustMerge",
    "lamp_merge": "LAMP-Merge",
}

RADAR_METHOD_ORDER = list(RADAR_METHOD_LABELS)
RADAR_AXES = ["ACC", "F1", "AUC"]
RADAR_METHOD_STYLES = {
    "Weight Avg": {"color": "#9E9E9E", "marker": "o", "linestyle": "-", "linewidth": 1.05},
    "TIES": {"color": "#6E9E5E", "marker": "s", "linestyle": "--", "linewidth": 1.05},
    "DARE-Linear": {"color": "#5A86C9", "marker": "^", "linestyle": "-.", "linewidth": 1.05},
    "DARE-TIES": {"color": "#3F6DB5", "marker": "v", "linestyle": ":", "linewidth": 1.05},
    "RegMean": {"color": "#D28B57", "marker": "D", "linestyle": "-", "linewidth": 1.05},
    "Fisher": {"color": "#B35C7B", "marker": "P", "linestyle": "--", "linewidth": 1.05},
    "Breadcrumbs": {"color": "#8A72B7", "marker": "X", "linestyle": "-.", "linewidth": 1.05},
    "Model Stock": {"color": "#2F9F9B", "marker": "h", "linestyle": ":", "linewidth": 1.05},
    "FROM": {"color": "#94703B", "marker": "<", "linestyle": "-", "linewidth": 1.05},
    "Iso-C": {"color": "#C45A51", "marker": ">", "linestyle": "--", "linewidth": 1.05},
    "FreeMerge": {"color": "#94C55F", "marker": "8", "linestyle": "-.", "linewidth": 1.05},
    "RobustMerge": {"color": "#7A5BBF", "marker": "p", "linestyle": ":", "linewidth": 1.05},
    "LAMP-Merge": {"color": "#FFC000", "marker": "*", "linestyle": "-", "linewidth": 2.9},
}


def load_dataset_radar_rows(csv_dir: Path) -> list[dict[str, str]]:
    auc_path = ROOT / "My_merge_ret" / "reports" / "prediction_diagnostics_auc.csv"
    vit_tiny_auc_path = ROOT / "My_merge_ret" / "reports" / "prediction_diagnostics_auc_vit_tiny.csv"
    diagnostics_path = (
        auc_path
        if auc_path.exists()
        else vit_tiny_auc_path
        if vit_tiny_auc_path.exists()
        else ROOT / "My_merge_ret" / "reports" / "prediction_diagnostics_full.csv"
    )
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with diagnostics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status", "").upper() != "OK":
                continue
            dataset = row.get("dataset", "")
            backbone = row.get("model", "")
            method = row.get("method", "")
            if dataset not in RADAR_DATASET_LABELS or backbone not in RADAR_BACKBONE_LABELS or method not in RADAR_METHOD_LABELS:
                continue
            if not row.get("accuracy") or not row.get("macro_f1"):
                continue
            if "macro_auc" not in row or row.get("macro_auc", "") == "":
                continue
            key = (RADAR_BACKBONE_LABELS[backbone], RADAR_DATASET_LABELS[dataset], RADAR_METHOD_LABELS[method])
            grouped[key]["ACC"].append(float(row["accuracy"]) * 100.0)
            grouped[key]["F1"].append(float(row["macro_f1"]) * 100.0)
            grouped[key]["AUC"].append(float(row["macro_auc"]) * 100.0)

    output_rows: list[dict[str, str]] = []
    for backbone in RADAR_BACKBONE_LABELS.values():
        for dataset in RADAR_DATASET_LABELS.values():
            for method_key in RADAR_METHOD_ORDER:
                method = RADAR_METHOD_LABELS[method_key]
                metrics = grouped.get((backbone, dataset, method))
                if not metrics:
                    continue
                output_rows.append(
                    {
                        "Backbone": backbone,
                        "Dataset": dataset,
                        "Method": method,
                        "Cases": str(len(metrics["ACC"])),
                        "ACC (%)": f"{np.mean(metrics['ACC']):.2f}",
                        "F1 (%)": f"{np.mean(metrics['F1']):.2f}",
                        "AUC (%)": f"{np.mean(metrics['AUC']):.2f}",
                    }
                )
    return output_rows


def write_dataset_radar_csv(rows: list[dict[str, str]], path: Path) -> None:
    fields = [
        "Backbone",
        "Dataset",
        "Method",
        "Cases",
        "ACC (%)",
        "F1 (%)",
        "AUC (%)",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        handle.write("说明,本表对应 backbone 级雷达图；每张图表示一个 backbone，图内三个子图分别表示 ACC、F1、AUC，雷达图五个顶点为五个医学数据集，曲线为 LAMP-Merge 及所有通用模型融合基线。数值对 K=3/5/7 和三个 beta 设置取平均。\n")
        writer.writeheader()
        writer.writerows(rows)


def radar_axis_limits(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    lower = max(0.0, 5.0 * np.floor((float(values.min()) - 5.0) / 5.0))
    upper = min(100.0, 5.0 * np.ceil((float(values.max()) + 5.0) / 5.0))
    if upper - lower < 15.0:
        center = (upper + lower) / 2.0
        lower = max(0.0, center - 7.5)
        upper = min(100.0, center + 7.5)
    ticks = np.linspace(lower, upper, 4)
    return lower, upper, ticks


def draw_metric_radar(
    axis: plt.Axes,
    metric: str,
    labels: list[str],
    series: dict[str, np.ndarray],
    show_legend: bool = False,
) -> None:
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    all_values = np.concatenate(list(series.values()))
    lower, upper, ticks = radar_axis_limits(all_values)
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_xticks(angles)
    axis.set_xticklabels(labels, fontsize=6.2)
    axis.tick_params(axis="x", pad=2.0)
    axis.set_ylim(lower, upper)
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{tick:.0f}" for tick in ticks], fontsize=5.3)
    axis.set_rlabel_position(10)
    axis.grid(True, linestyle="-", color="#D6D6D6", alpha=0.72, linewidth=0.55)
    axis.spines["polar"].set_color("#1F1F1F")
    axis.spines["polar"].set_linewidth(0.85)
    for name, values in series.items():
        closed_values = np.concatenate([values, values[:1]])
        style = RADAR_METHOD_STYLES[name]
        axis.plot(
            closed_angles,
            closed_values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=2.4 if name != "LAMP-Merge" else 4.8,
            linewidth=0.85 if name != "LAMP-Merge" else style["linewidth"],
            label=name,
            alpha=0.50 if name != "LAMP-Merge" else 1.0,
            zorder=5 if name == "LAMP-Merge" else 3,
        )
        if name == "LAMP-Merge":
            axis.fill(closed_angles, closed_values, color=style["color"], alpha=0.055, zorder=1)
    axis.set_title(metric, fontsize=8.8, fontweight="bold", pad=0.2)
    if show_legend:
        axis.legend(
            loc="center left",
            bbox_to_anchor=(1.18, 0.5),
            ncol=1,
            frameon=False,
            fontsize=7.1,
            handlelength=1.35,
            labelspacing=0.45,
        )


def plot_dataset_radars(csv_dir: Path, figure_dir: Path) -> None:
    rows = load_dataset_radar_rows(csv_dir)
    write_dataset_radar_csv(rows, csv_dir / "数据集雷达图.csv")
    backbones = list(RADAR_BACKBONE_LABELS.values())
    datasets = ["Ultrasound", "Derma", "Organ-C", "Organ-S", "Blood"]
    methods = [RADAR_METHOD_LABELS[key] for key in RADAR_METHOD_ORDER]
    row_by_key = {(row["Backbone"], row["Dataset"], row["Method"]): row for row in rows}
    setup_style("dashboard")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.titlesize": 12.0,
            "legend.fontsize": 8.2,
        }
    )

    out_dir = figure_dir / "backbone_radars"
    out_dir.mkdir(parents=True, exist_ok=True)
    first_out_base: Path | None = None
    for backbone in backbones:
        if not any(row["Backbone"] == backbone for row in rows):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.72), dpi=300, subplot_kw={"projection": "polar"})
        flat_axes = list(axes.ravel())
        figure_handles = None
        figure_labels = None
        for axis, metric in zip(flat_axes, RADAR_AXES):
            series = {
                method: np.asarray([float(row_by_key[(backbone, dataset, method)][f"{metric} (%)"]) for dataset in datasets])
                for method in methods
                if all((backbone, dataset, method) in row_by_key for dataset in datasets)
            }
            draw_metric_radar(axis, metric, datasets, series, show_legend=False)
            if figure_handles is None:
                figure_handles, figure_labels = axis.get_legend_handles_labels()
        if figure_handles is None or figure_labels is None:
            raise ValueError(f"No radar handles were created for {backbone}")
        fig.legend(
            figure_handles,
            figure_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=7,
            frameon=False,
            fontsize=5.35,
            handlelength=1.25,
            columnspacing=0.52,
            labelspacing=0.22,
        )
        fig.suptitle(backbone, fontsize=11.4, fontweight="bold", y=0.975)
        fig.subplots_adjust(left=0.04, right=0.985, bottom=0.245, top=0.75, wspace=0.055)
        out_base = out_dir / f"{backbone.lower().replace('-', '_')}_radar"
        save_png(fig, str(out_base), dpi=350)
        fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
        plt.close(fig)
        if first_out_base is None:
            first_out_base = out_base

    if first_out_base is not None:
        import shutil

        shutil.copyfile(str(first_out_base) + ".png", figure_dir / "09_dataset_radar_effects.png")
        shutil.copyfile(str(first_out_base) + ".pdf", figure_dir / "09_dataset_radar_effects.pdf")


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
    plot_dataset_radars(csv_dir, figure_dir)
    print(f"Wrote paper figures to {figure_dir}")


if __name__ == "__main__":
    main()
