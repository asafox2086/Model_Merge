#!/usr/bin/env python3
"""Plot separate TIES and LAMP-Merge BloodMNIST trajectories by backbone."""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import darken_color, polish_axes, save_png_pdf, setup_style


ROOT = Path(__file__).resolve().parent
LAMP_RESULTS = ROOT.parent / "new_lamp_merge_async_grid" / "async_grid_k1_to_k7.csv"
BACKBONES = ("resnet", "convnext", "vit_t", "swin_tiny")
LABELS = {"resnet": "ResNet", "convnext": "ConvNeXt", "vit_t": "ViT-T", "swin_tiny": "Swin-T"}
COLORS = {"resnet": "#4F81BD", "convnext": "#C0504D", "vit_t": "#9BBB59", "swin_tiny": "#8064A2"}
MARKERS = {"resnet": "o", "convnext": "D", "vit_t": "^", "swin_tiny": "s"}
LINESTYLES = {"resnet": "-", "convnext": "--", "vit_t": "-.", "swin_tiny": ":"}


def load_ties_rows():
    rows = []
    for backbone in BACKBONES:
        path = ROOT / "ties_prefixes" / backbone / "ties_k1_to_k7.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("backbone") != backbone or [row["k"] for row in payload["rows"]] != list(range(1, 8)):
            raise ValueError(f"Invalid TIES prefix result: {path}")
        for row in payload["rows"]:
            rows.append({"method": "TIES", "backbone": backbone, "k": int(row["k"]), "acc": float(row["acc"]), "macro_f1": float(row["macro_f1"])})
    return rows


def load_lamp_rows():
    rows = []
    with LAMP_RESULTS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "bloodmnist_224" and row["backbone"] in BACKBONES:
                rows.append({"method": "LAMP-Merge", "backbone": row["backbone"], "k": int(row["k"]), "acc": float(row["acc"]), "macro_f1": float(row["macro_f1"])})
    return rows


def validate(rows):
    expected = {(method, backbone) for method in ("TIES", "LAMP-Merge") for backbone in BACKBONES}
    actual = {(row["method"], row["backbone"]) for row in rows}
    if actual != expected:
        raise ValueError("Missing method/backbone trajectory")
    for method, backbone in expected:
        trajectory = [row for row in rows if row["method"] == method and row["backbone"] == backbone]
        if [row["k"] for row in trajectory] != list(range(1, 8)):
            raise ValueError(f"Invalid k ordering for {method}/{backbone}")
        for row in trajectory:
            if not all(math.isfinite(row[metric]) and 0 <= row[metric] <= 1 for metric in ("acc", "macro_f1")):
                raise ValueError(f"Invalid metric value for {method}/{backbone}")


def write_data(rows):
    path = ROOT / "ties_vs_lamp_merge_backbones_k1_to_k7.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "backbone", "k", "acc", "macro_f1"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({**row, "acc": f"{row['acc']:.12f}", "macro_f1": f"{row['macro_f1']:.12f}"} for row in rows)


def draw_panel(axis, rows, method, metric, title, ylabel):
    for backbone in BACKBONES:
        values = [row[metric] for row in rows if row["method"] == method and row["backbone"] == backbone]
        color = COLORS[backbone]
        axis.plot(
            range(1, 8), values, label=LABELS[backbone], color=color,
            linestyle=LINESTYLES[backbone], marker=MARKERS[backbone],
            markerfacecolor=color, markeredgecolor=darken_color(color, 0.65),
            markeredgewidth=1.1, linewidth=2.4, markersize=6.5, zorder=3,
        )
    axis.set_title(title, pad=5, fontweight="bold")
    axis.set_xlabel("Received clients (k)", labelpad=3)
    axis.set_ylabel(ylabel, labelpad=3)
    axis.set_xticks(range(1, 8))
    axis.set_ylim(0, 1.02)
    axis.tick_params(pad=3)
    polish_axes(axis)


def plot_metric(rows, metric, ylabel, output_base):
    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=300, sharex=True, sharey=True)
    draw_panel(axes[0], rows, "LAMP-Merge", metric, f"BloodMNIST: {ylabel} / LAMP-Merge", f"Test {ylabel}")
    draw_panel(axes[1], rows, "TIES", metric, f"BloodMNIST: {ylabel} / TIES", f"Test {ylabel}")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.015), handletextpad=0.35, columnspacing=1.1)
    figure.tight_layout(rect=(0, 0.11, 1, 1), pad=0.45)
    save_png_pdf(figure, str(output_base))
    plt.close(figure)


def plot_combined(rows):
    setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), dpi=300, sharex=True, sharey=True)
    draw_panel(axes[0, 0], rows, "LAMP-Merge", "acc", "BloodMNIST: Acc / LAMP-Merge", "Test Acc")
    draw_panel(axes[0, 1], rows, "TIES", "acc", "BloodMNIST: Acc / TIES", "Test Acc")
    draw_panel(axes[1, 0], rows, "LAMP-Merge", "macro_f1", "BloodMNIST: F1 / LAMP-Merge", "Test F1")
    draw_panel(axes[1, 1], rows, "TIES", "macro_f1", "BloodMNIST: F1 / TIES", "Test F1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005), handletextpad=0.35, columnspacing=1.1)
    figure.tight_layout(rect=(0, 0.055, 1, 1), pad=0.45)
    save_png_pdf(figure, str(ROOT / "ties_vs_lamp_merge_blood_acc_macro_f1_k1_to_k7"))
    plt.close(figure)


if __name__ == "__main__":
    all_rows = load_ties_rows() + load_lamp_rows()
    validate(all_rows)
    write_data(all_rows)
    plot_metric(all_rows, "acc", "Acc", ROOT / "ties_vs_lamp_merge_blood_acc_k1_to_k7")
    plot_metric(all_rows, "macro_f1", "F1", ROOT / "ties_vs_lamp_merge_blood_macro_f1_k1_to_k7")
    plot_combined(all_rows)
