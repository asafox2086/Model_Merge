#!/usr/bin/env python3
"""Plot prefix-wise BloodMNIST accuracy for TIES and LAMP-Merge."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import darken_color, polish_axes, save_png_pdf, setup_style


ROOT = Path(__file__).resolve().parent
LAMP_RESULTS = ROOT.parent / "new_lamp_merge_async_grid" / "async_grid_k1_to_k7.csv"


def load_series():
    ties_payload = json.loads((ROOT / "ties_k1_to_k7.json").read_text(encoding="utf-8"))
    ties = [float(row["acc"]) for row in ties_payload["rows"]]
    with LAMP_RESULTS.open(newline="", encoding="utf-8") as handle:
        lamp = [
            float(row["acc"])
            for row in csv.DictReader(handle)
            if row["dataset"] == "bloodmnist_224" and row["backbone"] == "resnet"
        ]
    if len(ties) != 7 or len(lamp) != 7:
        raise ValueError("Both methods must contain k=1..7 results")
    values = np.asarray([ties, lamp])
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Accuracy values must be finite probabilities")
    return ties, lamp


def save_data(ties, lamp):
    rows = []
    for method, values in (("TIES", ties), ("LAMP-Merge", lamp)):
        rows.extend({"method": method, "k": k, "acc": f"{acc:.12f}"} for k, acc in enumerate(values, start=1))
    with (ROOT / "ties_vs_lamp_merge_k1_to_k7.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "k", "acc"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(ties, lamp):
    setup_style()
    figure, axis = plt.subplots(figsize=(8.2, 5.2), dpi=300)
    clients = np.arange(1, 8)
    series = [
        ("TIES (baseline)", ties, "#C0504D", "--", "s"),
        ("LAMP-Merge (ours)", lamp, "#4F81BD", "-", "o"),
    ]
    for label, values, color, linestyle, marker in series:
        axis.plot(
            clients,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor=darken_color(color, 0.65),
            markeredgewidth=1.5,
            linewidth=3.0,
            markersize=10,
            zorder=3,
        )
    axis.set_title("BloodMNIST / ResNet: Acc", pad=5, fontweight="bold")
    axis.set_xlabel("Received clients (k)", labelpad=3)
    axis.set_ylabel("Test Acc", labelpad=3)
    axis.set_xticks(clients)
    axis.set_ylim(0, 0.9)
    axis.tick_params(pad=3)
    axis.legend(loc="center right", bbox_to_anchor=(0.99, 0.53), frameon=False, handletextpad=0.4, labelspacing=0.25)
    polish_axes(axis)
    figure.tight_layout(pad=0.45)
    save_png_pdf(figure, str(ROOT / "ties_vs_lamp_merge_acc_k1_to_k7"))
    plt.close(figure)


if __name__ == "__main__":
    ties_values, lamp_values = load_series()
    save_data(ties_values, lamp_values)
    plot(ties_values, lamp_values)
