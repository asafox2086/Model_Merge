#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "My_merge_ret/reports/lamp_merge_hparam_sensitivity.csv"
FIG_PATH = ROOT / "My_merge_ret/figures/lamp_merge_hparam_sensitivity.png"
PDF_PATH = ROOT / "My_merge_ret/figures/lamp_merge_hparam_sensitivity.pdf"
DEFAULT_VALUES = {
    ("M1", "prototype head scale"): 20.0,
    ("M2", "long-tail bias strength"): 6.0,
}


def read_rows():
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    grouped = defaultdict(list)
    for row in read_rows():
        grouped[(row["module"], row["parameter"])].append(row)

    order = [
        ("M1", "prototype head scale"),
        ("M2", "long-tail bias strength"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), dpi=220)
    for ax, key in zip(axes, order):
        rows = sorted(grouped[key], key=lambda item: float(item["value"]))
        xs = [float(row["value"]) for row in rows]
        ys = [float(row["mean_acc"]) for row in rows]
        module, parameter = key
        best_idx = max(range(len(ys)), key=lambda idx: ys[idx])
        default_value = DEFAULT_VALUES.get(key)
        ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=1.8, markersize=4.0)
        ax.scatter(
            [xs[best_idx]],
            [ys[best_idx]],
            marker="*",
            s=90,
            color="#d62728",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            label="grid best",
        )
        if default_value is not None and default_value in xs:
            default_idx = xs.index(default_value)
            ax.axvline(
                default_value,
                color="#2ca02c",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label="default",
            )
            ax.scatter(
                [default_value],
                [ys[default_idx]],
                marker="s",
                s=34,
                color="#2ca02c",
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
        ax.set_title(f"{module}: {parameter}", fontsize=9)
        ax.set_xlabel("hyperparameter value", fontsize=8)
        ax.set_ylabel("test accuracy", fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_ylim(min(ys) - 0.003, max(ys) + 0.003)
        ax.legend(loc="best", fontsize=7, frameon=False)
    fig.suptitle("Hyperparameter sensitivity on dermamnist_224 / ResNet / K=3 / beta=0.1", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(f"wrote {FIG_PATH}")
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
