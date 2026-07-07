#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "My_merge_ret/reports/lamp_merge_hparam_sensitivity.csv"
FIG_PATH = ROOT / "My_merge_ret/figures/lamp_merge_hparam_sensitivity.png"


def read_rows():
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    grouped = defaultdict(list)
    for row in read_rows():
        grouped[(row["module"], row["parameter"])].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=180)
    for ax, key in zip(axes, sorted(grouped)):
        rows = sorted(grouped[key], key=lambda item: float(item["value"]))
        xs = [float(row["value"]) for row in rows]
        ys = [float(row["mean_acc"]) for row in rows]
        module, parameter = key
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.8)
        ax.set_title(f"{module}: {parameter}", fontsize=9)
        ax.set_xlabel("value", fontsize=8)
        ax.set_ylabel("test accuracy", fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_ylim(min(ys) - 0.004, max(ys) + 0.004)
    fig.suptitle("LAMP-Merge hyperparameter sensitivity on dermamnist_224 / ResNet / K=3 / beta=0.1", fontsize=9)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, bbox_inches="tight")
    print(f"wrote {FIG_PATH}")


if __name__ == "__main__":
    main()
