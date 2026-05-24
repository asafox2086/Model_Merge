#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec


ABLATION_ORDER = ["full", "no_client_information", "no_fusion_selection", "avg_only"]
DATASET_LABELS = {
    "bloodmnist_224": "Blood",
    "dermamnist_224": "Derma",
    "organcmnist_224": "Organ-C",
    "organsmnist_224": "Organ-S",
    "chaoshengmnist_224": "Ultrasound",
}
ABLATION_LABELS = {
    "full": "Full",
    "no_client_information": "-M1",
    "no_fusion_selection": "-M2",
    "avg_only": "Avg",
}
COLORS = {
    "full": "#2166ac",
    "no_client_information": "#7b3294",
    "no_fusion_selection": "#b2182b",
    "avg_only": "#5f6b6d",
}


def parse_args():
    p = argparse.ArgumentParser("Plot one-page client weight dashboard")
    p.add_argument("--report-dir", default="My_merge_ret/reports")
    p.add_argument("--dest-png", default="My_merge_ret/figures/client_weight_dashboard.png")
    p.add_argument("--dest-svg", default="My_merge_ret/figures/client_weight_dashboard.svg")
    return p.parse_args()


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            converted = {}
            for key, value in row.items():
                num = safe_float(value)
                converted[key] = num if num is not None else value
            rows.append(converted)
        return rows


def by_key(rows, key):
    return {row[key]: row for row in rows}


def grouped_mean(rows, group_key, value_key, filter_fn=None):
    buckets = defaultdict(list)
    for row in rows:
        if filter_fn and not filter_fn(row):
            continue
        value = row.get(value_key)
        if isinstance(value, float):
            buckets[row[group_key]].append(value)
    return {key: sum(values) / len(values) for key, values in buckets.items() if values}


def plot_bar(ax, labels, values, color, ylabel=None, ylim=None):
    ax.bar(labels, values, color=color, edgecolor="#263238", linewidth=0.4)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="x", rotation=0)


def annotate_bars(ax, values, fmt="{:.3f}", ypad=0.006):
    for idx, value in enumerate(values):
        if value is None:
            continue
        ax.text(idx, value + ypad, fmt.format(value), ha="center", va="bottom", fontsize=8)


def draw_heatmap(ax, matrix, row_labels, col_labels, title, cmap="Blues", vmin=None, vmax=None):
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            text = "-" if value is None else f"{value:.3f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=8, color="#111827")
    return image


def top_cases(case_rows, n=10):
    full_rows = [
        row
        for row in case_rows
        if row.get("ablation") == "full" and isinstance(row.get("mean_abs_delta_morph"), float)
    ]
    return sorted(full_rows, key=lambda row: row["mean_abs_delta_morph"], reverse=True)[:n]


def main():
    args = parse_args()
    report_dir = Path(args.report_dir)
    by_ablation = by_key(read_csv(report_dir / "client_weight_by_ablation.csv"), "ablation")
    by_dataset = read_csv(report_dir / "client_weight_by_dataset.csv")
    by_model = read_csv(report_dir / "client_weight_by_model.csv")
    case_rows = read_csv(report_dir / "client_weight_case_summary.csv")

    fig = plt.figure(figsize=(16, 11.2), dpi=180)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.25, 1.25], hspace=0.48, wspace=0.34)
    fig.suptitle("Client Diagnostic Weight Dashboard", fontsize=18, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.955,
        "pi = prior merge weight, alpha_all = M1 diagnostic weight, alpha_morph = medical morphology weight",
        ha="center",
        fontsize=10,
        color="#455a64",
    )

    ax1 = fig.add_subplot(gs[0, 0])
    labels = [ABLATION_LABELS[a] for a in ABLATION_ORDER]
    acc = [by_ablation[a].get("mean_acc") for a in ABLATION_ORDER]
    plot_bar(ax1, labels, acc, [COLORS[a] for a in ABLATION_ORDER], ylabel="Mean accuracy", ylim=(0, max(acc) * 1.25))
    annotate_bars(ax1, acc)
    ax1.set_title("Performance by ablation", loc="left", fontsize=12, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    delta_all = [by_ablation[a].get("mean_abs_delta_all") for a in ABLATION_ORDER]
    delta_morph = [by_ablation[a].get("mean_abs_delta_morph") for a in ABLATION_ORDER]
    x = list(range(len(labels)))
    width = 0.36
    ax2.bar([i - width / 2 for i in x], [v or 0 for v in delta_all], width, label="alpha_all", color="#4d9221")
    ax2.bar([i + width / 2 for i in x], [v or 0 for v in delta_morph], width, label="alpha_morph", color="#c51b7d")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean |alpha - pi|")
    ax2.set_title("How much M1 changes weights", loc="left", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax2.legend(frameon=False, fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    entropy = [by_ablation[a].get("mean_entropy_alpha_all") for a in ABLATION_ORDER]
    plot_bar(ax3, labels, [v if isinstance(v, float) else 0 for v in entropy], [COLORS[a] for a in ABLATION_ORDER], ylabel="Normalized entropy", ylim=(0, 1.08))
    annotate_bars(ax3, [v if isinstance(v, float) else None for v in entropy], fmt="{:.2f}", ypad=0.015)
    ax3.set_title("Weight concentration", loc="left", fontsize=12, fontweight="bold")

    datasets = ["bloodmnist_224", "dermamnist_224", "organcmnist_224", "organsmnist_224", "chaoshengmnist_224"]
    dataset_labels = [DATASET_LABELS[d] for d in datasets]
    ablation_rows = ["full", "no_client_information", "no_fusion_selection"]
    ablation_labels = [ABLATION_LABELS[a] for a in ablation_rows]

    dataset_lookup = {(row["ablation"], row["dataset"]): row for row in by_dataset}
    matrix_delta = [
        [dataset_lookup.get((ablation, dataset), {}).get("mean_abs_delta_morph") for dataset in datasets]
        for ablation in ablation_rows
    ]
    ax4 = fig.add_subplot(gs[1, 0:2])
    draw_heatmap(ax4, matrix_delta, ablation_labels, dataset_labels, "Medical weight shift by dataset", cmap="YlGnBu", vmin=0, vmax=0.18)
    cbar = fig.colorbar(ax4.images[0], ax=ax4, fraction=0.022, pad=0.015)
    cbar.ax.set_ylabel("Mean |alpha_morph - pi|", rotation=270, labelpad=14)

    model_order = ["convnext", "resnet", "swin_tiny", "vit_t", "clip-vit-base-patch32"]
    model_labels = ["ConvNeXt", "ResNet", "Swin-T", "ViT-T", "CLIP"]
    model_lookup = {(row["ablation"], row["model"]): row for row in by_model}
    matrix_model = [
        [model_lookup.get((ablation, model), {}).get("mean_abs_delta_morph") for model in model_order]
        for ablation in ablation_rows
    ]
    ax5 = fig.add_subplot(gs[1, 2])
    draw_heatmap(ax5, matrix_model, ablation_labels, model_labels, "Medical weight shift by model", cmap="PuBuGn", vmin=0, vmax=0.12)

    full_by_dataset = [dataset_lookup.get(("full", dataset), {}).get("mean_acc") for dataset in datasets]
    no_m1_by_dataset = [dataset_lookup.get(("no_client_information", dataset), {}).get("mean_acc") for dataset in datasets]
    no_m2_by_dataset = [dataset_lookup.get(("no_fusion_selection", dataset), {}).get("mean_acc") for dataset in datasets]
    ax6 = fig.add_subplot(gs[2, 0])
    x = list(range(len(datasets)))
    ax6.plot(x, full_by_dataset, marker="o", label="Full", color=COLORS["full"], linewidth=2.0)
    ax6.plot(x, no_m1_by_dataset, marker="o", label="-M1", color=COLORS["no_client_information"], linewidth=1.8)
    ax6.plot(x, no_m2_by_dataset, marker="o", label="-M2", color=COLORS["no_fusion_selection"], linewidth=1.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(dataset_labels, rotation=25, ha="right")
    ax6.set_ylabel("Mean accuracy")
    ax6.set_title("Accuracy sensitivity by dataset", loc="left", fontsize=12, fontweight="bold")
    ax6.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax6.legend(frameon=False, fontsize=9)

    ax7 = fig.add_subplot(gs[2, 1:])
    cases = top_cases(case_rows, 10)
    case_labels = [
        f"{DATASET_LABELS.get(row['dataset'], row['dataset'])} | {row['model']} | c{int(row['num_clients'])}, b{row['beta']:g}"
        for row in cases
    ]
    values = [row["mean_abs_delta_morph"] for row in cases]
    colors = ["#2c7fb8" if row.get("test_acc", 0) >= 0.3 else "#f03b20" for row in cases]
    y = list(range(len(values)))
    ax7.barh(y, values, color=colors, edgecolor="#263238", linewidth=0.4)
    ax7.set_yticks(y)
    ax7.set_yticklabels(case_labels, fontsize=8)
    ax7.invert_yaxis()
    ax7.set_xlabel("Mean |alpha_morph - pi|")
    ax7.set_title("Top full-model cases with strongest medical reweighting", loc="left", fontsize=12, fontweight="bold")
    ax7.grid(axis="x", alpha=0.25, linewidth=0.7)

    fig.text(
        0.015,
        0.015,
        "Interpretation: full and -M2 share non-zero M1 weight shifts; -M1 collapses to prior weights. "
        "The performance gap between full and -M2 shows that selection/fusion is needed to turn weights into accuracy.",
        fontsize=9,
        color="#37474f",
    )

    for path in [Path(args.dest_png), Path(args.dest_svg)]:
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.dest_png, bbox_inches="tight")
    fig.savefig(args.dest_svg, bbox_inches="tight")
    print(f"wrote {args.dest_png}")
    print(f"wrote {args.dest_svg}")


if __name__ == "__main__":
    main()
