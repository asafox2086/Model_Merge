#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ABLATION_LABELS = {
    "full": "Full",
    "no_client_information": "-M1",
    "no_fusion_selection": "-M2",
    "no_adaptive_candidates": "-M3",
    "avg_only": "Avg",
}
DATASET_LABELS = {
    "bloodmnist_224": "Blood",
    "dermamnist_224": "Derma",
    "organcmnist_224": "Organ-C",
    "organsmnist_224": "Organ-S",
    "chaoshengmnist_224": "Ultrasound",
}
MODEL_LABELS = {
    "convnext": "ConvNeXt",
    "resnet": "ResNet",
    "swin_tiny": "Swin-T",
    "vit_t": "ViT-T",
    "clip-vit-base-patch32": "CLIP",
}


def parse_args():
    p = argparse.ArgumentParser("Plot simple client weight ratio analysis")
    p.add_argument("--detail-csv", default="My_merge_ret/reports/client_weight_detail.csv")
    p.add_argument("--dest-png", default="My_merge_ret/client_weight_ratio_analysis.png")
    p.add_argument("--dest-svg", default="My_merge_ret/client_weight_ratio_analysis.svg")
    p.add_argument("--dest-md", default="My_merge_ret/reports/client_weight_ratio_analysis.md")
    return p.parse_args()


def safe_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pi = safe_float(row.get("prior_weight_pi"))
            alpha_all = safe_float(row.get("diagnostic_weight_alpha_all"))
            alpha_morph = safe_float(row.get("medical_weight_alpha_morph"))
            if pi and pi > 0:
                row["ratio_all"] = alpha_all / pi if alpha_all is not None else None
                row["ratio_morph"] = alpha_morph / pi if alpha_morph is not None else None
            else:
                row["ratio_all"] = None
                row["ratio_morph"] = None
            row["num_clients"] = int(float(row["num_clients"]))
            row["beta"] = float(row["beta"])
            rows.append(row)
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def case_key(row):
    return (
        row["ablation"],
        row["task_type"],
        row["dataset"],
        row["model"],
        row["num_clients"],
        row["beta"],
        row["seed"],
    )


def case_spreads(rows, ratio_key):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(ratio_key)
        if value is not None:
            grouped[case_key(row)].append(value)
    out = []
    for key, values in grouped.items():
        if not values:
            continue
        min_value = min(values)
        max_value = max(values)
        spread = max_value / min_value if min_value > 0 else None
        ablation, task_type, dataset, model, num_clients, beta, seed = key
        out.append(
            {
                "ablation": ablation,
                "task_type": task_type,
                "dataset": dataset,
                "model": model,
                "num_clients": num_clients,
                "beta": beta,
                "seed": seed,
                "min_ratio": min_value,
                "max_ratio": max_value,
                "spread": spread,
            }
        )
    return out


def group_mean(rows, group_key, value_key, filter_fn=None):
    buckets = defaultdict(list)
    for row in rows:
        if filter_fn and not filter_fn(row):
            continue
        value = row.get(value_key)
        if value is not None:
            buckets[row[group_key]].append(value)
    return {key: mean(values) for key, values in buckets.items()}


def draw_boxplot(ax, rows, ratio_key, title):
    ablations = ["full", "no_client_information", "no_fusion_selection"]
    data = [[row[ratio_key] for row in rows if row["ablation"] == ablation and row[ratio_key] is not None] for ablation in ablations]
    labels = [ABLATION_LABELS[a] for a in ablations]
    boxplot_kwargs = {"patch_artist": True, "showfliers": False}
    try:
        box = ax.boxplot(data, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        box = ax.boxplot(data, labels=labels, **boxplot_kwargs)
    colors = ["#2166ac", "#7b3294", "#b2182b"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_ylabel("weight ratio")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.94, "1.0 = same as prior", transform=ax.transAxes, fontsize=9, color="#455a64")


def draw_bar(ax, labels, values, title, color):
    ax.bar(labels, values, color=color, edgecolor="#263238", linewidth=0.5)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_ylabel("mean max/min ratio")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)


def draw_heatmap(ax, spread_rows):
    datasets = ["bloodmnist_224", "dermamnist_224", "organcmnist_224", "organsmnist_224", "chaoshengmnist_224"]
    models = ["convnext", "resnet", "swin_tiny", "vit_t", "clip-vit-base-patch32"]
    buckets = defaultdict(list)
    for row in spread_rows:
        if row["ablation"] != "full":
            continue
        buckets[(row["dataset"], row["model"])].append(row["spread"])
    matrix = []
    for dataset in datasets:
        matrix.append([mean(buckets.get((dataset, model), [])) or 1.0 for model in models])
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=1.0, vmax=max(max(row) for row in matrix))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=25, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_title("Full: mean max/min ratio by dataset and backbone", loc="left", fontsize=12, fontweight="bold")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8)
    return image


def write_markdown(path, rows, spread_rows):
    lines = [
        "# Client Weight Ratio Analysis",
        "",
        "`ratio = alpha / pi`. `1.0` means the final weight is equal to the prior average weight.",
        "Values above `1.0` mean the client/checkpoint is up-weighted; values below `1.0` mean it is down-weighted.",
        "",
        "## Ablation Ratio Summary",
        "",
        "| ablation | rows | morph_ratio_p10 | morph_ratio_mean | morph_ratio_p90 | mean_case_max/min | max_case_max/min |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ablation in ["full", "no_client_information", "no_fusion_selection"]:
        values = [row["ratio_morph"] for row in rows if row["ablation"] == ablation and row["ratio_morph"] is not None]
        spreads = [row["spread"] for row in spread_rows if row["ablation"] == ablation and row["spread"] is not None]
        lines.append(
            f"| `{ablation}` | {len(values)} | {fmt(percentile(values, 0.1))} | {fmt(mean(values))} | {fmt(percentile(values, 0.9))} | {fmt(mean(spreads))} | {fmt(max(spreads) if spreads else None)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `no_client_information` stays at ratio `1.0`, so removing M1 makes all clients collapse back to prior weights.",
            "- `full` has a non-trivial ratio range and max/min spread, so the method is distinguishing clients/checkpoints.",
            "- `no_fusion_selection` keeps almost the same ratios as `full`, which means M1 still distinguishes clients, but the accuracy drop shows M2 is needed to use that distinction effectively.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = read_rows(args.detail_csv)
    spread_rows = case_spreads(rows, "ratio_morph")

    full_spreads = [row for row in spread_rows if row["ablation"] == "full"]
    by_dataset = group_mean(full_spreads, "dataset", "spread")
    by_model = group_mean(full_spreads, "model", "spread")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    fig.suptitle("Does my_merge distinguish client checkpoints?", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.945, "weight ratio = final diagnostic weight / prior weight; 1.0 means no distinction", ha="center", fontsize=10, color="#455a64")

    draw_boxplot(axes[0, 0], rows, "ratio_morph", "Distribution of medical weight ratios")
    image = draw_heatmap(axes[0, 1], spread_rows)
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04, label="mean max/min ratio")

    dataset_order = ["bloodmnist_224", "dermamnist_224", "organcmnist_224", "organsmnist_224", "chaoshengmnist_224"]
    draw_bar(
        axes[1, 0],
        [DATASET_LABELS[d] for d in dataset_order],
        [by_dataset.get(d, 1.0) for d in dataset_order],
        "Full: client distinction by dataset",
        "#2c7fb8",
    )

    model_order = ["convnext", "resnet", "swin_tiny", "vit_t", "clip-vit-base-patch32"]
    draw_bar(
        axes[1, 1],
        [MODEL_LABELS[m] for m in model_order],
        [by_model.get(m, 1.0) for m in model_order],
        "Full: client distinction by backbone",
        "#7b3294",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for dest in [args.dest_png, args.dest_svg]:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, bbox_inches="tight")
        print(f"wrote {dest}")
    write_markdown(args.dest_md, rows, spread_rows)
    print(f"wrote {args.dest_md}")


if __name__ == "__main__":
    main()
