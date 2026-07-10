#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "bloodmnist_224",
    "chaoshengmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
]

DATASET_LABELS = {
    "bloodmnist_224": "Blood",
    "chaoshengmnist_224": "Ultrasound",
    "dermamnist_224": "Derma",
    "organcmnist_224": "Organ-C",
    "organsmnist_224": "Organ-S",
}

GENERIC_METHODS = [
    "avg",
    "ties",
    "dare_linear",
    "dare_ties",
    "regmean",
    "fisher",
    "breadcrumbs",
    "model_stock",
    "from",
    "iso_c",
    "free_merge",
    "robustmerge",
]

PLOT_METHODS = [
    "LAMP-Merge",
    "M1 only",
    "Classifier-head aggregation",
    "Global-feature mean",
    "Support-only synthetic head",
    "Shuffled-label prototype",
    "Uniform client weight",
    "Binary support only",
    "Global client-size weight",
    "Smoothed prevalence prior",
]

SHORT_LABELS = {
    "LAMP-Merge": "LAMP-Merge",
    "M1 only": "M1 only",
    "Classifier-head aggregation": "Head aggregation",
    "Global-feature mean": "Global mean",
    "Support-only synthetic head": "Support only",
    "Shuffled-label prototype": "Shuffled prototype",
    "Uniform client weight": "Uniform-client",
    "Binary support only": "Binary support",
    "Global client-size weight": "Global-size",
    "Smoothed prevalence prior": "Smoothed prior",
}

DIAG_METHODS = {
    "LAMP-Merge": "lamp_merge:full",
    "M1 only": "lamp_merge:m1_only",
    "Classifier-head aggregation": "lamp_merge:prototype_head_agg",
    "Global-feature mean": "lamp_merge:global_feature_mean",
    "Support-only synthetic head": "lamp_merge:support_only",
    "Shuffled-label prototype": "lamp_merge:prototype_shuffle",
    "Uniform client weight": "lamp_merge:uniform_client_weight",
    "Binary support only": "lamp_merge:binary_support",
    "Global client-size weight": "lamp_merge:global_client_size_weight",
    "Smoothed prevalence prior": "lamp_merge:smoothed_prevalence",
}

METHOD_COLORS = {
    "LAMP-Merge": "#B83A4B",
    "M1 only": "#D88735",
    "Classifier-head aggregation": "#5E739B",
    "Global-feature mean": "#7687A5",
    "Support-only synthetic head": "#8F9DB7",
    "Shuffled-label prototype": "#4F6389",
    "Uniform client weight": "#4D8C73",
    "Binary support only": "#67A184",
    "Global client-size weight": "#7AAF91",
    "Smoothed prevalence prior": "#9E5764",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Plot final full-scope LAMP-Merge internal analyses.")
    parser.add_argument(
        "--internal-summary",
        type=Path,
        default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_internal_ablation_full.csv",
    )
    parser.add_argument(
        "--internal-by-dataset",
        type=Path,
        default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_internal_ablation_full_by_dataset.csv",
    )
    parser.add_argument(
        "--diagnostic-summary",
        type=Path,
        default=ROOT
        / "My_merge_ret"
        / "reports"
        / "prediction_diagnostics_full"
        / "prediction_diagnostics_dataset_summary.csv",
    )
    parser.add_argument(
        "--diagnostic-raw",
        type=Path,
        default=ROOT / "My_merge_ret" / "reports" / "prediction_diagnostics_full.csv",
    )
    parser.add_argument(
        "--geometry-overall",
        type=Path,
        default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_prototype_geometry_overall.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "My_merge_ret" / "figures" / "lamp_merge_internal_analysis",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def aggregate_diagnostics(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    metrics = [
        "mean_balanced_accuracy",
        "mean_macro_f1",
        "mean_collapse_ratio",
        "mean_effective_predicted_classes",
        "mean_pred_true_tv",
    ]
    output = {}
    for method, items in grouped.items():
        output[method] = {
            metric: mean(float(item[metric]) for item in items)
            for metric in metrics
        }
    return output


def plot_overall_metrics(
    internal_rows: list[dict[str, str]],
    diagnostic_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    plt = configure_matplotlib()
    accuracy = {
        row["label"]: float(row["client_average_mean_acc"])
        for row in internal_rows
        if row.get("client_average_mean_acc")
    }
    diagnostics = aggregate_diagnostics(diagnostic_rows)
    metric_specs = [
        ("Accuracy", "accuracy", True),
        ("Balanced accuracy", "mean_balanced_accuracy", True),
        ("Macro-F1", "mean_macro_f1", True),
        (r"Collapse ratio $\rho$", "mean_collapse_ratio", False),
        (r"Effective classes $C_{\mathrm{eff}}$", "mean_effective_predicted_classes", True),
        (r"$\mathrm{TV}(q,p_{\mathrm{test}})$", "mean_pred_true_tv", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    positions = np.arange(len(PLOT_METHODS))
    for ax, (title, metric, higher_is_better) in zip(axes.reshape(-1), metric_specs):
        if metric == "accuracy":
            values = [accuracy[method] for method in PLOT_METHODS]
        else:
            values = [diagnostics[DIAG_METHODS[method]][metric] for method in PLOT_METHODS]
        colors = [METHOD_COLORS[method] for method in PLOT_METHODS]
        bars = ax.barh(positions, values, color=colors, height=0.72)
        ax.set_yticks(positions, [SHORT_LABELS[method] for method in PLOT_METHODS])
        ax.invert_yaxis()
        ax.set_title(f"{title} ({'higher' if higher_is_better else 'lower'} is better)")
        ax.grid(axis="x", color="#D8DCE3", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        max_value = max(values)
        ax.set_xlim(0, max_value * 1.18 if max_value > 0 else 1)
        for bar, value in zip(bars, values):
            ax.text(
                value + max_value * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=8,
                color="#30343B",
            )
    fig.suptitle(
        "Full-scope internal ablation: performance and collapse diagnostics",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "internal_ablation_metrics")
    plt.close(fig)


def plot_dataset_accuracy_heatmap(rows: list[dict[str, str]], output_dir: Path) -> None:
    plt = configure_matplotlib()
    row_by_dataset = {row["dataset"]: row for row in rows}
    values = np.asarray(
        [
            [float(row_by_dataset[dataset][f"{method}_mean_acc"]) for dataset in DATASETS]
            for method in PLOT_METHODS
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(9.5, 7.2), constrained_layout=True)
    image = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=0.85, aspect="auto")
    ax.set_xticks(np.arange(len(DATASETS)), [DATASET_LABELS[item] for item in DATASETS])
    ax.set_yticks(np.arange(len(PLOT_METHODS)), [SHORT_LABELS[item] for item in PLOT_METHODS])
    ax.tick_params(axis="x", rotation=25)
    ax.get_yticklabels()[0].set_color(METHOD_COLORS["LAMP-Merge"])
    ax.get_yticklabels()[0].set_fontweight("bold")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            color = "white" if value > 0.52 else "#17202A"
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8.5)
    ax.set_title("Client-average accuracy across medical datasets", fontsize=14, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025, label="Accuracy")
    save_figure(fig, output_dir, "dataset_internal_ablation_accuracy")
    plt.close(fig)


def best_generic_by_dataset(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    output = {}
    for dataset in DATASETS:
        candidates = [
            row
            for row in rows
            if row["dataset"] == dataset and row["method"] in GENERIC_METHODS
        ]
        output[dataset] = max(candidates, key=lambda row: float(row["mean_accuracy"]))
    return output


def plot_collapse_recovery(rows: list[dict[str, str]], output_dir: Path) -> None:
    plt = configure_matplotlib()
    lookup = {(row["dataset"], row["method"]): row for row in rows}
    best_generic = best_generic_by_dataset(rows)
    methods = ["client_mean", "best_generic", "lamp_merge:m1_only", "lamp_merge:full"]
    labels = ["Client mean", "Best generic merge", "M1 only", "LAMP-Merge"]
    colors = ["#8D98A7", "#5E739B", "#D88735", "#B83A4B"]
    metric_specs = [
        ("mean_balanced_accuracy", "Balanced accuracy", True),
        ("mean_macro_f1", "Macro-F1", True),
        ("mean_collapse_ratio", r"Collapse ratio $\rho$", False),
        ("mean_effective_predicted_classes", r"Effective classes $C_{\mathrm{eff}}$", True),
        ("mean_pred_true_tv", r"$\mathrm{TV}(q,p_{\mathrm{test}})$", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.reshape(-1)
    x = np.arange(len(DATASETS))
    width = 0.19
    for ax, (metric, title, higher_is_better) in zip(axes, metric_specs):
        for method_idx, (method, label, color) in enumerate(zip(methods, labels, colors)):
            values = []
            for dataset in DATASETS:
                row = best_generic[dataset] if method == "best_generic" else lookup[(dataset, method)]
                values.append(float(row[metric]))
            ax.bar(x + (method_idx - 1.5) * width, values, width=width, label=label, color=color)
        ax.set_xticks(x, [DATASET_LABELS[item] for item in DATASETS], rotation=24, ha="right")
        ax.set_title(f"{title} ({'higher' if higher_is_better else 'lower'} is better)")
        ax.grid(axis="y", color="#D8DCE3", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.07), frameon=False, ncol=2)
    fig.suptitle(
        "Full-scope recovery of global diagnostic discrimination",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "collapse_recovery_by_dataset")
    plt.close(fig)


def parse_vector(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split()], dtype=np.float64)


def normalized_prediction(row: dict[str, str]) -> np.ndarray:
    counts = parse_vector(row["predicted_class_counts"])
    total = counts.sum()
    return counts / total if total > 0 else counts


def normalized_truth(row: dict[str, str]) -> np.ndarray:
    counts = parse_vector(row["true_counts"])
    total = counts.sum()
    return counts / total if total > 0 else counts


def case_key(row: dict[str, str]) -> tuple[str, str, int, str, int]:
    return (
        row["dataset"],
        row["model"],
        int(row["num_clients"]),
        format(float(row["beta"]), "g"),
        int(row["seed"]),
    )


def mean_distribution(rows: list[dict[str, str]]) -> np.ndarray:
    return np.mean(np.stack([normalized_prediction(row) for row in rows]), axis=0)


def client_mean_distribution(rows: list[dict[str, str]]) -> np.ndarray:
    grouped: dict[tuple[str, str, int, str, int], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if not row["method"].startswith("client_"):
            continue
        grouped[case_key(row)].append(normalized_prediction(row))
    per_case = [np.mean(np.stack(items), axis=0) for items in grouped.values() if items]
    return np.mean(np.stack(per_case), axis=0)


def plot_prediction_distributions(
    raw_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    plt = configure_matplotlib()
    best_generic = best_generic_by_dataset(summary_rows)
    for dataset in DATASETS:
        dataset_rows = [row for row in raw_rows if row["dataset"] == dataset and row.get("status") == "OK"]
        full_rows = [row for row in dataset_rows if row["method"] == "lamp_merge:full"]
        m1_rows = [row for row in dataset_rows if row["method"] == "lamp_merge:m1_only"]
        generic_method = best_generic[dataset]["method"]
        generic_rows = [row for row in dataset_rows if row["method"] == generic_method]
        true_distribution = normalized_truth(full_rows[0])
        distributions = np.stack(
            [
                true_distribution,
                client_mean_distribution(dataset_rows),
                mean_distribution(generic_rows),
                mean_distribution(m1_rows),
                mean_distribution(full_rows),
            ]
        )
        row_labels = [
            "True test distribution",
            "Client mean",
            f"Best generic ({generic_method})",
            "M1 only",
            "LAMP-Merge",
        ]
        fig, ax = plt.subplots(figsize=(max(8.5, distributions.shape[1] * 0.85), 4.6), constrained_layout=True)
        image = ax.imshow(distributions, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(distributions.shape[1]), [str(idx) for idx in range(distributions.shape[1])])
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.get_yticklabels()[-1].set_color(METHOD_COLORS["LAMP-Merge"])
        ax.get_yticklabels()[-1].set_fontweight("bold")
        for row_idx in range(distributions.shape[0]):
            for class_idx in range(distributions.shape[1]):
                value = distributions[row_idx, class_idx]
                text_color = "white" if value < 0.62 else "#111111"
                ax.text(class_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
        ax.set_xlabel("Diagnostic class index")
        ax.set_title(
            f"{DATASET_LABELS[dataset]}: full-scope predicted class distribution",
            fontsize=13,
            fontweight="bold",
        )
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025, label=r"Predicted fraction $q(c)$")
        save_figure(fig, output_dir, f"{dataset}_prediction_distribution")
        plt.close(fig)


def plot_prototype_geometry(rows: list[dict[str, str]], output_dir: Path) -> None:
    plt = configure_matplotlib()
    selected = [
        "LAMP-Merge",
        "Classifier-head aggregation",
        "Global-feature mean",
        "Shuffled-label prototype",
        "Uniform client weight",
        "Global client-size weight",
    ]
    lookup = {row["label"]: row for row in rows}
    metric_specs = [
        ("mean_mean_pairwise_distance", r"Pairwise distance $D_{\mathrm{pair}}$"),
        ("mean_mean_nearest_class_distance", r"Nearest-class distance $D_{\mathrm{nn}}$"),
        ("mean_prototype_consistency", r"Prototype consistency $A_{\mathrm{client}}$"),
        ("mean_prototype_to_client_alignment", r"Prototype-client alignment $A_{\mathrm{proto}}$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    positions = np.arange(len(selected))
    for ax, (metric, title) in zip(axes.reshape(-1), metric_specs):
        values = [float(lookup[method][metric]) for method in selected]
        colors = [METHOD_COLORS.get(method, "#5E739B") for method in selected]
        bars = ax.barh(positions, values, color=colors, height=0.72)
        ax.set_yticks(positions, [SHORT_LABELS.get(method, method) for method in selected])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", color="#D8DCE3", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        max_value = max(values)
        ax.set_xlim(min(0.0, min(values) * 1.1), max_value * 1.18 if max_value > 0 else 1.0)
        for bar, value in zip(bars, values):
            display_value = 0.0 if abs(value) < 5e-4 else value
            ax.text(
                value + max(max_value * 0.015, 0.006),
                bar.get_y() + bar.get_height() / 2,
                f"{display_value:.3f}",
                va="center",
                fontsize=8,
                color="#30343B",
            )
    fig.suptitle(
        "Full-scope prototype geometry in the shared reference space",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "prototype_geometry_key_comparison")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    internal_rows = read_csv(args.internal_summary)
    internal_dataset_rows = read_csv(args.internal_by_dataset)
    diagnostic_rows = read_csv(args.diagnostic_summary)
    diagnostic_raw_rows = read_csv(args.diagnostic_raw)
    geometry_rows = read_csv(args.geometry_overall)
    plot_overall_metrics(internal_rows, diagnostic_rows, args.output_dir)
    plot_dataset_accuracy_heatmap(internal_dataset_rows, args.output_dir)
    plot_collapse_recovery(diagnostic_rows, args.output_dir)
    plot_prediction_distributions(diagnostic_raw_rows, diagnostic_rows, args.output_dir)
    plot_prototype_geometry(geometry_rows, args.output_dir)
    print(f"Wrote LAMP-Merge internal-analysis figures to {args.output_dir}")


if __name__ == "__main__":
    main()
