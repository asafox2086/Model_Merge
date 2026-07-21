#!/usr/bin/env python3
"""Plot separate output-collapse diagnostics for the strict Derma/CIFAR-100 control."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "scripts" / "plot_templates"
if str(TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_DIR))

from style import darken_color, polish_axes, save_png_pdf, setup_style


def load_logit_module():
    module_path = ROOT / "scripts" / "plot_strict_domain_logit_tsne.py"
    spec = importlib.util.spec_from_file_location("strict_logit_tsne", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


LOGIT = load_logit_module()
STRICT = LOGIT.STRICT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medical-metrics", type=Path, required=True)
    parser.add_argument("--natural-metrics", type=Path, required=True)
    parser.add_argument("--medical-meta", type=Path, required=True)
    parser.add_argument("--natural-meta", type=Path, required=True)
    parser.add_argument("--medical-hub-dir", type=Path, required=True)
    parser.add_argument("--natural-hub-dir", type=Path, required=True)
    parser.add_argument("--medical-data-root", type=Path, required=True)
    parser.add_argument("--natural-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--perplexity", type=float, default=50.0)
    return parser.parse_args()


def load_avg_metrics(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("model") == "avg" and row.get("split") == "test"]
    if len(matches) != 1:
        raise ValueError(f"Expected one AVG test row in {path}, found {len(matches)}")
    row = matches[0]
    counts = np.asarray(ast.literal_eval(row["pred_counts"]), dtype=np.int64)
    support = np.asarray(ast.literal_eval(row["support"]), dtype=np.int64)
    if counts.ndim != 1 or support.shape != counts.shape or (counts < 0).any() or (support < 0).any():
        raise ValueError(f"Invalid prediction/support arrays in {path}")
    if int(counts.sum()) != int(support.sum()):
        raise ValueError(f"Prediction and support totals differ in {path}")
    return row, counts, support


def write_distribution_csv(path, counts):
    total = int(counts.sum())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("class", "prediction_count", "prediction_percent"), lineterminator="\n")
        writer.writeheader()
        for class_index, count in enumerate(counts):
            writer.writerow(
                {
                    "class": class_index,
                    "prediction_count": int(count),
                    "prediction_percent": f"{100 * count / total:.6f}",
                }
            )


def plot_distribution(output_base, counts, title, color, merge_label="AVG"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = int(counts.sum())
    percentages = counts * 100.0 / total
    positions = np.arange(len(counts))
    setup_style("bar")
    figure, axis = plt.subplots(figsize=(7.7, 5.2), dpi=350)
    bars = axis.bar(
        positions,
        percentages,
        width=0.64,
        color=color,
        edgecolor=darken_color(color, 0.65),
        linewidth=1.4,
        zorder=3,
    )
    for bar, count, percentage in zip(bars, counts, percentages):
        if count:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                percentage + 1.1,
                f"{count}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axis.set_xticks(positions)
    axis.set_xticklabels([f"Class {index}" for index in positions])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Test predictions (%)")
    axis.set_ylim(0, 108)
    axis.set_title(f"{title}\n{merge_label} ResNet / K=3 / test set (n={total:,})", fontweight="bold", pad=10)
    polish_axes(axis, y_grid=True, x_grid=False)
    figure.tight_layout()
    save_png_pdf(figure, str(output_base), dpi=350)
    plt.close(figure)


def plot_prediction_tsne(output_base, coordinates, predictions, domain_name, num_classes, merge_label="AVG"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    coordinates, predictions = STRICT.validate_embedding(coordinates, predictions)
    palette = plt.get_cmap("tab10")
    setup_style("scatter")
    figure, axis = plt.subplots(figsize=(8.8, 5.4), dpi=350)
    observed = []
    for class_index in range(num_classes):
        mask = predictions == class_index
        if not mask.any():
            continue
        observed.append(class_index)
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            alpha=0.64,
            color=palette(class_index),
            edgecolors="none",
            rasterized=True,
        )
    counts = np.bincount(predictions, minlength=num_classes)
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.set_title(
        f"{domain_name}\nCentered-logit t-SNE, colored by {merge_label} prediction",
        fontweight="bold",
        pad=10,
    )
    count_items = [f"Class {class_index}: {counts[class_index]}" for class_index in observed]
    count_lines = [", ".join(count_items[index:index + 3]) for index in range(0, len(count_items), 3)]
    axis.text(
        0.02,
        0.02,
        "\n".join(count_lines),
        transform=axis.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
    )
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette(class_index), markersize=7, label=f"Predicted Class {class_index}")
        for class_index in observed
    ]
    axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=10)
    polish_axes(axis, y_grid=False, x_grid=False)
    figure.tight_layout()
    save_png_pdf(figure, str(output_base), dpi=350)
    plt.close(figure)


def extract_domain(meta, hub_dir, data_root, device, batch_size, seed, perplexity):
    model, checkpoints, weights = STRICT.load_average_model(meta, hub_dir, device, "uniform")
    try:
        logits, labels, predictions, splits = LOGIT.extract_centered_logits(
            model, meta, data_root, device, batch_size, ("test",)
        )
    finally:
        del model
    coordinates = STRICT.run_tsne(logits, seed, perplexity)
    return {
        "coordinates": coordinates,
        "labels": labels,
        "predictions": predictions,
        "splits": splits,
        "checkpoint_sha256": {path.name: STRICT.state_hash(path) for path in checkpoints},
        "avg_weights": weights,
    }


def write_coordinates(path, domains):
    fields = ("domain", "sample_index", "true_label", "avg_prediction", "tsne_x", "tsne_y")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for domain_name, domain in domains.items():
            for index, (label, prediction, coordinate) in enumerate(
                zip(domain["labels"], domain["predictions"], domain["coordinates"])
            ):
                writer.writerow(
                    {
                        "domain": domain_name,
                        "sample_index": index,
                        "true_label": int(label),
                        "avg_prediction": int(prediction),
                        "tsne_x": f"{coordinate[0]:.6f}",
                        "tsne_y": f"{coordinate[1]:.6f}",
                    }
                )


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    medical_row, medical_counts, medical_support = load_avg_metrics(args.medical_metrics)
    natural_row, natural_counts, natural_support = load_avg_metrics(args.natural_metrics)
    if not np.array_equal(medical_support, natural_support):
        raise ValueError("Medical and natural test class supports are not identical")
    if medical_counts.shape != natural_counts.shape:
        raise ValueError("Medical and natural class counts do not have the same shape")
    medical_meta = STRICT.load_json(args.medical_meta)
    natural_meta = STRICT.load_json(args.natural_meta)
    medical_splits = LOGIT.load_npz_splits(args.medical_data_root / f"{medical_meta['dataset']}.npz")
    natural_splits = LOGIT.load_npz_splits(args.natural_data_root / f"{natural_meta['dataset']}.npz")
    alignment = STRICT.validate_alignment(medical_meta, natural_meta, medical_splits, natural_splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_distribution(args.output_dir / "dermamnist_avg_prediction_distribution", medical_counts, "Medical DermaMNIST", "#4F81BD")
    plot_distribution(args.output_dir / "cifar100_semantic7_avg_prediction_distribution", natural_counts, "Natural CIFAR-100 semantic 7-class control", "#C0504D")
    write_distribution_csv(args.output_dir / "dermamnist_avg_prediction_distribution.csv", medical_counts)
    write_distribution_csv(args.output_dir / "cifar100_semantic7_avg_prediction_distribution.csv", natural_counts)

    device = STRICT.resolve_device(args.device)
    domains = {
        "medical_dermamnist": extract_domain(medical_meta, args.medical_hub_dir, args.medical_data_root, device, args.batch_size, args.seed, args.perplexity),
        "natural_cifar100_semantic7": extract_domain(natural_meta, args.natural_hub_dir, args.natural_data_root, device, args.batch_size, args.seed, args.perplexity),
    }
    num_classes = int(medical_meta["num_classes"])
    plot_prediction_tsne(
        args.output_dir / "dermamnist_avg_centered_logit_prediction_tsne",
        domains["medical_dermamnist"]["coordinates"],
        domains["medical_dermamnist"]["predictions"],
        "Medical DermaMNIST (n=2,005)",
        num_classes,
    )
    plot_prediction_tsne(
        args.output_dir / "cifar100_semantic7_avg_centered_logit_prediction_tsne",
        domains["natural_cifar100_semantic7"]["coordinates"],
        domains["natural_cifar100_semantic7"]["predictions"],
        "Natural CIFAR-100 semantic 7-class control (n=2,005)",
        num_classes,
    )
    write_coordinates(args.output_dir / "centered_logit_prediction_tsne_coordinates.csv", domains)
    summary = {
        "analysis": "prediction distribution and separate prediction-colored centered-logit t-SNE embeddings of the uniform AVG model",
        "interpretation_boundary": "These are output/prediction-collapse diagnostics, not evidence of classifier-feature collapse.",
        "merge_weighting": "uniform",
        "tsne": {"seed": args.seed, "perplexity": args.perplexity, "init": "pca", "learning_rate": "auto"},
        "alignment": alignment,
        "avg_test": {"medical": medical_row, "natural": natural_row},
        "domains": {
            name: {
                "prediction_counts": np.bincount(domain["predictions"], minlength=num_classes).astype(int).tolist(),
                "checkpoint_sha256": domain["checkpoint_sha256"],
                "avg_weights": domain["avg_weights"],
            }
            for name, domain in domains.items()
        },
    }
    (args.output_dir / "plot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote strict-domain diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
