#!/usr/bin/env python3
"""Plot matched-domain ResNet feature t-SNEs with GT and AVG predictions."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEMPLATE_DIR = ROOT / "scripts" / "plot_templates"
if str(TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_DIR))

from dataset import NpzTensorDataset, load_npz_splits
from style import polish_axes, save_png_pdf, setup_style


def load_strict_plot_module():
    module_path = ROOT / "scripts" / "plot_strict_domain_tsne.py"
    spec = importlib.util.spec_from_file_location("strict_domain_tsne", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STRICT = load_strict_plot_module()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medical-meta", type=Path, required=True)
    parser.add_argument("--medical-hub-dir", type=Path, required=True)
    parser.add_argument("--medical-data-root", type=Path, required=True)
    parser.add_argument("--natural-meta", type=Path, required=True)
    parser.add_argument("--natural-hub-dir", type=Path, required=True)
    parser.add_argument("--natural-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--perplexity", type=float, default=30.0)
    return parser.parse_args()


def extract_features_and_predictions(model, meta, data_root, device, batch_size):
    splits = load_npz_splits(Path(data_root) / f"{meta['dataset']}.npz")
    data_split = splits["test"]
    dataset = NpzTensorDataset(
        data_split.images,
        data_split.labels,
        transform=STRICT.build_transform(meta, data_split.images),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    classifier = model.get_classifier()
    if not isinstance(classifier, torch.nn.Module):
        raise TypeError(f"Expected a classifier module, got {type(classifier).__name__}")
    captured = {}

    def capture_features(_module, inputs):
        captured["features"] = inputs[0].detach()

    handle = classifier.register_forward_pre_hook(capture_features)
    features = []
    labels = []
    predictions = []
    try:
        with torch.no_grad():
            for images, targets in loader:
                logits = model(images.to(device, non_blocking=True))
                features.append(captured["features"].cpu().float().numpy())
                labels.append(targets.numpy().reshape(-1))
                predictions.append(logits.argmax(dim=1).cpu().numpy())
    finally:
        handle.remove()
    features = np.concatenate(features)
    labels = np.concatenate(labels).astype(np.int64)
    predictions = np.concatenate(predictions).astype(np.int64)
    if features.shape[0] != labels.shape[0] or labels.shape != predictions.shape:
        raise ValueError("Feature, label, and prediction counts must match")
    if not np.isfinite(features).all():
        raise ValueError("Classifier-input features contain non-finite values")
    return features, labels, predictions


def class_counts(labels, num_classes):
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(int).tolist()


def draw_embedding(axis, coordinates, color_labels, panel_label, title, label_name, num_classes):
    import matplotlib.pyplot as plt

    coordinates, color_labels = STRICT.validate_embedding(coordinates, color_labels)
    palette = plt.get_cmap("tab10")
    for class_index in range(num_classes):
        mask = color_labels == class_index
        if not mask.any():
            continue
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            alpha=0.72,
            color=palette(class_index),
            edgecolors="none",
            rasterized=True,
            label=f"Class {class_index}",
        )
    axis.set_title(title, fontweight="bold", fontsize=11, pad=7)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(0.02, 0.98, panel_label, transform=axis.transAxes, va="top", ha="left", fontweight="bold")
    counts = Counter(int(value) for value in color_labels)
    count_text = ", ".join(f"{label_name} {key}: {value}" for key, value in sorted(counts.items()))
    axis.text(0.02, 0.02, count_text, transform=axis.transAxes, va="bottom", ha="left", fontsize=8.5)
    polish_axes(axis, y_grid=False, x_grid=False)


def plot_dashboard(output_base, medical, natural, num_classes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_style("scatter")
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 9.0), dpi=350)
    panels = (
        (axes[0, 0], medical, "labels", "(a)", "Medical DermaMNIST: feature t-SNE / GT", "GT"),
        (axes[0, 1], medical, "predictions", "(b)", "Medical DermaMNIST: same t-SNE / AVG prediction", "Pred"),
        (axes[1, 0], natural, "labels", "(c)", "Natural CIFAR: feature t-SNE / GT", "GT"),
        (axes[1, 1], natural, "predictions", "(d)", "Natural CIFAR: same t-SNE / AVG prediction", "Pred"),
    )
    for axis, domain, color_key, panel_label, title, label_name in panels:
        draw_embedding(
            axis,
            domain["coordinates"],
            domain[color_key],
            panel_label,
            title,
            label_name,
            num_classes,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=num_classes, frameon=False)
    figure.tight_layout(rect=(0, 0.065, 1, 1))
    save_png_pdf(figure, str(output_base), dpi=350)
    plt.close(figure)


def write_coordinates(path, domains):
    fields = ["domain", "dataset", "sample_index", "true_label", "avg_prediction", "tsne_x", "tsne_y"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for domain_name, domain in domains.items():
            for index in range(len(domain["labels"])):
                writer.writerow(
                    {
                        "domain": domain_name,
                        "dataset": domain["dataset"],
                        "sample_index": index,
                        "true_label": int(domain["labels"][index]),
                        "avg_prediction": int(domain["predictions"][index]),
                        "tsne_x": f"{domain['coordinates'][index, 0]:.6f}",
                        "tsne_y": f"{domain['coordinates'][index, 1]:.6f}",
                    }
                )


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = STRICT.resolve_device(args.device)
    medical_meta = STRICT.load_json(args.medical_meta)
    natural_meta = STRICT.load_json(args.natural_meta)
    medical_splits = load_npz_splits(args.medical_data_root / f"{medical_meta['dataset']}.npz")
    natural_splits = load_npz_splits(args.natural_data_root / f"{natural_meta['dataset']}.npz")
    alignment = STRICT.validate_alignment(medical_meta, natural_meta, medical_splits, natural_splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    domains = {}
    inputs = (
        ("medical_derma", medical_meta, args.medical_hub_dir, args.medical_data_root),
        ("natural_cifar", natural_meta, args.natural_hub_dir, args.natural_data_root),
    )
    for domain_name, meta, hub_dir, data_root in inputs:
        model, checkpoint_paths, normalized_weights = STRICT.load_average_model(meta, hub_dir, device, "uniform")
        features, labels, predictions = extract_features_and_predictions(
            model, meta, data_root, device, args.batch_size
        )
        coordinates = STRICT.run_tsne(features, args.seed, args.perplexity)
        domains[domain_name] = {
            "dataset": meta["dataset"],
            "coordinates": coordinates,
            "labels": labels,
            "predictions": predictions,
            "feature_dimension": int(features.shape[1]),
            "checkpoint_sha256": {path.name: STRICT.state_hash(path) for path in checkpoint_paths},
            "avg_weights": normalized_weights,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    num_classes = int(medical_meta["num_classes"])
    plot_dashboard(args.output_dir / "strict_domain_resnet_k3_feature_prediction_tsne", domains["medical_derma"], domains["natural_cifar"], num_classes)
    write_coordinates(args.output_dir / "strict_domain_resnet_k3_feature_prediction_tsne_coordinates.csv", domains)
    summary = {
        "analysis": "separate full-test feature t-SNE embeddings, each shown with GT and the same AVG-model predictions",
        "feature_source": "classifier-input features (512D)",
        "test_samples_per_domain": int(len(domains["medical_derma"]["labels"])),
        "merge_weighting": "uniform",
        "tsne": {"seed": args.seed, "perplexity": args.perplexity, "init": "pca", "learning_rate": "auto"},
        "alignment": alignment,
        "domains": {
            domain_name: {
                "dataset": domain["dataset"],
                "feature_dimension": domain["feature_dimension"],
                "true_class_counts": class_counts(domain["labels"], num_classes),
                "prediction_counts": class_counts(domain["predictions"], num_classes),
                "checkpoint_sha256": domain["checkpoint_sha256"],
                "avg_weights": domain["avg_weights"],
            }
            for domain_name, domain in domains.items()
        },
    }
    with (args.output_dir / "strict_domain_resnet_k3_feature_prediction_tsne_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote prediction-aware t-SNE to {args.output_dir}")


if __name__ == "__main__":
    main()
