#!/usr/bin/env python3
"""Plot matched-domain ResNet feature t-SNE visualizations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEMPLATE_DIR = ROOT / "scripts" / "plot_templates"
if str(TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_DIR))

from dataset import NpzTensorDataset, load_npz_splits
from model import build_model
from utils.io import load_checkpoint, load_json
from utils.state_dict import extract_state_dict
from style import polish_axes, save_png_pdf, setup_style


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
    parser.add_argument("--merge-weighting", choices=("uniform", "sample"), default="uniform")
    return parser.parse_args()


def resolve_device(value):
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def state_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average_state_dicts(state_dicts, weights):
    if len(state_dicts) != len(weights):
        raise ValueError("state dicts and weights must have the same length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("client sample weights must sum to a positive value")
    normalized_weights = [float(weight) / total for weight in weights]
    merged = OrderedDict()
    for key in state_dicts[0]:
        values = [state_dict[key] for state_dict in state_dicts]
        if torch.is_floating_point(values[0]):
            value = values[0].detach().clone() * normalized_weights[0]
            for next_value, weight in zip(values[1:], normalized_weights[1:]):
                value.add_(next_value.detach(), alpha=weight)
            merged[key] = value
        else:
            merged[key] = values[0].detach().clone()
    return merged, normalized_weights


def build_transform(meta, images):
    if images.ndim == 3:
        height, width = images.shape[1:3]
    else:
        height, width = images.shape[1:3]
    target = int(meta["image_size"])
    if (height, width) == (target, target):
        return None
    return transforms.Resize((target, target), antialias=True)


def class_counts(labels):
    return np.bincount(labels.astype(np.int64), minlength=int(labels.max()) + 1).astype(int).tolist()


def client_counts(meta, train_labels):
    counts = {}
    for client in meta["clients"]:
        client_id = str(client["client_id"])
        classes = [int(value) for value in client["classes"]]
        counts[client_id] = {
            str(class_index): int((train_labels == class_index).sum())
            for class_index in classes
        }
    return counts


def validate_alignment(medical_meta, natural_meta, medical_splits, natural_splits):
    required_keys = [
        "model",
        "num_clients",
        "num_classes",
        "in_channels",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "image_size",
        "pretrained",
        "seed",
        "client_classes",
    ]
    mismatches = {
        key: [medical_meta.get(key), natural_meta.get(key)]
        for key in required_keys
        if medical_meta.get(key) != natural_meta.get(key)
    }
    medical_train_counts = class_counts(medical_splits["train"].labels)
    natural_train_counts = class_counts(natural_splits["train"].labels)
    split_counts = {
        split: [class_counts(medical_splits[split].labels), class_counts(natural_splits[split].labels)]
        for split in ("train", "val", "test")
    }
    for split, values in split_counts.items():
        if values[0] != values[1]:
            mismatches[f"{split}_class_counts"] = values
    medical_client_counts = client_counts(medical_meta, medical_splits["train"].labels)
    natural_client_counts = client_counts(natural_meta, natural_splits["train"].labels)
    if medical_client_counts != natural_client_counts:
        mismatches["client_class_counts"] = [medical_client_counts, natural_client_counts]
    medical_samples = [int(client["num_samples"]) for client in medical_meta["clients"]]
    natural_samples = [int(client["num_samples"]) for client in natural_meta["clients"]]
    if medical_samples != natural_samples:
        mismatches["client_num_samples"] = [medical_samples, natural_samples]
    if mismatches:
        raise ValueError(f"Strict-domain control mismatch: {json.dumps(mismatches, sort_keys=True)}")
    return {
        "shared_hyperparameters": {key: medical_meta[key] for key in required_keys if key != "client_classes"},
        "client_classes": medical_meta["client_classes"],
        "client_class_counts": medical_client_counts,
        "split_class_counts": {split: values[0] for split, values in split_counts.items()},
    }


def load_average_model(meta, hub_dir, device, merge_weighting):
    checkpoint_paths = [Path(hub_dir) / client["checkpoint"] for client in meta["clients"]]
    state_dicts = [extract_state_dict(load_checkpoint(path, device="cpu")) for path in checkpoint_paths]
    if merge_weighting == "uniform":
        weights = [1] * len(checkpoint_paths)
    else:
        weights = [int(client["num_samples"]) for client in meta["clients"]]
    merged_state, normalized_weights = average_state_dicts(state_dicts, weights)
    model, _, _, _ = build_model(
        name=meta["model"],
        num_classes=int(meta["num_classes"]),
        in_channels=int(meta["in_channels"]),
        pretrained=False,
    )
    model.load_state_dict(merged_state, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint_paths, normalized_weights


def extract_classifier_features(model, meta, data_root, device, batch_size):
    splits = load_npz_splits(Path(data_root) / f"{meta['dataset']}.npz")
    test_split = splits["test"]
    dataset = NpzTensorDataset(
        test_split.images,
        test_split.labels,
        transform=build_transform(meta, test_split.images),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    classifier = model.get_classifier()
    if not isinstance(classifier, torch.nn.Module):
        raise TypeError(f"Expected a classifier module, got {type(classifier).__name__}")
    captured = {}

    def capture_features(_module, inputs):
        captured["features"] = inputs[0].detach()

    handle = classifier.register_forward_pre_hook(capture_features)
    features = []
    labels = []
    try:
        with torch.no_grad():
            for images, targets in loader:
                _ = model(images.to(device, non_blocking=True))
                features.append(captured["features"].cpu().float().numpy())
                labels.append(targets.numpy().reshape(-1))
    finally:
        handle.remove()
    return np.concatenate(features), np.concatenate(labels).astype(np.int64)


def run_tsne(features, seed, perplexity):
    if not 1 < perplexity < len(features):
        raise ValueError(f"perplexity must be in (1, {len(features)}), got {perplexity}")
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
        method="barnes_hut",
        n_jobs=-1,
    ).fit_transform(features)


def validate_embedding(coordinates, labels):
    coordinates = np.asarray(coordinates)
    labels = np.asarray(labels)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"Expected t-SNE coordinates shaped (n, 2), got {coordinates.shape}")
    if labels.shape != (coordinates.shape[0],):
        raise ValueError(f"Expected one label per coordinate, got {labels.shape} for {coordinates.shape}")
    if not np.isfinite(coordinates).all():
        raise ValueError("t-SNE coordinates contain non-finite values")
    return coordinates, labels


def draw_embedding(axis, coordinates, labels, title, panel_label=None):
    import matplotlib.pyplot as plt

    coordinates, labels = validate_embedding(coordinates, labels)
    class_colors = plt.get_cmap("tab10")(np.arange(7))
    for class_index in range(7):
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
    axis.set_title(title, fontweight="bold", fontsize=11, pad=7)
    axis.set_xticks([])
    axis.set_yticks([])
    if panel_label:
        axis.text(0.02, 0.98, panel_label, transform=axis.transAxes, va="top", ha="left", fontweight="bold")
    polish_axes(axis, y_grid=False, x_grid=False)


def plot_embedding(out_base, coordinates, labels, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_style("scatter")
    figure, axis = plt.subplots(figsize=(5.1, 4.6), dpi=350)
    draw_embedding(axis, coordinates, labels, title)
    axis.legend(title="True label", loc="best", frameon=False, markerscale=1.2)
    figure.tight_layout()
    save_png_pdf(figure, str(out_base), dpi=350)
    plt.close(figure)


def plot_domain_comparison(out_base, embeddings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_style("scatter")
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.25), dpi=350)
    for axis, (title, coordinates, labels), panel_label in zip(axes, embeddings, ("(a)", "(b)")):
        draw_embedding(axis, coordinates, labels, title, panel_label)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=7, frameon=False)
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    save_png_pdf(figure, str(out_base), dpi=350)
    plt.close(figure)


def write_coordinates(path, rows):
    fields = ["domain", "dataset", "sample_index", "true_label", "tsne_x", "tsne_y"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = resolve_device(args.device)
    medical_meta = load_json(args.medical_meta)
    natural_meta = load_json(args.natural_meta)
    medical_splits = load_npz_splits(args.medical_data_root / f"{medical_meta['dataset']}.npz")
    natural_splits = load_npz_splits(args.natural_data_root / f"{natural_meta['dataset']}.npz")
    alignment = validate_alignment(medical_meta, natural_meta, medical_splits, natural_splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summaries = {}
    embeddings = []
    domains = [
        ("medical_dermamnist", "Medical DermaMNIST", medical_meta, args.medical_hub_dir, args.medical_data_root),
        ("natural_cifar", "Natural CIFAR", natural_meta, args.natural_hub_dir, args.natural_data_root),
    ]
    for domain, display_name, meta, hub_dir, data_root in domains:
        model, checkpoint_paths, normalized_weights = load_average_model(meta, hub_dir, device, args.merge_weighting)
        features, labels = extract_classifier_features(model, meta, data_root, device, args.batch_size)
        coordinates = run_tsne(features, args.seed, args.perplexity)
        plot_embedding(
            args.output_dir / f"{domain}_resnet_k3_avg_feature_tsne",
            coordinates,
            labels,
            f"{display_name}\nResNet / K=3 / {args.merge_weighting} AVG",
        )
        embeddings.append((f"{display_name}\nResNet / K=3 / {args.merge_weighting} AVG", coordinates, labels))
        rows.extend(
            {
                "domain": domain,
                "dataset": meta["dataset"],
                "sample_index": sample_index,
                "true_label": int(labels[sample_index]),
                "tsne_x": f"{coordinates[sample_index, 0]:.6f}",
                "tsne_y": f"{coordinates[sample_index, 1]:.6f}",
            }
            for sample_index in range(len(labels))
        )
        summaries[domain] = {
            "dataset": meta["dataset"],
            "test_samples": int(len(labels)),
            "feature_dimension": int(features.shape[1]),
            "checkpoint_sha256": {path.name: state_hash(path) for path in checkpoint_paths},
            "avg_weights": normalized_weights,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_domain_comparison(args.output_dir / "strict_domain_resnet_k3_avg_feature_tsne", embeddings)
    write_coordinates(args.output_dir / "strict_domain_resnet_k3_avg_feature_tsne_coordinates.csv", rows)
    summary = {
        "analysis": "separate t-SNE embeddings of AVG classifier-input features",
        "merge_weighting": args.merge_weighting,
        "tsne": {"seed": args.seed, "perplexity": args.perplexity, "init": "pca", "learning_rate": "auto"},
        "alignment": alignment,
        "domains": summaries,
    }
    with (args.output_dir / "strict_domain_resnet_k3_avg_feature_tsne_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote t-SNE figures and metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
