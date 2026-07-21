#!/usr/bin/env python3
"""Plot compact prediction-colored t-SNEs of matched-domain AVG logits."""

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
    parser.add_argument("--medical-name", required=True)
    parser.add_argument("--natural-meta", type=Path, required=True)
    parser.add_argument("--natural-hub-dir", type=Path, required=True)
    parser.add_argument("--natural-data-root", type=Path, required=True)
    parser.add_argument("--natural-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("test",))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--perplexity", type=float, default=50.0)
    return parser.parse_args()


def extract_centered_logits(model, meta, data_root, device, batch_size, split_names):
    splits = load_npz_splits(Path(data_root) / f"{meta['dataset']}.npz")
    logits_all = []
    labels_all = []
    predictions_all = []
    split_values = []
    for split_name in split_names:
        data_split = splits[split_name]
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
        with torch.no_grad():
            for images, targets in loader:
                logits = model(images.to(device, non_blocking=True))
                logits_all.append((logits - logits.mean(dim=1, keepdim=True)).cpu().float().numpy())
                labels_all.append(targets.numpy().reshape(-1))
                predictions_all.append(logits.argmax(dim=1).cpu().numpy())
                split_values.extend([split_name] * len(targets))
    centered_logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all).astype(np.int64)
    predictions = np.concatenate(predictions_all).astype(np.int64)
    if centered_logits.shape[0] != labels.shape[0] or labels.shape != predictions.shape:
        raise ValueError("Logit, label, and prediction counts must match")
    if not np.isfinite(centered_logits).all():
        raise ValueError("Centered logits contain non-finite values")
    return centered_logits, labels, predictions, split_values


def prediction_counts(predictions, num_classes):
    return np.bincount(predictions, minlength=num_classes).astype(int).tolist()


def draw_embedding(axis, coordinates, predictions, title, panel_label, num_classes):
    import matplotlib.pyplot as plt

    coordinates, predictions = STRICT.validate_embedding(coordinates, predictions)
    palette = plt.get_cmap("tab10")
    for class_index in range(num_classes):
        mask = predictions == class_index
        if not mask.any():
            continue
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=5,
            alpha=0.62,
            color=palette(class_index),
            edgecolors="none",
            rasterized=True,
        )
    axis.set_title(title, fontweight="bold", fontsize=12, pad=8)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(0.02, 0.98, panel_label, transform=axis.transAxes, va="top", ha="left", fontweight="bold")
    nonzero_counts = Counter(int(value) for value in predictions)
    counts_text = ", ".join(f"Pred {class_index}: {count}" for class_index, count in sorted(nonzero_counts.items()))
    axis.text(0.02, 0.02, counts_text, transform=axis.transAxes, va="bottom", ha="left", fontsize=9)
    polish_axes(axis, y_grid=False, x_grid=False)


def plot_pair(output_base, medical, natural, medical_name, natural_name, num_classes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    setup_style("scatter")
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), dpi=350)
    draw_embedding(axes[0], medical["coordinates"], medical["predictions"], f"{medical_name}: centered-logit t-SNE / AVG prediction", "(a)", num_classes)
    draw_embedding(axes[1], natural["coordinates"], natural["predictions"], f"{natural_name}: centered-logit t-SNE / AVG prediction", "(b)", num_classes)
    observed = sorted(set(medical["predictions"].tolist()) | set(natural["predictions"].tolist()))
    palette = plt.get_cmap("tab10")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=palette(index), markersize=7, label=f"Predicted Class {index}") for index in observed]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(handles), frameon=False)
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    save_png_pdf(figure, str(output_base), dpi=350)
    plt.close(figure)


def write_coordinates(path, domains):
    fields = ["domain", "dataset", "split", "sample_index", "true_label", "avg_prediction", "tsne_x", "tsne_y"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for domain_name, domain in domains.items():
            for index in range(len(domain["labels"])):
                writer.writerow(
                    {
                        "domain": domain_name,
                        "dataset": domain["dataset"],
                        "split": domain["splits"][index],
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
        ("medical", medical_meta, args.medical_hub_dir, args.medical_data_root),
        ("natural", natural_meta, args.natural_hub_dir, args.natural_data_root),
    )
    for domain_name, meta, hub_dir, data_root in inputs:
        model, checkpoint_paths, normalized_weights = STRICT.load_average_model(meta, hub_dir, device, "uniform")
        centered_logits, labels, predictions, split_names = extract_centered_logits(
            model, meta, data_root, device, args.batch_size, args.splits
        )
        coordinates = STRICT.run_tsne(centered_logits, args.seed, args.perplexity)
        domains[domain_name] = {
            "dataset": meta["dataset"],
            "coordinates": coordinates,
            "labels": labels,
            "predictions": predictions,
            "splits": split_names,
            "checkpoint_sha256": {path.name: STRICT.state_hash(path) for path in checkpoint_paths},
            "avg_weights": normalized_weights,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    num_classes = int(medical_meta["num_classes"])
    if num_classes != int(natural_meta["num_classes"]):
        raise ValueError("Domains must have equal class counts")
    output_base = args.output_dir / "strict_domain_resnet_k3_centered_logit_prediction_tsne"
    plot_pair(output_base, domains["medical"], domains["natural"], args.medical_name, args.natural_name, num_classes)
    write_coordinates(args.output_dir / "strict_domain_resnet_k3_centered_logit_prediction_tsne_coordinates.csv", domains)
    summary = {
        "analysis": "separate prediction-colored t-SNE embeddings of per-sample centered AVG logits",
        "logit_transform": "subtract the per-sample mean logit; this retains all pairwise class margins and predictions",
        "sample_splits": list(args.splits),
        "samples_per_domain": int(len(domains["medical"]["labels"])),
        "merge_weighting": "uniform",
        "tsne": {"seed": args.seed, "perplexity": args.perplexity, "init": "pca", "learning_rate": "auto"},
        "alignment": alignment,
        "domains": {
            domain_name: {
                "dataset": domain["dataset"],
                "prediction_counts": prediction_counts(domain["predictions"], num_classes),
                "checkpoint_sha256": domain["checkpoint_sha256"],
                "avg_weights": domain["avg_weights"],
            }
            for domain_name, domain in domains.items()
        },
    }
    with (args.output_dir / "strict_domain_resnet_k3_centered_logit_prediction_tsne_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote centered-logit prediction t-SNE to {args.output_dir}")


if __name__ == "__main__":
    main()
