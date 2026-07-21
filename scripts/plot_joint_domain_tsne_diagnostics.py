#!/usr/bin/env python3
"""Compare medical and natural domains in shared t-SNE coordinate systems."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
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
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--perplexity", type=float, default=50.0)
    return parser.parse_args()


def extract_features_and_logits(model, model_meta, dataset_name, data_root, device, batch_size, split_names):
    splits = load_npz_splits(Path(data_root) / f"{dataset_name}.npz")
    classifier = model.get_classifier()
    captured = {}

    def capture_features(_module, inputs):
        captured["features"] = inputs[0].detach()

    handle = classifier.register_forward_pre_hook(capture_features)
    features = []
    logits = []
    labels = []
    sample_splits = []
    try:
        for split_name in split_names:
            data_split = splits[split_name]
            dataset = NpzTensorDataset(
                data_split.images,
                data_split.labels,
                transform=STRICT.build_transform(model_meta, data_split.images),
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
                    output = model(images.to(device, non_blocking=True))
                    features.append(captured["features"].cpu().float().numpy())
                    logits.append(output.cpu().float().numpy())
                    labels.append(targets.numpy().reshape(-1))
                    sample_splits.extend([split_name] * len(targets))
    finally:
        handle.remove()
    return (
        np.concatenate(features),
        np.concatenate(logits),
        np.concatenate(labels).astype(np.int64),
        sample_splits,
    )


def validate_joint_array(values, expected_rows, name):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != expected_rows:
        raise ValueError(f"{name} has shape {values.shape}, expected ({expected_rows}, d)")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def draw_domain_embedding(axis, coordinates, domains, title, panel_label):
    import matplotlib.pyplot as plt

    colors = {"Medical": "#1F77B4", "Natural": "#D95F02"}
    for domain in ("Medical", "Natural"):
        mask = domains == domain
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=4,
            alpha=0.28,
            color=colors[domain],
            edgecolors="none",
            label=domain,
            rasterized=True,
        )
    axis.set_title(title, fontweight="bold", fontsize=11, pad=7)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(0.02, 0.98, panel_label, transform=axis.transAxes, va="top", ha="left", fontweight="bold")
    polish_axes(axis, y_grid=False, x_grid=False)


def plot_dashboard(output_base, embeddings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_style("scatter")
    figure, axes = plt.subplots(1, len(embeddings), figsize=(13.2, 4.35), dpi=350)
    for axis, (title, coordinates, domains), panel_label in zip(axes, embeddings, ("(a)", "(b)", "(c)")):
        draw_domain_embedding(axis, coordinates, domains, title, panel_label)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    save_png_pdf(figure, str(output_base), dpi=350)
    plt.close(figure)


def write_coordinates(path, rows):
    fields = ["variant", "domain", "dataset", "split", "sample_index", "true_label", "tsne_x", "tsne_y"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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

    medical_model, medical_paths, medical_weights = STRICT.load_average_model(
        medical_meta, args.medical_hub_dir, device, "uniform"
    )
    natural_model, natural_paths, natural_weights = STRICT.load_average_model(
        natural_meta, args.natural_hub_dir, device, "uniform"
    )
    medical_features, medical_logits, medical_labels, medical_split_names = extract_features_and_logits(
        medical_model,
        medical_meta,
        medical_meta["dataset"],
        args.medical_data_root,
        device,
        args.batch_size,
        args.splits,
    )
    natural_features_on_medical, _, natural_labels, natural_split_names = extract_features_and_logits(
        medical_model,
        medical_meta,
        natural_meta["dataset"],
        args.natural_data_root,
        device,
        args.batch_size,
        args.splits,
    )
    natural_features, natural_logits, _, _ = extract_features_and_logits(
        natural_model,
        natural_meta,
        natural_meta["dataset"],
        args.natural_data_root,
        device,
        args.batch_size,
        args.splits,
    )
    medical_features_on_natural, _, _, _ = extract_features_and_logits(
        natural_model,
        natural_meta,
        medical_meta["dataset"],
        args.medical_data_root,
        device,
        args.batch_size,
        args.splits,
    )

    sample_count = len(medical_labels)
    if sample_count != len(natural_labels):
        raise ValueError(f"Domain sample counts differ: {sample_count} != {len(natural_labels)}")
    domains = np.array(["Medical"] * sample_count + ["Natural"] * sample_count)
    datasets = np.array([medical_meta["dataset"]] * sample_count + [natural_meta["dataset"]] * sample_count)
    labels = np.concatenate((medical_labels, natural_labels))
    split_names = medical_split_names + natural_split_names
    variants = {
        "medical_encoder_features": np.concatenate((medical_features, natural_features_on_medical)),
        "natural_encoder_features": np.concatenate((medical_features_on_natural, natural_features)),
        "model_logits": np.concatenate((medical_logits, natural_logits)),
    }
    titles = {
        "medical_encoder_features": "Joint features: Medical AVG encoder",
        "natural_encoder_features": "Joint features: Natural AVG encoder",
        "model_logits": "Joint logits: domain-specific AVG models",
    }
    rows = []
    embeddings = []
    for variant, values in variants.items():
        values = validate_joint_array(values, 2 * sample_count, variant)
        coordinates = STRICT.run_tsne(values, args.seed, args.perplexity)
        embeddings.append((titles[variant], coordinates, domains))
        rows.extend(
            {
                "variant": variant,
                "domain": domains[index],
                "dataset": datasets[index],
                "split": split_names[index],
                "sample_index": index % sample_count,
                "true_label": int(labels[index]),
                "tsne_x": f"{coordinates[index, 0]:.6f}",
                "tsne_y": f"{coordinates[index, 1]:.6f}",
            }
            for index in range(2 * sample_count)
        )
    plot_dashboard(args.output_dir / "joint_domain_tsne_diagnostics", embeddings)
    write_coordinates(args.output_dir / "joint_domain_tsne_diagnostics_coordinates.csv", rows)
    summary = {
        "analysis": "joint domain t-SNE variants using all paired split samples",
        "tsne": {"seed": args.seed, "perplexity": args.perplexity, "init": "pca", "learning_rate": "auto"},
        "sample_splits": list(args.splits),
        "samples_per_domain": sample_count,
        "alignment": alignment,
        "variants": list(variants),
        "medical_checkpoint_sha256": {path.name: STRICT.state_hash(path) for path in medical_paths},
        "natural_checkpoint_sha256": {path.name: STRICT.state_hash(path) for path in natural_paths},
        "medical_avg_weights": medical_weights,
        "natural_avg_weights": natural_weights,
    }
    with (args.output_dir / "joint_domain_tsne_diagnostics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote joint domain diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
