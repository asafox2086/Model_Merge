#!/usr/bin/env python3
"""Plot separate RegMean prediction diagnostics for the strict-domain control."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_ROOT = Path(
    os.environ.get("MODEL_MERGE_IMPLEMENTATION_ROOT", "/data2/liyapeng_grp/program/MedMNISTMerge")
).resolve()
if (IMPLEMENTATION_ROOT / "methods" / "avg.py").exists():
    sys.path.insert(0, str(IMPLEMENTATION_ROOT))
elif str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import NpzTensorDataset, load_npz_splits
from merge import METHOD_DEFAULTS, merge_with_method
from model import build_model
from utils import extract_state_dict, load_checkpoint, load_json, set_seed


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PLOT = load_module("strict_domain_avg_plot", ROOT / "scripts" / "plot_derma_cifar100_semantic_strict.py")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medical-metrics", type=Path, required=True)
    parser.add_argument("--natural-metrics", type=Path, required=True)
    parser.add_argument("--medical-meta", type=Path, required=True)
    parser.add_argument("--natural-meta", type=Path, required=True)
    parser.add_argument("--medical-hub-dir", type=Path, required=True)
    parser.add_argument("--natural-hub-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stats-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--perplexity", type=float, default=50.0)
    return parser.parse_args()


def load_regmean_counts(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["method"] == "regmean" and row["status"] == "ok"]
    if len(matches) != 1:
        raise ValueError(f"Expected one successful RegMean row in {path}, found {len(matches)}")
    counts = np.asarray(ast.literal_eval(matches[0]["prediction_counts"]), dtype=np.int64)
    if counts.ndim != 1 or (counts < 0).any():
        raise ValueError(f"Invalid RegMean prediction counts in {path}")
    return matches[0], counts


def clone_state_dict(state_dict):
    return OrderedDict((key, value.detach().clone()) for key, value in state_dict.items())


def merge_regmean(meta, hub_dir, data_root, device, batch_size, stats_batch_size):
    checkpoint_paths = [hub_dir / client["checkpoint"] for client in meta["clients"]]
    checkpoints = [load_checkpoint(path, device="cpu") for path in checkpoint_paths]
    state_dicts = [clone_state_dict(extract_state_dict(checkpoint)) for checkpoint in checkpoints]
    config = {
        **METHOD_DEFAULTS,
        "task_type": meta["task_type"],
        "dataset": meta["dataset"],
        "model": meta["model"],
        "num_clients": meta["num_clients"],
        "beta": meta["beta"],
        "seed": meta["seed"],
        "method": "regmean",
        "merge_weight_mode": "equal",
        "data_root": str(data_root),
        "device": str(device),
        "batch_size": batch_size,
        "num_workers": 0,
        "stats_split": "val",
        "stats_batch_size": stats_batch_size,
        "stats_num_workers": 0,
        "regmean_max_batches": 1,
        "regmean_max_dim": 1024,
    }
    set_seed(int(meta["seed"]))
    merged_state, _ = merge_with_method("regmean", state_dicts, [1.0] * len(state_dicts), meta, checkpoints, config)
    model, _, _, _ = build_model(
        name=meta["model"],
        num_classes=int(meta["num_classes"]),
        in_channels=int(meta["in_channels"]),
        pretrained=False,
    )
    model.load_state_dict(merged_state, strict=True)
    model.to(device)
    return model, checkpoint_paths


@torch.no_grad()
def extract_centered_logits(model, meta, data_root, device, batch_size):
    test_split = load_npz_splits(str(data_root / f"{meta['dataset']}.npz"))["test"]
    source_height, source_width = test_split.images.shape[1:3]
    target_size = int(meta["image_size"])
    transform = None
    if (source_height, source_width) != (target_size, target_size):
        transform = transforms.Resize((target_size, target_size), antialias=True)
    dataset = NpzTensorDataset(test_split.images, test_split.labels, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    logits_chunks = []
    labels_chunks = []
    predictions_chunks = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(inputs)
        logits_chunks.append((logits - logits.mean(dim=1, keepdim=True)).cpu().float().numpy())
        labels_chunks.append(targets.numpy().reshape(-1))
        predictions_chunks.append(logits.argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(logits_chunks),
        np.concatenate(labels_chunks).astype(np.int64),
        np.concatenate(predictions_chunks).astype(np.int64),
    )


def extract_domain(meta, hub_dir, data_root, device, batch_size, stats_batch_size, seed, perplexity):
    model, checkpoint_paths = merge_regmean(meta, hub_dir, data_root, device, batch_size, stats_batch_size)
    try:
        logits, labels, predictions = extract_centered_logits(model, meta, data_root, device, batch_size)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    coordinates = PLOT.STRICT.run_tsne(logits, seed, perplexity)
    return {
        "coordinates": coordinates,
        "labels": labels,
        "predictions": predictions,
        "splits": ["test"] * len(labels),
        "checkpoint_sha256": {path.name: PLOT.STRICT.state_hash(path) for path in checkpoint_paths},
    }


def write_coordinates(path, domains):
    fields = ("domain", "sample_index", "true_label", "regmean_prediction", "tsne_x", "tsne_y")
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
                        "regmean_prediction": int(prediction),
                        "tsne_x": f"{coordinate[0]:.6f}",
                        "tsne_y": f"{coordinate[1]:.6f}",
                    }
                )


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.stats_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    medical_row, medical_counts = load_regmean_counts(args.medical_metrics)
    natural_row, natural_counts = load_regmean_counts(args.natural_metrics)
    if medical_counts.shape != natural_counts.shape:
        raise ValueError("Medical and natural RegMean predictions have different class counts")
    medical_meta = load_json(args.medical_meta)
    natural_meta = load_json(args.natural_meta)
    medical_splits = load_npz_splits(str(args.data_root / f"{medical_meta['dataset']}.npz"))
    natural_splits = load_npz_splits(str(args.data_root / f"{natural_meta['dataset']}.npz"))
    alignment = PLOT.STRICT.validate_alignment(medical_meta, natural_meta, medical_splits, natural_splits)
    expected_support = np.asarray(alignment["split_class_counts"]["test"], dtype=np.int64)
    if int(medical_counts.sum()) != int(expected_support.sum()) or int(natural_counts.sum()) != int(expected_support.sum()):
        raise ValueError("RegMean predictions do not cover the full matched test support")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    PLOT.plot_distribution(
        args.output_dir / "dermamnist_regmean_prediction_distribution",
        medical_counts,
        "Medical DermaMNIST",
        "#4F81BD",
        merge_label="RegMean",
    )
    PLOT.plot_distribution(
        args.output_dir / "cifar100_semantic7_regmean_prediction_distribution",
        natural_counts,
        "Natural CIFAR-100 semantic 7-class control",
        "#C0504D",
        merge_label="RegMean",
    )
    PLOT.write_distribution_csv(args.output_dir / "dermamnist_regmean_prediction_distribution.csv", medical_counts)
    PLOT.write_distribution_csv(args.output_dir / "cifar100_semantic7_regmean_prediction_distribution.csv", natural_counts)

    device = PLOT.STRICT.resolve_device(args.device)
    domains = {
        "medical_dermamnist": extract_domain(medical_meta, args.medical_hub_dir, args.data_root, device, args.batch_size, args.stats_batch_size, args.seed, args.perplexity),
        "natural_cifar100_semantic7": extract_domain(natural_meta, args.natural_hub_dir, args.data_root, device, args.batch_size, args.stats_batch_size, args.seed, args.perplexity),
    }
    for domain_name, expected_counts in (("medical_dermamnist", medical_counts), ("natural_cifar100_semantic7", natural_counts)):
        observed_counts = np.bincount(domains[domain_name]["predictions"], minlength=len(expected_counts))
        if not np.array_equal(observed_counts, expected_counts):
            raise ValueError(
                f"RegMean re-merge predictions differ for {domain_name}: "
                f"recorded={expected_counts.tolist()} observed={observed_counts.tolist()}"
            )
    num_classes = int(medical_meta["num_classes"])
    PLOT.plot_prediction_tsne(
        args.output_dir / "dermamnist_regmean_centered_logit_prediction_tsne",
        domains["medical_dermamnist"]["coordinates"],
        domains["medical_dermamnist"]["predictions"],
        "Medical DermaMNIST (n=2,005)",
        num_classes,
        merge_label="RegMean",
    )
    PLOT.plot_prediction_tsne(
        args.output_dir / "cifar100_semantic7_regmean_centered_logit_prediction_tsne",
        domains["natural_cifar100_semantic7"]["coordinates"],
        domains["natural_cifar100_semantic7"]["predictions"],
        "Natural CIFAR-100 semantic 7-class control (n=2,005)",
        num_classes,
        merge_label="RegMean",
    )
    write_coordinates(args.output_dir / "regmean_centered_logit_prediction_tsne_coordinates.csv", domains)
    summary = {
        "analysis": "separate RegMean prediction distributions and prediction-colored centered-logit t-SNE embeddings",
        "interpretation_boundary": "These are output/prediction-collapse diagnostics, not evidence of classifier-feature collapse.",
        "merge_method": "regmean",
        "merge_weight_mode": "equal",
        "formal_stats_budget": {"split": "val", "batch_size": args.stats_batch_size, "regmean_max_batches": 1},
        "tsne": {"seed": args.seed, "perplexity": args.perplexity, "init": "pca", "learning_rate": "auto"},
        "alignment": alignment,
        "recorded_metrics": {"medical": medical_row, "natural": natural_row},
        "domains": {
            name: {
                "prediction_counts": np.bincount(domain["predictions"], minlength=num_classes).astype(int).tolist(),
                "checkpoint_sha256": domain["checkpoint_sha256"],
            }
            for name, domain in domains.items()
        },
    }
    (args.output_dir / "plot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote RegMean strict-domain diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
