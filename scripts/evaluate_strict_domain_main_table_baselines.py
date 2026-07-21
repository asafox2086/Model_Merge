#!/usr/bin/env python3
"""Evaluate the formal-table merge baselines on a strict-domain checkpoint set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
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


FORMAL_METHODS = (
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
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--hub-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stats-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=FORMAL_METHODS, default=FORMAL_METHODS)
    return parser.parse_args()


def build_loader(meta, data_root, batch_size, num_workers, device):
    splits = load_npz_splits(str(data_root / f"{meta['dataset']}.npz"))
    test_split = splits["test"]
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
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return loader, np.bincount(test_split.labels, minlength=int(meta["num_classes"])).astype(int).tolist()


@torch.no_grad()
def evaluate_predictions(model, loader, device, num_classes):
    model.eval()
    prediction_chunks = []
    target_chunks = []
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = functional.cross_entropy(logits, targets)
        total_loss += float(loss.item()) * targets.numel()
        prediction_chunks.append(logits.argmax(dim=1).cpu())
        target_chunks.append(targets.cpu())
    predictions = torch.cat(prediction_chunks)
    targets = torch.cat(target_chunks)
    prediction_counts = torch.bincount(predictions, minlength=num_classes)
    total = int(targets.numel())
    return {
        "accuracy": float((predictions == targets).float().mean().item()),
        "loss": total_loss / total,
        "num_samples": total,
        "prediction_counts": prediction_counts.tolist(),
        "collapse_ratio": float(prediction_counts.max().item() / total),
        "effective_predicted_classes": int((prediction_counts > 0).sum().item()),
        "top_predicted_class": int(prediction_counts.argmax().item()),
    }


def clone_state_dict(state_dict):
    return OrderedDict((key, value.detach().clone()) for key, value in state_dict.items())


def build_method_config(meta, args, method):
    return {
        **METHOD_DEFAULTS,
        "task_type": meta["task_type"],
        "dataset": meta["dataset"],
        "model": meta["model"],
        "num_clients": meta["num_clients"],
        "beta": meta["beta"],
        "seed": meta["seed"],
        "method": method,
        "merge_weight_mode": "equal",
        "data_root": str(args.data_root),
        "device": str(args.device),
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "stats_split": "val",
        "stats_batch_size": args.stats_batch_size,
        "stats_num_workers": args.num_workers,
        "fisher_max_batches": 1,
        "regmean_max_batches": 1,
        "regmean_max_dim": 1024,
    }


def write_csv(path, rows):
    fields = (
        "method",
        "status",
        "accuracy",
        "loss",
        "num_samples",
        "collapse_ratio",
        "effective_predicted_classes",
        "top_predicted_class",
        "prediction_counts",
        "seconds",
        "error",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.stats_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    meta = load_json(args.meta)
    checkpoint_paths = [args.hub_dir / client["checkpoint"] for client in meta["clients"]]
    if not all(path.exists() for path in checkpoint_paths):
        missing = [str(path) for path in checkpoint_paths if not path.exists()]
        raise FileNotFoundError(f"Missing client checkpoints: {missing}")
    checkpoints = [load_checkpoint(path, device="cpu") for path in checkpoint_paths]
    original_states = [extract_state_dict(checkpoint) for checkpoint in checkpoints]
    loader, support = build_loader(meta, args.data_root, args.batch_size, args.num_workers, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    method_details = {}
    for method in args.methods:
        started_at = time.monotonic()
        try:
            set_seed(int(meta["seed"]))
            state_dicts = [clone_state_dict(state) for state in original_states]
            config = build_method_config(meta, args, method)
            merged_state, method_info = merge_with_method(
                method,
                state_dicts,
                [1.0] * len(state_dicts),
                meta,
                checkpoints,
                config,
            )
            model, _, _, _ = build_model(
                name=meta["model"],
                num_classes=int(meta["num_classes"]),
                in_channels=int(meta["in_channels"]),
                pretrained=False,
            )
            model.load_state_dict(merged_state, strict=True)
            metrics = evaluate_predictions(model.to(device), loader, device, int(meta["num_classes"]))
            seconds = time.monotonic() - started_at
            rows.append(
                {
                    "method": method,
                    "status": "ok",
                    "accuracy": f"{metrics['accuracy']:.6f}",
                    "loss": f"{metrics['loss']:.6f}",
                    "num_samples": metrics["num_samples"],
                    "collapse_ratio": f"{metrics['collapse_ratio']:.6f}",
                    "effective_predicted_classes": metrics["effective_predicted_classes"],
                    "top_predicted_class": metrics["top_predicted_class"],
                    "prediction_counts": json.dumps(metrics["prediction_counts"]),
                    "seconds": f"{seconds:.3f}",
                    "error": "",
                }
            )
            method_details[method] = {
                "metrics": metrics,
                "method_info_keys": sorted(method_info) if isinstance(method_info, dict) else [],
            }
            del model, merged_state
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            seconds = time.monotonic() - started_at
            rows.append(
                {
                    "method": method,
                    "status": "error",
                    "accuracy": "",
                    "loss": "",
                    "num_samples": "",
                    "collapse_ratio": "",
                    "effective_predicted_classes": "",
                    "top_predicted_class": "",
                    "prediction_counts": "",
                    "seconds": f"{seconds:.3f}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(args.output_dir / "main_table_baseline_collapse_metrics.csv", rows)
    payload = {
        "analysis": "formal main-table baselines under the strict DermaMNIST/CIFAR-100 control",
        "methods": list(args.methods),
        "main_table_implementation_root": str(IMPLEMENTATION_ROOT),
        "merge_weight_mode": "equal",
        "formal_stats_budget": {"split": "val", "batch_size": args.stats_batch_size, "fisher_max_batches": 1, "regmean_max_batches": 1},
        "test_support": support,
        "rows": rows,
        "method_details": method_details,
    }
    (args.output_dir / "main_table_baseline_collapse_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    successful = [row for row in rows if row["status"] == "ok"]
    print(f"Completed {len(successful)}/{len(rows)} methods: {args.output_dir}")


if __name__ == "__main__":
    main()
