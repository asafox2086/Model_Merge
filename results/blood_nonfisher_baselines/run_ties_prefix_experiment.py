#!/usr/bin/env python3
"""Evaluate TIES after each prefix of a BloodMNIST client-arrival order."""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


RESULT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, required=True)
    parser.add_argument("--model-hub-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", choices=("resnet", "convnext", "vit_t", "swin_tiny"), default="resnet")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    return parser.parse_args()


def evaluate_checkpoint(meta, state_dict, data_root, device, batch_size, num_workers):
    from dataset import NpzTensorDataset, load_npz_splits
    from model import build_model

    splits = load_npz_splits(f"{data_root}/{meta['dataset']}.npz")
    images = splits["test"].images
    source_size = (int(images.shape[1]), int(images.shape[2]))
    target_size = int(meta.get("image_size") or 0)
    transform = None
    if target_size > 0 and source_size != (target_size, target_size):
        transform = transforms.Resize((target_size, target_size), antialias=True)
    loader = DataLoader(
        NpzTensorDataset(images, splits["test"].labels, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model, _, _, _ = build_model(
        name=meta["model"],
        num_classes=int(meta["num_classes"]),
        in_channels=int(meta["in_channels"]),
        pretrained=False,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    total_loss = 0.0
    correct = 0
    total = 0
    targets = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            total_loss += torch.nn.functional.cross_entropy(logits, labels).item() * labels.size(0)
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            targets.append(labels.cpu())
            predictions.append(predicted.cpu())
    target = torch.cat(targets)
    prediction = torch.cat(predictions)
    num_classes = int(max(target.max().item(), prediction.max().item()) + 1)
    confusion = torch.bincount(
        target * num_classes + prediction,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes).float()
    true_positive = confusion.diag()
    macro_f1 = (
        2.0 * true_positive
        / (2.0 * true_positive + (confusion.sum(0) - true_positive) + (confusion.sum(1) - true_positive)).clamp_min(1.0)
    ).mean().item()
    return {
        "acc": correct / total,
        "macro_f1": macro_f1,
        "loss": total_loss / total,
        "num_samples": total,
    }


def main():
    args = parse_args()
    framework_root = args.framework_root.resolve()
    if str(framework_root) not in sys.path:
        sys.path.insert(0, str(framework_root))

    from merge import METHOD_DEFAULTS, merge_with_method
    from utils import ensure_state_dicts_compatible, extract_state_dict, load_checkpoint, load_hub_meta, set_seed

    exp_dir = (
        args.model_hub_root
        / "small"
        / "bloodmnist_224"
        / args.backbone
        / "clients_7"
        / "beta_0"
        / "seed_42"
    )
    meta = load_hub_meta(exp_dir)
    checkpoints = [load_checkpoint(exp_dir / client["checkpoint"], device="cpu") for client in meta["clients"]]
    state_dicts = [extract_state_dict(checkpoint) for checkpoint in checkpoints]
    ensure_state_dicts_compatible(state_dicts)
    cfg = {**METHOD_DEFAULTS, "method": "ties", "merge_weight_mode": "equal", "seed": 42}
    device = torch.device(args.device)
    rows = []

    for k in range(1, 8):
        step_meta = copy.deepcopy(meta)
        step_meta["num_clients"] = k
        step_meta["clients"] = step_meta["clients"][:k]
        set_seed(42)
        merged_state, _ = merge_with_method(
            "ties",
            state_dicts[:k],
            [1] * k,
            step_meta,
            checkpoints[:k],
            cfg,
        )
        result = evaluate_checkpoint(
            meta=step_meta,
            data_root=str(args.data_root),
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            state_dict=merged_state,
        )
        rows.append(
            {
                "k": k,
                "received_client_ids": list(range(k)),
                "acc": float(result["acc"]),
                "macro_f1": float(result["macro_f1"]),
                "loss": float(result["loss"]),
                "num_samples": int(result["num_samples"]),
                "amp_enabled": False,
            }
        )
        print(f"k={k} acc={result['acc']:.6f} macro_f1={result['macro_f1']:.6f}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.result_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.result_root / "ties_k1_to_k7.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "method": "ties",
        "dataset": "bloodmnist_224",
        "backbone": args.backbone,
        "num_clients": 7,
        "beta": 0.0,
        "seed": 42,
        "delivery_order": list(range(7)),
        "amp_enabled": False,
        "rows": rows,
    }
    (args.result_root / "ties_k1_to_k7.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
