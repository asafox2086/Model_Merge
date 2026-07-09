#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import NpzTensorDataset, load_npz_splits
from model import build_model
from utils import load_checkpoint, load_json


def parse_args():
    p = argparse.ArgumentParser("Diagnose per-class prediction collapse for a small checkpoint")
    p.add_argument("--merged-dir", type=str, required=True)
    p.add_argument("--data-root", type=str, default=str(ROOT / "Med_data"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def image_hw(images):
    if images.ndim == 3:
        return int(images.shape[1]), int(images.shape[2])
    if images.ndim >= 4:
        return int(images.shape[1]), int(images.shape[2])
    raise ValueError(f"Unexpected image shape: {images.shape}")


def build_transform(meta, images):
    source_h, source_w = image_hw(images)
    target = int(meta.get("image_size") or 0)
    if target > 0 and (source_h != target or source_w != target):
        return transforms.Resize((target, target), antialias=True)
    return None


def main():
    args = parse_args()
    merged_dir = Path(args.merged_dir)
    meta = load_json(merged_dir / "meta.json")
    checkpoint = load_checkpoint(merged_dir / "merged.pt", device="cpu")
    if meta.get("task_type") != "small":
        raise ValueError("Only small image checkpoints are supported.")

    splits = load_npz_splits(str(Path(args.data_root) / f"{meta['dataset']}.npz"))
    split = splits[args.split]
    transform = build_transform(meta, split.images)
    dataset = NpzTensorDataset(split.images, split.labels, transform=transform)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    model, _, _, _ = build_model(
        name=meta["model"],
        num_classes=int(meta["num_classes"]),
        in_channels=int(meta.get("in_channels", 3)),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    num_classes = int(meta["num_classes"])
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    loss_sum = 0.0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += torch.nn.functional.cross_entropy(logits, y).item() * int(y.numel())
            pred = logits.argmax(dim=1)
            for t, p in zip(y.detach().cpu().tolist(), pred.detach().cpu().tolist()):
                confusion[int(t), int(p)] += 1
            total += int(y.numel())

    correct = int(confusion.diag().sum().item())
    true_counts = confusion.sum(dim=1).clamp_min(1)
    pred_counts = confusion.sum(dim=0)
    recall = confusion.diag().float() / true_counts.float()
    payload = {
        "dataset": meta["dataset"],
        "model": meta["model"],
        "num_clients": int(meta["num_clients"]),
        "beta": float(meta["beta"]),
        "split": args.split,
        "acc": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "true_counts": [int(x) for x in confusion.sum(dim=1).tolist()],
        "pred_counts": [int(x) for x in pred_counts.tolist()],
        "recall": [float(x) for x in recall.tolist()],
        "confusion": [[int(v) for v in row] for row in confusion.tolist()],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
