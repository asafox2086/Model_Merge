#!/usr/bin/env python3
"""Train an equal-client-count control case for collapse diagnostics.

This creates a new isolated hub-like directory. It keeps the original DermaMNIST
client class groups but downsamples every client to the same total number of
training examples, preserving each client's within-group class proportions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.medmnist_npz import load_npz_splits
from model import build_model
from utils.io import load_json
from utils.seed import set_seed
from utils.state_dict import average_state_dicts


DEFAULT_CASE = "model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42"


class ArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray | None = None):
        self.images = images
        self.labels = labels.reshape(-1).astype(np.int64)
        self.indices = np.arange(len(self.labels), dtype=np.int64) if indices is None else indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        img = self.images[idx]
        if img.ndim == 2:
            img = img[:, :, None]
        x = torch.from_numpy(img).float()
        if x.ndim == 3:
            x = x.permute(2, 0, 1)
        x = x / 255.0
        y = int(self.labels[idx])
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", default=DEFAULT_CASE)
    parser.add_argument("--data-root", default="Med_data")
    parser.add_argument("--public-root", default="PublicMedFingerprint_data")
    parser.add_argument("--out-dir", default="medmerge_empirical_study/results/experiment7_equal_count_clients")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling-mode",
        choices=("equal_total_proportional", "balanced_local"),
        default="equal_total_proportional",
        help=(
            "equal_total_proportional keeps client totals equal and preserves each local class ratio; "
            "balanced_local samples the same count per seen class inside each client."
        ),
    )
    return parser.parse_args()


def proportional_downsample(labels: np.ndarray, classes: list[int], target_total: int, seed: int) -> tuple[np.ndarray, dict[int, int]]:
    rng = np.random.default_rng(seed)
    pools = {cls: np.where(labels.reshape(-1) == cls)[0] for cls in classes}
    counts = {cls: int(len(pools[cls])) for cls in classes}
    total = sum(counts.values())
    if target_total > total:
        raise ValueError(f"target_total={target_total} exceeds available={total} for classes={classes}")

    raw = {cls: target_total * counts[cls] / total for cls in classes}
    take = {cls: int(math.floor(raw[cls])) for cls in classes}
    remainder = target_total - sum(take.values())
    order = sorted(classes, key=lambda cls: (raw[cls] - take[cls], counts[cls]), reverse=True)
    for cls in order[:remainder]:
        take[cls] += 1

    sampled = []
    for cls in classes:
        cls_indices = pools[cls]
        if take[cls] > len(cls_indices):
            raise ValueError(f"not enough class {cls}: take={take[cls]}, available={len(cls_indices)}")
        chosen = rng.choice(cls_indices, size=take[cls], replace=False)
        sampled.append(chosen)
    out = np.concatenate(sampled)
    rng.shuffle(out)
    return out.astype(np.int64), take


def balanced_local_downsample(labels: np.ndarray, classes: list[int], per_class: int, seed: int) -> tuple[np.ndarray, dict[int, int]]:
    rng = np.random.default_rng(seed)
    sampled = []
    take = {}
    for cls in classes:
        cls_indices = np.where(labels.reshape(-1) == cls)[0]
        if per_class > len(cls_indices):
            raise ValueError(f"not enough class {cls}: take={per_class}, available={len(cls_indices)}")
        chosen = rng.choice(cls_indices, size=per_class, replace=False)
        sampled.append(chosen)
        take[cls] = int(per_class)
    out = np.concatenate(sampled)
    rng.shuffle(out)
    return out.astype(np.int64), take


def build_initial_model(meta: dict, device: torch.device) -> torch.nn.Module:
    model, model_name, _img_size, used_pretrained = build_model(
        name=meta["model"],
        num_classes=int(meta["num_classes"]),
        in_channels=int(meta.get("in_channels", 3)),
        pretrained=False,
    )
    cache_path = REPO_ROOT / "reference_cache" / "small" / (
        f"{meta['model']}__cls{meta['num_classes']}__in{meta.get('in_channels', 3)}__"
        f"pretrained{int(bool(meta.get('pretrained', False)))}__seed{meta.get('seed', 42)}.pt"
    )
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        model.load_state_dict(cached["state_dict"], strict=True)
        used_pretrained = bool(cached.get("meta", {}).get("pretrained", meta.get("pretrained", False)))
        model_name = cached.get("meta", {}).get("model", model_name)
    elif meta.get("pretrained", False):
        model, model_name, _img_size, used_pretrained = build_model(
            name=meta["model"],
            num_classes=int(meta["num_classes"]),
            in_channels=int(meta.get("in_channels", 3)),
            pretrained=True,
        )
    model._hub_model_name = model_name  # type: ignore[attr-defined]
    model._hub_used_pretrained = used_pretrained  # type: ignore[attr-defined]
    return model.to(device)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int, amp: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    pred_counts = torch.zeros(num_classes, dtype=torch.long)
    support = torch.zeros(num_classes, dtype=torch.long)
    confidence_sum = 0.0
    margin_sum = 0.0
    min_margin = None
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        probs = torch.softmax(logits.float(), dim=1)
        sorted_logits, _ = torch.sort(logits.float(), dim=1, descending=True)
        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * y.numel()
        correct += int((preds == y).sum().item())
        total += int(y.numel())
        pred_counts += torch.bincount(preds.detach().cpu(), minlength=num_classes)
        support += torch.bincount(y.detach().cpu(), minlength=num_classes)
        confidence_sum += float(probs.max(dim=1).values.sum().item())
        margins = sorted_logits[:, 0] - sorted_logits[:, 1]
        margin_sum += float(margins.sum().item())
        batch_min = float(margins.min().item())
        min_margin = batch_min if min_margin is None else min(min_margin, batch_min)
    recalls = []
    for cls in range(num_classes):
        denom = int(support[cls].item())
        if denom:
            # Re-evaluate recall from predictions is not stored per class here; do a confusion-free second pass would be wasteful.
            recalls.append(float("nan"))
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "num_samples": total,
        "pred_counts": pred_counts.tolist(),
        "support": support.tolist(),
        "collapse_ratio": int(pred_counts.max().item()) / max(total, 1),
        "effective_pred_classes": int((pred_counts > 0).sum().item()),
        "top_pred_class": int(torch.argmax(pred_counts).item()),
        "mean_confidence": confidence_sum / max(total, 1),
        "mean_top1_top2_margin": margin_sum / max(total, 1),
        "min_top1_top2_margin": float(min_margin or 0.0),
    }


@torch.no_grad()
def evaluate_full(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int, amp: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    logits_chunks = []
    label_chunks = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * y.numel()
        logits_chunks.append(logits.detach().float().cpu())
        label_chunks.append(y.detach().cpu())
    logits = torch.cat(logits_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    preds = logits.argmax(dim=1)
    pred_counts = torch.bincount(preds, minlength=num_classes)
    support = torch.bincount(labels, minlength=num_classes)
    recalls = []
    f1s = []
    precisions = []
    for cls in range(num_classes):
        tp = int(((preds == cls) & (labels == cls)).sum().item())
        fp = int(((preds == cls) & (labels != cls)).sum().item())
        fn = int(((preds != cls) & (labels == cls)).sum().item())
        denom_r = int((labels == cls).sum().item())
        rec = tp / denom_r if denom_r else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)
    probs = torch.softmax(logits, dim=1)
    sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
    margins = sorted_logits[:, 0] - sorted_logits[:, 1]
    return {
        "loss": total_loss / max(int(labels.numel()), 1),
        "accuracy": float((preds == labels).float().mean().item()),
        "balanced_accuracy": float(sum(recalls) / num_classes),
        "macro_f1": float(sum(f1s) / num_classes),
        "num_samples": int(labels.numel()),
        "pred_counts": pred_counts.tolist(),
        "support": support.tolist(),
        "collapse_ratio": int(pred_counts.max().item()) / max(int(labels.numel()), 1),
        "effective_pred_classes": int((pred_counts > 0).sum().item()),
        "top_pred_class": int(torch.argmax(pred_counts).item()),
        "mean_confidence": float(probs.max(dim=1).values.mean().item()),
        "mean_top1_top2_margin": float(margins.mean().item()),
        "min_top1_top2_margin": float(margins.min().item()),
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "per_class_f1": f1s,
    }


def train_one_client(
    *,
    client_id: int,
    meta: dict,
    train_ds: Dataset,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    amp: bool,
) -> tuple[dict, dict, dict]:
    set_seed(int(meta.get("seed", 42)) + client_id)
    model = build_initial_model(meta, device)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    best_val_acc = -1.0
    best_state = None
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * y.numel()
            train_total += int(y.numel())
            train_correct += int((logits.argmax(dim=1) == y).sum().item())
        val_metrics = evaluate_full(model, val_loader, device, int(meta["num_classes"]), amp=amp)
        train_acc = train_correct / max(train_total, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_total, 1),
            "train_acc": train_acc,
            "val_acc": val_metrics["accuracy"],
            "val_bacc": val_metrics["balanced_accuracy"],
            "val_collapse": val_metrics["collapse_ratio"],
            "seconds": time.time() - start,
        }
        history.append(row)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
        print(
            f"client={client_id} epoch={epoch:03d}/{epochs} "
            f"train_acc={train_acc:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f} collapse={val_metrics['collapse_ratio']:.3f}",
            flush=True,
        )
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    test_metrics = evaluate_full(model, test_loader, device, int(meta["num_classes"]), amp=amp)
    checkpoint = {
        "state_dict": best_state,
        "model_name": getattr(model, "_hub_model_name", meta["model"]),
        "used_pretrained": bool(getattr(model, "_hub_used_pretrained", meta.get("pretrained", False))),
        "num_classes": int(meta["num_classes"]),
        "in_channels": int(meta.get("in_channels", 3)),
        "client_id": client_id,
        "best_val_acc": best_val_acc,
    }
    return checkpoint, {"history": history, "test": test_metrics}, test_metrics


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compact_row(model_name: str, split_name: str, metrics: dict) -> dict:
    return {
        "model": model_name,
        "split": split_name,
        "accuracy": f"{metrics['accuracy']:.6f}",
        "balanced_accuracy": f"{metrics['balanced_accuracy']:.6f}",
        "macro_f1": f"{metrics['macro_f1']:.6f}",
        "collapse_ratio": f"{metrics['collapse_ratio']:.6f}",
        "effective_pred_classes": metrics["effective_pred_classes"],
        "top_pred_class": metrics["top_pred_class"],
        "mean_confidence": f"{metrics['mean_confidence']:.6f}",
        "mean_top1_top2_margin": f"{metrics['mean_top1_top2_margin']:.6f}",
        "pred_counts": json.dumps(metrics["pred_counts"], ensure_ascii=False),
        "support": json.dumps(metrics["support"], ensure_ascii=False),
    }


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    hub_dir = out_dir / "equal_count_hub" / "small" / "dermamnist_224" / "resnet" / "clients_3" / args.sampling_mode / "seed_42"
    hub_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    set_seed(args.seed)

    case_dir = Path(args.case_dir)
    source_meta = load_json(case_dir / "meta.json")
    epochs = int(args.epochs or source_meta.get("epochs", 50))
    batch_size = int(args.batch_size or source_meta.get("batch_size", 64))
    lr = float(args.lr or source_meta.get("lr", 1e-3))
    weight_decay = float(args.weight_decay or source_meta.get("weight_decay", 1e-4))
    num_classes = int(source_meta["num_classes"])

    splits = load_npz_splits(str(Path(args.data_root) / f"{source_meta['dataset']}.npz"))
    public_splits = load_npz_splits(str(Path(args.public_root) / f"{source_meta['dataset']}.npz"))
    train_labels = splits["train"].labels.reshape(-1)
    client_classes = {int(k): [int(x) for x in v] for k, v in source_meta["client_classes"].items()}
    available = {
        client_id: int(sum((train_labels == cls).sum() for cls in classes))
        for client_id, classes in client_classes.items()
    }
    target_total = min(available.values())
    balanced_per_class = min(
        int((train_labels == cls).sum())
        for classes in client_classes.values()
        for cls in classes
    )

    val_loader = DataLoader(
        ArrayDataset(splits["val"].images, splits["val"].labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    source_test_loader = DataLoader(
        ArrayDataset(splits["test"].images, splits["test"].labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    public_test_loader = DataLoader(
        ArrayDataset(public_splits["test"].images, public_splits["test"].labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    client_meta = []
    state_dicts = []
    client_rows = []
    metric_rows = []
    for client_id in sorted(client_classes):
        classes = client_classes[client_id]
        if args.sampling_mode == "equal_total_proportional":
            indices, per_class_counts = proportional_downsample(
                train_labels,
                classes,
                target_total,
                seed=args.seed + client_id * 101,
            )
        else:
            indices, per_class_counts = balanced_local_downsample(
                train_labels,
                classes,
                balanced_per_class,
                seed=args.seed + client_id * 101,
            )
        train_ds = ArrayDataset(splits["train"].images, splits["train"].labels, indices=indices)
        checkpoint, train_info, source_test_metrics = train_one_client(
            client_id=client_id,
            meta=source_meta,
            train_ds=train_ds,
            val_loader=val_loader,
            test_loader=source_test_loader,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            num_workers=args.num_workers,
            amp=args.amp,
        )
        ckpt_name = f"client_{client_id}.pt"
        torch.save(checkpoint, hub_dir / ckpt_name)
        state_dicts.append(checkpoint["state_dict"])
        with (hub_dir / f"client_{client_id}_history.json").open("w", encoding="utf-8") as handle:
            json.dump(train_info, handle, ensure_ascii=False, indent=2)
        eval_model = build_initial_model(source_meta, device)
        eval_model.load_state_dict(checkpoint["state_dict"], strict=True)
        public_metrics = evaluate_full(eval_model, public_test_loader, device, num_classes, amp=args.amp)
        del eval_model
        client_meta.append(
            {
                "client_id": client_id,
                "num_samples": int(len(indices)),
                "classes": classes,
                "available_num_samples": available[client_id],
                "sampled_class_counts": {str(k): int(v) for k, v in per_class_counts.items()},
                "best_val_acc": checkpoint["best_val_acc"],
                "test_acc": source_test_metrics["accuracy"],
                "test_loss": source_test_metrics["loss"],
                "checkpoint": ckpt_name,
                "source_checkpoint": "",
            }
        )
        client_rows.append(
            {
                "client_id": client_id,
                "classes": json.dumps(classes),
                "available_num_samples": available[client_id],
                "sampled_num_samples": int(len(indices)),
                "sampled_class_counts": json.dumps(per_class_counts, ensure_ascii=False),
                "best_val_acc": f"{checkpoint['best_val_acc']:.6f}",
                "source_test_acc": f"{source_test_metrics['accuracy']:.6f}",
                "source_test_bacc": f"{source_test_metrics['balanced_accuracy']:.6f}",
                "source_test_pred_counts": json.dumps(source_test_metrics["pred_counts"], ensure_ascii=False),
                "public_test_acc": f"{public_metrics['accuracy']:.6f}",
                "public_test_bacc": f"{public_metrics['balanced_accuracy']:.6f}",
                "public_test_pred_counts": json.dumps(public_metrics["pred_counts"], ensure_ascii=False),
            }
        )
        metric_rows.append(compact_row(f"client_{client_id}", "source_test", source_test_metrics))
        metric_rows.append(compact_row(f"client_{client_id}", "public_test", public_metrics))

    meta = dict(source_meta)
    meta.update(
        {
            "train_mode": "equal_count_control",
            "equal_count_variant": f"{args.sampling_mode}_with_original_client_classes",
            "source_case_dir": str(case_dir),
            "num_clients": len(client_meta),
            "beta": source_meta.get("beta", 0.0),
            "seed": args.seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "clients": client_meta,
            "client_classes": {str(k): v for k, v in client_classes.items()},
        }
    )
    with (hub_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    avg_state, avg_weights = average_state_dicts(state_dicts, [1.0] * len(state_dicts))
    avg_ckpt = {
        "state_dict": avg_state,
        "model_name": "resnet18",
        "used_pretrained": bool(source_meta.get("pretrained", True)),
        "num_classes": num_classes,
        "in_channels": int(source_meta.get("in_channels", 3)),
        "merge_method": "avg",
        "weights": avg_weights,
    }
    torch.save(avg_ckpt, hub_dir / "avg_merged.pt")
    avg_model = build_initial_model(source_meta, device)
    avg_model.load_state_dict(avg_state, strict=True)
    avg_source = evaluate_full(avg_model, source_test_loader, device, num_classes, amp=args.amp)
    avg_public = evaluate_full(avg_model, public_test_loader, device, num_classes, amp=args.amp)
    metric_rows.append(compact_row("avg", "source_test", avg_source))
    metric_rows.append(compact_row("avg", "public_test", avg_public))

    save_csv(out_dir / "equal_count_client_rows.csv", client_rows)
    save_csv(out_dir / "equal_count_metrics.csv", metric_rows)
    with (out_dir / "equal_count_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "hub_dir": str(hub_dir),
                "target_total_per_client": target_total,
                "balanced_per_class": balanced_per_class,
                "sampling_mode": args.sampling_mode,
                "available_per_client": available,
                "avg_source": avg_source,
                "avg_public": avg_public,
                "client_rows": client_rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    md = []
    md.append("# Equal-Count Client Control\n")
    md.append(f"- Source case: `{case_dir}`")
    md.append(f"- Output hub: `{hub_dir}`")
    if args.sampling_mode == "equal_total_proportional":
        md.append(f"- Control: every client is downsampled to `{target_total}` train samples.")
        md.append("- Within-client class proportions are preserved.")
    else:
        md.append(f"- Control: every seen class inside every client is downsampled to `{balanced_per_class}` train samples.")
        md.append("- Client totals differ when clients have different numbers of seen classes.")
    md.append("- Client class groups are unchanged.\n")
    md.append("## Client Sampling and Single-Client Evaluation\n")
    md.append(
        markdown_table(
            client_rows,
            [
                "client_id",
                "classes",
                "available_num_samples",
                "sampled_num_samples",
                "sampled_class_counts",
                "best_val_acc",
                "source_test_acc",
                "source_test_bacc",
                "source_test_pred_counts",
                "public_test_acc",
                "public_test_bacc",
                "public_test_pred_counts",
            ],
        )
    )
    md.append("\n## Equal-Count AVG Evaluation\n")
    avg_rows = [row for row in metric_rows if row["model"] == "avg"]
    md.append(
        markdown_table(
            avg_rows,
            [
                "model",
                "split",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "collapse_ratio",
                "effective_pred_classes",
                "top_pred_class",
                "mean_confidence",
                "pred_counts",
                "support",
            ],
        )
    )
    md.append("\n## Reading\n")
    if args.sampling_mode == "equal_total_proportional":
        md.append(
            "- This isolates total client sample count. It does not make the data IID because each client still sees only its original class subset."
        )
    else:
        md.append(
            "- This isolates local class imbalance within each client. It still does not make the data IID because label support remains disjoint."
        )
    md.append(
        "- If collapse persists or class-wise metrics remain low, equal total sample count is insufficient; label-support mismatch is the core issue."
    )
    (out_dir / "equal_count_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'equal_count_summary.md'}")
    print(f"Wrote {out_dir / 'equal_count_metrics.csv'}")
    print(f"Hub directory: {hub_dir}")


if __name__ == "__main__":
    main()
