#!/usr/bin/env python3
"""Probe whether partial-label natural-image clients collapse after averaging."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.medmnist_npz import load_npz_splits
from model import build_model
from utils.seed import set_seed
from utils.state_dict import average_state_dicts


DEFAULT_GROUPS = {
    "cifar10_32": "0,1;2,3;4,5;6,7;8,9",
    "svhn_32": "0,1;2,3;4,5;6,7;8,9",
    "cifar100_32": ";".join(",".join(str(x) for x in range(i, i + 10)) for i in range(0, 100, 10)),
}


class IndexedArrayDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray | None = None,
        *,
        image_size: int | None = None,
    ) -> None:
        self.images = images
        self.labels = labels.reshape(-1).astype(np.int64)
        self.indices = np.arange(len(self.labels), dtype=np.int64) if indices is None else indices.astype(np.int64)
        self.resize = None
        if image_size:
            h = int(images.shape[1])
            w = int(images.shape[2])
            if h != image_size or w != image_size:
                self.resize = transforms.Resize((image_size, image_size), antialias=True)

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
        if self.resize is not None:
            x = self.resize(x)
        return x, int(self.labels[idx])


def parse_groups(raw: str) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for client_id, part in enumerate(raw.split(";")):
        classes = [int(x.strip()) for x in part.split(",") if x.strip() != ""]
        if not classes:
            raise ValueError(f"empty class group at client {client_id}: {raw!r}")
        groups[client_id] = classes
    return groups


def parse_group_class_counts(raw: str, client_classes: dict[int, list[int]]) -> dict[int, dict[int, int]]:
    counts_by_client: dict[int, dict[int, int]] = {}
    parts = raw.split(";")
    if len(parts) != len(client_classes):
        raise ValueError(
            f"--group-class-counts has {len(parts)} client groups, expected {len(client_classes)}."
        )
    for client_id, part in enumerate(parts):
        values = [int(x.strip()) for x in part.split(",") if x.strip() != ""]
        classes = client_classes[client_id]
        if len(values) != len(classes):
            raise ValueError(
                f"--group-class-counts client {client_id} has {len(values)} counts, "
                f"expected {len(classes)} for classes={classes}."
            )
        counts_by_client[client_id] = {int(cls): int(count) for cls, count in zip(classes, values)}
    return counts_by_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10_32")
    parser.add_argument("--data-root", default="Med_data")
    parser.add_argument("--model", default="resnet")
    parser.add_argument("--groups", default=None, help="Client class groups, e.g. '0,1;2,3;4,5;6,7;8,9'.")
    parser.add_argument("--out-dir", default="medmerge_empirical_study/results/experiment9_natural_image_collapse_probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--max-train-per-class", type=int, default=0)
    parser.add_argument(
        "--group-class-counts",
        default=None,
        help=(
            "Optional per-client class counts aligned with --groups, e.g. "
            "'80,80;80,80;80,80;80,80;4000,80'. Overrides --max-train-per-class."
        ),
    )
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def infer_num_classes(labels: np.ndarray) -> int:
    labels = labels.reshape(-1)
    return int(labels.max()) + 1


def sample_indices_for_classes(
    labels: np.ndarray,
    classes: list[int],
    *,
    max_per_class: int,
    class_counts: dict[int, int] | None,
    seed: int,
) -> tuple[np.ndarray, dict[int, int]]:
    rng = np.random.default_rng(seed)
    labels = labels.reshape(-1)
    out = []
    counts = {}
    for cls in classes:
        pool = np.where(labels == cls)[0]
        if class_counts is not None:
            take = int(class_counts[int(cls)])
        else:
            take = int(len(pool))
        if max_per_class > 0 and class_counts is None:
            take = min(take, max_per_class)
        if take > int(len(pool)):
            raise ValueError(f"class {cls} requested {take} samples, but only {len(pool)} are available.")
        chosen = rng.choice(pool, size=take, replace=False)
        out.append(chosen)
        counts[int(cls)] = int(take)
    indices = np.concatenate(out).astype(np.int64)
    rng.shuffle(indices)
    return indices, counts


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
def evaluate_full(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    amp: bool,
) -> dict:
    model.eval()
    logits_chunks = []
    label_chunks = []
    total_loss = 0.0
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
    precisions = []
    f1s = []
    for cls in range(num_classes):
        tp = int(((preds == cls) & (labels == cls)).sum().item())
        fp = int(((preds == cls) & (labels != cls)).sum().item())
        fn = int(((preds != cls) & (labels == cls)).sum().item())
        rec = tp / int(support[cls].item()) if int(support[cls].item()) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)
    probs = torch.softmax(logits, dim=1)
    sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
    margins = sorted_logits[:, 0] - sorted_logits[:, 1]
    total = max(int(labels.numel()), 1)
    return {
        "loss": total_loss / total,
        "accuracy": float((preds == labels).float().mean().item()),
        "balanced_accuracy": float(sum(recalls) / num_classes),
        "macro_f1": float(sum(f1s) / num_classes),
        "num_samples": int(labels.numel()),
        "pred_counts": pred_counts.tolist(),
        "support": support.tolist(),
        "collapse_ratio": int(pred_counts.max().item()) / total,
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
    set_seed(int(meta["seed"]) + client_id)
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
        train_total = 0
        train_correct = 0
        train_loss = 0.0
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
            train_total += int(y.numel())
            train_correct += int((logits.argmax(dim=1) == y).sum().item())
            train_loss += float(loss.item()) * y.numel()
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
        "min_top1_top2_margin": f"{metrics['min_top1_top2_margin']:.6f}",
        "pred_counts": json.dumps(metrics["pred_counts"], ensure_ascii=False),
        "support": json.dumps(metrics["support"], ensure_ascii=False),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    set_seed(args.seed)

    splits = load_npz_splits(str(Path(args.data_root) / f"{args.dataset}.npz"))
    train_images = splits["train"].images
    train_labels = splits["train"].labels.reshape(-1)
    num_classes = infer_num_classes(train_labels)
    in_channels = int(train_images.shape[-1]) if train_images.ndim >= 4 else 1
    raw_groups = args.groups or DEFAULT_GROUPS.get(args.dataset)
    if raw_groups is None:
        raise ValueError(f"No default groups for dataset={args.dataset}; pass --groups.")
    client_classes = parse_groups(raw_groups)
    group_class_counts = (
        parse_group_class_counts(args.group_class_counts, client_classes)
        if args.group_class_counts
        else None
    )

    meta = {
        "task_type": "small",
        "dataset": args.dataset,
        "model": args.model,
        "train_mode": "natural_partial_label_probe",
        "num_clients": len(client_classes),
        "beta": 0.0,
        "seed": args.seed,
        "num_classes": num_classes,
        "observed_num_classes": num_classes,
        "in_channels": in_channels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "pretrained": bool(args.pretrained),
        "client_classes": {str(k): v for k, v in client_classes.items()},
        "group_class_counts": (
            {str(cid): {str(cls): int(count) for cls, count in counts.items()} for cid, counts in group_class_counts.items()}
            if group_class_counts
            else None
        ),
    }

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(
        IndexedArrayDataset(splits["val"].images, splits["val"].labels, image_size=args.image_size),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        IndexedArrayDataset(splits["test"].images, splits["test"].labels, image_size=args.image_size),
        **loader_kwargs,
    )

    hub_dir = out_dir / "natural_probe_hub" / "small" / args.dataset / args.model / f"clients_{len(client_classes)}" / "partial_label" / f"seed_{args.seed}"
    hub_dir.mkdir(parents=True, exist_ok=True)
    state_dicts = []
    client_rows = []
    metric_rows = []
    client_meta = []
    for client_id in sorted(client_classes):
        classes = client_classes[client_id]
        indices, sampled_counts = sample_indices_for_classes(
            train_labels,
            classes,
            max_per_class=args.max_train_per_class,
            class_counts=group_class_counts.get(client_id) if group_class_counts else None,
            seed=args.seed + client_id * 101,
        )
        train_ds = IndexedArrayDataset(splits["train"].images, splits["train"].labels, indices, image_size=args.image_size)
        checkpoint, train_info, test_metrics = train_one_client(
            client_id=client_id,
            meta=meta,
            train_ds=train_ds,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            amp=args.amp,
        )
        ckpt_name = f"client_{client_id}.pt"
        torch.save(checkpoint, hub_dir / ckpt_name)
        (hub_dir / f"client_{client_id}_history.json").write_text(
            json.dumps(train_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state_dicts.append(checkpoint["state_dict"])
        client_meta.append(
            {
                "client_id": client_id,
                "num_samples": int(len(indices)),
                "classes": classes,
                "sampled_class_counts": {str(k): int(v) for k, v in sampled_counts.items()},
                "best_val_acc": checkpoint["best_val_acc"],
                "test_acc": test_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "checkpoint": ckpt_name,
            }
        )
        client_rows.append(
            {
                "client_id": client_id,
                "classes": json.dumps(classes),
                "sampled_num_samples": int(len(indices)),
                "sampled_class_counts": json.dumps(sampled_counts, ensure_ascii=False),
                "best_val_acc": f"{checkpoint['best_val_acc']:.6f}",
                "test_acc": f"{test_metrics['accuracy']:.6f}",
                "test_bacc": f"{test_metrics['balanced_accuracy']:.6f}",
                "test_collapse": f"{test_metrics['collapse_ratio']:.6f}",
                "effective_pred_classes": test_metrics["effective_pred_classes"],
                "top_pred_class": test_metrics["top_pred_class"],
                "test_pred_counts": json.dumps(test_metrics["pred_counts"], ensure_ascii=False),
            }
        )
        metric_rows.append(compact_row(f"client_{client_id}", "test", test_metrics))

    meta["clients"] = client_meta
    (hub_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_state, avg_weights = average_state_dicts(state_dicts, [1.0] * len(state_dicts))
    avg_ckpt = {
        "state_dict": avg_state,
        "model_name": args.model,
        "used_pretrained": bool(args.pretrained),
        "num_classes": num_classes,
        "in_channels": in_channels,
        "merge_method": "avg",
        "weights": avg_weights,
    }
    torch.save(avg_ckpt, hub_dir / "avg_merged.pt")
    avg_model = build_initial_model(meta, device)
    avg_model.load_state_dict(avg_state, strict=True)
    avg_test = evaluate_full(avg_model, test_loader, device, num_classes, amp=args.amp)
    metric_rows.append(compact_row("avg", "test", avg_test))

    save_csv(out_dir / "natural_collapse_metrics.csv", metric_rows)
    save_csv(out_dir / "natural_client_rows.csv", client_rows)
    summary = {
        "hub_dir": str(hub_dir),
        "dataset": args.dataset,
        "model": args.model,
        "groups": client_classes,
        "image_size": args.image_size,
        "max_train_per_class": args.max_train_per_class,
        "group_class_counts": group_class_counts,
        "avg_test": avg_test,
        "client_rows": client_rows,
    }
    (out_dir / "natural_collapse_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_rows = metric_rows
    md = [
        "# Natural-Image Partial-Label Collapse Probe",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Model: `{args.model}`",
        f"- Clients: `{len(client_classes)}`",
        f"- Class groups: `{raw_groups}`",
        f"- Image size: `{args.image_size}`",
        f"- Epochs: `{args.epochs}`",
        f"- Max train per class: `{args.max_train_per_class or 'all'}`",
        f"- Group class counts: `{args.group_class_counts or 'not set'}`",
        f"- Output hub: `{hub_dir}`",
        "",
        "## Client Sampling and Single-Client Behavior",
        "",
        markdown_table(
            client_rows,
            [
                "client_id",
                "classes",
                "sampled_num_samples",
                "sampled_class_counts",
                "best_val_acc",
                "test_acc",
                "test_bacc",
                "test_collapse",
                "effective_pred_classes",
                "top_pred_class",
                "test_pred_counts",
            ],
        ),
        "",
        "## AVG Behavior",
        "",
        markdown_table(
            [row for row in md_rows if row["model"] == "avg"],
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
                "mean_top1_top2_margin",
                "pred_counts",
                "support",
            ],
        ),
        "",
        "## All Rows",
        "",
        markdown_table(
            md_rows,
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "collapse_ratio",
                "effective_pred_classes",
                "top_pred_class",
                "pred_counts",
            ],
        ),
        "",
        "## Reading",
        "",
        "- `collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.",
        "- `effective_pred_classes` counts how many classes receive at least one prediction.",
        "- A medical-style single-class collapse would have `collapse_ratio=1.0` and `effective_pred_classes=1`.",
    ]
    (out_dir / "natural_collapse_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'natural_collapse_summary.md'}")
    print(f"Wrote {out_dir / 'natural_collapse_metrics.csv'}")
    print(f"Hub directory: {hub_dir}")


if __name__ == "__main__":
    main()
