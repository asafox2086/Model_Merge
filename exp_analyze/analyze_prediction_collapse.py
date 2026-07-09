#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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


SETTINGS = [
    (3, 0.0, "c3_b0"),
    (3, 0.01, "c3_b0.01"),
    (3, 0.1, "c3_b0.1"),
    (5, 0.0, "c5_b0"),
    (5, 0.01, "c5_b0.01"),
    (5, 0.1, "c5_b0.1"),
    (7, 0.0, "c7_b0"),
    (7, 0.01, "c7_b0.01"),
    (7, 0.1, "c7_b0.1"),
]


def parse_label_roots(items):
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--root expects LABEL=PATH, got: {item}")
        label, path = item.split("=", 1)
        label = label.strip()
        path = Path(path.strip())
        if not label or not path:
            raise SystemExit(f"--root expects LABEL=PATH, got: {item}")
        out.append((label, path))
    return out


def parse_args():
    p = argparse.ArgumentParser("Analyze prediction collapse from merged small-model checkpoints.")
    p.add_argument("--root", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--merged-dir", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--data-root", default=str(ROOT / "Med_data"))
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--num-clients", nargs="*", type=int, default=None)
    p.add_argument("--betas", nargs="*", type=float, default=None)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dest-csv", required=True)
    p.add_argument("--dest-md", required=True)
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


def beta_label(beta):
    text = f"{float(beta):g}".replace(".", "p")
    return "beta_" + text


def setting_label(num_clients, beta):
    for c, b, label in SETTINGS:
        if int(num_clients) == c and abs(float(beta) - float(b)) < 1e-12:
            return label
    return f"c{int(num_clients)}_b{float(beta):g}"


def iter_merged_dirs(label, root):
    merged_root = root / "merged"
    if not merged_root.exists():
        return
    for ckpt in sorted(merged_root.rglob("merged.pt")):
        meta = ckpt.parent / "meta.json"
        if meta.exists():
            yield label, ckpt.parent


def iter_explicit_dirs(items):
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--merged-dir expects LABEL=PATH, got: {item}")
        label, path = item.split("=", 1)
        path = Path(path.strip())
        if not (path / "merged.pt").exists() or not (path / "meta.json").exists():
            raise FileNotFoundError(f"merged-dir must contain merged.pt and meta.json: {path}")
        yield label.strip(), path


def passes_filters(meta, method, args):
    if meta.get("task_type") != "small":
        return False
    if args.datasets and meta.get("dataset") not in set(args.datasets):
        return False
    if args.models and meta.get("model") not in set(args.models):
        return False
    if args.num_clients and int(meta.get("num_clients")) not in set(args.num_clients):
        return False
    if args.betas:
        wanted = {format(float(x), "g") for x in args.betas}
        if format(float(meta.get("beta")), "g") not in wanted:
            return False
    if args.methods and method not in set(args.methods):
        return False
    return True


def safe_distribution(counts):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    total = counts.sum().clamp_min(1.0)
    return counts / total


def entropy(dist):
    dist = dist.clamp_min(1e-12)
    return float(-(dist * dist.log()).sum().item())


def normalized_entropy(dist):
    if int(dist.numel()) <= 1:
        return 0.0
    return entropy(dist) / math.log(int(dist.numel()))


def kl_div(p, q):
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    return float((p * (p.log() - q.log())).sum().item())


def tv_distance(p, q):
    return float(0.5 * (p - q).abs().sum().item())


def diagnose_one(label, merged_dir, args, device):
    meta = load_json(merged_dir / "meta.json")
    checkpoint = load_checkpoint(merged_dir / "merged.pt", device="cpu")
    ckpt_method = checkpoint.get("meta", {}).get("method") or merged_dir.name
    method = str(ckpt_method)
    if not passes_filters(meta, method, args):
        return None

    splits = load_npz_splits(str(Path(args.data_root) / f"{meta['dataset']}.npz"))
    split = splits[args.split]
    transform = build_transform(meta, split.images)
    dataset = NpzTensorDataset(split.images, split.labels, transform=transform)
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
    prob_sum = torch.zeros(num_classes, dtype=torch.float64)
    max_prob_sum = 0.0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            loss_sum += torch.nn.functional.cross_entropy(logits, y).item() * int(y.numel())
            prob_sum += probs.detach().cpu().sum(dim=0).to(torch.float64)
            max_prob_sum += float(probs.max(dim=1).values.detach().cpu().sum().item())
            for t, p in zip(y.detach().cpu().tolist(), pred.detach().cpu().tolist()):
                confusion[int(t), int(p)] += 1
            total += int(y.numel())

    true_counts = confusion.sum(dim=1)
    pred_counts = confusion.sum(dim=0)
    true_dist = safe_distribution(true_counts)
    pred_dist = safe_distribution(pred_counts)
    prob_dist = safe_distribution(prob_sum)
    correct = int(confusion.diag().sum().item())
    dominant_pred = int(pred_counts.argmax().item()) if total else -1
    dominant_true = int(true_counts.argmax().item()) if total else -1
    row = {
        "label": label,
        "method": method,
        "dataset": meta["dataset"],
        "model": meta["model"],
        "setting": setting_label(meta["num_clients"], meta["beta"]),
        "num_clients": int(meta["num_clients"]),
        "beta": float(meta["beta"]),
        "acc": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "num_samples": int(total),
        "num_classes": int(num_classes),
        "true_majority_class": dominant_true,
        "true_majority_ratio": float(true_dist.max().item()),
        "pred_majority_class": dominant_pred,
        "pred_majority_ratio": float(pred_dist.max().item()),
        "pred_effective_classes": float(torch.exp(torch.tensor(entropy(pred_dist))).item()),
        "pred_nonzero_classes": int((pred_counts > 0).sum().item()),
        "pred_entropy_norm": normalized_entropy(pred_dist),
        "prob_entropy_norm": normalized_entropy(prob_dist),
        "mean_max_probability": max_prob_sum / max(total, 1),
        "pred_true_tv": tv_distance(pred_dist, true_dist),
        "pred_true_kl": kl_div(pred_dist, true_dist),
        "prob_true_tv": tv_distance(prob_dist, true_dist),
        "true_counts": json.dumps([int(x) for x in true_counts.tolist()], ensure_ascii=False),
        "pred_counts": json.dumps([int(x) for x in pred_counts.tolist()], ensure_ascii=False),
        "prob_mean": json.dumps([round(float(x), 6) for x in prob_dist.tolist()], ensure_ascii=False),
        "merged_dir": str(merged_dir),
    }
    return row


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "method",
        "dataset",
        "model",
        "setting",
        "num_clients",
        "beta",
        "acc",
        "loss",
        "num_samples",
        "num_classes",
        "true_majority_class",
        "true_majority_ratio",
        "pred_majority_class",
        "pred_majority_ratio",
        "pred_effective_classes",
        "pred_nonzero_classes",
        "pred_entropy_norm",
        "prob_entropy_norm",
        "mean_max_probability",
        "pred_true_tv",
        "pred_true_kl",
        "prob_true_tv",
        "true_counts",
        "pred_counts",
        "prob_mean",
        "merged_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def group_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["label"], row["method"])].append(row)
    out = []
    for (label, method), items in sorted(groups.items()):
        out.append({
            "label": label,
            "method": method,
            "n": len(items),
            "acc": sum(x["acc"] for x in items) / len(items),
            "pred_majority_ratio": sum(x["pred_majority_ratio"] for x in items) / len(items),
            "pred_effective_classes": sum(x["pred_effective_classes"] for x in items) / len(items),
            "pred_entropy_norm": sum(x["pred_entropy_norm"] for x in items) / len(items),
            "pred_true_tv": sum(x["pred_true_tv"] for x in items) / len(items),
        })
    return out


def fmt(value, digits=4):
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def write_markdown(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Prediction Collapse Diagnostics",
        "",
        "This report measures whether a merged model collapses to a small number of predicted classes.",
        "",
        "Metrics:",
        "- `pred_majority_ratio`: fraction of test samples predicted as the most frequent predicted class.",
        "- `pred_effective_classes`: exp(entropy(predicted-label distribution)); lower means stronger collapse.",
        "- `pred_entropy_norm`: predicted-label entropy divided by log(number of classes).",
        "- `pred_true_tv`: total variation distance between predicted-label distribution and true test-label distribution.",
        "",
        "## Group Summary",
        "",
        "| label | method | n | acc | pred_majority_ratio | pred_effective_classes | pred_entropy_norm | pred_true_tv |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_summary(rows):
        lines.append(
            "| {label} | {method} | {n} | {acc} | {pred_majority_ratio} | "
            "{pred_effective_classes} | {pred_entropy_norm} | {pred_true_tv} |".format(
                label=row["label"],
                method=row["method"],
                n=row["n"],
                acc=fmt(row["acc"]),
                pred_majority_ratio=fmt(row["pred_majority_ratio"]),
                pred_effective_classes=fmt(row["pred_effective_classes"]),
                pred_entropy_norm=fmt(row["pred_entropy_norm"]),
                pred_true_tv=fmt(row["pred_true_tv"]),
            )
        )
    lines.extend([
        "",
        "## Case Detail",
        "",
        "| label | method | dataset | model | setting | acc | true_majority | pred_majority | effective_classes | pred_counts |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ])
    for row in sorted(rows, key=lambda x: (x["dataset"], x["model"], x["setting"], x["label"], x["method"])):
        true_majority = f"{row['true_majority_class']}:{row['true_majority_ratio']:.3f}"
        pred_majority = f"{row['pred_majority_class']}:{row['pred_majority_ratio']:.3f}"
        lines.append(
            "| {label} | {method} | {dataset} | {model} | {setting} | {acc} | {true_majority} | "
            "{pred_majority} | {effective} | `{pred_counts}` |".format(
                label=row["label"],
                method=row["method"],
                dataset=row["dataset"],
                model=row["model"],
                setting=row["setting"],
                acc=fmt(row["acc"]),
                true_majority=true_majority,
                pred_majority=pred_majority,
                effective=fmt(row["pred_effective_classes"]),
                pred_counts=row["pred_counts"],
            )
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    args = parse_args()
    label_roots = parse_label_roots(args.root)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    candidates = []
    for label, root in label_roots:
        candidates.extend(iter_merged_dirs(label, root))
    candidates.extend(iter_explicit_dirs(args.merged_dir))

    rows = []
    seen = set()
    for label, merged_dir in candidates:
        key = (label, str(merged_dir.resolve()))
        if key in seen:
            continue
        seen.add(key)
        row = diagnose_one(label, merged_dir, args, device)
        if row is None:
            continue
        rows.append(row)
        print(
            f"{label}:{row['method']} {row['dataset']} {row['model']} {row['setting']} "
            f"acc={row['acc']:.4f} pred_majority={row['pred_majority_ratio']:.3f}"
        )
        if args.limit and len(rows) >= args.limit:
            break

    write_csv(args.dest_csv, rows)
    write_markdown(args.dest_md, rows)
    print(f"wrote {args.dest_csv}")
    print(f"wrote {args.dest_md}")


if __name__ == "__main__":
    main()
