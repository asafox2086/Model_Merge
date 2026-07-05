#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import NpzTensorDataset, load_npz_splits
from model import build_model
from utils import load_json
from utils.hub import beta_to_dirname
from utils.runtime import build_reference_bundle


def parse_args():
    p = argparse.ArgumentParser("Export client-side class prototype statistics for my_merge")
    p.add_argument("--model-hub-root", type=str, default=str(ROOT / "model_hub"))
    p.add_argument("--data-root", type=str, default=str(ROOT / "Med_data"))
    p.add_argument("--output-root", type=str, default=str(ROOT / "ClientPrototypeStats"))
    p.add_argument("--task-type", choices=["small"], default="small")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--small-models", nargs="*", default=None)
    p.add_argument("--num-clients", nargs="*", type=int, default=None)
    p.add_argument("--betas", nargs="*", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "val", "trainval"], default="train")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples-per-client", type=int, default=4096)
    p.add_argument("--max-samples-per-class", type=int, default=1024)
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


def iter_meta_paths(args):
    root = Path(args.model_hub_root) / "small"
    datasets = args.datasets or sorted(path.name for path in root.iterdir() if path.is_dir())
    for dataset in datasets:
        ds_dir = root / dataset
        if not ds_dir.exists():
            continue
        models = args.small_models or sorted(path.name for path in ds_dir.iterdir() if path.is_dir())
        for model in models:
            model_dir = ds_dir / model
            if not model_dir.exists():
                continue
            client_dirs = [f"clients_{n}" for n in args.num_clients] if args.num_clients else sorted(path.name for path in model_dir.iterdir() if path.is_dir())
            for client_dir in client_dirs:
                cdir = model_dir / client_dir
                if not cdir.exists():
                    continue
                beta_dirs = [beta_to_dirname(beta) for beta in args.betas] if args.betas else sorted(path.name for path in cdir.iterdir() if path.is_dir())
                for beta_dir in beta_dirs:
                    meta_path = cdir / beta_dir / f"seed_{int(args.seed)}" / "meta.json"
                    if meta_path.exists():
                        yield meta_path


def load_split_arrays(data_root, dataset, split):
    splits = load_npz_splits(str(Path(data_root) / f"{dataset}.npz"))
    if split == "trainval":
        images = np.concatenate([splits["train"].images, splits["val"].images], axis=0)
        labels = np.concatenate([splits["train"].labels, splits["val"].labels], axis=0)
        return images, labels
    return splits[split].images, splits[split].labels


def output_path(output_root, meta):
    return (
        Path(output_root)
        / str(meta["task_type"])
        / str(meta["dataset"])
        / str(meta["model"])
        / f"clients_{int(meta['num_clients'])}"
        / beta_to_dirname(meta["beta"])
        / f"seed_{int(meta['seed'])}"
        / "prototype_stats.pt"
    )


def select_client_indices(labels, classes, client_idx, max_samples, max_per_class):
    rng = np.random.default_rng(1009 + int(client_idx))
    chosen = []
    for cls in classes:
        cls_indices = np.flatnonzero(labels == int(cls))
        if max_per_class > 0 and cls_indices.size > max_per_class:
            cls_indices = rng.choice(cls_indices, size=max_per_class, replace=False)
        chosen.append(cls_indices)
    if not chosen:
        return np.array([], dtype=np.int64)
    indices = np.concatenate(chosen).astype(np.int64)
    if max_samples > 0 and indices.size > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False).astype(np.int64)
    rng.shuffle(indices)
    return indices


def client_prevalence_counts(labels, classes, num_classes, num_samples=None):
    counts = np.bincount(np.asarray(labels).reshape(-1).astype(np.int64), minlength=num_classes)
    out = np.zeros(num_classes, dtype=np.int64)
    for cls in classes:
        cls = int(cls)
        if 0 <= cls < num_classes:
            out[cls] = int(counts[cls])
    if num_samples is not None and out.sum() > 0:
        scaled = out.astype(np.float64) * (float(num_samples) / float(out.sum()))
        rounded = np.floor(scaled).astype(np.int64)
        remainder = int(num_samples) - int(rounded.sum())
        if remainder > 0:
            order = np.argsort(-(scaled - rounded))
            rounded[order[:remainder]] += 1
        out = rounded
    return out


def extract_features(model, x):
    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        z = model.forward_features(x)
        try:
            feat = model.forward_head(z, pre_logits=True)
        except TypeError:
            feat = model.forward_head(z)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return feat

    captured = {}
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if classifier is None:
        raise ValueError("Model does not expose forward_features/forward_head or get_classifier().")

    def hook(_module, inputs):
        captured["features"] = inputs[0]

    handle = classifier.register_forward_pre_hook(hook)
    try:
        _ = model(x)
        feat = captured["features"]
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return feat
    finally:
        handle.remove()


def export_one(meta_path, args):
    meta = load_json(meta_path)
    if meta.get("task_type") != "small":
        return None
    out_path = output_path(args.output_root, meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images, labels = load_split_arrays(args.data_root, meta["dataset"], args.split)
    transform = build_transform(meta, images)
    dataset = NpzTensorDataset(images, labels, transform=transform)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    clients_payload = []

    reference_model, _, _, _ = build_model(
        name=meta["model"],
        num_classes=int(meta["num_classes"]),
        in_channels=int(meta.get("in_channels", 3)),
        pretrained=False,
    )
    reference_state, _ = build_reference_bundle(meta, device="cpu")
    reference_model.load_state_dict(reference_state, strict=True)
    reference_model.to(device)
    reference_model.eval()

    for client_idx, client in enumerate(meta.get("clients", [])):
        classes = [int(c) for c in client.get("classes", [])]
        prevalence_counts = client_prevalence_counts(
            labels,
            classes,
            int(meta["num_classes"]),
            num_samples=int(client.get("num_samples", 0) or 0),
        )
        indices = select_client_indices(
            labels,
            classes,
            client_idx,
            max_samples=int(args.max_samples_per_client),
            max_per_class=int(args.max_samples_per_class),
        )
        loader = DataLoader(
            Subset(dataset, indices.tolist()),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=str(device).startswith("cuda"),
        )
        num_classes = int(meta["num_classes"])
        sums = None
        counts = torch.zeros(num_classes, dtype=torch.float32)

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                feat = extract_features(reference_model, x).detach().float().cpu()
                y_cpu = y.detach().cpu()
                if sums is None:
                    sums = torch.zeros(num_classes, feat.shape[1], dtype=torch.float32)
                for cls in y_cpu.unique().tolist():
                    cls = int(cls)
                    mask = y_cpu == cls
                    sums[cls] += feat[mask].sum(dim=0)
                    counts[cls] += float(mask.sum().item())

        if sums is None:
            raise ValueError(f"No samples selected for client {client_idx}: {meta_path}")
        means = sums / counts.clamp_min(1.0).view(-1, 1)
        clients_payload.append(
            {
                "client_id": int(client.get("client_id", client_idx)),
                "classes": classes,
                "num_selected_samples": int(counts.sum().item()),
                "class_counts": counts.tolist(),
                "class_feature_counts": counts.tolist(),
                "class_prevalence_counts": prevalence_counts.tolist(),
                "class_feature_mean": means.tolist(),
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "format": "my_merge_client_prototype_stats_v1",
        "privacy": "client-side aggregate statistics only; no raw image and no per-sample feature is stored",
        "feature_space": "reference_model",
        "source_split": args.split,
        "meta_path": str(meta_path),
        "task_type": meta["task_type"],
        "dataset": meta["dataset"],
        "model": meta["model"],
        "num_clients": int(meta["num_clients"]),
        "beta": float(meta["beta"]),
        "seed": int(meta["seed"]),
        "num_classes": int(meta["num_classes"]),
        "clients": clients_payload,
    }
    torch.save(payload, out_path)
    return out_path


def main():
    args = parse_args()
    paths = list(iter_meta_paths(args))
    print(f"prototype export start | tasks={len(paths)} | output_root={args.output_root}")
    done = 0
    for idx, meta_path in enumerate(paths, start=1):
        out_path = export_one(meta_path, args)
        if out_path is not None:
            done += 1
            print(f"done ({idx}/{len(paths)}) {meta_path} -> {out_path}")
    print(f"prototype export done | written={done}")


if __name__ == "__main__":
    main()
