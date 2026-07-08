#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import load_npz_splits
from utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Copy reference prototype statistics and attach client-side class prevalence counts."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--data-root", default=str(ROOT / "Med_data"))
    parser.add_argument("--split", choices=["train", "val", "trainval"], default="train")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_labels(data_root: Path, dataset: str, split: str) -> np.ndarray:
    splits = load_npz_splits(str(data_root / f"{dataset}.npz"))
    if split == "trainval":
        return np.concatenate([splits["train"].labels, splits["val"].labels], axis=0)
    return splits[split].labels


def prevalence_counts(labels: np.ndarray, classes: list[int], num_classes: int, num_samples: int) -> list[int]:
    label_counts = np.bincount(labels.reshape(-1).astype(np.int64), minlength=num_classes)
    restricted = np.zeros(num_classes, dtype=np.int64)
    for cls in classes:
        if 0 <= int(cls) < num_classes:
            restricted[int(cls)] = int(label_counts[int(cls)])

    if num_samples > 0 and restricted.sum() > 0:
        scaled = restricted.astype(np.float64) * (float(num_samples) / float(restricted.sum()))
        rounded = np.floor(scaled).astype(np.int64)
        remainder = int(num_samples) - int(rounded.sum())
        if remainder > 0:
            order = np.argsort(-(scaled - rounded))
            rounded[order[:remainder]] += 1
        restricted = rounded
    return [int(x) for x in restricted.tolist()]


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    dest_root = Path(args.dest_root)
    data_root = Path(args.data_root)

    paths = sorted(source_root.glob("small/*/*/clients_*/*/seed_*/prototype_stats.pt"))
    if not paths:
        raise FileNotFoundError(f"No prototype_stats.pt files found under {source_root}")

    label_cache: dict[tuple[str, str], np.ndarray] = {}
    written = 0
    for src in paths:
        rel = src.relative_to(source_root)
        dst = dest_root / rel
        if dst.exists() and not args.overwrite:
            written += 1
            continue

        stats = torch.load(src, map_location="cpu")
        if not isinstance(stats, dict):
            raise ValueError(f"Invalid prototype stats payload: {src}")

        meta_path = Path(str(stats.get("meta_path", "")))
        if not meta_path.exists():
            parts = rel.parts
            dataset, model, clients_dir, beta_dir, seed_dir = parts[1], parts[2], parts[3], parts[4], parts[5]
            meta_path = ROOT / "model_hub" / "small" / dataset / model / clients_dir / beta_dir / seed_dir / "meta.json"
        meta = load_json(meta_path)
        dataset = str(meta["dataset"])
        num_classes = int(meta["num_classes"])

        key = (dataset, args.split)
        if key not in label_cache:
            label_cache[key] = load_labels(data_root, dataset, args.split)
        labels = label_cache[key]

        clients = stats.get("clients", [])
        meta_clients = meta.get("clients", [])
        if len(clients) != len(meta_clients):
            raise ValueError(f"Client count mismatch for {src}: stats={len(clients)} meta={len(meta_clients)}")

        for idx, item in enumerate(clients):
            if "class_feature_counts" not in item:
                if "class_counts" not in item:
                    raise ValueError(f"Missing prototype support counts for client {idx}: {src}")
                item["class_feature_counts"] = item["class_counts"]
            client_meta = meta_clients[idx]
            classes = [int(c) for c in client_meta.get("classes", [])]
            item["class_prevalence_counts"] = prevalence_counts(
                labels,
                classes,
                num_classes,
                int(client_meta.get("num_samples", 0) or 0),
            )

        stats["format"] = "my_merge_client_prototype_stats_v1"
        stats["privacy"] = "client-side aggregate statistics only; no raw image and no per-sample feature is stored"
        stats["prevalence_source"] = f"client_label_counts:{args.split}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        torch.save(stats, tmp)
        shutil.move(str(tmp), str(dst))
        written += 1

    print(f"wrote {written}/{len(paths)} prototype stats to {dest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
