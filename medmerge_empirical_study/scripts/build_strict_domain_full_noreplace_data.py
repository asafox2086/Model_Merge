#!/usr/bin/env python3
"""Build a full-size DermaMNIST/SVHN strict control without replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGET_COUNTS = {
    "train": [228, 359, 769, 80, 779, 4693, 99],
    "val": [33, 52, 110, 12, 111, 671, 14],
    "test": [66, 103, 220, 23, 223, 1341, 29],
}
SVHN_SOURCE_CLASSES = [0, 2, 3, 4, 5, 1, 6]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--medical-output-name", default="dermamnist_224_strict_full_noreplace")
    parser.add_argument("--natural-output-name", default="svhn7_derma_strict_full_noreplace")
    return parser.parse_args()


def load_splits(path):
    archive = np.load(path, allow_pickle=False)
    return {
        split: {
            "images": archive[f"{split}_images"],
            "labels": archive[f"{split}_labels"].reshape(-1).astype(np.int64),
        }
        for split in ("train", "val", "test")
    }


def count_by_class(labels, num_classes):
    return np.bincount(labels, minlength=num_classes).astype(int).tolist()


def sample_dataset(source_splits, target_to_source, seed):
    dataset = {}
    source_indices = {}
    for split_index, split in enumerate(("train", "val", "test")):
        images = source_splits[split]["images"]
        labels = source_splits[split]["labels"]
        rng = np.random.default_rng(seed + split_index * 1009)
        selected = []
        remapped_labels = []
        per_class = {}
        for target_class, requested in enumerate(TARGET_COUNTS[split]):
            source_class = target_to_source[target_class]
            candidates = np.flatnonzero(labels == source_class)
            if requested > len(candidates):
                raise ValueError(
                    f"{split} target class {target_class}: requested {requested}, "
                    f"available {len(candidates)} from source class {source_class}"
                )
            chosen = rng.choice(candidates, size=requested, replace=False)
            if len(np.unique(chosen)) != len(chosen):
                raise RuntimeError(f"{split} target class {target_class}: duplicate source indices")
            selected.append(chosen)
            remapped_labels.append(np.full(requested, target_class, dtype=np.int64))
            per_class[str(target_class)] = {
                "source_class": int(source_class),
                "indices": chosen.astype(int).tolist(),
            }
        indices = np.concatenate(selected).astype(np.int64)
        output_labels = np.concatenate(remapped_labels)
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"{split}: duplicate source indices")
        permutation = rng.permutation(len(indices))
        indices = indices[permutation]
        output_labels = output_labels[permutation]
        observed_counts = count_by_class(output_labels, len(target_to_source))
        if observed_counts != TARGET_COUNTS[split]:
            raise RuntimeError(f"{split}: sampled counts {observed_counts} != target {TARGET_COUNTS[split]}")
        dataset[f"{split}_images"] = images[indices]
        dataset[f"{split}_labels"] = output_labels.reshape(-1, 1)
        source_indices[split] = per_class
    return dataset, source_indices


def save_dataset(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def main():
    args = parse_args()
    medical_source = args.data_root / "dermamnist_224.npz"
    natural_source = args.data_root / "svhn_32.npz"
    if not medical_source.exists() or not natural_source.exists():
        raise FileNotFoundError("Both dermamnist_224.npz and svhn_32.npz must exist under --data-root")

    medical_splits = load_splits(medical_source)
    natural_splits = load_splits(natural_source)
    medical_dataset, medical_indices = sample_dataset(medical_splits, list(range(7)), args.seed)
    natural_dataset, natural_indices = sample_dataset(natural_splits, SVHN_SOURCE_CLASSES, args.seed + 100_003)

    medical_output = args.data_root / f"{args.medical_output_name}.npz"
    natural_output = args.data_root / f"{args.natural_output_name}.npz"
    save_dataset(medical_output, medical_dataset)
    save_dataset(natural_output, natural_dataset)

    manifest = {
        "sampling": "without_replacement",
        "seed": args.seed,
        "scale_fraction": "1/1",
        "rounding": "none",
        "target_counts": TARGET_COUNTS,
        "natural_target_to_source_classes": {
            str(target): source for target, source in enumerate(SVHN_SOURCE_CLASSES)
        },
        "datasets": {
            "medical": {
                "source": medical_source.name,
                "output": medical_output.name,
                "source_counts": {
                    split: count_by_class(payload["labels"], 7)
                    for split, payload in medical_splits.items()
                },
                "source_indices": medical_indices,
            },
            "natural": {
                "source": natural_source.name,
                "output": natural_output.name,
                "source_counts": {
                    split: count_by_class(payload["labels"], 10)
                    for split, payload in natural_splits.items()
                },
                "source_indices": natural_indices,
            },
        },
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {medical_output}")
    print(f"Wrote {natural_output}")
    print(f"Wrote {args.manifest_path}")


if __name__ == "__main__":
    main()
