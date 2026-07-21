#!/usr/bin/env python3
"""Build a no-replacement seven-class CIFAR-100 control matched to DermaMNIST."""

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

CIFAR100_GROUPS = {
    0: [10],
    1: [47, 52],
    2: [0, 53, 57],
    3: [37],
    4: [8, 13, 48],
    5: [1, 3, 4, 6, 7, 14, 15, 18, 19, 21, 24, 26, 27, 30],
    6: [60],
}

CIFAR100_GROUP_NAMES = {
    0: "bowl",
    1: "trees",
    2: "fruit",
    3: "house",
    4: "vehicles",
    5: "animals",
    6: "plain",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-name", default="cifar100_semantic7_derma_strict")
    parser.add_argument("--seed", type=int, default=42)
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


def count_target_labels(labels):
    return np.bincount(labels, minlength=len(CIFAR100_GROUPS)).astype(int).tolist()


def validate_groups():
    source_classes = [source_class for group in CIFAR100_GROUPS.values() for source_class in group]
    if len(source_classes) != len(set(source_classes)):
        raise ValueError("CIFAR-100 source groups must be disjoint")
    if sorted(CIFAR100_GROUPS) != list(range(len(CIFAR100_GROUPS))):
        raise ValueError("Target labels must be consecutive from zero")


def build_dataset(source_splits, seed):
    target_dataset = {}
    source_indices = {}
    for split_index, split in enumerate(("train", "val", "test")):
        source = source_splits[split]
        rng = np.random.default_rng(seed + split_index * 1009)
        selected = []
        class_indices = {}
        for target_class, requested_count in enumerate(TARGET_COUNTS[split]):
            source_classes = np.asarray(CIFAR100_GROUPS[target_class], dtype=np.int64)
            candidates = np.flatnonzero(np.isin(source["labels"], source_classes))
            if requested_count > len(candidates):
                raise ValueError(
                    f"{split} target class {target_class}: requested {requested_count}, available {len(candidates)}"
                )
            chosen = rng.choice(candidates, size=requested_count, replace=False)
            if len(np.unique(chosen)) != len(chosen):
                raise RuntimeError(f"{split} target class {target_class}: duplicate source indices")
            selected.append(chosen)
            class_indices[str(target_class)] = chosen.astype(int).tolist()
        indices = np.concatenate(selected).astype(np.int64)
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"{split}: duplicate source indices")
        indices = rng.permutation(indices)
        target_labels = np.empty(len(indices), dtype=np.int64)
        for target_class, source_classes in CIFAR100_GROUPS.items():
            target_labels[np.isin(source["labels"][indices], np.asarray(source_classes, dtype=np.int64))] = target_class
        observed_counts = count_target_labels(target_labels)
        if observed_counts != TARGET_COUNTS[split]:
            raise RuntimeError(f"{split}: target counts {observed_counts} != {TARGET_COUNTS[split]}")
        target_dataset[f"{split}_images"] = source["images"][indices]
        target_dataset[f"{split}_labels"] = target_labels.reshape(-1, 1)
        source_indices[split] = class_indices
    return target_dataset, source_indices


def main():
    args = parse_args()
    validate_groups()
    source_path = args.data_root / "cifar100_32.npz"
    medical_path = args.data_root / "dermamnist_224.npz"
    if not source_path.exists() or not medical_path.exists():
        raise FileNotFoundError("cifar100_32.npz and dermamnist_224.npz must exist under --data-root")
    source_splits = load_splits(source_path)
    medical_splits = load_splits(medical_path)
    medical_counts = {
        split: np.bincount(payload["labels"], minlength=len(CIFAR100_GROUPS)).astype(int).tolist()
        for split, payload in medical_splits.items()
    }
    if medical_counts != TARGET_COUNTS:
        raise ValueError(f"DermaMNIST counts changed: {medical_counts}")
    dataset, source_indices = build_dataset(source_splits, args.seed)
    output_path = args.data_root / f"{args.output_name}.npz"
    np.savez_compressed(output_path, **dataset)
    manifest = {
        "sampling": "without_replacement",
        "seed": args.seed,
        "scale_fraction": "1/1",
        "medical_source": medical_path.name,
        "natural_source": source_path.name,
        "natural_output": output_path.name,
        "target_counts": TARGET_COUNTS,
        "natural_target_groups": {
            str(target_class): {
                "name": CIFAR100_GROUP_NAMES[target_class],
                "source_fine_labels": source_classes,
            }
            for target_class, source_classes in CIFAR100_GROUPS.items()
        },
        "source_indices": source_indices,
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {args.manifest_path}")


if __name__ == "__main__":
    main()
