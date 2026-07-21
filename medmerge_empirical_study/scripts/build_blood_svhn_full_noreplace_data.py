#!/usr/bin/env python3
"""Build a full BloodMNIST/SVHN eight-class control without replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--natural-output-name", default="svhn8_blood_strict_full_noreplace")
    return parser.parse_args()


def load_splits(path, include_images=True):
    archive = np.load(path, allow_pickle=False)
    return {
        split: {
            "images": archive[f"{split}_images"] if include_images else None,
            "labels": archive[f"{split}_labels"].reshape(-1).astype(np.int64),
        }
        for split in ("train", "val", "test")
    }


def count_by_class(labels, num_classes):
    return np.bincount(labels, minlength=num_classes).astype(int).tolist()


def all_source_indices(splits, num_classes):
    indices = {}
    for split, payload in splits.items():
        indices[split] = {
            str(class_index): {
                "source_class": class_index,
                "indices": np.flatnonzero(payload["labels"] == class_index).astype(int).tolist(),
            }
            for class_index in range(num_classes)
        }
    return indices


def sample_svhn(splits, target_counts, seed):
    dataset = {}
    source_indices = {}
    num_classes = len(target_counts["train"])
    for split_index, split in enumerate(("train", "val", "test")):
        images = splits[split]["images"]
        labels = splits[split]["labels"]
        rng = np.random.default_rng(seed + split_index * 1009)
        selected = []
        output_labels = []
        per_class = {}
        for class_index, requested in enumerate(target_counts[split]):
            candidates = np.flatnonzero(labels == class_index)
            if requested > len(candidates):
                raise ValueError(
                    f"{split} class {class_index}: requested {requested}, available {len(candidates)}"
                )
            chosen = rng.choice(candidates, size=requested, replace=False)
            if len(np.unique(chosen)) != len(chosen):
                raise RuntimeError(f"{split} class {class_index}: duplicate source indices")
            selected.append(chosen)
            output_labels.append(np.full(requested, class_index, dtype=np.int64))
            per_class[str(class_index)] = {
                "source_class": class_index,
                "indices": chosen.astype(int).tolist(),
            }
        indices = np.concatenate(selected).astype(np.int64)
        remapped_labels = np.concatenate(output_labels)
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"{split}: duplicate source indices")
        permutation = rng.permutation(len(indices))
        indices = indices[permutation]
        remapped_labels = remapped_labels[permutation]
        observed = count_by_class(remapped_labels, num_classes)
        if observed != target_counts[split]:
            raise RuntimeError(f"{split}: sampled counts {observed} != target {target_counts[split]}")
        dataset[f"{split}_images"] = images[indices]
        dataset[f"{split}_labels"] = remapped_labels.reshape(-1, 1)
        source_indices[split] = per_class
    return dataset, source_indices


def main():
    args = parse_args()
    blood_source = args.data_root / "bloodmnist_224.npz"
    svhn_source = args.data_root / "svhn_32.npz"
    if not blood_source.exists() or not svhn_source.exists():
        raise FileNotFoundError("Both bloodmnist_224.npz and svhn_32.npz must exist under --data-root")

    blood_splits = load_splits(blood_source, include_images=False)
    svhn_splits = load_splits(svhn_source)
    target_counts = {split: count_by_class(payload["labels"], 8) for split, payload in blood_splits.items()}
    if any(len(counts) != 8 for counts in target_counts.values()):
        raise ValueError(f"Expected eight BloodMNIST classes, got {target_counts}")
    natural_dataset, natural_indices = sample_svhn(svhn_splits, target_counts, args.seed + 100_003)
    natural_output = args.data_root / f"{args.natural_output_name}.npz"
    np.savez_compressed(natural_output, **natural_dataset)

    manifest = {
        "sampling": "without_replacement",
        "seed": args.seed,
        "scale_fraction": "1/1",
        "rounding": "none",
        "target_counts": target_counts,
        "natural_target_to_source_classes": {str(class_index): class_index for class_index in range(8)},
        "datasets": {
            "medical": {
                "source": blood_source.name,
                "output": blood_source.name,
                "uses_source_archive": True,
                "source_counts": target_counts,
                "source_indices": all_source_indices(blood_splits, 8),
            },
            "natural": {
                "source": svhn_source.name,
                "output": natural_output.name,
                "source_counts": {
                    split: count_by_class(payload["labels"], 10)
                    for split, payload in svhn_splits.items()
                },
                "source_indices": natural_indices,
            },
        },
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {natural_output}")
    print(f"Wrote {args.manifest_path}")


if __name__ == "__main__":
    main()
