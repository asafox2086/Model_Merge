#!/usr/bin/env python3
"""Build proportional strict-domain control datasets without replacement."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


BASE_COUNTS = {
    "train": [228, 359, 769, 80, 779, 4693, 99],
    "val": [33, 52, 110, 12, 111, 671, 14],
    "test": [66, 103, 220, 23, 223, 1341, 29],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--medical-output-name", default="dermamnist_224_strict_noreplace")
    parser.add_argument("--natural-output-name", default="cifar7_derma_strict_noreplace")
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


def scale_fraction(natural_splits):
    candidates = []
    for split, required_counts in BASE_COUNTS.items():
        available_counts = count_by_class(natural_splits[split]["labels"], len(required_counts))
        for class_index, (available, required) in enumerate(zip(available_counts, required_counts)):
            if required > 0:
                candidates.append((Fraction(int(available), int(required)), split, class_index))
    return min(candidates, key=lambda item: item[0])


def scaled_counts(factor):
    return {
        split: [int(Fraction(count) * factor) for count in counts]
        for split, counts in BASE_COUNTS.items()
    }


def sample_dataset(source_splits, target_counts, seed):
    dataset = {}
    source_indices = {}
    num_classes = len(target_counts["train"])
    for split_index, split in enumerate(("train", "val", "test")):
        images = source_splits[split]["images"]
        labels = source_splits[split]["labels"]
        rng = np.random.default_rng(seed + split_index * 1009)
        selected = []
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
            per_class[str(class_index)] = chosen.astype(int).tolist()
        indices = np.concatenate(selected).astype(np.int64)
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"{split}: duplicate source indices")
        indices = rng.permutation(indices)
        selected_labels = labels[indices]
        observed_counts = count_by_class(selected_labels, num_classes)
        if observed_counts != target_counts[split]:
            raise RuntimeError(f"{split}: sampled counts {observed_counts} != target {target_counts[split]}")
        dataset[f"{split}_images"] = images[indices]
        dataset[f"{split}_labels"] = selected_labels.reshape(-1, 1)
        source_indices[split] = per_class
    return dataset, source_indices


def save_dataset(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def main():
    args = parse_args()
    medical_source = args.data_root / "dermamnist_224.npz"
    natural_source = args.data_root / "cifar10_32.npz"
    if not medical_source.exists() or not natural_source.exists():
        raise FileNotFoundError("Both dermamnist_224.npz and cifar10_32.npz must exist under --data-root")

    medical_splits = load_splits(medical_source)
    natural_splits = load_splits(natural_source)
    factor, limiting_split, limiting_class = scale_fraction(natural_splits)
    target_counts = scaled_counts(factor)
    medical_dataset, medical_indices = sample_dataset(medical_splits, target_counts, args.seed)
    natural_dataset, natural_indices = sample_dataset(natural_splits, target_counts, args.seed + 100_003)

    medical_output = args.data_root / f"{args.medical_output_name}.npz"
    natural_output = args.data_root / f"{args.natural_output_name}.npz"
    save_dataset(medical_output, medical_dataset)
    save_dataset(natural_output, natural_dataset)

    manifest = {
        "sampling": "without_replacement",
        "seed": args.seed,
        "scale_fraction": f"{factor.numerator}/{factor.denominator}",
        "scale_float": float(factor),
        "rounding": "floor",
        "limiting_source": {"split": limiting_split, "class": limiting_class},
        "base_counts": BASE_COUNTS,
        "target_counts": target_counts,
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
                    split: count_by_class(payload["labels"], 7)
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
