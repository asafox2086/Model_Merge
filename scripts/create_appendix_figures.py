#!/usr/bin/env python3
"""Create appendix reference figures from the exact benchmark files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "blood": ("bloodmnist_224.npz", 8),
    "derma": ("dermamnist_224.npz", 7),
    "organ_c": ("organcmnist_224.npz", 11),
    "organ_s": ("organsmnist_224.npz", 11),
    "ultrasound": ("chaoshengmnist_224.npz", 8),
}


def create_class_samples() -> None:
    for dataset, (filename, num_classes) in DATASETS.items():
        with np.load(ROOT / "Med_data" / filename) as data:
            images = data["train_images"]
            labels = np.asarray(data["train_labels"]).reshape(-1)
            if images.shape[0] != labels.shape[0] or set(np.unique(labels)) != set(range(num_classes)):
                raise ValueError(f"Unexpected {dataset} training data")
            sample_indices = [int(np.flatnonzero(labels == class_id)[0]) for class_id in range(num_classes)]
            samples = [images[index] for index in sample_indices]
        for class_id, image in enumerate(samples):
            kwargs = {"cmap": "gray"} if image.ndim == 2 else {}
            plt.imsave(ROOT / "figures" / f"appendix_sample_{dataset}_{class_id}.png", image, **kwargs)


if __name__ == "__main__":
    create_class_samples()
