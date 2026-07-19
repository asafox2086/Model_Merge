#!/usr/bin/env python3
"""Create appendix reference figures from the exact benchmark files."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "scripts" / "plot_templates"
if str(TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_DIR))

from style import save_png, setup_style  # noqa: E402


BLOOD_CLASSES = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "Immature granulocyte",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet",
]


def create_blood_reference_sheet() -> None:
    data_path = ROOT / "Med_data" / "bloodmnist_224.npz"
    output_path = ROOT / "figures" / "appendix_blood_classes"
    with np.load(data_path) as data:
        images = data["train_images"]
        labels = np.asarray(data["train_labels"]).reshape(-1)
        if images.shape[0] != labels.shape[0] or set(np.unique(labels)) != set(range(8)):
            raise ValueError("Unexpected BloodMNIST training data")
        sample_indices = [int(np.flatnonzero(labels == class_id)[0]) for class_id in range(8)]
        samples = [images[index] for index in sample_indices]

    setup_style("dashboard")
    plt.rcParams.update({"font.size": 8.0, "axes.titlesize": 9.0})
    figure, axes = plt.subplots(2, 4, figsize=(7.15, 3.75), dpi=300)
    for class_id, (axis, image, name) in enumerate(zip(axes.flat, samples, BLOOD_CLASSES)):
        axis.imshow(image)
        axis.set_title(f"{class_id}: {name}", pad=5, fontweight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#595959")
    figure.tight_layout(pad=0.55, w_pad=0.45, h_pad=0.65)
    save_png(figure, str(output_path), dpi=350)
    plt.close(figure)


if __name__ == "__main__":
    create_blood_reference_sheet()
