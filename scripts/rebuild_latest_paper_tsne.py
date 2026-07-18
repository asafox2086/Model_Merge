#!/usr/bin/env python3
"""Rebuild the paper t-SNE coordinates with the formal LAMP checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FORMAL_CHECKPOINT = (
    ROOT
    / "outputs"
    / "lamp_merge_internal_ablation_full_20260718_formal_gamma055_s18p75_tau2p5_lambda4p25"
    / "full"
    / "merged"
    / "small"
    / "bloodmnist_224"
    / "resnet"
    / "clients_3"
    / "beta_0p01"
    / "seed_42"
    / "lamp_merge"
    / "merged.pt"
)


def read_annotated_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first = next(reader)
        header = next(reader) if first and first[0] == "说明" else first
        return [dict(zip(header, row)) for row in reader if row]


def write_annotated_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "说明",
                "本表对应 Blood/ResNet/K=3/beta=0.01 的 output-probability t-SNE；LAMP-Merge 概率由 gamma=0.55、s=18.75、tau=2.5、lambda=4.25 正式 checkpoint 重新推理，三个通用基线保持论文原实验概率向量，并在共享 t-SNE 平面重新拟合。",
            ]
        )
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([row.get(field, "") for field in fieldnames])


def find_manifest_meta() -> Path:
    manifest = ROOT / "model_hub" / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("dataset") == "bloodmnist_224"
                and row.get("model") == "resnet"
                and row.get("num_clients") == "3"
                and row.get("beta") == "0.01"
                and row.get("seed") == "42"
            ):
                return ROOT / "model_hub" / row["meta_path"]
    raise FileNotFoundError("Blood/ResNet/K=3/beta=0.01 metadata not found")


def infer_formal_lamp_probabilities(checkpoint_path: Path) -> tuple[object, object]:
    import numpy as np
    import torch
    from utils import load_checkpoint, load_json
    from utils.runtime import build_runtime
    from utils.state_dict import extract_state_dict

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    meta = load_json(find_manifest_meta())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runtime = build_runtime(
        meta=meta,
        data_root=str(ROOT / "Med_data"),
        split="test",
        batch_size=256,
        num_workers=0,
        device=device,
    )
    model = runtime["model"]
    loader = runtime["loader"]
    forward_fn = runtime["forward_fn"]
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.to(device)
    model.eval()

    probabilities = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            logits = forward_fn(model, images.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits, dim=1).detach().cpu().float().numpy())
            labels.append(targets.detach().cpu().numpy().reshape(-1))
    probability_array = np.concatenate(probabilities, axis=0)
    label_array = np.concatenate(labels, axis=0).astype(int)
    if probability_array.shape != (3421, 8) or label_array.shape != (3421,):
        raise ValueError(
            f"Unexpected formal inference shapes: probabilities={probability_array.shape}, labels={label_array.shape}"
        )
    if not np.isfinite(probability_array).all():
        raise ValueError("Formal LAMP probabilities contain non-finite values")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probability_array, label_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-csv", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, default=ROOT / "论文实验数据")
    parser.add_argument("--formal-checkpoint", type=Path, default=FORMAL_CHECKPOINT)
    args = parser.parse_args()

    import numpy as np
    from sklearn.manifold import TSNE

    historical_rows = read_annotated_csv(args.historical_csv.resolve())
    historical_by_method: dict[str, list[dict[str, str]]] = {}
    for method in ["LAMP-Merge", "avg", "TIES-Merging", "DARE-Linear"]:
        rows = [row for row in historical_rows if row["method"] == method]
        rows.sort(key=lambda row: int(row["sample_index"]))
        if len(rows) != 3421:
            raise ValueError(f"Historical method {method} has {len(rows)} rows")
        historical_by_method[method] = rows

    lamp_probabilities, lamp_labels = infer_formal_lamp_probabilities(args.formal_checkpoint.resolve())
    historical_labels = np.asarray(
        [int(row["true_label"]) for row in historical_by_method["LAMP-Merge"]], dtype=int
    )
    if not np.array_equal(lamp_labels, historical_labels):
        raise ValueError("Formal loader sample order does not match archived paper probabilities")

    method_specs = [
        ("LAMP-Merge", "LAMP-Merge"),
        ("avg", "Weight Averaging"),
        ("TIES-Merging", "TIES-Merging"),
        ("DARE-Linear", "DARE-Linear"),
    ]
    probabilities_by_method = []
    for source_method, _ in method_specs:
        if source_method == "LAMP-Merge":
            probabilities = lamp_probabilities
        else:
            probabilities = np.asarray(
                [
                    [float(row[f"prob_{class_index}"]) for class_index in range(8)]
                    for row in historical_by_method[source_method]
                ],
                dtype=float,
            )
        if probabilities.shape != (3421, 8) or not np.isfinite(probabilities).all():
            raise ValueError(f"Invalid probability matrix for {source_method}: {probabilities.shape}")
        probabilities_by_method.append(probabilities)

    all_probabilities = np.vstack(probabilities_by_method)
    coordinates = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=40,
        random_state=1701,
        method="barnes_hut",
        n_jobs=-1,
    ).fit_transform(all_probabilities)
    if coordinates.shape != (13684, 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"Invalid t-SNE coordinate matrix: {coordinates.shape}")

    output_rows: list[dict[str, object]] = []
    offset = 0
    for probabilities, (_, paper_method) in zip(probabilities_by_method, method_specs):
        for sample_index in range(3421):
            row: dict[str, object] = {
                "Dataset": "Blood",
                "Backbone": "ResNet",
                "K": 3,
                "Beta": 0.01,
                "Method": paper_method,
                "Sample index": sample_index,
                "True label": int(lamp_labels[sample_index]),
                "t-SNE x": f"{coordinates[offset + sample_index, 0]:.6f}",
                "t-SNE y": f"{coordinates[offset + sample_index, 1]:.6f}",
            }
            for class_index in range(8):
                row[f"Probability class {class_index}"] = f"{probabilities[sample_index, class_index]:.8f}"
            output_rows.append(row)
        offset += 3421

    fieldnames = [
        "Dataset",
        "Backbone",
        "K",
        "Beta",
        "Method",
        "Sample index",
        "True label",
        "t-SNE x",
        "t-SNE y",
    ] + [f"Probability class {class_index}" for class_index in range(8)]
    write_annotated_csv(args.csv_dir.resolve() / "tSNE输出概率坐标.csv", output_rows, fieldnames)
    accuracy = float(np.mean(np.argmax(lamp_probabilities, axis=1) == lamp_labels))
    print(f"Formal LAMP t-SNE source accuracy: {accuracy:.8f}")
    print(f"Wrote {len(output_rows)} t-SNE rows")


if __name__ == "__main__":
    main()
