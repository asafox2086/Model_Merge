#!/usr/bin/env python3
"""Compare private-train and public medical image domain fingerprints.

The target fingerprint is computed only from the benchmark/client train split.
Public candidates may come from prepared npz files or from raw public medical
archives cached under ``.cache/public_labeled_probe``. The script reports a
standardized distance; smaller means the public reference looks more like the
client task distribution.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


IMAGE_SIZE = 224
DEFAULT_MAX_SAMPLES = 768
FEATURE_NAMES = [
    "mean",
    "std",
    "q05",
    "q50",
    "q95",
    "chroma",
    "edge",
    "texture",
    "fg_area",
    "evidence_contrast",
    "aspect_fg",
]

DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "chaoshengmnist_224",
    "organcmnist_224",
    "organsmnist_224",
]

KU_BLOOD_MAP = {
    "Basophil": 0,
    "Eosinophil": 1,
    "Erythroblast": 2,
    "Band Neutrophil": 3,
    "Metamyelocyte": 3,
    "Myelocyte": 3,
    "Lymphocyte": 4,
    "Monocyte": 5,
    "Segmented Neutrophil": 6,
    "Giant Platelet": 7,
    "Platelet Cluster": 7,
}

PAD_DERMA_MAP = {
    "ACK": 0,
    "BCC": 1,
    "SEK": 2,
    "MEL": 4,
    "NEV": 5,
}

LOW_RESOURCE_FETAL_MAP = {
    "Fetal abdomen": 0,
    "Fetal brain": 1,
    "Fetal femur": 2,
    "Fetal thorax": 3,
}


def labels1(labels):
    return np.asarray(labels).reshape(-1).astype(np.int64)


def stable_rng():
    return np.random.default_rng(123)


def stratified_indices(labels, max_samples, rng):
    labels = labels1(labels)
    classes = np.unique(labels[labels >= 0])
    if classes.size == 0:
        n = min(int(max_samples), int(labels.size))
        return np.sort(rng.choice(labels.size, n, replace=False)) if labels.size > n else np.arange(labels.size)
    per_class = max(1, int(math.ceil(float(max_samples) / float(classes.size))))
    indices = []
    for cls in classes:
        cls_indices = np.flatnonzero(labels == cls)
        take = min(per_class, cls_indices.size)
        indices.extend(rng.choice(cls_indices, take, replace=False).tolist())
    if len(indices) > max_samples:
        indices = rng.choice(np.asarray(indices), max_samples, replace=False).tolist()
    return np.asarray(sorted(indices), dtype=np.int64)


def image_to_array(image):
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, dict) and image.get("bytes") is not None:
        pil = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, (bytes, bytearray)):
        pil = Image.open(BytesIO(image))
    else:
        raise TypeError(f"Unsupported image payload: {type(image)!r}")
    return np.asarray(pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), dtype=np.uint8)


def fingerprint_features(images, labels=None, max_samples=DEFAULT_MAX_SAMPLES):
    rng = stable_rng()
    if labels is None:
        n = min(max_samples, len(images))
        indices = np.sort(rng.choice(len(images), n, replace=False)) if len(images) > n else np.arange(len(images))
    else:
        indices = stratified_indices(labels, min(max_samples, len(images)), rng)

    arr = images[indices]
    if arr.ndim == 3:
        arr = arr[:, :, :, None]
    arr = arr[:, ::4, ::4, ...].astype(np.float32) / 255.0
    if arr.shape[-1] == 1:
        rgb = np.repeat(arr, 3, axis=3)
    else:
        rgb = arr[..., :3]

    gray = rgb.mean(axis=3)
    flat = gray.reshape(gray.shape[0], -1)
    q04, q05, q50, q95, q96 = np.quantile(flat, [0.04, 0.05, 0.50, 0.95, 0.96], axis=1)
    norm = np.clip((gray - q04[:, None, None]) / (q96[:, None, None] - q04[:, None, None] + 1e-6), 0.0, 1.0)

    dx = np.diff(norm, axis=2, append=norm[:, :, -1:])
    dy = np.diff(norm, axis=1, append=norm[:, -1:, :])
    edge = np.sqrt(dx * dx + dy * dy)
    texture = 0.5 * (
        np.abs(np.diff(norm, axis=2)).mean(axis=(1, 2))
        + np.abs(np.diff(norm, axis=1)).mean(axis=(1, 2))
    )
    contrast = np.abs(norm - norm.mean(axis=(1, 2))[:, None, None])
    evidence = 0.45 * edge + 0.35 * contrast + 0.20 * texture[:, None, None]
    threshold = evidence.mean(axis=(1, 2)) + 0.50 * evidence.std(axis=(1, 2))
    mask = evidence >= threshold[:, None, None]
    fg_area = mask.mean(axis=(1, 2))
    evidence_contrast = (contrast * mask).sum(axis=(1, 2)) / (mask.sum(axis=(1, 2)) + 1e-6)
    chroma = np.sqrt(((rgb - rgb.mean(axis=3, keepdims=True)) ** 2).mean(axis=3)).mean(axis=(1, 2))

    aspect = []
    for item in mask:
        ys, xs = np.where(item)
        if ys.size == 0:
            aspect.append(0.0)
            continue
        height = float(ys.max() - ys.min() + 1) / float(item.shape[0])
        width = float(xs.max() - xs.min() + 1) / float(item.shape[1])
        aspect.append(min(height, width) / (max(height, width) + 1e-6))

    return np.stack(
        [
            gray.mean(axis=(1, 2)),
            gray.std(axis=(1, 2)),
            q05,
            q50,
            q95,
            chroma,
            edge.mean(axis=(1, 2)),
            texture,
            fg_area,
            evidence_contrast,
            np.asarray(aspect),
        ],
        axis=1,
    ).astype(np.float32), indices


def aggregate(features):
    return {
        "mean": features.mean(axis=0),
        "std": features.std(axis=0) + 1e-6,
        "n": int(features.shape[0]),
    }


def fingerprint_distance(target, candidate):
    mean_z = (candidate["mean"] - target["mean"]) / target["std"]
    std_z = np.log((candidate["std"] + 1e-6) / (target["std"] + 1e-6))
    return float(np.sqrt(np.mean(mean_z * mean_z) + 0.25 * np.mean(std_z * std_z)))


def load_npz_split(path, split):
    data = np.load(path, allow_pickle=False)
    images = data[f"{split}_images"]
    labels = labels1(data[f"{split}_labels"])
    return images, labels


def class_count(labels):
    labels = labels1(labels)
    labels = labels[labels >= 0]
    if labels.size == 0:
        return {}
    return {str(i): int(v) for i, v in enumerate(np.bincount(labels, minlength=int(labels.max()) + 1)) if int(v) > 0}


def candidate_row(name, dataset, source, images, labels, target, max_samples):
    features, indices = fingerprint_features(images, labels=labels, max_samples=max_samples)
    fp = aggregate(features)
    labels = labels1(labels)
    nonneg = labels[labels >= 0]
    return {
        "candidate": name,
        "dataset": dataset,
        "source": source,
        "distance": fingerprint_distance(target, fp),
        "sampled": int(indices.size),
        "total": int(labels.size),
        "classes": int(np.unique(nonneg).size) if nonneg.size else 0,
        "class_counts": json.dumps(class_count(labels), ensure_ascii=False),
        "mean": json.dumps([round(float(x), 6) for x in fp["mean"].tolist()]),
    }


def iter_npz_public_candidates(public_roots, datasets):
    for root in public_roots:
        root = Path(root)
        for dataset in datasets:
            path = root / f"{dataset}.npz"
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=False)
            for split in ("train", "val"):
                image_key = f"{split}_images"
                label_key = f"{split}_labels"
                if image_key not in data or label_key not in data or data[label_key].size == 0:
                    continue
                yield f"{root.name}/{dataset}/{split}", dataset, str(path), data[image_key], labels1(data[label_key])


def load_ku_optofil(cache_root, per_class_limit):
    path = Path(cache_root) / "KU_Optofil_PBC" / "dataset.zip"
    if not path.exists():
        return None
    buckets = defaultdict(list)
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith((".jpg", ".jpeg", ".png"))]
        for name in names:
            parts = name.split("/")
            if len(parts) < 4:
                continue
            label = KU_BLOOD_MAP.get(parts[2])
            if label is None or len(buckets[label]) >= per_class_limit:
                continue
            buckets[label].append(image_to_array(archive.read(name)))
            if all(len(buckets[label]) >= per_class_limit for label in range(8)):
                break
    if sorted(buckets) != list(range(8)):
        return None
    images, labels = [], []
    for label in range(8):
        for image in buckets[label]:
            images.append(image)
            labels.append(label)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def load_low_resource_fetal(cache_root):
    root = Path(cache_root) / "low_resource_fetal_ultrasound" / "extracted" / "Zenodo_dataset"
    csv_path = root / "African_planes_database.csv"
    if not csv_path.exists():
        return None
    images, labels = [], []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            label = LOW_RESOURCE_FETAL_MAP.get(row.get("Plane", ""))
            if label is None:
                continue
            path = root / row["Center"] / f"{row['Filename']}.png"
            if not path.exists():
                continue
            images.append(image_to_array(Image.open(path)))
            labels.append(label)
    if not images:
        return None
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def load_pad_ufes(cache_root, max_shards):
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    images, labels = [], []
    skipped = Counter()
    for shard in range(max_shards):
        filename = f"data/train-{shard:05d}-of-00008.parquet"
        try:
            path = Path(
                hf_hub_download(
                    repo_id="OctoMed/PAD_UFES_20",
                    repo_type="dataset",
                    filename=filename,
                    cache_dir=cache_root,
                    local_files_only=True,
                )
            )
        except Exception:
            continue
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=["image", "answer"])
            for image_payload, answer in zip(table["image"].to_pylist(), table["answer"].to_pylist()):
                match = re.search(r"\(([^)]+)\)", str(answer))
                source_label = match.group(1) if match else ""
                label = PAD_DERMA_MAP.get(source_label)
                if label is None:
                    skipped[source_label or "unparsed"] += 1
                    continue
                images.append(image_to_array(image_payload))
                labels.append(label)
    if not images:
        return None
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser("Analyze medical public-domain fingerprints.")
    parser.add_argument("--target-root", default="Med_data")
    parser.add_argument("--public-roots", nargs="*", default=["PublicMedLabeled_data", "PublicMed_data"])
    parser.add_argument("--cache-root", default=".cache/public_labeled_probe")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--per-class-limit", type=int, default=180)
    parser.add_argument("--pad-shards", type=int, default=2)
    parser.add_argument("--output-csv", default="outputs/public_domain_fingerprint_report.csv")
    args = parser.parse_args()

    targets = {}
    for dataset in args.datasets:
        path = Path(args.target_root) / f"{dataset}.npz"
        if not path.exists():
            continue
        images, labels = load_npz_split(path, "train")
        features, indices = fingerprint_features(images, labels=labels, max_samples=args.max_samples)
        targets[dataset] = aggregate(features)
        print(f"[target] {dataset} sampled={indices.size}/{labels.size} mean={np.round(targets[dataset]['mean'], 4).tolist()}")

    rows = []
    for name, dataset, source, images, labels in iter_npz_public_candidates(args.public_roots, args.datasets):
        if dataset not in targets:
            continue
        rows.append(candidate_row(name, dataset, source, images, labels, targets[dataset], args.max_samples))

    ku = load_ku_optofil(args.cache_root, args.per_class_limit)
    if ku is not None and "bloodmnist_224" in targets:
        rows.append(candidate_row("KU_Optofil_PBC/mapped8", "bloodmnist_224", "Zenodo 17333317", ku[0], ku[1], targets["bloodmnist_224"], args.max_samples))

    pad = load_pad_ufes(args.cache_root, args.pad_shards)
    if pad is not None and "dermamnist_224" in targets:
        rows.append(candidate_row("PAD_UFES_20/mapped5", "dermamnist_224", "OctoMed/PAD_UFES_20", pad[0], pad[1], targets["dermamnist_224"], args.max_samples))

    fetal = load_low_resource_fetal(args.cache_root)
    if fetal is not None and "chaoshengmnist_224" in targets:
        rows.append(candidate_row("LowResourceFetalUS/mapped4", "chaoshengmnist_224", "Zenodo 7540448", fetal[0], fetal[1], targets["chaoshengmnist_224"], args.max_samples))

    rows.sort(key=lambda item: (item["dataset"], item["distance"], item["candidate"]))
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "candidate", "distance", "sampled", "total", "classes", "source", "class_counts", "mean"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[fingerprint] wrote {out}")
    for row in rows:
        print(
            f"{row['dataset']}\t{row['candidate']}\tdist={row['distance']:.3f}\t"
            f"classes={row['classes']}\ttotal={row['total']}"
        )


if __name__ == "__main__":
    main()
