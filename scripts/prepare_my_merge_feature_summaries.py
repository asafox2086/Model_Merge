#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURE_NAMES = [
    "intensity_mean",
    "intensity_std",
    "intensity_range",
    "foreground_fraction",
    "foreground_mean",
    "center_x",
    "center_y",
    "gradient_mean",
    "edge_density",
    "gradient_anisotropy",
    "high_frequency_energy",
    "channel_contrast",
    "fourier_low_energy",
    "fourier_mid_energy",
    "fourier_high_energy",
    "fourier_low_ratio",
    "fourier_slope",
]


def feature_summary_path(meta, root):
    model_name = meta.get("model") if meta.get("task_type") == "small" else str(meta.get("clip_model", meta.get("model", ""))).split("/")[-1]
    beta = str(meta.get("beta", "0")).replace(".", "p")
    name = f"{meta.get('task_type')}__{meta.get('dataset')}__{model_name}__c{meta.get('num_clients')}__b{beta}__s{meta.get('seed')}.json"
    return Path(root) / name


def parse_args():
    p = argparse.ArgumentParser("Prepare client-side medical feature summaries for my_merge.")
    p.add_argument("--model-hub-root", default=str(ROOT / "model_hub"))
    p.add_argument("--data-root", default=str(ROOT / "Med_data"))
    p.add_argument("--output-root", default=str(ROOT / "model_hub" / "my_merge_feature_summaries"))
    p.add_argument("--manifest", default="")
    p.add_argument("--task-type", choices=["all", "small", "vlm"], default="small")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--small-models", nargs="*", default=None)
    p.add_argument("--clip-models", nargs="*", default=None)
    p.add_argument("--stats-split", default="train")
    p.add_argument("--stats-batch-size", type=int, default=256)
    p.add_argument("--stats-num-workers", type=int, default=0, help="Kept for CLI compatibility; numpy batches are used.")
    p.add_argument("--device", default="cpu", help="Kept for CLI compatibility; feature extraction is numpy-only.")
    p.add_argument("--max-samples-per-client", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_manifest(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def keep(row, args):
    if args.task_type != "all" and row["task_type"] != args.task_type:
        return False
    if args.datasets and row["dataset"] not in set(args.datasets):
        return False
    if args.small_models and row["task_type"] == "small" and row["model"] not in set(args.small_models):
        return False
    if args.clip_models and row["task_type"] == "vlm" and row["clip_model"] not in set(args.clip_models):
        return False
    return True


def tensor_to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: tensor_to_list(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_to_list(item) for item in value]
    if isinstance(value, tuple):
        return [tensor_to_list(item) for item in value]
    return value


def _single_label(labels):
    if labels.ndim == 1:
        return labels.astype(np.int64)
    if labels.ndim == 2 and labels.shape[1] == 1:
        return labels.reshape(-1).astype(np.int64)
    raise ValueError(f"Only single-label arrays are supported, got shape {labels.shape}")


def _load_npz_split(npz_path, split):
    with np.load(npz_path, allow_pickle=False, mmap_mode="r") as arr:
        image_key = f"{split}_images"
        label_key = f"{split}_labels"
        if image_key not in arr or label_key not in arr:
            raise KeyError(f"Missing {image_key}/{label_key} in {npz_path}")
        images = arr[image_key]
        labels = _single_label(arr[label_key])
    return images, labels


def _as_nchw_float(images):
    x = images.astype(np.float32, copy=False) / 255.0
    if x.ndim == 3:
        x = x[..., None]
    if x.ndim != 4:
        raise ValueError(f"Unexpected image batch shape: {x.shape}")
    return np.moveaxis(x, -1, 1)


def medical_feature_batch(images):
    x = _as_nchw_float(images)
    if x.shape[1] == 1:
        gray = x[:, 0]
        channel_contrast = np.zeros(x.shape[0], dtype=np.float32)
    else:
        gray = x.mean(axis=1)
        channel_contrast = x.std(axis=1).mean(axis=(1, 2))

    flat = gray.reshape(gray.shape[0], -1)
    intensity_mean = flat.mean(axis=1)
    intensity_std = flat.std(axis=1)
    intensity_range = flat.max(axis=1) - flat.min(axis=1)

    threshold = intensity_mean[:, None, None] + 0.25 * intensity_std[:, None, None]
    foreground = (gray > threshold).astype(np.float32)
    foreground_fraction = foreground.mean(axis=(1, 2))
    foreground_sum = np.maximum(foreground.sum(axis=(1, 2)), 1.0)
    foreground_mean = (gray * foreground).sum(axis=(1, 2)) / foreground_sum

    h, w = gray.shape[-2], gray.shape[-1]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(1, h, 1)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, 1, w)
    mass = np.maximum(gray - gray.min(axis=(1, 2), keepdims=True), 0.0) + 1e-4
    mass_sum = np.maximum(mass.sum(axis=(1, 2)), 1e-4)
    center_x = (mass * xx).sum(axis=(1, 2)) / mass_sum
    center_y = (mass * yy).sum(axis=(1, 2)) / mass_sum

    dx = gray[:, :, 1:] - gray[:, :, :-1]
    dy = gray[:, 1:, :] - gray[:, :-1, :]
    dx_abs = np.abs(dx).mean(axis=(1, 2))
    dy_abs = np.abs(dy).mean(axis=(1, 2))
    grad_x = np.pad(np.abs(dx), ((0, 0), (0, 0), (0, 1)))
    grad_y = np.pad(np.abs(dy), ((0, 0), (0, 1), (0, 0)))
    grad = grad_x + grad_y
    gradient_mean = grad.mean(axis=(1, 2))
    edge_threshold = gradient_mean[:, None, None] + grad.std(axis=(1, 2))[:, None, None]
    edge_density = (grad > edge_threshold).mean(axis=(1, 2))
    gradient_anisotropy = np.abs(dx_abs - dy_abs) / (dx_abs + dy_abs + 1e-6)

    if h > 2 and w > 2:
        lap = (
            4.0 * gray[:, 1:-1, 1:-1]
            - gray[:, :-2, 1:-1]
            - gray[:, 2:, 1:-1]
            - gray[:, 1:-1, :-2]
            - gray[:, 1:-1, 2:]
        )
        high_frequency_energy = np.abs(lap).mean(axis=(1, 2))
    else:
        high_frequency_energy = np.zeros_like(intensity_mean)

    step_y = max(1, h // 64)
    step_x = max(1, w // 64)
    gray_fft = gray[:, ::step_y, ::step_x]
    h_fft, w_fft = gray_fft.shape[-2], gray_fft.shape[-1]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray_fft, axes=(-2, -1)), axes=(-2, -1)))
    log_spectrum = np.log1p(spectrum).astype(np.float32)
    yy_freq = np.linspace(-1.0, 1.0, h_fft, dtype=np.float32).reshape(1, h_fft, 1)
    xx_freq = np.linspace(-1.0, 1.0, w_fft, dtype=np.float32).reshape(1, 1, w_fft)
    radius = np.sqrt(xx_freq * xx_freq + yy_freq * yy_freq)
    low_mask = radius <= 0.18
    mid_mask = (radius > 0.18) & (radius <= 0.45)
    high_mask = radius > 0.45
    fourier_low_energy = log_spectrum[:, low_mask[0]].mean(axis=1)
    fourier_mid_energy = log_spectrum[:, mid_mask[0]].mean(axis=1)
    fourier_high_energy = log_spectrum[:, high_mask[0]].mean(axis=1)
    total_energy = fourier_low_energy + fourier_mid_energy + fourier_high_energy + 1e-6
    fourier_low_ratio = fourier_low_energy / total_energy
    fourier_slope = (fourier_low_energy - fourier_high_energy) / (fourier_low_energy + fourier_high_energy + 1e-6)

    features = np.stack(
        [
            intensity_mean,
            intensity_std,
            intensity_range,
            foreground_fraction,
            foreground_mean,
            center_x,
            center_y,
            gradient_mean,
            edge_density,
            gradient_anisotropy,
            high_frequency_energy,
            channel_contrast,
            fourier_low_energy,
            fourier_mid_energy,
            fourier_high_energy,
            fourier_low_ratio,
            fourier_slope,
        ],
        axis=1,
    ).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _aggregate_features(features):
    if features.size == 0:
        width = len(FEATURE_NAMES)
        return np.zeros(width, dtype=np.float32), np.zeros(width, dtype=np.float32)
    return features.mean(axis=0), features.var(axis=0)


def _select_indices(labels, classes, limit, seed):
    mask = np.isin(labels, np.asarray(classes, dtype=np.int64))
    indices = np.nonzero(mask)[0]
    if limit > 0 and indices.size > limit:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=limit, replace=False)
        indices.sort()
    return indices


def _extract_features_for_indices(images, indices, args):
    if indices.size == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    chunks = []
    batch_size = max(1, int(args.stats_batch_size))
    for start in range(0, indices.size, batch_size):
        batch_idx = indices[start : start + batch_size]
        chunks.append(medical_feature_batch(images[batch_idx]))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)


def _load_or_cache_split(meta, args, split_cache):
    cache_key = (str(meta["dataset"]), str(args.stats_split))
    if cache_key in split_cache:
        return split_cache[cache_key]
    images, labels = _load_npz_split(Path(args.data_root) / f"{meta['dataset']}.npz", args.stats_split)
    split_cache[cache_key] = (images, labels)
    return images, labels


def collect_medical_feature_summary(meta, args, split_cache=None):
    if split_cache is None:
        split_cache = {}
    images, labels = _load_or_cache_split(meta, args, split_cache)
    num_clients = int(meta.get("num_clients", len(meta.get("clients", []))))
    num_classes = int(meta.get("num_classes", int(labels.max()) + 1))

    client_feature_mean = []
    client_feature_var = []

    for client_idx, client in enumerate(meta.get("clients", [])[:num_clients]):
        classes = [int(x) for x in client.get("classes", []) if 0 <= int(x) < num_classes]
        indices = _select_indices(labels, classes, int(args.max_samples_per_client), int(args.seed) + 997 * client_idx)
        features = _extract_features_for_indices(images, indices, args)
        mean, var = _aggregate_features(features)
        client_feature_mean.append(mean)
        client_feature_var.append(var)

    client_feature_mean = np.stack(client_feature_mean, axis=0) if client_feature_mean else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    client_feature_var = np.stack(client_feature_var, axis=0) if client_feature_var else np.zeros_like(client_feature_mean)

    return {
        "source": "client_uploaded_medical_feature_summary_simulation",
        "feature_names": FEATURE_NAMES,
        "client_feature_mean": client_feature_mean,
        "client_feature_var": client_feature_var,
        "split": args.stats_split,
        "max_samples_per_client": int(args.max_samples_per_client),
        "privacy_note": (
            "Each client uploads only aggregate intensity, texture, morphology, and frequency feature moments. "
            "No raw image, per-sample feature, logit, activation, class-conditional prototype, or candidate feedback is stored."
        ),
    }


def feature_cache_key(meta, args):
    clients = []
    for client in meta.get("clients", [])[: int(meta.get("num_clients", len(meta.get("clients", []))))]:
        classes = tuple(int(x) for x in client.get("classes", []))
        clients.append(classes)
    return (
        str(meta.get("dataset")),
        int(meta.get("num_classes", 0) or 0),
        int(meta.get("num_clients", len(clients))),
        tuple(clients),
        str(args.stats_split),
        int(args.max_samples_per_client),
        int(args.seed),
    )


def main():
    args = parse_args()
    manifest_path = Path(args.manifest) if args.manifest else Path(args.model_hub_root) / "manifest.csv"
    rows = [row for row in load_manifest(manifest_path) if keep(row, args)]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    written = 0
    split_cache = {}
    summary_cache = {}
    rows = sorted(rows, key=lambda item: (item["dataset"], item.get("task_type", ""), item.get("model", ""), item.get("clip_model", "")))
    for row in rows:
        meta = load_json(Path(args.model_hub_root) / row["meta_path"])
        cache_key = feature_cache_key(meta, args)
        if cache_key not in summary_cache:
            summary_cache[cache_key] = collect_medical_feature_summary(meta, args, split_cache)
        payload = summary_cache[cache_key]
        payload = {key: tensor_to_list(value) for key, value in payload.items()}
        payload.update(
            {
                "task_type": meta.get("task_type"),
                "dataset": meta.get("dataset"),
                "model": meta.get("model"),
                "clip_model": meta.get("clip_model"),
                "num_clients": int(meta.get("num_clients", 0)),
                "beta": meta.get("beta"),
                "seed": int(meta.get("seed", 0)),
            }
        )
        path = feature_summary_path(meta, out_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        written += 1
        print(f"[medical-summary] {written}/{len(rows)} {path}")
    print(f"[medical-summary] done | written={written} | output_root={out_root}")


if __name__ == "__main__":
    main()
