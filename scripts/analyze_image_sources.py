#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8


DEFAULT_DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
]


def parse_args():
    p = argparse.ArgumentParser("Analyze source image files used by MedMNISTMerge.")
    p.add_argument("--data-root", default=str(ROOT / "Med_data"))
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--sample-per-split", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(ROOT / "My_merge_ret" / "reports" / "source_image_analysis"))
    return p.parse_args()


def safe_mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def safe_std(values):
    return float(np.std(values)) if len(values) else float("nan")


def q(values, quantile):
    return float(np.quantile(values, quantile)) if len(values) else float("nan")


def labels_1d(labels):
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)
    return labels.astype(np.int64)


def to_tensor(images):
    x = torch.from_numpy(images).float() / 255.0
    if x.ndim == 3:
        x = x.unsqueeze(-1)
    x = x.permute(0, 3, 1, 2).contiguous()
    return x


def safe_quantile(values, quantile):
    return torch.quantile(values.flatten(1), q=quantile, dim=1, keepdim=True).view(-1, 1, 1)


def robust_rescale01(gray, low_q=0.04, high_q=0.96):
    low = safe_quantile(gray, low_q)
    high = safe_quantile(gray, high_q)
    return torch.clamp((gray - low) / (high - low + EPS), min=0.0, max=1.0)


def avg_pool(gray, kernel):
    return F.avg_pool2d(gray.unsqueeze(1), kernel_size=kernel, stride=1, padding=kernel // 2).squeeze(1)


def local_variance(gray, kernel=9):
    mean = avg_pool(gray, kernel)
    mean_sq = avg_pool(gray.square(), kernel)
    return (mean_sq - mean.square()).clamp_min(0.0)


def sobel_edges(gray):
    sx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
    ).view(1, 1, 3, 3)
    sy = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=gray.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray.unsqueeze(1), sx, padding=1)
    gy = F.conv2d(gray.unsqueeze(1), sy, padding=1)
    return torch.sqrt(gx.square() + gy.square() + EPS).squeeze(1)


def speckle_reduced_gray(gray):
    mean3 = avg_pool(gray, 3)
    mean5 = avg_pool(gray, 5)
    return torch.clamp(0.55 * mean3 + 0.45 * mean5, min=0.0, max=1.0)


def spatial_eccentricity(mask):
    side_h, side_w = mask.shape[-2], mask.shape[-1]
    xs = torch.linspace(-1.0, 1.0, side_w, device=mask.device)
    ys = torch.linspace(-1.0, 1.0, side_h, device=mask.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    weight = mask.float()
    mass = weight.sum(dim=(1, 2)) + EPS
    mx = (weight * xx).sum(dim=(1, 2)) / mass
    my = (weight * yy).sum(dim=(1, 2)) / mass
    dx = xx - mx[:, None, None]
    dy = yy - my[:, None, None]
    cov_xx = (weight * dx.square()).sum(dim=(1, 2)) / mass
    cov_yy = (weight * dy.square()).sum(dim=(1, 2)) / mass
    cov_xy = (weight * dx * dy).sum(dim=(1, 2)) / mass
    trace = cov_xx + cov_yy
    det = torch.sqrt(torch.clamp((cov_xx - cov_yy).square() + 4.0 * cov_xy.square(), min=0.0))
    eig_1 = 0.5 * (trace + det)
    eig_2 = 0.5 * (trace - det)
    return eig_2 / (eig_1 + EPS)


def corr_per_image(a, b):
    af = a.flatten(1)
    bf = b.flatten(1)
    ac = af - af.mean(dim=1, keepdim=True)
    bc = bf - bf.mean(dim=1, keepdim=True)
    denom = torch.sqrt((ac.square().sum(dim=1) + EPS) * (bc.square().sum(dim=1) + EPS))
    return (ac * bc).sum(dim=1) / denom


def current_evidence_features(x):
    gray_raw = x[:, 0] if x.shape[1] == 1 else x.mean(dim=1)
    dark_fraction = (gray_raw < (8.0 / 255.0)).float().flatten(1).mean(dim=1)
    bright_fraction = (gray_raw > (230.0 / 255.0)).float().flatten(1).mean(dim=1)
    gray = robust_rescale01(gray_raw)
    smooth = speckle_reduced_gray(gray)
    edge = sobel_edges(smooth)
    local_mean = avg_pool(smooth, 11)
    contrast = (smooth - local_mean).abs()
    texture = torch.sqrt(local_variance(smooth, kernel=9) + EPS)
    residual = torch.sqrt(local_variance(gray - smooth, kernel=5) + EPS)
    denom = edge.flatten(1).mean(dim=1) + contrast.flatten(1).mean(dim=1) + texture.flatten(1).mean(dim=1) + EPS
    speckle_ratio = torch.clamp(residual.flatten(1).mean(dim=1) / denom, min=0.0, max=1.0)
    sr = speckle_ratio.view(-1, 1, 1)
    edge_weight = 0.34 - 0.18 * sr
    contrast_weight = 0.36 + 0.07 * sr
    texture_weight = 1.0 - edge_weight - contrast_weight
    evidence = edge_weight * edge + contrast_weight * contrast + texture_weight * texture
    mask = torch.clamp(evidence / (safe_quantile(evidence, 0.88) + EPS), min=0.0, max=1.0)
    mass = mask.sum(dim=(1, 2)) + EPS
    area = mask.mean(dim=(1, 2))
    boundary_strength = (edge * mask).sum(dim=(1, 2)) / mass
    local_contrast = (contrast * mask).sum(dim=(1, 2)) / mass
    texture_heterogeneity = (texture * mask).sum(dim=(1, 2)) / mass
    compactness = spatial_eccentricity(mask)
    reliability = torch.clamp(
        (0.50 * local_contrast + 0.35 * boundary_strength + 0.15 * compactness)
        / (0.45 + speckle_ratio + EPS),
        min=0.05,
        max=2.0,
    )
    salience = (
        (0.35 + boundary_strength)
        * (0.35 + local_contrast)
        * (0.35 + texture_heterogeneity)
        * (0.65 + area)
        * (0.75 + compactness)
        * reliability
    )
    evidence_mean = evidence.flatten(1).mean(dim=1)
    evidence_std = evidence.flatten(1).std(dim=1)
    h, w = evidence.shape[-2], evidence.shape[-1]
    border = max(8, int(round(min(h, w) * 0.08)))
    border_mask = torch.zeros((h, w), dtype=torch.bool, device=evidence.device)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    center_mask = ~border_mask
    border_mask_f = border_mask.view(1, h, w).float()
    center_mask_f = center_mask.view(1, h, w).float()
    border_evidence_fraction = (evidence * border_mask_f).sum(dim=(1, 2)) / (evidence.sum(dim=(1, 2)) + EPS)
    border_edge_fraction = (edge * border_mask_f).sum(dim=(1, 2)) / (edge.sum(dim=(1, 2)) + EPS)
    center_evidence_fraction = (evidence * center_mask_f).sum(dim=(1, 2)) / (evidence.sum(dim=(1, 2)) + EPS)
    top = evidence >= safe_quantile(evidence, 0.90)
    rest = ~top
    top_mean = (evidence * top).flatten(1).sum(dim=1) / (top.flatten(1).sum(dim=1) + EPS)
    rest_mean = (evidence * rest).flatten(1).sum(dim=1) / (rest.flatten(1).sum(dim=1) + EPS)
    return {
        "raw_mean": gray_raw.flatten(1).mean(dim=1),
        "raw_std": gray_raw.flatten(1).std(dim=1),
        "raw_dynamic_p95_p05": safe_quantile(gray_raw, 0.95).flatten() - safe_quantile(gray_raw, 0.05).flatten(),
        "dark_pixel_fraction": dark_fraction,
        "bright_pixel_fraction": bright_fraction,
        "edge_mean": edge.flatten(1).mean(dim=1),
        "local_contrast_mean": contrast.flatten(1).mean(dim=1),
        "texture_mean": texture.flatten(1).mean(dim=1),
        "residual_noise_mean": residual.flatten(1).mean(dim=1),
        "speckle_ratio": speckle_ratio,
        "edge_residual_corr": corr_per_image(edge, residual),
        "border_evidence_fraction": border_evidence_fraction,
        "border_edge_fraction": border_edge_fraction,
        "center_evidence_fraction": center_evidence_fraction,
        "evidence_mean": evidence_mean,
        "evidence_cv": evidence_std / (evidence_mean + EPS),
        "evidence_top10_rest_ratio": top_mean / (rest_mean + EPS),
        "soft_mask_area": area,
        "boundary_strength": boundary_strength,
        "masked_local_contrast": local_contrast,
        "masked_texture": texture_heterogeneity,
        "shape_compactness": compactness,
        "diagnostic_salience": salience,
        "evidence_reliability": reliability,
        "edge_weight": edge_weight.flatten(),
        "contrast_weight": contrast_weight.flatten(),
        "texture_weight": texture_weight.flatten(),
    }, {
        "gray": gray,
        "edge": edge,
        "residual": residual,
        "evidence": evidence,
        "mask": mask,
    }


def collect_channel_metrics(x):
    out = {}
    if x.shape[1] >= 3:
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        out["channel_abs_diff"] = ((r - g).abs() + (r - b).abs() + (g - b).abs()).flatten(1).mean(dim=1) / 3.0
        out["rgb_channel_std"] = x[:, :3].std(dim=1).flatten(1).mean(dim=1)
    else:
        z = torch.zeros(x.shape[0], device=x.device)
        out["channel_abs_diff"] = z
        out["rgb_channel_std"] = z
    return out


def sample_indices(n, limit, rng):
    if n <= limit:
        return np.arange(n)
    return np.sort(rng.choice(n, size=limit, replace=False))


def dataset_samples_for_montage(images, labels, count, rng):
    idx = sample_indices(len(images), count, rng)
    return images[idx], labels[idx]


def append_metric(storage, name, values):
    storage.setdefault(name, []).extend(values.detach().cpu().numpy().astype(float).tolist())


def analyze_dataset(npz_path, sample_per_split, batch_size, rng):
    npz = np.load(npz_path, allow_pickle=False)
    metric_values = {}
    class_counts = {}
    shape_info = {}
    montage_images = []
    montage_labels = []
    class_examples = {}
    evidence_examples = None

    for split in ("train", "val", "test"):
        images = npz[f"{split}_images"]
        labels = labels_1d(npz[f"{split}_labels"])
        shape_info[split] = {
            "images": list(images.shape),
            "labels": list(labels.shape),
            "dtype": str(images.dtype),
            "min": int(images.min()),
            "max": int(images.max()),
        }
        for cls, count in zip(*np.unique(labels, return_counts=True)):
            class_counts[str(int(cls))] = class_counts.get(str(int(cls)), 0) + int(count)
        for cls in sorted(np.unique(labels).astype(int).tolist()):
            existing = class_examples.setdefault(cls, [])
            if len(existing) >= 4:
                continue
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) == 0:
                continue
            take = cls_idx[: max(0, 4 - len(existing))]
            existing.extend([images[i].copy() for i in take])

        if split == "test":
            m_img, m_lab = dataset_samples_for_montage(images, labels, 8, rng)
            montage_images.extend(list(m_img))
            montage_labels.extend([int(x) for x in m_lab])

        idx = sample_indices(len(images), sample_per_split, rng)
        sampled_images = images[idx]
        sampled_labels = labels[idx]
        for start in range(0, len(sampled_images), batch_size):
            batch = sampled_images[start : start + batch_size]
            x = to_tensor(batch)
            with torch.no_grad():
                metrics, maps = current_evidence_features(x)
                metrics.update(collect_channel_metrics(x))
            for name, values in metrics.items():
                append_metric(metric_values, name, values)

            if evidence_examples is None and split == "test":
                keep = min(6, x.shape[0])
                evidence_examples = {
                    "images": batch[:keep].copy(),
                    "labels": [int(v) for v in sampled_labels[start : start + keep]],
                    "gray": maps["gray"][:keep].cpu().numpy(),
                    "edge": maps["edge"][:keep].cpu().numpy(),
                    "residual": maps["residual"][:keep].cpu().numpy(),
                    "evidence": maps["evidence"][:keep].cpu().numpy(),
                    "mask": maps["mask"][:keep].cpu().numpy(),
                }

    summary = {
        "dataset": npz_path.stem,
        "num_samples_analyzed": len(next(iter(metric_values.values()))) if metric_values else 0,
        "shape_info": shape_info,
        "class_counts": class_counts,
        "metrics": {},
    }
    for name, values in metric_values.items():
        arr = np.asarray(values, dtype=np.float64)
        summary["metrics"][name] = {
            "mean": safe_mean(arr),
            "std": safe_std(arr),
            "p10": q(arr, 0.10),
            "p50": q(arr, 0.50),
            "p90": q(arr, 0.90),
        }

    sal = np.asarray(metric_values.get("diagnostic_salience", []), dtype=np.float64)
    summary["metrics"]["diagnostic_salience_between_sample_cv"] = {
        "mean": float(np.std(sal) / (np.mean(sal) + EPS)) if len(sal) else float("nan"),
        "std": 0.0,
        "p10": float("nan"),
        "p50": float("nan"),
        "p90": float("nan"),
    }
    return summary, {
        "images": montage_images[:8],
        "labels": montage_labels[:8],
        "class_examples": class_examples,
        "evidence_examples": evidence_examples,
    }


def image_to_pil(img):
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def map_to_pil(values):
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.quantile(arr, 0.02))
    hi = float(np.quantile(arr, 0.98))
    arr = np.clip((arr - lo) / (hi - lo + EPS), 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def get_font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def save_montage(samples_by_dataset, out_path):
    cell = 96
    label_h = 22
    name_w = 170
    datasets = list(samples_by_dataset.keys())
    width = name_w + 8 * cell
    height = len(datasets) * (cell + label_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = get_font(13)
    for row, dataset in enumerate(datasets):
        y = row * (cell + label_h)
        draw.text((8, y + 35), dataset, fill=(20, 20, 20), font=font)
        data = samples_by_dataset[dataset]
        for col, img in enumerate(data["images"][:8]):
            x = name_w + col * cell
            tile = image_to_pil(img).resize((cell, cell))
            canvas.paste(tile, (x, y))
            label = data["labels"][col] if col < len(data["labels"]) else ""
            draw.text((x + 4, y + cell + 2), f"y={label}", fill=(40, 40, 40), font=font)
    canvas.save(out_path)


def save_evidence_grid(dataset, evidence_examples, out_path):
    if evidence_examples is None:
        return
    rows = [
        ("source", [image_to_pil(x) for x in evidence_examples["images"]]),
        ("edge", [map_to_pil(x) for x in evidence_examples["edge"]]),
        ("residual", [map_to_pil(x) for x in evidence_examples["residual"]]),
        ("evidence", [map_to_pil(x) for x in evidence_examples["evidence"]]),
        ("soft_mask", [map_to_pil(x) for x in evidence_examples["mask"]]),
    ]
    cell = 112
    name_w = 110
    label_h = 24
    width = name_w + len(rows[0][1]) * cell
    height = len(rows) * cell + label_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = get_font(13)
    title_font = get_font(15)
    draw.text((8, 4), f"{dataset} evidence maps", fill=(20, 20, 20), font=title_font)
    for col, label in enumerate(evidence_examples["labels"]):
        draw.text((name_w + col * cell + 4, 4), f"y={label}", fill=(40, 40, 40), font=font)
    for row_idx, (name, imgs) in enumerate(rows):
        y = label_h + row_idx * cell
        draw.text((8, y + 44), name, fill=(20, 20, 20), font=font)
        for col, img in enumerate(imgs):
            tile = img.resize((cell, cell))
            canvas.paste(tile, (name_w + col * cell, y))
    canvas.save(out_path)


def save_class_montage(dataset, class_examples, out_path):
    if not class_examples:
        return
    classes = sorted(class_examples)
    cell = 96
    label_w = 90
    cols = max(len(v) for v in class_examples.values())
    width = label_w + cols * cell
    height = len(classes) * cell
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = get_font(13)
    for row, cls in enumerate(classes):
        y = row * cell
        draw.text((8, y + 40), f"class {cls}", fill=(20, 20, 20), font=font)
        for col, img in enumerate(class_examples[cls]):
            x = label_w + col * cell
            canvas.paste(image_to_pil(img).resize((cell, cell)), (x, y))
    canvas.save(out_path)


def metric(summary, name):
    return summary["metrics"][name]["mean"]


def save_metric_bars(summaries, out_path):
    metric_names = [
        "raw_dynamic_p95_p05",
        "dark_pixel_fraction",
        "border_evidence_fraction",
        "edge_mean",
        "speckle_ratio",
        "edge_residual_corr",
        "evidence_reliability",
        "diagnostic_salience_between_sample_cv",
    ]
    labels = {
        "raw_dynamic_p95_p05": "dynamic",
        "dark_pixel_fraction": "dark pixels",
        "border_evidence_fraction": "border evidence",
        "edge_mean": "edge",
        "speckle_ratio": "speckle",
        "edge_residual_corr": "edge-noise corr",
        "evidence_reliability": "reliability",
        "diagnostic_salience_between_sample_cv": "salience cv",
    }
    datasets = [s["dataset"] for s in summaries]
    width = 1260
    row_h = 145
    left = 260
    right = 40
    top = 58
    bar_h = 15
    gap = 4
    height = top + len(metric_names) * row_h + 30
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = get_font(13)
    title_font = get_font(17)
    draw.text((18, 16), "Source image metric comparison", fill=(20, 20, 20), font=title_font)
    palette = ["#385E72", "#C95F3F", "#4F7F52", "#8A6F3A", "#6A5A8C"]
    for row_idx, name in enumerate(metric_names):
        y0 = top + row_idx * row_h
        values = [metric(s, name) for s in summaries]
        max_v = max(max(values), EPS)
        min_v = min(values)
        draw.text((18, y0 + 4), labels[name], fill=(20, 20, 20), font=font)
        if min_v < 0:
            max_abs = max(abs(min_v), abs(max_v), EPS)
            zero_x = left + int((0.0 + max_abs) / (2 * max_abs) * (width - left - right))
            draw.line((zero_x, y0 + 8, zero_x, y0 + 96), fill=(210, 210, 210))
            scale = (width - left - right) / (2 * max_abs)
            for i, (dataset, value) in enumerate(zip(datasets, values)):
                y = y0 + 30 + i * (bar_h + gap)
                x1 = zero_x
                x2 = zero_x + int(value * scale)
                draw.rectangle((min(x1, x2), y, max(x1, x2), y + bar_h), fill=palette[i % len(palette)])
                draw.text((52, y - 1), dataset.replace("mnist_224", ""), fill=(70, 70, 70), font=font)
                draw.text((max(x1, x2) + 5, y - 1), f"{value:.3f}", fill=(40, 40, 40), font=font)
        else:
            for i, (dataset, value) in enumerate(zip(datasets, values)):
                y = y0 + 30 + i * (bar_h + gap)
                x2 = left + int(value / max_v * (width - left - right))
                draw.rectangle((left, y, x2, y + bar_h), fill=palette[i % len(palette)])
                draw.text((52, y - 1), dataset.replace("mnist_224", ""), fill=(70, 70, 70), font=font)
                draw.text((x2 + 5, y - 1), f"{value:.3f}", fill=(40, 40, 40), font=font)
    canvas.save(out_path)


def write_csv(summaries, out_path):
    metric_names = sorted({name for s in summaries for name in s["metrics"]})
    fields = ["dataset", "num_samples_analyzed"]
    for name in metric_names:
        fields.extend([f"{name}_mean", f"{name}_std", f"{name}_p10", f"{name}_p50", f"{name}_p90"])
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            row = {"dataset": s["dataset"], "num_samples_analyzed": s["num_samples_analyzed"]}
            for name in metric_names:
                stats = s["metrics"].get(name, {})
                for stat in ("mean", "std", "p10", "p50", "p90"):
                    row[f"{name}_{stat}"] = stats.get(stat, "")
            writer.writerow(row)


def fmt(x):
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.4f}"


def write_report(summaries, paths, out_path):
    by_name = {s["dataset"]: s for s in summaries}
    chao = by_name.get("chaoshengmnist_224")
    others = [s for s in summaries if s["dataset"] != "chaoshengmnist_224"]
    lines = [
        "# Source Image Analysis",
        "",
        "This report is generated from NPZ source images, not from model predictions.",
        "",
        "## Files",
        "",
    ]
    for label, path in paths.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(["", "## Dataset Structure", ""])
    lines.append("| dataset | train shape | val shape | test shape | classes | analyzed samples |")
    lines.append("| --- | --- | --- | --- | ---: | ---: |")
    for s in summaries:
        info = s["shape_info"]
        n_cls = len(s["class_counts"])
        lines.append(
            f"| `{s['dataset']}` | `{info['train']['images']}` | `{info['val']['images']}` | "
            f"`{info['test']['images']}` | {n_cls} | {s['num_samples_analyzed']} |"
        )

    key_metrics = [
        ("channel_abs_diff", "RGB channel difference"),
        ("dark_pixel_fraction", "dark pixel fraction"),
        ("border_evidence_fraction", "border evidence fraction"),
        ("raw_dynamic_p95_p05", "p95-p05 dynamic range"),
        ("edge_mean", "Sobel edge mean"),
        ("residual_noise_mean", "residual noise"),
        ("speckle_ratio", "speckle ratio"),
        ("edge_residual_corr", "edge-noise correlation"),
        ("evidence_reliability", "M1 evidence reliability"),
        ("diagnostic_salience_between_sample_cv", "salience between-sample CV"),
    ]
    lines.extend(["", "## Metric Summary", ""])
    lines.append("| dataset | " + " | ".join(label for _, label in key_metrics) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in key_metrics) + " |")
    for s in summaries:
        values = [fmt(s["metrics"][name]["mean"]) for name, _ in key_metrics]
        lines.append(f"| `{s['dataset']}` | " + " | ".join(values) + " |")

    lines.extend(["", "## Main Findings", ""])
    if chao:
        ch = chao["metrics"]
        def other_mean(name):
            return float(np.mean([metric(s, name) for s in others])) if others else float("nan")

        lines.append(
            f"- `chaoshengmnist_224` is stored as RGB but has near-zero channel difference "
            f"({fmt(ch['channel_abs_diff']['mean'])}); it is effectively grayscale replicated into 3 channels."
        )
        lines.append(
            f"- Its edge-noise correlation is {fmt(ch['edge_residual_corr']['mean'])}, compared with "
            f"other-dataset average {fmt(other_mean('edge_residual_corr'))}. High correlation means Sobel edges are strongly coupled with high-frequency residuals."
        )
        lines.append(
            f"- Its speckle ratio is {fmt(ch['speckle_ratio']['mean'])}, compared with other-dataset average "
            f"{fmt(other_mean('speckle_ratio'))}. This makes an edge-heavy M1 feature likely to score acquisition noise as diagnostic evidence."
        )
        lines.append(
            f"- Its dark-pixel fraction is {fmt(ch['dark_pixel_fraction']['mean'])}, and border evidence fraction is "
            f"{fmt(ch['border_evidence_fraction']['mean'])}. Ultrasound frames and machine overlays can therefore receive non-trivial evidence mass."
        )
        lines.append(
            f"- Its M1 evidence reliability is {fmt(ch['evidence_reliability']['mean'])}, compared with other-dataset average "
            f"{fmt(other_mean('evidence_reliability'))}. Lower reliability should gate client-specific weighting more aggressively."
        )
        lines.append(
            f"- Its salience between-sample CV is {fmt(ch['diagnostic_salience_between_sample_cv']['mean'])}. "
            "If this is high while evidence is noisy, M1 can over-amplify arbitrary ultrasound texture differences between clients."
        )

    lines.extend(
        [
            "",
            "## Optimization Direction",
            "",
            "1. Treat ultrasound as a low-reliability evidence domain unless the source-image reliability score is high.",
            "2. Reduce the role of Sobel edge strength when edge-noise correlation or speckle ratio is high.",
            "3. Use structure-preserving smoothing before evidence extraction; median/SRAD-like despeckling is more appropriate than raw edge maps.",
            "4. Prefer robust texture and coarse ROI statistics over hard foreground masks for ultrasound.",
            "5. In M2, do not anchor strongly to one client when M1 evidence reliability is low; keep routed weights close to average/consensus.",
            "",
            "## Suggested Code-Level Rule",
            "",
            "For each dataset or validation split, compute `speckle_ratio`, `edge_residual_corr`, and `evidence_reliability` from source images. "
            "If `speckle_ratio` and `edge_residual_corr` are high, apply an ultrasound-safe profile: lower edge weight, higher smoothing, lower M1 gate, and no hard anchor.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    figure_dir = ROOT / "My_merge_ret" / "figures" / "source_image_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summaries = []
    samples = {}
    for dataset in args.datasets:
        npz_path = data_root / f"{dataset}.npz"
        if not npz_path.exists():
            print(f"[skip] missing {npz_path}")
            continue
        print(f"[analyze] {npz_path}")
        summary, sample_info = analyze_dataset(npz_path, args.sample_per_split, args.batch_size, rng)
        summaries.append(summary)
        samples[dataset] = sample_info

    metrics_csv = out_dir / "image_source_metrics.csv"
    summary_json = out_dir / "image_source_summary.json"
    report_md = out_dir / "image_source_analysis.md"
    montage_png = figure_dir / "source_samples.png"
    bars_png = figure_dir / "source_metric_bars.png"
    chao_evidence_png = figure_dir / "chaosheng_evidence_maps.png"
    chao_class_png = figure_dir / "chaosheng_class_samples.png"

    write_csv(summaries, metrics_csv)
    summary_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    save_montage(samples, montage_png)
    save_metric_bars(summaries, bars_png)
    if "chaoshengmnist_224" in samples:
        save_evidence_grid(
            "chaoshengmnist_224",
            samples["chaoshengmnist_224"]["evidence_examples"],
            chao_evidence_png,
        )
        save_class_montage(
            "chaoshengmnist_224",
            samples["chaoshengmnist_224"]["class_examples"],
            chao_class_png,
        )
    write_report(
        summaries,
        {
            "metrics_csv": metrics_csv,
            "summary_json": summary_json,
            "sample_montage": montage_png,
            "metric_bars": bars_png,
            "chaosheng_evidence_maps": chao_evidence_png,
            "chaosheng_class_samples": chao_class_png,
        },
        report_md,
    )
    print(f"[done] {report_md}")
    print(f"[done] {metrics_csv}")
    print(f"[done] {montage_png}")
    print(f"[done] {bars_png}")
    print(f"[done] {chao_evidence_png}")
    print(f"[done] {chao_class_png}")


if __name__ == "__main__":
    main()
