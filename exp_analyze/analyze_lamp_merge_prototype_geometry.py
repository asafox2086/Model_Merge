#!/usr/bin/env python3
"""Analyze LAMP-Merge prototype geometry on the formal full grid.

The main ablation table measures final accuracy. This script inspects the
    uploaded class-level prototype information itself: class separation, client
    consistency, and support coverage. These are geometry diagnostics computed
    from uploaded statistics under each ablation definition; they are not a
    substitute for full accuracy evaluation of the corresponding ablation.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import NpzTensorDataset, load_npz_splits
from methods.lamp_merge_analysis import (
    _classifier_weight_means,
    _client_feature_means,
    _client_stat_matrix,
    _evidence_matrix,
    _global_feature_mean_proxy,
    _num_clients,
    _support_only_proxy,
)
from model import build_model
from utils import load_checkpoint, load_json
from utils.hub import beta_to_dirname
from utils.lamp_merge_stats import default_prototype_root, validate_lamp_merge_stats_payload
from utils.runtime import build_reference_bundle
from utils.state_dict import extract_state_dict


FORMAL_SMALL_DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
]

FORMAL_SMALL_MODELS = ["resnet", "convnext", "vit_t", "swin_tiny"]

PROTOTYPE_MODES = [
    "full",
    "m1_only",
    "prototype_head_agg",
    "global_feature_mean",
    "support_only",
    "prototype_shuffle",
    "uniform_client_weight",
    "binary_support",
    "global_client_size_weight",
    "uniform_prevalence",
    "smoothed_prevalence",
]

MODE_LABELS = {
    "full": "LAMP-Merge",
    "m1_only": "M1 only",
    "no_prevalence": "No prevalence calibration",
    "prototype_head_agg": "Classifier-head aggregation",
    "head_agg": "Classifier-head aggregation",
    "global_feature_mean": "Global-feature mean",
    "support_only": "Support-only synthetic head",
    "prototype_shuffle": "Shuffled-label prototype",
    "uniform_client_weight": "Uniform client weight",
    "binary_support": "Binary support only",
    "global_client_size_weight": "Global client-size weight",
    "uniform_prevalence": "Uniform prevalence prior",
    "smoothed_prevalence": "Smoothed prevalence prior",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Full-scope LAMP-Merge prototype geometry analysis.")
    parser.add_argument("--model-hub-root", type=Path, default=ROOT / "model_hub")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Med_data")
    parser.add_argument(
        "--prototype-root",
        type=Path,
        default=default_prototype_root(ROOT),
    )
    parser.add_argument("--output-csv", type=Path, default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_prototype_geometry_full.csv")
    parser.add_argument(
        "--overall-csv",
        type=Path,
        default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_prototype_geometry_overall.csv",
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_prototype_geometry_by_dataset.csv",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "My_merge_ret" / "reports" / "lamp_merge_prototype_geometry_summary.md",
    )
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "My_merge_ret" / "figures" / "lamp_merge_prototype_geometry")
    parser.add_argument("--datasets", nargs="*", default=FORMAL_SMALL_DATASETS)
    parser.add_argument("--small-models", nargs="*", default=FORMAL_SMALL_MODELS)
    parser.add_argument("--num-clients", nargs="*", type=int, default=[3, 5, 7])
    parser.add_argument("--betas", nargs="*", type=float, default=[0.0, 0.01, 0.1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="*", default=PROTOTYPE_MODES)
    parser.add_argument("--proto-count-power", type=float, default=0.55)
    parser.add_argument("--ablation-seed", type=int, default=1701)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plot-tsne", action="store_true")
    parser.add_argument("--tsne-datasets", nargs="*", default=[])
    parser.add_argument("--tsne-model", type=str, default="resnet")
    parser.add_argument("--tsne-num-clients", type=int, default=3)
    parser.add_argument("--tsne-max-samples-per-class", type=int, default=120)
    parser.add_argument(
        "--plot-full-tsne",
        action="store_true",
        help=(
            "Plot full-scope t-SNE diagnostics for every selected dataset/backbone. "
            "For each dataset/backbone/mode, prototypes from all selected client counts "
            "and beta values are projected together with reference-backbone test features."
        ),
    )
    parser.add_argument(
        "--full-tsne-modes",
        nargs="*",
        default=[
            "full",
            "prototype_head_agg",
            "global_feature_mean",
            "prototype_shuffle",
            "uniform_client_weight",
        ],
    )
    return parser.parse_args()


def beta_key(value: object) -> str:
    return format(float(value), "g")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_manifest(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    datasets = set(args.datasets)
    models = set(args.small_models)
    clients = set(int(x) for x in args.num_clients)
    betas = {beta_key(x) for x in args.betas}
    out = [
        row
        for row in rows
        if row["task_type"] == "small"
        and row["dataset"] in datasets
        and row["model"] in models
        and int(row["num_clients"]) in clients
        and beta_key(row["beta"]) in betas
        and int(row["seed"]) == int(args.seed)
    ]
    out.sort(key=lambda row: (row["dataset"], row["model"], int(row["num_clients"]), float(row["beta"])))
    return out


def prototype_stats_path(args: argparse.Namespace, row: dict[str, str]) -> Path:
    return (
        args.prototype_root
        / "small"
        / row["dataset"]
        / row["model"]
        / f"clients_{int(row['num_clients'])}"
        / beta_to_dirname(row["beta"])
        / f"seed_{int(row['seed'])}"
        / "prototype_stats.pt"
    )


def load_state_dicts(args: argparse.Namespace, row: dict[str, str], meta: dict[str, object]) -> list[dict[str, torch.Tensor]]:
    exp_dir = args.model_hub_root / row["hub_dir"]
    state_dicts = []
    for idx, client in enumerate(meta.get("clients", [])):
        ckpt_path = exp_dir / str(client.get("checkpoint", f"client_{idx}.pt"))
        state_dicts.append(extract_state_dict(load_checkpoint(ckpt_path, device="cpu")))
    return state_dicts


def ablation_components(mode: str) -> tuple[str, str]:
    if mode in {"full", "m1_only", "no_prevalence", "uniform_prevalence", "smoothed_prevalence"}:
        return "reference_prototype", "support_power"
    if mode == "prototype_head_agg":
        return "classifier_head_aggregation", "support_power"
    if mode == "head_agg":
        return "classifier_head_aggregation", "support_power"
    if mode == "global_feature_mean":
        return "global_feature_mean", "support_power"
    if mode == "support_only":
        return "support_only_synthetic", "support_power"
    if mode == "prototype_shuffle":
        return "reference_prototype_shuffle", "support_power"
    if mode == "uniform_client_weight":
        return "reference_prototype", "uniform_present_client"
    if mode == "binary_support":
        return "reference_prototype", "uniform_present_client"
    if mode == "global_client_size_weight":
        return "reference_prototype", "global_client_size"
    raise ValueError(f"unsupported prototype geometry mode: {mode}")


def build_mode_geometry(
    mode: str,
    proto_stats: dict[str, object],
    meta: dict[str, object],
    cfg: dict[str, object],
    state_dicts: list[dict[str, torch.Tensor]] | None,
) -> dict[str, torch.Tensor | str | float | int | list[int]]:
    num_classes = int(meta["num_classes"])
    num_clients = _num_clients(meta, proto_stats)
    means = _client_feature_means(proto_stats, num_clients, num_classes)
    feature_counts = _client_stat_matrix(proto_stats, ("class_feature_counts", "class_counts"), num_clients, num_classes)
    prototype_mode, evidence_mode = ablation_components(mode)

    if prototype_mode == "reference_prototype" or prototype_mode == "reference_prototype_shuffle":
        inputs = means
    elif prototype_mode == "classifier_head_aggregation":
        if state_dicts is None:
            raise ValueError("classifier-head aggregation requires client state_dicts")
        inputs = _classifier_weight_means(state_dicts, proto_stats, meta)
    elif prototype_mode == "global_feature_mean":
        inputs = _global_feature_mean_proxy(means, feature_counts)
    elif prototype_mode == "support_only_synthetic":
        inputs = _support_only_proxy(num_clients, num_classes, int(means.shape[-1]), cfg)
    else:
        raise ValueError(f"unsupported prototype mode: {prototype_mode}")

    evidence, gamma = _evidence_matrix(feature_counts, proto_stats, meta, cfg, evidence_mode)
    evidence_per_class = evidence.sum(dim=0)
    weights = evidence / evidence_per_class.view(1, -1).clamp_min(1e-8)
    prototypes = (inputs * weights.unsqueeze(-1)).sum(dim=0)
    permutation = list(range(num_classes))
    if prototype_mode == "reference_prototype_shuffle":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(cfg["lamp_merge_ablation_seed"]) + int(meta.get("seed", 0)) + num_classes)
        perm = torch.randperm(num_classes, generator=generator)
        prototypes = prototypes[perm]
        permutation = [int(x) for x in perm.tolist()]

    return {
        "prototypes": prototypes.float(),
        "prototype_inputs": inputs.float(),
        "feature_counts": feature_counts.float(),
        "weights": weights.float(),
        "evidence": evidence.float(),
        "evidence_gamma": gamma if gamma is not None else "",
        "prototype_mode": prototype_mode,
        "evidence_mode": evidence_mode,
        "permutation": permutation,
    }


def cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    z = F.normalize(x.float(), dim=1)
    return z @ z.t()


def prototype_metrics(geometry: dict[str, object]) -> dict[str, object]:
    prototypes = geometry["prototypes"]
    inputs = geometry["prototype_inputs"]
    counts = geometry["feature_counts"]
    weights = geometry["weights"]
    assert isinstance(prototypes, torch.Tensor)
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(counts, torch.Tensor)
    assert isinstance(weights, torch.Tensor)

    num_classes = int(prototypes.shape[0])
    sim = cosine_matrix(prototypes)
    off_mask = ~torch.eye(num_classes, dtype=torch.bool)
    off_diag = sim[off_mask]
    nearest = off_diag.view(num_classes, num_classes - 1).max(dim=1).values if num_classes > 1 else torch.zeros(1)
    separation = 1.0 - off_diag
    present = counts > 0

    consistency_values = []
    weighted_consistency_values = []
    for c in range(num_classes):
        idx = torch.nonzero(present[:, c], as_tuple=False).view(-1)
        if int(idx.numel()) <= 1:
            continue
        client_vecs = inputs[idx, c, :]
        client_sim = cosine_matrix(client_vecs)
        pair_mask = ~torch.eye(int(idx.numel()), dtype=torch.bool)
        consistency_values.extend(client_sim[pair_mask].tolist())
        proto_dir = F.normalize(prototypes[c].view(1, -1), dim=1)
        client_dir = F.normalize(client_vecs, dim=1)
        weighted_consistency_values.extend((client_dir @ proto_dir.t()).view(-1).tolist())

    support = counts.sum(dim=0)
    evidence_entropy = []
    for c in range(num_classes):
        column = weights[:, c]
        nz = column[column > 0]
        if int(nz.numel()) == 0:
            continue
        entropy = float(-(nz * torch.log(nz.clamp_min(1e-12))).sum().item())
        denom = math.log(max(int(nz.numel()), 2))
        evidence_entropy.append(entropy / denom)

    return {
        "num_classes": num_classes,
        "covered_classes": int((support > 0).sum().item()),
        "mean_class_support": float(support.mean().item()),
        "min_class_support": float(support.min().item()),
        "max_class_support": float(support.max().item()),
        "mean_pairwise_cosine": float(off_diag.mean().item()) if int(off_diag.numel()) else 0.0,
        "mean_pairwise_distance": float(separation.mean().item()) if int(separation.numel()) else 0.0,
        "min_pairwise_distance": float(separation.min().item()) if int(separation.numel()) else 0.0,
        "mean_nearest_class_cosine": float(nearest.mean().item()) if int(nearest.numel()) else 0.0,
        "mean_nearest_class_distance": float((1.0 - nearest).mean().item()) if int(nearest.numel()) else 0.0,
        "prototype_consistency": mean(consistency_values) if consistency_values else "",
        "prototype_to_client_alignment": mean(weighted_consistency_values) if weighted_consistency_values else "",
        "mean_evidence_entropy_norm": mean(evidence_entropy) if evidence_entropy else "",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["mode"]))].append(row)
    metric_names = [
        "covered_classes",
        "mean_class_support",
        "min_class_support",
        "max_class_support",
        "mean_pairwise_cosine",
        "mean_pairwise_distance",
        "min_pairwise_distance",
        "mean_nearest_class_cosine",
        "mean_nearest_class_distance",
        "prototype_consistency",
        "prototype_to_client_alignment",
        "mean_evidence_entropy_norm",
    ]
    out = []
    for (dataset, mode), items in sorted(grouped.items()):
        row = {"dataset": dataset, "mode": mode, "label": MODE_LABELS.get(mode, mode), "cases": len(items)}
        for metric in metric_names:
            vals = [float(item[metric]) for item in items if item.get(metric) not in {"", None}]
            row[f"mean_{metric}"] = mean(vals) if vals else ""
        out.append(row)
    return out


def group_overall_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    metric_names = [
        "covered_classes",
        "mean_class_support",
        "min_class_support",
        "max_class_support",
        "mean_pairwise_cosine",
        "mean_pairwise_distance",
        "min_pairwise_distance",
        "mean_nearest_class_cosine",
        "mean_nearest_class_distance",
        "prototype_consistency",
        "prototype_to_client_alignment",
        "mean_evidence_entropy_norm",
    ]
    out = []
    for mode, items in sorted(grouped.items(), key=lambda item: mode_sort_key(item[0])):
        row = {"mode": mode, "label": MODE_LABELS.get(mode, mode), "cases": len(items)}
        for metric in metric_names:
            vals = [float(item[metric]) for item in items if item.get(metric) not in {"", None}]
            row[f"mean_{metric}"] = mean(vals) if vals else ""
        out.append(row)
    return out


def fmt(value: object, digits: int = 4) -> str:
    if value in {"", None}:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def mode_sort_key(mode: str) -> tuple[int, str]:
    order = {mode: idx for idx, mode in enumerate(PROTOTYPE_MODES)}
    return order.get(mode, len(order)), mode


def write_geometry_table(handle, rows: list[dict[str, object]]) -> None:
    handle.write(
        "| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |\n"
    )
    handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for row in sorted(rows, key=lambda item: mode_sort_key(str(item["mode"]))):
        handle.write(
            f"| {row['label']} | {row['cases']} | "
            f"{fmt(row['mean_mean_pairwise_distance'])} | "
            f"{fmt(row['mean_mean_nearest_class_distance'])} | "
            f"{fmt(row['mean_prototype_consistency'])} | "
            f"{fmt(row['mean_prototype_to_client_alignment'])} | "
            f"{fmt(row['mean_mean_evidence_entropy_norm'])} |\n"
        )


def write_summary(
    path: Path,
    rows: list[dict[str, object]],
    dataset_rows: list[dict[str, object]],
    overall_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in dataset_rows:
        by_dataset[str(row["dataset"])].append(row)

    total_expected = len(args.datasets) * len(args.small_models) * len(args.num_clients) * len(args.betas)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# LAMP-Merge Prototype Geometry Analysis\n\n")
        handle.write(
            "This report analyzes uploaded class-level prototype statistics on the formal full grid. "
            "The numerical rows aggregate five medical datasets, four backbones, three client counts, "
            "and three Dirichlet skew levels unless explicitly filtered.\n\n"
        )
        handle.write(f"Expected cases per mode: `{total_expected}`.\n\n")
        handle.write("## Overall Geometry\n\n")
        write_geometry_table(handle, overall_rows)
        handle.write("\n")
        handle.write("## Dataset-Level Geometry\n\n")
        for dataset in sorted(by_dataset):
            handle.write(f"### {dataset}\n\n")
            write_geometry_table(handle, by_dataset[dataset])
            handle.write("\n")
        handle.write("## Interpretation\n\n")
        handle.write(
            "Prototype separation measures whether class directions occupy distinct positions in the shared "
            "reference feature space. Prototype consistency measures whether clients that contain the same "
            "diagnosis class agree on its reference-space direction. Evidence entropy measures whether the "
            "server assigns class-specific evidence to several clients or concentrates it on one reliable "
            "client. These quantities are diagnostic analyses of the uploaded aggregate statistics; they do "
            "not require the server to read raw client images.\n\n"
        )
        handle.write(
            "Prevalence-only variants share the same prototype geometry as "
            "LAMP-Merge because they modify the score bias rather than the class prototype construction.\n"
        )
        handle.write(
            "These geometry rows are not accuracy ablation results. They are computed directly from the "
            "uploaded prototype and support statistics under each ablation definition; full-scope accuracy "
            "completion must be checked in the main ablation table.\n"
        )


def plot_geometry_bars(dataset_rows: list[dict[str, object]], overall_rows: list[dict[str, object]], figure_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("mean_mean_pairwise_distance", "Mean pairwise distance"),
        ("mean_mean_nearest_class_distance", "Nearest-class distance"),
        ("mean_prototype_consistency", "Prototype consistency"),
        ("mean_prototype_to_client_alignment", "Prototype-client alignment"),
        ("mean_mean_evidence_entropy_norm", "Evidence entropy"),
    ]
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in dataset_rows:
        by_dataset[str(row["dataset"])].append(row)

    def plot_one(rows: list[dict[str, object]], title: str, path: Path) -> None:
        rows = sorted(rows, key=lambda item: mode_sort_key(str(item["mode"])))
        labels = [str(row["label"]) for row in rows]
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
        flat_axes = axes.reshape(-1)
        for ax, (metric, metric_title) in zip(flat_axes, metrics):
            values = []
            for row in rows:
                value = row.get(metric, "")
                values.append(float(value) if value not in {"", None} else np.nan)
            colors = ["#B23A48" if row["mode"] == "full" else "#5B6C8F" for row in rows]
            ax.barh(labels, values, color=colors)
            ax.set_title(metric_title)
            ax.tick_params(axis="y", labelsize=8)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.25)
        for ax in flat_axes[len(metrics) :]:
            ax.axis("off")
        fig.suptitle(title, fontsize=14)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    plot_one(
        overall_rows,
        "All datasets: full-grid prototype-geometry diagnostics",
        figure_dir / "overall_prototype_geometry.png",
    )
    for dataset, rows in sorted(by_dataset.items()):
        plot_one(
            rows,
            f"{dataset}: full-grid prototype-geometry diagnostics",
            figure_dir / f"{dataset}_prototype_geometry.png",
        )


def image_hw(images: np.ndarray) -> tuple[int, int]:
    if images.ndim == 3:
        return int(images.shape[1]), int(images.shape[2])
    if images.ndim >= 4:
        return int(images.shape[1]), int(images.shape[2])
    raise ValueError(f"unexpected image shape: {images.shape}")


def build_transform(meta: dict[str, object], images: np.ndarray):
    from torchvision import transforms

    source_h, source_w = image_hw(images)
    target = int(meta.get("image_size") or 0)
    if target > 0 and (source_h != target or source_w != target):
        return transforms.Resize((target, target), antialias=True)
    return None


def extract_features(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        z = model.forward_features(x)
        try:
            feat = model.forward_head(z, pre_logits=True)
        except TypeError:
            feat = model.forward_head(z)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return feat

    captured = {}
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if classifier is None:
        raise ValueError("model does not expose a feature extraction interface")

    def hook(_module, inputs):
        captured["features"] = inputs[0]

    handle = classifier.register_forward_pre_hook(hook)
    try:
        _ = model(x)
        feat = captured["features"]
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return feat
    finally:
        handle.remove()


def sample_test_indices(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    flat = labels.reshape(-1).astype(np.int64)
    for cls in sorted(np.unique(flat).tolist()):
        idx = np.flatnonzero(flat == int(cls))
        if max_per_class > 0 and idx.size > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        out.append(idx)
    if not out:
        return np.array([], dtype=np.int64)
    result = np.concatenate(out).astype(np.int64)
    rng.shuffle(result)
    return result


def reference_sample_features(args: argparse.Namespace, meta: dict[str, object], dataset: str) -> tuple[np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader, Subset

    split = load_npz_splits(str(args.data_root / f"{dataset}.npz"))["test"]
    indices = sample_test_indices(split.labels, int(args.tsne_max_samples_per_class), int(meta.get("seed", 42)))
    transform = build_transform(meta, split.images)
    ds = NpzTensorDataset(split.images, split.labels, transform=transform)
    loader = DataLoader(
        Subset(ds, indices.tolist()),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=str(args.device).startswith("cuda"),
    )
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model, _, _, _ = build_model(
        name=str(meta["model"]),
        num_classes=int(meta["num_classes"]),
        in_channels=int(meta.get("in_channels", 3)),
        pretrained=False,
    )
    reference_state, _ = build_reference_bundle(meta, device="cpu")
    model.load_state_dict(reference_state, strict=True)
    model.to(device)
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            feat = extract_features(model, x).detach().cpu().float()
            features.append(feat.numpy())
            labels.append(y.detach().cpu().numpy().reshape(-1))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def plot_tsne(args: argparse.Namespace, rows_by_key: dict[tuple[str, str, int, str], dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    tsne_dir = args.figure_dir / "tsne"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.tsne_datasets:
        key = (dataset, args.tsne_model, int(args.tsne_num_clients), "0.01")
        row = rows_by_key.get(key)
        if row is None:
            continue
        meta = row["meta"]
        proto_stats = row["proto_stats"]
        cfg = row["cfg"]
        assert isinstance(meta, dict)
        assert isinstance(proto_stats, dict)
        assert isinstance(cfg, dict)
        sample_features, sample_labels = reference_sample_features(args, meta, dataset)
        for mode in ["full", "prototype_head_agg", "prototype_shuffle", "uniform_client_weight"]:
            state_dicts = row["state_dicts"] if mode == "prototype_head_agg" else None
            geometry = build_mode_geometry(mode, proto_stats, meta, cfg, state_dicts)
            prototypes = geometry["prototypes"]
            assert isinstance(prototypes, torch.Tensor)
            proto_np = prototypes.detach().cpu().float().numpy()
            all_points = np.vstack([sample_features, proto_np])
            perplexity = max(5, min(30, (all_points.shape[0] - 1) // 3))
            embed = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=1701).fit_transform(all_points)
            sample_embed = embed[: sample_features.shape[0]]
            proto_embed = embed[sample_features.shape[0] :]
            fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
            scatter = ax.scatter(
                sample_embed[:, 0],
                sample_embed[:, 1],
                c=sample_labels,
                s=8,
                alpha=0.55,
                cmap="tab20",
                linewidths=0,
            )
            ax.scatter(
                proto_embed[:, 0],
                proto_embed[:, 1],
                c=np.arange(proto_embed.shape[0]),
                s=130,
                marker="X",
                cmap="tab20",
                edgecolors="black",
                linewidths=0.8,
            )
            for cls_idx, (x_coord, y_coord) in enumerate(proto_embed):
                ax.text(x_coord, y_coord, str(cls_idx), fontsize=8, weight="bold", ha="center", va="center")
            ax.set_title(
                f"{dataset} / {args.tsne_model} / K={args.tsne_num_clients} / beta=0.01 / {MODE_LABELS.get(mode, mode)}"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02, label="true class")
            fig.savefig(tsne_dir / f"{dataset}_{args.tsne_model}_c{args.tsne_num_clients}_b0p01_{mode}_tsne.png", dpi=180)
            plt.close(fig)


def plot_full_scope_tsne(
    args: argparse.Namespace,
    tsne_records: list[dict[str, object]],
    meta_by_dataset_model: dict[tuple[str, str], dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    tsne_dir = args.figure_dir / "full_scope_tsne_comparison"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in tsne_records:
        grouped[(str(record["dataset"]), str(record["model"]))].append(record)

    mode_order = [mode for mode in args.full_tsne_modes if mode in set(args.modes)]
    for (dataset, model_name), items in sorted(grouped.items()):
        meta = meta_by_dataset_model.get((dataset, model_name))
        if meta is None:
            continue
        sample_features, sample_labels = reference_sample_features(args, meta, dataset)
        mode_items = {
            mode: sorted(
                [item for item in items if item["mode"] == mode],
                key=lambda item: (int(item["num_clients"]), float(item["beta"])),
            )
            for mode in mode_order
        }
        mode_items = {mode: records for mode, records in mode_items.items() if records}
        if not mode_items:
            continue
        prototype_blocks = []
        block_sizes = {}
        for mode, records in mode_items.items():
            block = np.concatenate(
                [np.asarray(item["prototypes"], dtype=np.float32) for item in records],
                axis=0,
            )
            prototype_blocks.append(block)
            block_sizes[mode] = int(block.shape[0])
        all_points = np.vstack([sample_features, *prototype_blocks])
        perplexity = max(5, min(40, (all_points.shape[0] - 1) // 4))
        embed = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=1701,
        ).fit_transform(all_points)
        sample_embed = embed[: sample_features.shape[0]]
        prototype_embeddings = {}
        offset = sample_features.shape[0]
        for mode in mode_items:
            size = block_sizes[mode]
            prototype_embeddings[mode] = embed[offset : offset + size]
            offset += size

        num_classes = int(meta["num_classes"])
        cmap = plt.get_cmap("tab20", num_classes)
        fig, axes = plt.subplots(
            1,
            len(mode_items),
            figsize=(4.5 * len(mode_items), 4.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        x_min, x_max = float(embed[:, 0].min()), float(embed[:, 0].max())
        y_min, y_max = float(embed[:, 1].min()), float(embed[:, 1].max())
        x_pad = max((x_max - x_min) * 0.04, 1e-3)
        y_pad = max((y_max - y_min) * 0.04, 1e-3)
        scatter = None
        for ax, mode in zip(axes, mode_items):
            scatter = ax.scatter(
                sample_embed[:, 0],
                sample_embed[:, 1],
                c=sample_labels,
                s=6,
                alpha=0.30,
                cmap=cmap,
                vmin=-0.5,
                vmax=num_classes - 0.5,
                linewidths=0,
            )
            proto_embed = prototype_embeddings[mode]
            records = mode_items[mode]
            proto_labels = np.concatenate(
                [np.arange(int(item["num_classes"]), dtype=np.int64) for item in records]
            )
            ax.scatter(
                proto_embed[:, 0],
                proto_embed[:, 1],
                c=proto_labels,
                s=55,
                marker="X",
                alpha=0.90,
                cmap=cmap,
                vmin=-0.5,
                vmax=num_classes - 0.5,
                edgecolors="#1E2228",
                linewidths=0.45,
            )
            for class_id in range(num_classes):
                class_indices = np.flatnonzero(proto_labels == class_id)
                if class_indices.size == 0:
                    continue
                center = proto_embed[class_indices].mean(axis=0)
                ax.text(
                    float(center[0]),
                    float(center[1]),
                    str(class_id),
                    fontsize=7,
                    weight="bold",
                    ha="center",
                    va="center",
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
                )
            ax.set_title(MODE_LABELS.get(mode, mode), fontsize=10, fontweight="bold")
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#D6DAE0")
        fig.suptitle(
            f"{dataset} / {model_name}: joint full-scope prototype t-SNE",
            fontsize=14,
            fontweight="bold",
        )
        if scatter is not None:
            colorbar = fig.colorbar(scatter, ax=axes.tolist(), fraction=0.018, pad=0.012)
            colorbar.set_label("True diagnostic class")
            colorbar.set_ticks(np.arange(num_classes))
        stem = f"{dataset}_{model_name}_joint_full_scope_tsne"
        fig.savefig(tsne_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(tsne_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "lamp_merge_proto_count_power": float(args.proto_count_power),
        "lamp_merge_ablation_seed": int(args.ablation_seed),
    }
    manifest = filter_manifest(load_manifest(args.model_hub_root / "manifest.csv"), args)
    if not manifest:
        raise SystemExit("No matching manifest rows.")

    rows = []
    rows_by_key: dict[tuple[str, str, int, str], dict[str, object]] = {}
    tsne_records: list[dict[str, object]] = []
    meta_by_dataset_model: dict[tuple[str, str], dict[str, object]] = {}
    tsne_dataset_set = set(args.tsne_datasets)
    full_tsne_modes = set(args.full_tsne_modes)
    for row in manifest:
        meta = load_json(args.model_hub_root / row["meta_path"])
        meta_by_dataset_model.setdefault((row["dataset"], row["model"]), meta)
        proto_path = prototype_stats_path(args, row)
        if not proto_path.exists():
            print(f"[WARN] missing prototype stats: {proto_path}", file=sys.stderr)
            continue
        proto_stats = torch.load(proto_path, map_location="cpu")
        validate_lamp_merge_stats_payload(proto_stats, proto_path)
        state_dicts = None
        needs_head_agg = "prototype_head_agg" in args.modes
        needs_tsne_cache = (
            row["dataset"] in tsne_dataset_set
            and row["model"] == args.tsne_model
            and int(row["num_clients"]) == int(args.tsne_num_clients)
            and beta_key(row["beta"]) == "0.01"
        )
        if needs_head_agg:
            state_dicts = load_state_dicts(args, row, meta)
        if needs_tsne_cache:
            rows_by_key[(row["dataset"], row["model"], int(row["num_clients"]), beta_key(row["beta"]))] = {
                "meta": meta,
                "proto_stats": proto_stats,
                "state_dicts": state_dicts,
                "cfg": cfg,
            }
        for mode in args.modes:
            mode_state_dicts = state_dicts if mode == "prototype_head_agg" else None
            try:
                geometry = build_mode_geometry(mode, proto_stats, meta, cfg, mode_state_dicts)
                metrics = prototype_metrics(geometry)
                out = {
                    "task_type": "small",
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "num_clients": int(row["num_clients"]),
                    "beta": beta_key(row["beta"]),
                    "seed": int(row["seed"]),
                    "mode": mode,
                    "label": MODE_LABELS.get(mode, mode),
                    "prototype_mode": geometry["prototype_mode"],
                    "evidence_mode": geometry["evidence_mode"],
                }
                out.update(metrics)
                rows.append(out)
                if args.plot_full_tsne and mode in full_tsne_modes:
                    prototypes = geometry["prototypes"]
                    assert isinstance(prototypes, torch.Tensor)
                    tsne_records.append(
                        {
                            "dataset": row["dataset"],
                            "model": row["model"],
                            "num_clients": int(row["num_clients"]),
                            "beta": beta_key(row["beta"]),
                            "mode": mode,
                            "num_classes": int(meta["num_classes"]),
                            "prototypes": prototypes.detach().cpu().float().numpy(),
                        }
                    )
            except Exception as exc:
                rows.append(
                    {
                        "task_type": "small",
                        "dataset": row["dataset"],
                        "model": row["model"],
                        "num_clients": int(row["num_clients"]),
                        "beta": beta_key(row["beta"]),
                        "seed": int(row["seed"]),
                        "mode": mode,
                        "label": MODE_LABELS.get(mode, mode),
                        "error": str(exc),
                    }
                )

    dataset_rows = group_rows(rows)
    overall_rows = group_overall_rows(rows)
    write_csv(args.output_csv, rows)
    write_csv(args.dataset_csv, dataset_rows)
    write_csv(args.overall_csv, overall_rows)
    write_summary(args.summary_md, rows, dataset_rows, overall_rows, args)
    plot_geometry_bars(dataset_rows, overall_rows, args.figure_dir)
    if args.plot_tsne:
        plot_tsne(args, rows_by_key)
    if args.plot_full_tsne:
        plot_full_scope_tsne(args, tsne_records, meta_by_dataset_model)
    print(f"Wrote prototype geometry CSV: {args.output_csv}")
    print(f"Wrote prototype geometry summary: {args.summary_md}")
    print(f"Wrote prototype geometry figures: {args.figure_dir}")


if __name__ == "__main__":
    main()
