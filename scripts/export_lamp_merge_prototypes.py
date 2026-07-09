#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import NpzTensorDataset, load_npz_splits
from model import build_model
from utils import load_json
from utils.hub import beta_to_dirname
from utils.lamp_merge_stats import default_prototype_root
from utils.runtime import build_reference_bundle


def parse_args():
    p = argparse.ArgumentParser("Export client-side class prototype statistics for LAMP-Merge")
    p.add_argument("--model-hub-root", type=str, default=str(ROOT / "model_hub"))
    p.add_argument("--data-root", type=str, default=str(ROOT / "Med_data"))
    p.add_argument("--output-root", type=str, default=str(default_prototype_root(ROOT)))
    p.add_argument(
        "--partition-root",
        type=str,
        default="",
        help=(
            "Optional root containing true client partition artifacts. The exporter looks for "
            "client_indices.json/npz or client_<id>_indices.npy/json under the same relative path "
            "as each meta.json. If omitted, indices must be present in meta.json/client entries or "
            "the client entries must already contain uploaded aggregate statistics."
        ),
    )
    p.add_argument("--task-type", choices=["small"], default="small")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--small-models", nargs="*", default=None)
    p.add_argument("--num-clients", nargs="*", type=int, default=None)
    p.add_argument("--betas", nargs="*", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "val", "trainval"], default="train")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--max-samples-per-client",
        type=int,
        default=0,
        help="Optional client-side feature budget. 0 means use the full local client dataset.",
    )
    p.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help="Optional per-class feature budget inside one client. 0 means no class cap.",
    )
    return p.parse_args()


def image_hw(images):
    if images.ndim == 3:
        return int(images.shape[1]), int(images.shape[2])
    if images.ndim >= 4:
        return int(images.shape[1]), int(images.shape[2])
    raise ValueError(f"Unexpected image shape: {images.shape}")


def build_transform(meta, images):
    source_h, source_w = image_hw(images)
    target = int(meta.get("image_size") or 0)
    if target > 0 and (source_h != target or source_w != target):
        return transforms.Resize((target, target), antialias=True)
    return None


def iter_meta_paths(args):
    root = Path(args.model_hub_root) / "small"
    datasets = args.datasets or sorted(path.name for path in root.iterdir() if path.is_dir())
    for dataset in datasets:
        ds_dir = root / dataset
        if not ds_dir.exists():
            continue
        models = args.small_models or sorted(path.name for path in ds_dir.iterdir() if path.is_dir())
        for model in models:
            model_dir = ds_dir / model
            if not model_dir.exists():
                continue
            client_dirs = [f"clients_{n}" for n in args.num_clients] if args.num_clients else sorted(path.name for path in model_dir.iterdir() if path.is_dir())
            for client_dir in client_dirs:
                cdir = model_dir / client_dir
                if not cdir.exists():
                    continue
                beta_dirs = [beta_to_dirname(beta) for beta in args.betas] if args.betas else sorted(path.name for path in cdir.iterdir() if path.is_dir())
                for beta_dir in beta_dirs:
                    meta_path = cdir / beta_dir / f"seed_{int(args.seed)}" / "meta.json"
                    if meta_path.exists():
                        yield meta_path


def load_split_arrays(data_root, dataset, split):
    splits = load_npz_splits(str(Path(data_root) / f"{dataset}.npz"))
    if split == "trainval":
        images = np.concatenate([splits["train"].images, splits["val"].images], axis=0)
        labels = np.concatenate([splits["train"].labels, splits["val"].labels], axis=0)
        return images, labels
    return splits[split].images, splits[split].labels


def output_path(output_root, meta):
    return (
        Path(output_root)
        / str(meta["task_type"])
        / str(meta["dataset"])
        / str(meta["model"])
        / f"clients_{int(meta['num_clients'])}"
        / beta_to_dirname(meta["beta"])
        / f"seed_{int(meta['seed'])}"
        / "prototype_stats.pt"
    )


def count_vector(value, num_classes, *, field_name):
    out = np.zeros(num_classes, dtype=np.int64)
    if value is None:
        return None
    if isinstance(value, dict):
        for cls_key, count in value.items():
            cls = int(cls_key)
            if 0 <= cls < num_classes:
                out[cls] = int(count)
        return out
    arr = np.asarray(value).reshape(-1)
    if arr.size != num_classes:
        raise ValueError(f"{field_name} must contain {num_classes} values, got {arr.size}.")
    return arr.astype(np.int64)


def reported_prevalence_counts(client, num_classes):
    for field in (
        "class_prevalence_counts",
        "class_label_counts",
        "label_counts",
        "prevalence_counts",
        "class_counts",
    ):
        if field in client and client.get(field) is not None:
            return count_vector(client.get(field), num_classes, field_name=field), field
    return None, ""


def uploaded_feature_stats(client, num_classes):
    means = client.get("class_feature_mean")
    if means is None:
        return None, None
    counts = None
    for field in ("class_feature_counts", "prototype_counts", "class_counts"):
        if field in client and client.get(field) is not None:
            counts = count_vector(client.get(field), num_classes, field_name=field)
            break
    if counts is None:
        raise ValueError("class_feature_mean was provided without class_feature_counts.")
    means_tensor = torch.as_tensor(means, dtype=torch.float32)
    if means_tensor.ndim != 2 or int(means_tensor.shape[0]) != int(num_classes):
        raise ValueError(
            f"class_feature_mean must have shape [{num_classes}, feature_dim], got {list(means_tensor.shape)}."
        )
    return means_tensor, counts


def _indices_from_value(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _load_indices_json(path, client_id):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return _indices_from_value(data)
    for key in (str(client_id), f"client_{client_id}", f"client{client_id}"):
        if isinstance(data, dict) and key in data:
            return _indices_from_value(data[key])
    return None


def _load_indices_npz(path, client_id):
    data = np.load(path, allow_pickle=False)
    for key in (str(client_id), f"client_{client_id}", f"client{client_id}"):
        if key in data:
            return _indices_from_value(data[key])
    return None


def _partition_candidates(meta_path, args, client_id):
    local_dir = meta_path.parent
    roots = [local_dir]
    if args.partition_root:
        rel_dir = meta_path.parent.relative_to(Path(args.model_hub_root))
        roots.insert(0, Path(args.partition_root) / rel_dir)
    for root in roots:
        yield root / f"client_{client_id}_indices.npy"
        yield root / f"client_{client_id}_indices.json"
        yield root / f"client{client_id}_indices.npy"
        yield root / f"client{client_id}_indices.json"
        yield root / "client_indices.npz"
        yield root / "client_indices.json"


def load_client_indices(meta_path, args, client, client_idx, num_samples):
    for field in ("client_indices", "train_indices", "indices", "sample_indices"):
        if field in client and client.get(field) is not None:
            indices = _indices_from_value(client.get(field))
            source = f"meta.clients[{client_idx}].{field}"
            break
    else:
        indices = None
        source = ""
        client_id = int(client.get("client_id", client_idx))
        for path in _partition_candidates(meta_path, args, client_id):
            if not path.exists():
                continue
            if path.suffix == ".npy":
                indices = _indices_from_value(np.load(path, allow_pickle=False))
            elif path.suffix == ".npz":
                indices = _load_indices_npz(path, client_id)
            elif path.suffix == ".json":
                indices = _load_indices_json(path, client_id)
            if indices is not None:
                source = str(path)
                break

    if indices is None:
        return None, ""
    if indices.size <= 0:
        raise ValueError(f"Empty client indices for client {client_idx}: {meta_path}")
    if int(indices.min()) < 0 or int(indices.max()) >= int(num_samples):
        raise ValueError(
            f"Client indices out of range for client {client_idx}: min={int(indices.min())}, "
            f"max={int(indices.max())}, dataset_size={num_samples}."
        )
    return indices.astype(np.int64), source


def local_class_counts(labels, indices, num_classes):
    selected = np.asarray(labels).reshape(-1)[indices]
    return np.bincount(selected.astype(np.int64), minlength=num_classes).astype(np.int64)


def select_feature_indices(labels, local_indices, client_idx, max_samples, max_per_class):
    indices = np.asarray(local_indices, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(1009 + int(client_idx))
    labels = np.asarray(labels).reshape(-1)
    if max_per_class > 0:
        chosen = []
        for cls in sorted(np.unique(labels[indices]).astype(np.int64).tolist()):
            cls_indices = indices[labels[indices] == int(cls)]
            if cls_indices.size > max_per_class:
                cls_indices = rng.choice(cls_indices, size=max_per_class, replace=False)
            chosen.append(cls_indices)
        indices = np.concatenate(chosen).astype(np.int64) if chosen else np.array([], dtype=np.int64)
    if max_samples > 0 and indices.size > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False).astype(np.int64)
    rng.shuffle(indices)
    return indices


def extract_features(model, x):
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
        raise ValueError("Model does not expose forward_features/forward_head or get_classifier().")

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


def export_one(meta_path, args):
    meta = load_json(meta_path)
    if meta.get("task_type") != "small":
        return None
    out_path = output_path(args.output_root, meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images, labels = load_split_arrays(args.data_root, meta["dataset"], args.split)
    transform = build_transform(meta, images)
    dataset = NpzTensorDataset(images, labels, transform=transform)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    clients_payload = []

    reference_model, _, _, _ = build_model(
        name=meta["model"],
        num_classes=int(meta["num_classes"]),
        in_channels=int(meta.get("in_channels", 3)),
        pretrained=False,
    )
    reference_state, _ = build_reference_bundle(meta, device="cpu")
    reference_model.load_state_dict(reference_state, strict=True)
    reference_model.to(device)
    reference_model.eval()

    for client_idx, client in enumerate(meta.get("clients", [])):
        num_classes = int(meta["num_classes"])
        classes = [int(c) for c in client.get("classes", [])]
        reported_counts, reported_count_field = reported_prevalence_counts(client, num_classes)
        local_indices, local_index_source = load_client_indices(meta_path, args, client, client_idx, len(labels))
        uploaded_means, uploaded_feature_counts = uploaded_feature_stats(client, num_classes)

        if local_indices is not None:
            prevalence_counts = local_class_counts(labels, local_indices, num_classes)
            prevalence_source = "client_local_dataset:" + args.split
            if reported_counts is not None and not np.array_equal(reported_counts, prevalence_counts):
                raise ValueError(
                    f"Reported class counts do not match local D_i for client {client_idx}: {meta_path}. "
                    f"field={reported_count_field}"
                )
        elif reported_counts is not None:
            prevalence_counts = reported_counts
            prevalence_source = "client_uploaded_label_counts"
        else:
            raise ValueError(
                "LAMP-Merge requires each client to upload true class_prevalence_counts or "
                f"provide true local indices for D_i. Missing client {client_idx}: {meta_path}"
            )

        if uploaded_means is not None:
            means = uploaded_means
            counts = torch.as_tensor(uploaded_feature_counts, dtype=torch.float32)
            feature_source = "client_uploaded_reference_prototypes"
            num_selected_samples = int(counts.sum().item())
        else:
            if local_indices is None:
                raise ValueError(
                    "Reference prototype export requires true local D_i indices unless the client "
                    f"has already uploaded class_feature_mean. Missing client {client_idx}: {meta_path}"
                )
            indices = select_feature_indices(
                labels,
                local_indices,
                client_idx,
                max_samples=int(args.max_samples_per_client),
                max_per_class=int(args.max_samples_per_class),
            )
            loader = DataLoader(
                Subset(dataset, indices.tolist()),
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=str(device).startswith("cuda"),
            )
            sums = None
            counts = torch.zeros(num_classes, dtype=torch.float32)

            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    feat = extract_features(reference_model, x).detach().float().cpu()
                    y_cpu = y.detach().cpu()
                    if sums is None:
                        sums = torch.zeros(num_classes, feat.shape[1], dtype=torch.float32)
                    for cls in y_cpu.unique().tolist():
                        cls = int(cls)
                        mask = y_cpu == cls
                        sums[cls] += feat[mask].sum(dim=0)
                        counts[cls] += float(mask.sum().item())

            if sums is None:
                raise ValueError(f"No samples selected for client {client_idx}: {meta_path}")
            means = sums / counts.clamp_min(1.0).view(-1, 1)
            feature_source = "client_local_dataset:" + args.split
            num_selected_samples = int(counts.sum().item())

        if not classes:
            classes = np.flatnonzero(prevalence_counts > 0).astype(int).tolist()
        clients_payload.append(
            {
                "client_id": int(client.get("client_id", client_idx)),
                "classes": classes,
                "num_local_samples": int(prevalence_counts.sum()),
                "num_selected_samples": num_selected_samples,
                "class_counts": counts.tolist(),
                "class_feature_counts": counts.tolist(),
                "class_prevalence_counts": prevalence_counts.tolist(),
                "local_index_source": local_index_source,
                "feature_source": feature_source,
                "prevalence_source": prevalence_source,
                "class_feature_mean": means.tolist(),
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "format": "lamp_merge_client_prototype_stats_v2",
        "privacy": "client-side aggregate statistics only; no raw image and no per-sample feature is stored",
        "feature_space": "reference_model",
        "client_dataset_source": "client_local_dataset_or_uploaded_aggregates",
        "prevalence_source": "client_local_dataset_or_uploaded_label_counts",
        "feature_source": "client_local_dataset_or_uploaded_reference_prototypes",
        "source_split": args.split,
        "meta_path": str(meta_path),
        "task_type": meta["task_type"],
        "dataset": meta["dataset"],
        "model": meta["model"],
        "num_clients": int(meta["num_clients"]),
        "beta": float(meta["beta"]),
        "seed": int(meta["seed"]),
        "num_classes": int(meta["num_classes"]),
        "clients": clients_payload,
    }
    torch.save(payload, out_path)
    return out_path


def main():
    args = parse_args()
    paths = list(iter_meta_paths(args))
    print(f"prototype export start | tasks={len(paths)} | output_root={args.output_root}")
    done = 0
    for idx, meta_path in enumerate(paths, start=1):
        out_path = export_one(meta_path, args)
        if out_path is not None:
            done += 1
            print(f"done ({idx}/{len(paths)}) {meta_path} -> {out_path}")
    print(f"prototype export done | written={done}")


if __name__ == "__main__":
    main()
