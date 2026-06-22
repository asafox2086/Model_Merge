#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.my_merge import FEATURE_NAMES, _feature_summary_path
from utils import ensure_checkpoint_files, extract_state_dict, find_hub_experiment_dir, load_checkpoint, load_json


def parse_args():
    p = argparse.ArgumentParser("Prepare my_merge BN moment summaries.")
    p.add_argument("--model-hub-root", default=str(ROOT / "model_hub"))
    p.add_argument("--data-root", default=str(ROOT / "Med_data"), help="Kept for CLI compatibility; BN-only summaries do not read data.")
    p.add_argument("--output-root", default=str(ROOT / "model_hub" / "my_merge_feature_summaries"))
    p.add_argument("--manifest", default="")
    p.add_argument("--task-type", choices=["all", "small", "vlm"], default="all")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--small-models", nargs="*", default=None)
    p.add_argument("--clip-models", nargs="*", default=None)
    p.add_argument("--stats-split", default="val", help="Kept for CLI compatibility; ignored in BN-only mode.")
    p.add_argument("--stats-batch-size", type=int, default=32, help="Kept for CLI compatibility; ignored in BN-only mode.")
    p.add_argument("--stats-num-workers", type=int, default=0, help="Kept for CLI compatibility; ignored in BN-only mode.")
    p.add_argument("--device", default="cpu", help="Kept for CLI compatibility; checkpoints are read on CPU.")
    p.add_argument("--my-merge-stats-max-batches", type=int, default=0, help="Kept for CLI compatibility; ignored in BN-only mode.")
    p.add_argument("--include-client-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--diagnostic-max-batches", type=int, default=0, help="Kept for CLI compatibility; ignored in BN-only mode.")
    p.add_argument("--include-bn-moments", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-class-metadata", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def load_manifest(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: tensor_to_list(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_to_list(item) for item in value]
    if isinstance(value, tuple):
        return [tensor_to_list(item) for item in value]
    return value


def meta_to_config(meta):
    cfg = {
        "task_type": meta["task_type"],
        "dataset": meta["dataset"],
        "num_clients": int(meta["num_clients"]),
        "beta": float(meta["beta"]),
        "seed": int(meta["seed"]),
    }
    if meta["task_type"] == "small":
        cfg["model"] = meta["model"]
    else:
        cfg["clip_model"] = meta.get("clip_model", meta.get("model", ""))
    return cfg


def bn_pairs(state_dicts):
    if not state_dicts:
        return []
    common = set(state_dicts[0].keys())
    for state in state_dicts[1:]:
        common &= set(state.keys())
    pairs = []
    for key in state_dicts[0].keys():
        if not key.endswith("running_mean"):
            continue
        var_key = key[: -len("running_mean")] + "running_var"
        num_key = key[: -len("running_mean")] + "num_batches_tracked"
        if key not in common or var_key not in common:
            continue
        shape = tuple(state_dicts[0][key].shape)
        if any(tuple(state[key].shape) != shape or tuple(state[var_key].shape) != shape for state in state_dicts):
            continue
        pairs.append((key, var_key, num_key))
    return pairs


def collect_bn_summary(meta, args):
    num_clients = int(meta.get("num_clients", len(meta.get("clients", []))))
    exp_dir = find_hub_experiment_dir(args.model_hub_root, meta_to_config(meta))
    ckpt_paths = ensure_checkpoint_files(exp_dir, meta)
    state_dicts = []
    for ckpt_path in ckpt_paths[:num_clients]:
        checkpoint = load_checkpoint(ckpt_path, device="cpu")
        state_dicts.append(extract_state_dict(checkpoint))

    payload = {
        "source": "client_uploaded_bn_moment_summary_simulation",
        "feature_names": FEATURE_NAMES,
        "bn_layer_count": 0,
        "bn_layers": [],
    }
    if args.include_bn_moments:
        means = {}
        variances = {}
        num_batches = {}
        for mean_key, var_key, num_key in bn_pairs(state_dicts):
            means[mean_key] = torch.stack([state[mean_key].detach().cpu() for state in state_dicts], dim=0)
            variances[var_key] = torch.stack([state[var_key].detach().cpu() for state in state_dicts], dim=0)
            if all(num_key in state for state in state_dicts):
                num_batches[num_key] = torch.stack([state[num_key].detach().cpu() for state in state_dicts], dim=0)
        payload.update(
            {
                "bn_layer_count": len(means),
                "bn_layers": list(means.keys()),
                "client_bn_running_mean": means,
                "client_bn_running_var": variances,
                "client_bn_num_batches_tracked": num_batches,
            }
        )

    if args.include_class_metadata:
        num_classes = int(meta.get("num_classes", 0))
        class_counts = torch.zeros(num_classes, dtype=torch.float32)
        client_classes = []
        client_num_samples = []
        for client in meta.get("clients", [])[:num_clients]:
            seen = [int(x) for x in client.get("classes", []) if 0 <= int(x) < num_classes]
            samples = float(client.get("num_samples", 0))
            client_classes.append(seen)
            client_num_samples.append(samples)
            if seen:
                per_class = samples / float(len(seen))
                for cls_idx in seen:
                    class_counts[cls_idx] += per_class
        payload.update(
            {
                "client_classes": client_classes,
                "client_num_samples": client_num_samples,
                "class_counts_from_metadata": class_counts,
            }
        )

    return payload


def main():
    args = parse_args()
    manifest_path = Path(args.manifest) if args.manifest else Path(args.model_hub_root) / "manifest.csv"
    rows = [row for row in load_manifest(manifest_path) if keep(row, args)]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = {
        "my_merge_feature_summary_root": str(out_root),
    }
    written = 0
    for row in rows:
        meta = load_json(Path(args.model_hub_root) / row["meta_path"])
        payload = collect_bn_summary(meta, args)
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
                "privacy_note": (
                    "Only aggregate BN running moments and task metadata are exported; no raw images, "
                    "morphology features, per-sample logits, per-sample activations, or candidate feedback are stored."
                ),
            }
        )
        path = _feature_summary_path(meta, cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        written += 1
        print(f"[bn-summary] {written}/{len(rows)} {path} | bn_layers={payload.get('bn_layer_count', 0)}")
    print(f"[bn-summary] done | written={written} | output_root={out_root}")


if __name__ == "__main__":
    main()
