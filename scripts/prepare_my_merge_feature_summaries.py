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

from methods.my_merge import _collect_feature_summary, _feature_summary_path
from utils import load_json


def parse_args():
    p = argparse.ArgumentParser("Prepare five-dimensional my_merge medical feature summaries.")
    p.add_argument("--model-hub-root", default=str(ROOT / "model_hub"))
    p.add_argument("--data-root", default=str(ROOT / "Med_data"))
    p.add_argument("--output-root", default=str(ROOT / "model_hub" / "my_merge_feature_summaries"))
    p.add_argument("--manifest", default="")
    p.add_argument("--task-type", choices=["all", "small", "vlm"], default="all")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--small-models", nargs="*", default=None)
    p.add_argument("--clip-models", nargs="*", default=None)
    p.add_argument("--stats-split", default="val")
    p.add_argument("--stats-batch-size", type=int, default=32)
    p.add_argument("--stats-num-workers", type=int, default=0)
    p.add_argument("--my-merge-stats-max-batches", type=int, default=16)
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
    return value


def main():
    args = parse_args()
    manifest_path = Path(args.manifest) if args.manifest else Path(args.model_hub_root) / "manifest.csv"
    rows = [row for row in load_manifest(manifest_path) if keep(row, args)]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = {
        "data_root": args.data_root,
        "stats_split": args.stats_split,
        "stats_batch_size": args.stats_batch_size,
        "stats_num_workers": args.stats_num_workers,
        "my_merge_stats_max_batches": args.my_merge_stats_max_batches,
        "my_merge_feature_summary_root": str(out_root),
    }
    written = 0
    for row in rows:
        meta = load_json(Path(args.model_hub_root) / row["meta_path"])
        payload = _collect_feature_summary(meta, cfg)
        payload = {key: tensor_to_list(value) for key, value in payload.items()}
        payload.update({
            "task_type": meta.get("task_type"),
            "dataset": meta.get("dataset"),
            "model": meta.get("model"),
            "clip_model": meta.get("clip_model"),
            "num_clients": int(meta.get("num_clients", 0)),
            "beta": meta.get("beta"),
            "seed": int(meta.get("seed", 0)),
            "feature_names": ["boundary", "contrast", "texture", "salience", "reliability"],
            "privacy_note": "Only five-dimensional summary statistics are exported; no raw images or labels are stored.",
        })
        path = _feature_summary_path(meta, cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        written += 1
        print(f"[feature-summary] {written}/{len(rows)} {path}")
    print(f"[feature-summary] done | written={written} | output_root={out_root}")


if __name__ == "__main__":
    main()
