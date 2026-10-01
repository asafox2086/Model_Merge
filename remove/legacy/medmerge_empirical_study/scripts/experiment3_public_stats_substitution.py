#!/usr/bin/env python3
"""Test whether same-type public medical data can replace source validation images.

The key control is fixed checkpoints and fixed source test set. Only the image
statistics / calibration data source changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
import torch


STATS_METHODS = ["fisher", "regmean", "adamerging"]
CALIBRATION_SOURCES = ["source_val_512", "public_labeled_val_512", "public_labeled_reweighted_512"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 3: source validation vs same-type public statistics.")
    parser.add_argument("--repo-root", type=Path, default=Path("program/MedMNISTMerge"))
    parser.add_argument("--model-hub-root", type=Path, default=None)
    parser.add_argument("--source-data-root", type=Path, default=None)
    parser.add_argument("--public-labeled-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("medmerge_empirical_study/results/experiment3"))
    parser.add_argument("--task-type", choices=["small"], default="small")
    parser.add_argument("--datasets", nargs="*", default=["dermamnist"])
    parser.add_argument("--small-models", nargs="*", default=["resnet"])
    parser.add_argument("--num-clients", nargs="*", type=int, default=[3])
    parser.add_argument("--betas", nargs="*", type=float, default=[0.0, 0.01, 0.1])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--methods", nargs="*", default=STATS_METHODS)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stats-batch-size", type=int, default=256)
    parser.add_argument("--stats-num-workers", type=int, default=2)
    parser.add_argument("--adamerging-epochs", type=int, default=50)
    parser.add_argument("--adamerging-max-batches", type=int, default=1)
    parser.add_argument("--merge-weight-mode", choices=["equal", "sample"], default="equal")
    parser.add_argument("--delete-merged", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def setup_repo_imports(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_LOCAL_FILES_ONLY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_dataset_names(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    out = set()
    for value in values:
        out.add(value)
        if not value.endswith("_224"):
            out.add(f"{value}_224")
    return out


def filter_manifest(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = [row for row in rows if row["task_type"] == args.task_type]
    datasets = normalize_dataset_names(args.datasets)
    if datasets is not None:
        selected = [row for row in selected if row["dataset"] in datasets]
    if args.small_models:
        models = set(args.small_models)
        selected = [row for row in selected if row["model"] in models]
    if args.num_clients:
        clients = set(args.num_clients)
        selected = [row for row in selected if int(row["num_clients"]) in clients]
    if args.betas:
        betas = {format(float(item), "g") for item in args.betas}
        selected = [row for row in selected if format(float(row["beta"]), "g") in betas]
    if args.seeds:
        seeds = set(args.seeds)
        selected = [row for row in selected if int(row["seed"]) in seeds]
    selected.sort(key=lambda row: (row["dataset"], row["model"], int(row["num_clients"]), float(row["beta"]), int(row["seed"])))
    return selected


def empty_images_like(images: np.ndarray) -> np.ndarray:
    return np.empty((0,) + tuple(images.shape[1:]), dtype=images.dtype)


def empty_labels_like(labels: np.ndarray) -> np.ndarray:
    return np.empty((0,), dtype=labels.dtype)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    flat = labels.reshape(-1).astype(int)
    values, counts = np.unique(flat, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def stratified_indices(labels: np.ndarray, total: int, rng: np.random.Generator, *, replace_shortage: bool = False) -> np.ndarray:
    flat = labels.reshape(-1).astype(int)
    classes, counts = np.unique(flat, return_counts=True)
    proportions = counts.astype(float) / float(counts.sum())
    raw = proportions * int(total)
    targets = np.floor(raw).astype(int)
    remainder = int(total) - int(targets.sum())
    if remainder > 0:
        order = np.argsort(-(raw - targets))
        for idx in order[:remainder]:
            targets[idx] += 1

    selected = []
    for cls, target in zip(classes, targets):
        candidates = np.where(flat == cls)[0]
        replace = bool(replace_shortage and target > candidates.size)
        if target > candidates.size and not replace:
            target = candidates.size
        if target > 0:
            selected.append(rng.choice(candidates, size=int(target), replace=replace))
    if not selected:
        raise ValueError("No calibration samples selected")
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out.astype(int)


def reweighted_public_indices(public_labels: np.ndarray, source_labels: np.ndarray, total: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    source_flat = source_labels.reshape(-1).astype(int)
    public_flat = public_labels.reshape(-1).astype(int)
    classes, counts = np.unique(source_flat, return_counts=True)
    proportions = counts.astype(float) / float(counts.sum())
    raw = proportions * int(total)
    targets = np.floor(raw).astype(int)
    remainder = int(total) - int(targets.sum())
    if remainder > 0:
        order = np.argsort(-(raw - targets))
        for idx in order[:remainder]:
            targets[idx] += 1

    selected = []
    used_replacement = False
    for cls, target in zip(classes, targets):
        candidates = np.where(public_flat == cls)[0]
        if candidates.size == 0:
            continue
        replace = int(target) > int(candidates.size)
        used_replacement = used_replacement or replace
        selected.append(rng.choice(candidates, size=int(target), replace=replace))
    if not selected:
        raise ValueError("No public samples selected for reweighting")
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out.astype(int), used_replacement


def save_calibration_npz(root: Path, dataset: str, images: np.ndarray, labels: np.ndarray) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / f"{dataset}.npz"
    empty_images = empty_images_like(images)
    empty_labels = empty_labels_like(labels)
    np.savez_compressed(
        out_path,
        train_images=empty_images,
        train_labels=empty_labels,
        val_images=images,
        val_labels=labels.reshape(-1).astype(np.int64),
        test_images=empty_images,
        test_labels=empty_labels,
    )
    return out_path


def prepare_calibration_roots(args: argparse.Namespace, datasets: list[str]) -> dict[str, Path]:
    rng = np.random.default_rng(42)
    roots = {
        "source_val_512": args.out_dir / "calibration_roots" / "source_val_512",
        "public_labeled_val_512": args.out_dir / "calibration_roots" / "public_labeled_val_512",
        "public_labeled_reweighted_512": args.out_dir / "calibration_roots" / "public_labeled_reweighted_512",
    }
    manifest = []
    for dataset in sorted(set(datasets)):
        source_npz = np.load(args.source_data_root / f"{dataset}.npz", allow_pickle=False)
        public_npz = np.load(args.public_labeled_root / f"{dataset}.npz", allow_pickle=False)

        source_val_images = source_npz["val_images"]
        source_val_labels = source_npz["val_labels"].reshape(-1).astype(np.int64)
        public_val_images = public_npz["val_images"]
        public_val_labels = public_npz["val_labels"].reshape(-1).astype(np.int64)

        source_idx = stratified_indices(source_val_labels, args.sample_count, rng, replace_shortage=False)
        public_idx = stratified_indices(public_val_labels, min(args.sample_count, len(public_val_labels)), rng, replace_shortage=False)
        public_reweighted_idx, replacement = reweighted_public_indices(public_val_labels, source_val_labels, args.sample_count, rng)

        specs = [
            ("source_val_512", source_val_images[source_idx], source_val_labels[source_idx], False),
            ("public_labeled_val_512", public_val_images[public_idx], public_val_labels[public_idx], False),
            ("public_labeled_reweighted_512", public_val_images[public_reweighted_idx], public_val_labels[public_reweighted_idx], replacement),
        ]
        for source_name, images, labels, used_replacement in specs:
            path = save_calibration_npz(roots[source_name], dataset, images, labels)
            manifest.append(
                {
                    "dataset": dataset,
                    "calibration_source": source_name,
                    "path": str(path),
                    "num_samples": int(labels.shape[0]),
                    "class_counts": class_counts(labels),
                    "used_replacement": bool(used_replacement),
                }
            )
    manifest_path = args.out_dir / "calibration_roots" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return roots


def base_cfg_from_manifest_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    return {
        "task_type": row["task_type"],
        "dataset": row["dataset"],
        "model": row["model"],
        "clip_model": row.get("clip_model", ""),
        "num_clients": int(row["num_clients"]),
        "beta": float(row["beta"]),
        "seed": int(row["seed"]),
        "model_hub_root": str(args.model_hub_root),
        "device": args.device,
        "stats_device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "stats_split": "val",
        "stats_batch_size": args.stats_batch_size,
        "stats_num_workers": args.stats_num_workers,
        "merge_weight_mode": args.merge_weight_mode,
        "adamerging_epochs": args.adamerging_epochs,
        "adamerging_max_batches": args.adamerging_max_batches,
        "amp": str(args.device).startswith("cuda"),
    }


def evaluate_checkpoint(meta: dict[str, object], checkpoint: dict[str, object], cfg: dict[str, object]) -> dict[str, object]:
    from collect_prediction_metrics import evaluate_checkpoint_predictions

    eval_cfg = dict(cfg)
    eval_cfg["data_root"] = str(cfg["source_data_root"])
    eval_cfg["split"] = cfg.get("split", "test")
    return evaluate_checkpoint_predictions(meta, checkpoint, eval_cfg)


def reset_and_recalibrate_bn(meta: dict[str, object], checkpoint: dict[str, object], calibration_root: Path, cfg: dict[str, object]) -> dict[str, object]:
    from utils.runtime import build_runtime

    device = torch.device(str(cfg.get("device", "cuda:0")))
    runtime = build_runtime(
        meta=meta,
        data_root=str(calibration_root),
        split="val",
        batch_size=int(cfg.get("stats_batch_size", cfg.get("batch_size", 128))),
        num_workers=int(cfg.get("stats_num_workers", 0)),
        device=device,
    )
    model = runtime["model"]
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    for param in model.parameters():
        param.requires_grad_(False)
    bn_layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            module.momentum = None
            bn_layers.append(module)
    if not bn_layers:
        raise RuntimeError("No BatchNorm layers found for BN recalibration")
    model.eval()
    for module in bn_layers:
        module.train()
    forward_fn = runtime["forward_fn"]
    with torch.no_grad():
        for x, _ in runtime["loader"]:
            x = x.to(device, non_blocking=True)
            _ = forward_fn(model, x)
    return {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "meta": checkpoint.get("meta", {})}


def append_csv(path: Path, row: dict[str, object], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def make_result_row(base: dict[str, object], method: str, calibration_source: str, metrics: dict[str, object], seconds: float) -> dict[str, object]:
    row = dict(base)
    row.update(metrics)
    row.update(
        {
            "status": "OK",
            "error": "",
            "method": method,
            "calibration_source": calibration_source,
            "seconds": f"{seconds:.2f}",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return row


def make_fail_row(base: dict[str, object], method: str, calibration_source: str, error: Exception, seconds: float) -> dict[str, object]:
    row = dict(base)
    row.update(
        {
            "status": "FAIL",
            "error": str(error),
            "method": method,
            "calibration_source": calibration_source,
            "seconds": f"{seconds:.2f}",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return row


def write_summary(metrics_csv: Path, out_path: Path) -> None:
    if not metrics_csv.exists():
        return
    with metrics_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "OK"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["calibration_source"])].append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# 实验三：同类型公开数据替代源域验证集\n\n")
        handle.write("## 工作流核对\n\n")
        handle.write("对应 `experiment_workflow.md` 第 6 和第 7 节：固定客户端 checkpoint 与源域 test set，只替换 Fisher/RegMean/AdaMerging/BN recalibration 的统计或校准图像来源。\n\n")
        handle.write("## 结果摘要\n\n")
        handle.write("| method | calibration source | cases | mean acc | mean balanced acc | mean macro F1 | mean collapse ratio | median effective classes |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for (method, source), items in sorted(grouped.items()):
            eff = sorted(float(row["effective_predicted_classes"]) for row in items)
            median_eff = eff[len(eff) // 2]
            handle.write(
                f"| {method} | {source} | {len(items)} | "
                f"{mean(float(row['accuracy']) for row in items):.4f} | "
                f"{mean(float(row['balanced_accuracy']) for row in items):.4f} | "
                f"{mean(float(row['macro_f1']) for row in items):.4f} | "
                f"{mean(float(row['majority_prediction_ratio']) for row in items):.4f} | "
                f"{median_eff:.2f} |\n"
            )
        handle.write("\n## 结论读取方式\n\n")
        handle.write("重点比较同一 method 下 `source_val_512`、`public_labeled_val_512`、`public_labeled_reweighted_512` 与 `no_image_stats`。如果 source validation 明显高于 public，说明同类型公开医学数据不能稳定替代源域图像统计。\n")


def main() -> None:
    args = parse_args()
    setup_repo_imports(args.repo_root)

    from merge import METHOD_DEFAULTS, run_merge
    from utils import load_checkpoint, load_json

    args.model_hub_root = args.model_hub_root or args.repo_root / "model_hub"
    args.source_data_root = args.source_data_root or args.repo_root / "Med_data"
    args.public_labeled_root = args.public_labeled_root or args.repo_root / "PublicMedLabeled_data"

    manifest = filter_manifest(load_manifest(args.model_hub_root / "manifest.csv"), args)
    if not manifest:
        raise SystemExit("No manifest rows matched the requested filters.")

    calibration_roots = prepare_calibration_roots(args, [row["dataset"] for row in manifest])
    metrics_csv = args.out_dir / "public_stats_substitution_metrics.csv"
    if metrics_csv.exists():
        metrics_csv.unlink()

    fields = [
        "status",
        "error",
        "seconds",
        "task_type",
        "dataset",
        "model",
        "clip_model",
        "num_clients",
        "beta",
        "seed",
        "split",
        "method",
        "calibration_source",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mean_precision",
        "loss",
        "num_samples",
        "majority_prediction_ratio",
        "effective_predicted_classes",
        "prediction_entropy",
        "per_class_recall",
        "per_class_precision",
        "per_class_f1",
        "class_support",
        "predicted_class_counts",
        "confusion_matrix_json",
        "updated_at",
    ]

    for row in manifest:
        base_cfg = base_cfg_from_manifest_row(row, args)
        base_eval = {
            "task_type": row["task_type"],
            "dataset": row["dataset"],
            "model": row["model"],
            "clip_model": row.get("clip_model", ""),
            "num_clients": int(row["num_clients"]),
            "beta": format(float(row["beta"]), "g"),
            "seed": int(row["seed"]),
            "split": args.split,
        }

        avg_checkpoint = None
        avg_meta = None
        avg_cfg = dict(METHOD_DEFAULTS)
        avg_cfg.update(base_cfg)
        avg_cfg.update(
            {
                "method": "avg",
                "data_root": str(args.source_data_root),
                "source_data_root": str(args.source_data_root),
                "output_root": str(args.out_dir / "merge_runs" / "no_image_stats"),
                "split": args.split,
            }
        )
        start = time.time()
        try:
            print(f"merge+eval no_image_stats avg {base_eval}")
            avg_info = run_merge(avg_cfg)
            avg_meta = load_json(Path(avg_info["meta_path"]))
            avg_checkpoint = load_checkpoint(Path(avg_info["merged_checkpoint"]), device="cpu")
            metrics = evaluate_checkpoint(avg_meta, avg_checkpoint, avg_cfg)
            append_csv(metrics_csv, make_result_row(base_eval, "avg", "no_image_stats", metrics, time.time() - start), fields)
            append_csv(metrics_csv, make_result_row(base_eval, "bn_recalibration", "no_recalibration", metrics, time.time() - start), fields)
        except Exception as exc:
            append_csv(metrics_csv, make_fail_row(base_eval, "avg", "no_image_stats", exc, time.time() - start), fields)

        for method in args.methods:
            for source_name in CALIBRATION_SOURCES:
                cfg = dict(METHOD_DEFAULTS)
                cfg.update(base_cfg)
                cfg.update(
                    {
                        "method": method,
                        "data_root": str(calibration_roots[source_name]),
                        "source_data_root": str(args.source_data_root),
                        "output_root": str(args.out_dir / "merge_runs" / source_name),
                        "split": args.split,
                    }
                )
                start = time.time()
                try:
                    print(f"merge+eval {method} stats={source_name} {base_eval}")
                    merge_info = run_merge(cfg)
                    meta = load_json(Path(merge_info["meta_path"]))
                    checkpoint_path = Path(merge_info["merged_checkpoint"])
                    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
                    metrics = evaluate_checkpoint(meta, checkpoint, cfg)
                    append_csv(metrics_csv, make_result_row(base_eval, method, source_name, metrics, time.time() - start), fields)
                    if args.delete_merged and checkpoint_path.exists():
                        checkpoint_path.unlink()
                except Exception as exc:
                    append_csv(metrics_csv, make_fail_row(base_eval, method, source_name, exc, time.time() - start), fields)
                    print(f"FAIL {method} stats={source_name}: {exc}")

        if avg_checkpoint is not None and avg_meta is not None:
            for source_name in CALIBRATION_SOURCES:
                start = time.time()
                try:
                    print(f"bn_recalibration stats={source_name} {base_eval}")
                    recalibrated = reset_and_recalibrate_bn(avg_meta, avg_checkpoint, calibration_roots[source_name], avg_cfg)
                    metrics = evaluate_checkpoint(avg_meta, recalibrated, avg_cfg)
                    append_csv(metrics_csv, make_result_row(base_eval, "bn_recalibration", source_name, metrics, time.time() - start), fields)
                except Exception as exc:
                    append_csv(metrics_csv, make_fail_row(base_eval, "bn_recalibration", source_name, exc, time.time() - start), fields)
                    print(f"FAIL bn_recalibration stats={source_name}: {exc}")

    write_summary(metrics_csv, args.out_dir / "experiment3_public_stats_summary.md")
    print(f"Wrote experiment 3 metrics to {metrics_csv}")


if __name__ == "__main__":
    main()
