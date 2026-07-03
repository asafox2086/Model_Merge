#!/usr/bin/env python3
"""Re-merge selected cases and collect prediction-level metrics.

The formal markdown tables only contain accuracy. This script recomputes
selected merged models and records balanced accuracy, macro F1, per-class
recall, and prediction-collapse diagnostics required by experiment 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
import torch.nn.functional as F


FORMAL_METHODS = [
    "avg",
    "ties",
    "dare_linear",
    "dare_ties",
    "regmean",
    "fisher",
    "breadcrumbs",
    "model_stock",
    "from",
    "iso_c",
    "free_merge",
    "robustmerge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect prediction-level metrics for selected merge cases.")
    parser.add_argument("--repo-root", type=Path, default=Path("program/MedMNISTMerge"))
    parser.add_argument("--model-hub-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--eval-data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("medmerge_empirical_study/results/experiment1/prediction_runs"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("medmerge_empirical_study/results/experiment1/prediction_metrics.csv"))
    parser.add_argument("--task-type", choices=["small", "vlm"], default="small")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--small-models", nargs="*", default=None)
    parser.add_argument("--clip-models", nargs="*", default=None)
    parser.add_argument("--num-clients", nargs="*", type=int, default=None)
    parser.add_argument("--betas", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--methods", nargs="*", default=FORMAL_METHODS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--small-batch-size", type=int, default=128)
    parser.add_argument("--vlm-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stats-batch-size", type=int, default=0)
    parser.add_argument("--stats-num-workers", type=int, default=0)
    parser.add_argument("--merge-weight-mode", choices=["sample", "equal"], default="equal")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--delete-merged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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
        if not value.endswith("_224") and value not in {"cifar10_32", "cifar100_32", "svhn_32", "tinyimagenet_64"}:
            out.add(f"{value}_224")
    return out


def filter_manifest(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = [row for row in rows if row["task_type"] == args.task_type]
    datasets = normalize_dataset_names(args.datasets)
    if datasets is not None:
        selected = [row for row in selected if row["dataset"] in datasets]
    if args.small_models:
        models = set(args.small_models)
        selected = [row for row in selected if row["task_type"] != "small" or row["model"] in models]
    if args.clip_models:
        clips = {item.split("/")[-1] for item in args.clip_models}
        selected = [row for row in selected if row["task_type"] != "vlm" or row["clip_model"].split("/")[-1] in clips]
    if args.num_clients:
        clients = set(args.num_clients)
        selected = [row for row in selected if int(row["num_clients"]) in clients]
    if args.betas:
        betas = {format(float(item), "g") for item in args.betas}
        selected = [row for row in selected if format(float(row["beta"]), "g") in betas]
    if args.seeds:
        seeds = set(args.seeds)
        selected = [row for row in selected if int(row["seed"]) in seeds]
    selected.sort(
        key=lambda row: (
            row["task_type"],
            row["dataset"],
            row["model"],
            row["clip_model"],
            int(row["num_clients"]),
            float(row["beta"]),
            int(row["seed"]),
        )
    )
    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def read_existing_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = set()
    for row in rows:
        keys.add(
            (
                row.get("task_type", ""),
                row.get("dataset", ""),
                row.get("model", ""),
                row.get("clip_model", ""),
                row.get("num_clients", ""),
                row.get("beta", ""),
                row.get("seed", ""),
                row.get("method", ""),
                row.get("split", ""),
            )
        )
    return keys


def append_csv(path: Path, row: dict[str, object], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def build_cfg(row: dict[str, str], method: str, args: argparse.Namespace, method_defaults: dict[str, object]) -> dict[str, object]:
    cfg: dict[str, object] = dict(method_defaults)
    cfg.update(
        {
            "task_type": row["task_type"],
            "dataset": row["dataset"],
            "num_clients": int(row["num_clients"]),
            "beta": float(row["beta"]),
            "seed": int(row["seed"]),
            "method": method,
            "merge_weight_mode": args.merge_weight_mode,
            "model_hub_root": str(args.model_hub_root),
            "data_root": str(args.data_root),
            "device": args.device,
            "num_workers": args.num_workers,
            "output_root": str(args.output_root),
            "split": args.split,
            "amp": str(args.device).startswith("cuda"),
            "stats_batch_size": args.stats_batch_size,
            "stats_num_workers": args.stats_num_workers,
        }
    )
    if row["task_type"] == "small":
        cfg["model"] = row["model"]
        cfg["batch_size"] = args.small_batch_size
    else:
        cfg["clip_model"] = row["clip_model"]
        cfg["batch_size"] = args.vlm_batch_size
    return cfg


def classification_metrics(confusion: np.ndarray) -> dict[str, object]:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.divide(true_positive, support, out=np.zeros_like(true_positive, dtype=float), where=support > 0)
        precisions = np.divide(true_positive, predicted, out=np.zeros_like(true_positive, dtype=float), where=predicted > 0)
        f1 = np.divide(
            2.0 * precisions * recalls,
            precisions + recalls,
            out=np.zeros_like(recalls, dtype=float),
            where=(precisions + recalls) > 0,
        )
    valid = support > 0
    pred_total = max(int(predicted.sum()), 1)
    pred_distribution = predicted.astype(float) / float(pred_total)
    nonzero = pred_distribution[pred_distribution > 0]
    pred_entropy = float(-(nonzero * np.log(nonzero)).sum()) if nonzero.size else 0.0
    return {
        "balanced_accuracy": float(recalls[valid].mean()) if valid.any() else 0.0,
        "macro_f1": float(f1[valid].mean()) if valid.any() else 0.0,
        "mean_precision": float(precisions[valid].mean()) if valid.any() else 0.0,
        "per_class_recall": " ".join(f"{x:.6f}" for x in recalls.tolist()),
        "per_class_precision": " ".join(f"{x:.6f}" for x in precisions.tolist()),
        "per_class_f1": " ".join(f"{x:.6f}" for x in f1.tolist()),
        "class_support": " ".join(str(int(x)) for x in support.tolist()),
        "predicted_class_counts": " ".join(str(int(x)) for x in predicted.tolist()),
        "majority_prediction_ratio": float(predicted.max() / pred_total) if predicted.size else 0.0,
        "effective_predicted_classes": float(np.exp(pred_entropy)),
        "prediction_entropy": pred_entropy,
    }


def evaluate_checkpoint_predictions(meta: dict[str, object], checkpoint: dict[str, object], cfg: dict[str, object]) -> dict[str, object]:
    from utils.runtime import build_runtime

    device = torch.device(str(cfg.get("device", "cpu")))
    runtime = build_runtime(
        meta=meta,
        data_root=str(cfg["data_root"]),
        split=str(cfg.get("split", "test")),
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=int(cfg.get("num_workers", 4)),
        device=device,
    )
    model = runtime["model"]
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    forward_fn = runtime["forward_fn"]
    loader = runtime["loader"]
    num_classes = int(meta.get("num_classes") or len(meta.get("class_names", [])))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    total = 0
    correct = 0
    amp_enabled = bool(cfg.get("amp", False)) and device.type == "cuda"
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = forward_fn(model, x)
                    loss = F.cross_entropy(logits, y, reduction="sum")
            else:
                logits = forward_fn(model, x)
                loss = F.cross_entropy(logits, y, reduction="sum")
            preds = logits.argmax(dim=1)
            total_loss += float(loss.detach().cpu())
            total += int(y.numel())
            correct += int((preds == y).sum().detach().cpu())
            y_np = y.detach().cpu().numpy().astype(int)
            p_np = preds.detach().cpu().numpy().astype(int)
            for target, pred in zip(y_np, p_np):
                if 0 <= target < num_classes and 0 <= pred < num_classes:
                    confusion[target, pred] += 1
    metrics = classification_metrics(confusion)
    metrics.update(
        {
            "accuracy": float(correct / max(total, 1)),
            "loss": float(total_loss / max(total, 1)),
            "num_samples": int(total),
            "confusion_matrix_json": json.dumps(confusion.tolist(), separators=(",", ":")),
        }
    )
    return metrics


def summarize_prediction_metrics(path: Path, out_path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "OK":
            by_method[row["method"]].append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# 实验一逐样本指标摘要\n\n")
        handle.write("| method | cases | mean acc | mean balanced acc | mean macro F1 | mean collapse ratio | median effective classes |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for method, items in sorted(by_method.items()):
            acc = [float(item["accuracy"]) for item in items]
            bacc = [float(item["balanced_accuracy"]) for item in items]
            f1 = [float(item["macro_f1"]) for item in items]
            collapse = [float(item["majority_prediction_ratio"]) for item in items]
            eff = [float(item["effective_predicted_classes"]) for item in items]
            handle.write(
                f"| {method} | {len(items)} | {mean(acc):.4f} | {mean(bacc):.4f} | "
                f"{mean(f1):.4f} | {mean(collapse):.4f} | {median(eff):.2f} |\n"
            )


def main() -> None:
    args = parse_args()
    setup_repo_imports(args.repo_root)

    from merge import METHOD_DEFAULTS, run_merge
    from utils import load_checkpoint, load_json

    args.model_hub_root = args.model_hub_root or args.repo_root / "model_hub"
    args.data_root = args.data_root or args.repo_root / "Med_data"
    args.eval_data_root = args.eval_data_root or args.data_root
    manifest = filter_manifest(load_manifest(args.model_hub_root / "manifest.csv"), args)
    if not manifest:
        raise SystemExit("No manifest rows matched the requested filters.")

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
        "method",
        "split",
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
        "merged_checkpoint",
        "merge_data_root",
        "eval_data_root",
        "updated_at",
    ]
    existing_keys = read_existing_keys(args.metrics_csv) if args.resume else set()
    total_cases = len(manifest) * len(args.methods)
    done = 0
    print(f"Selected manifest rows={len(manifest)} methods={len(args.methods)} total_cases={total_cases}")
    for row in manifest:
        for method in args.methods:
            cfg = build_cfg(row, method, args, METHOD_DEFAULTS)
            key = (
                str(cfg["task_type"]),
                str(cfg["dataset"]),
                str(cfg.get("model", "")),
                str(cfg.get("clip_model", "")),
                str(cfg["num_clients"]),
                format(float(cfg["beta"]), "g"),
                str(cfg["seed"]),
                method,
                str(cfg["split"]),
            )
            done += 1
            if key in existing_keys:
                print(f"[{done}/{total_cases}] skip existing {key}")
                continue
            start = time.time()
            base_row: OrderedDict[str, object] = OrderedDict(
                {
                    "task_type": cfg["task_type"],
                    "dataset": cfg["dataset"],
                    "model": cfg.get("model", ""),
                    "clip_model": cfg.get("clip_model", ""),
                    "num_clients": cfg["num_clients"],
                    "beta": format(float(cfg["beta"]), "g"),
                    "seed": cfg["seed"],
                    "method": method,
                    "split": cfg["split"],
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            try:
                print(f"[{done}/{total_cases}] merge+eval {key}")
                merge_info = run_merge(cfg)
                checkpoint_path = Path(str(merge_info["merged_checkpoint"]))
                meta = load_json(Path(str(merge_info["meta_path"])))
                checkpoint = load_checkpoint(checkpoint_path, device="cpu")
                eval_cfg = dict(cfg)
                eval_cfg["data_root"] = str(args.eval_data_root)
                metrics = evaluate_checkpoint_predictions(meta, checkpoint, eval_cfg)
                result = dict(base_row)
                result.update(metrics)
                result.update(
                    {
                        "status": "OK",
                        "error": "",
                        "seconds": f"{time.time() - start:.2f}",
                        "merged_checkpoint": str(checkpoint_path),
                        "merge_data_root": str(args.data_root),
                        "eval_data_root": str(args.eval_data_root),
                    }
                )
                append_csv(args.metrics_csv, result, fields)
                existing_keys.add(key)
                if args.delete_merged and checkpoint_path.exists():
                    checkpoint_path.unlink()
            except Exception as exc:
                result = dict(base_row)
                result.update(
                    {
                        "status": "FAIL",
                        "error": str(exc),
                        "seconds": f"{time.time() - start:.2f}",
                        "merge_data_root": str(args.data_root),
                        "eval_data_root": str(args.eval_data_root),
                    }
                )
                append_csv(args.metrics_csv, result, fields)
                print(f"[{done}/{total_cases}] FAIL {key}: {exc}")
    summarize_prediction_metrics(args.metrics_csv, args.metrics_csv.with_name("prediction_metrics_summary.md"))
    print(f"Wrote prediction metrics to {args.metrics_csv}")


if __name__ == "__main__":
    main()
