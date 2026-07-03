#!/usr/bin/env python3
"""Evaluate client pool strength and compare it with merged models.

Experiment 2 asks whether merge failures come from weak clients or from
weight-space fusion losing usable client knowledge. This script evaluates
single clients, prediction ensembles, and oracle client selectors for selected
model_hub settings.
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
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate client pool baselines for experiment 2.")
    parser.add_argument("--repo-root", type=Path, default=Path("program/MedMNISTMerge"))
    parser.add_argument("--model-hub-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("medmerge_empirical_study/results/experiment2"))
    parser.add_argument("--merged-metrics-csv", type=Path, default=Path("medmerge_empirical_study/results/experiment1/prediction_metrics.csv"))
    parser.add_argument("--task-type", choices=["small", "vlm"], default="small")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--small-models", nargs="*", default=None)
    parser.add_argument("--clip-models", nargs="*", default=None)
    parser.add_argument("--num-clients", nargs="*", type=int, default=None)
    parser.add_argument("--betas", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
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
        "confusion_matrix_json": json.dumps(confusion.tolist(), separators=(",", ":")),
    }


def add_confusion(confusion: np.ndarray, targets: np.ndarray, preds: np.ndarray) -> None:
    num_classes = confusion.shape[0]
    for target, pred in zip(targets.astype(int), preds.astype(int)):
        if 0 <= target < num_classes and 0 <= pred < num_classes:
            confusion[target, pred] += 1


def row_key(row: dict[str, object]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(row["task_type"]),
        str(row["dataset"]),
        str(row.get("model", "")),
        str(row.get("clip_model", "")),
        str(row["num_clients"]),
        format(float(row["beta"]), "g"),
        str(row["seed"]),
    )


def evaluate_client_pool(meta: dict[str, object], checkpoint_paths: list[Path], cfg: dict[str, object]) -> list[dict[str, object]]:
    from utils import extract_state_dict, load_checkpoint
    from utils.runtime import build_runtime

    device = torch.device(str(cfg.get("device", "cuda:0")))
    runtime = build_runtime(
        meta=meta,
        data_root=str(cfg["data_root"]),
        split=str(cfg.get("split", "test")),
        batch_size=int(cfg.get("batch_size", 128)),
        num_workers=int(cfg.get("num_workers", 4)),
        device=device,
    )
    loader = runtime["loader"]
    num_classes = int(meta.get("num_classes") or len(meta.get("class_names", [])))

    model_entries = []
    for ckpt_path in checkpoint_paths:
        model_runtime = build_runtime(
            meta=meta,
            data_root=str(cfg["data_root"]),
            split=str(cfg.get("split", "test")),
            batch_size=int(cfg.get("batch_size", 128)),
            num_workers=0,
            device=device,
        )
        model = model_runtime["model"]
        model.load_state_dict(extract_state_dict(load_checkpoint(ckpt_path, device="cpu")), strict=True)
        model.to(device)
        model.eval()
        model_entries.append((model, model_runtime["forward_fn"]))

    num_clients = len(model_entries)
    single_confusions = [np.zeros((num_classes, num_classes), dtype=np.int64) for _ in range(num_clients)]
    prob_ensemble_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    logit_ensemble_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    oracle_any_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    oracle_loss_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    single_loss = [0.0 for _ in range(num_clients)]
    prob_ensemble_loss = 0.0
    logit_ensemble_loss = 0.0
    oracle_loss_value = 0.0
    single_correct = [0 for _ in range(num_clients)]
    prob_ensemble_correct = 0
    logit_ensemble_correct = 0
    oracle_any_correct = 0
    oracle_loss_correct = 0
    total = 0

    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits_by_client = []
            losses_by_client = []
            preds_by_client = []
            for client_idx, (model, client_forward_fn) in enumerate(model_entries):
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        logits = client_forward_fn(model, x)
                else:
                    logits = client_forward_fn(model, x)
                logits_by_client.append(logits.float())
                per_sample_loss = F.cross_entropy(logits.float(), y, reduction="none")
                losses_by_client.append(per_sample_loss)
                preds = logits.argmax(dim=1)
                preds_by_client.append(preds)
                single_loss[client_idx] += float(per_sample_loss.sum().detach().cpu())
                single_correct[client_idx] += int((preds == y).sum().detach().cpu())
                add_confusion(single_confusions[client_idx], y.detach().cpu().numpy(), preds.detach().cpu().numpy())

            logits_stack = torch.stack(logits_by_client, dim=0)
            preds_stack = torch.stack(preds_by_client, dim=0)
            losses_stack = torch.stack(losses_by_client, dim=0)
            probs_stack = torch.softmax(logits_stack, dim=-1)

            prob_ensemble = probs_stack.mean(dim=0)
            prob_preds = prob_ensemble.argmax(dim=1)
            prob_ensemble_loss += float((-torch.log(prob_ensemble.clamp_min(1e-12).gather(1, y[:, None]).squeeze(1))).sum().detach().cpu())
            prob_ensemble_correct += int((prob_preds == y).sum().detach().cpu())
            add_confusion(prob_ensemble_confusion, y.detach().cpu().numpy(), prob_preds.detach().cpu().numpy())

            logit_ensemble = logits_stack.mean(dim=0)
            logit_preds = logit_ensemble.argmax(dim=1)
            logit_ensemble_loss += float(F.cross_entropy(logit_ensemble, y, reduction="sum").detach().cpu())
            logit_ensemble_correct += int((logit_preds == y).sum().detach().cpu())
            add_confusion(logit_ensemble_confusion, y.detach().cpu().numpy(), logit_preds.detach().cpu().numpy())

            any_correct_mask = (preds_stack == y.unsqueeze(0)).any(dim=0)
            min_loss_idx = losses_stack.argmin(dim=0)
            sample_index = torch.arange(y.numel(), device=device)
            oracle_loss_preds = preds_stack[min_loss_idx, sample_index]
            oracle_loss_value += float(losses_stack[min_loss_idx, sample_index].sum().detach().cpu())
            oracle_loss_correct += int((oracle_loss_preds == y).sum().detach().cpu())
            add_confusion(oracle_loss_confusion, y.detach().cpu().numpy(), oracle_loss_preds.detach().cpu().numpy())

            oracle_any_preds = oracle_loss_preds.clone()
            oracle_any_preds[any_correct_mask] = y[any_correct_mask]
            oracle_any_correct += int(any_correct_mask.sum().detach().cpu())
            add_confusion(oracle_any_confusion, y.detach().cpu().numpy(), oracle_any_preds.detach().cpu().numpy())
            total += int(y.numel())

    rows: list[dict[str, object]] = []
    for client_idx, confusion in enumerate(single_confusions):
        metrics = classification_metrics(confusion)
        metrics.update(
            {
                "baseline": f"single_client_{client_idx}",
                "client_id": client_idx,
                "accuracy": float(single_correct[client_idx] / max(total, 1)),
                "loss": float(single_loss[client_idx] / max(total, 1)),
                "num_samples": total,
            }
        )
        rows.append(metrics)

    best = max(rows, key=lambda item: float(item["accuracy"]))
    best_row = dict(best)
    best_row["baseline"] = "best_single_client"
    rows.append(best_row)

    mean_row: dict[str, object] = {
        "baseline": "mean_single_client",
        "client_id": "",
        "accuracy": mean(float(row["accuracy"]) for row in rows if str(row["baseline"]).startswith("single_client_")),
        "balanced_accuracy": mean(float(row["balanced_accuracy"]) for row in rows if str(row["baseline"]).startswith("single_client_")),
        "macro_f1": mean(float(row["macro_f1"]) for row in rows if str(row["baseline"]).startswith("single_client_")),
        "mean_precision": mean(float(row["mean_precision"]) for row in rows if str(row["baseline"]).startswith("single_client_")),
        "loss": mean(float(row["loss"]) for row in rows if str(row["baseline"]).startswith("single_client_")),
        "num_samples": total,
        "majority_prediction_ratio": mean(float(row["majority_prediction_ratio"]) for row in rows if str(row["baseline"]).startswith("single_client_")),
        "effective_predicted_classes": mean(float(row["effective_predicted_classes"]) for row in rows if str(row["baseline"]).startswith("single_client_")),
        "prediction_entropy": mean(float(row["prediction_entropy"]) for row in rows if str(row["baseline"]).startswith("single_client_")),
        "per_class_recall": "",
        "per_class_precision": "",
        "per_class_f1": "",
        "class_support": "",
        "predicted_class_counts": "",
        "confusion_matrix_json": "",
    }
    rows.append(mean_row)

    for name, confusion, correct, loss in [
        ("prob_ensemble", prob_ensemble_confusion, prob_ensemble_correct, prob_ensemble_loss),
        ("logit_ensemble", logit_ensemble_confusion, logit_ensemble_correct, logit_ensemble_loss),
        ("oracle_any_correct", oracle_any_confusion, oracle_any_correct, 0.0),
        ("oracle_min_loss", oracle_loss_confusion, oracle_loss_correct, oracle_loss_value),
    ]:
        metrics = classification_metrics(confusion)
        metrics.update(
            {
                "baseline": name,
                "client_id": "",
                "accuracy": float(correct / max(total, 1)),
                "loss": float(loss / max(total, 1)) if name != "oracle_any_correct" else "",
                "num_samples": total,
            }
        )
        rows.append(metrics)

    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def load_merged_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "OK"]


def build_gap_rows(client_rows: list[dict[str, object]], merged_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    baselines_by_key: dict[tuple[str, str, str, str, str, str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in client_rows:
        baselines_by_key[row_key(row)][str(row["baseline"])] = row

    out: list[dict[str, object]] = []
    targets = ["best_single_client", "prob_ensemble", "logit_ensemble", "oracle_any_correct", "oracle_min_loss"]
    for merged in merged_rows:
        key = row_key(merged)
        if key not in baselines_by_key:
            continue
        for target in targets:
            baseline = baselines_by_key[key].get(target)
            if baseline is None:
                continue
            row = {
                "task_type": merged["task_type"],
                "dataset": merged["dataset"],
                "model": merged.get("model", ""),
                "clip_model": merged.get("clip_model", ""),
                "num_clients": merged["num_clients"],
                "beta": format(float(merged["beta"]), "g"),
                "seed": merged["seed"],
                "merged_method": merged["method"],
                "comparison_baseline": target,
            }
            for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
                merged_value = float(merged[metric])
                baseline_value = float(baseline[metric])
                row[f"merged_{metric}"] = merged_value
                row[f"baseline_{metric}"] = baseline_value
                row[f"gap_{metric}"] = merged_value - baseline_value
            out.append(row)
    return out


def write_summary(out_path: Path, client_rows: list[dict[str, object]], gap_rows: list[dict[str, object]]) -> None:
    by_baseline: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in client_rows:
        by_baseline[str(row["baseline"])].append(row)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# 实验二：客户端能力与融合损失分析\n\n")
        handle.write("## 工作流核对\n\n")
        handle.write("对应 `experiment_workflow.md` 第 5 节：评估 single client、best single client、mean single client、prediction ensemble 和 oracle per-sample client。\n\n")
        handle.write("## Baseline 摘要\n\n")
        handle.write("| baseline | cases | mean acc | mean balanced acc | mean macro F1 | mean collapse ratio |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for baseline in sorted(by_baseline):
            rows = by_baseline[baseline]
            handle.write(
                f"| {baseline} | {len(rows)} | "
                f"{mean(float(row['accuracy']) for row in rows):.4f} | "
                f"{mean(float(row['balanced_accuracy']) for row in rows):.4f} | "
                f"{mean(float(row['macro_f1']) for row in rows):.4f} | "
                f"{mean(float(row['majority_prediction_ratio']) for row in rows):.4f} |\n"
            )
        handle.write("\n## 融合模型相对客户端池上界的 Gap\n\n")
        if gap_rows:
            for target in ["best_single_client", "prob_ensemble", "logit_ensemble", "oracle_any_correct", "oracle_min_loss"]:
                rows = [row for row in gap_rows if row["comparison_baseline"] == target]
                if not rows:
                    continue
                handle.write(
                    f"- merged - {target}: "
                    f"mean acc gap {mean(float(row['gap_accuracy']) for row in rows):+.4f}, "
                    f"mean balanced acc gap {mean(float(row['gap_balanced_accuracy']) for row in rows):+.4f}, "
                    f"mean macro F1 gap {mean(float(row['gap_macro_f1']) for row in rows):+.4f}\n"
                )
        else:
            handle.write("未找到可匹配的融合模型逐样本结果，未生成 gap。\n")
        handle.write("\n## 结论\n\n")
        handle.write("如果 oracle 或 prediction ensemble 明显高于权重融合，说明客户端池中存在可利用信息；融合失败不能简单归因于所有客户端都弱，而应归因于权重空间融合在医学异质客户端中无法稳定恢复这些信息。\n")


def main() -> None:
    args = parse_args()
    setup_repo_imports(args.repo_root)

    from utils import ensure_checkpoint_files, ensure_task_matches_config, find_hub_experiment_dir, load_hub_meta

    args.model_hub_root = args.model_hub_root or args.repo_root / "model_hub"
    args.data_root = args.data_root or args.repo_root / "Med_data"
    manifest = filter_manifest(load_manifest(args.model_hub_root / "manifest.csv"), args)
    if not manifest:
        raise SystemExit("No manifest rows matched the requested filters.")

    all_rows: list[dict[str, object]] = []
    start_all = time.time()
    for idx, row in enumerate(manifest, start=1):
        cfg: dict[str, object] = {
            "task_type": row["task_type"],
            "dataset": row["dataset"],
            "model": row.get("model", ""),
            "clip_model": row.get("clip_model", ""),
            "num_clients": int(row["num_clients"]),
            "beta": float(row["beta"]),
            "seed": int(row["seed"]),
            "model_hub_root": str(args.model_hub_root),
            "data_root": str(args.data_root),
            "device": args.device,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "split": args.split,
        }
        print(f"[{idx}/{len(manifest)}] evaluate client pool {row_key(cfg)}")
        start = time.time()
        exp_dir = find_hub_experiment_dir(args.model_hub_root, cfg)
        meta = load_hub_meta(exp_dir)
        ensure_task_matches_config(meta, cfg)
        checkpoint_paths = ensure_checkpoint_files(exp_dir, meta)
        rows = evaluate_client_pool(meta, checkpoint_paths, cfg)
        for out_row in rows:
            out_row.update(
                {
                    "task_type": cfg["task_type"],
                    "dataset": cfg["dataset"],
                    "model": cfg.get("model", ""),
                    "clip_model": cfg.get("clip_model", ""),
                    "num_clients": cfg["num_clients"],
                    "beta": format(float(cfg["beta"]), "g"),
                    "seed": cfg["seed"],
                    "split": cfg["split"],
                    "seconds": f"{time.time() - start:.2f}",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            all_rows.append(out_row)

    fields = [
        "task_type",
        "dataset",
        "model",
        "clip_model",
        "num_clients",
        "beta",
        "seed",
        "split",
        "baseline",
        "client_id",
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
        "seconds",
        "updated_at",
    ]
    client_csv = args.out_dir / "client_pool_metrics.csv"
    write_csv(client_csv, all_rows, fields)

    gap_rows = build_gap_rows(all_rows, load_merged_rows(args.merged_metrics_csv))
    gap_fields = [
        "task_type",
        "dataset",
        "model",
        "clip_model",
        "num_clients",
        "beta",
        "seed",
        "merged_method",
        "comparison_baseline",
        "merged_accuracy",
        "baseline_accuracy",
        "gap_accuracy",
        "merged_balanced_accuracy",
        "baseline_balanced_accuracy",
        "gap_balanced_accuracy",
        "merged_macro_f1",
        "baseline_macro_f1",
        "gap_macro_f1",
    ]
    gap_csv = args.out_dir / "merged_vs_client_pool_gaps.csv"
    write_csv(gap_csv, gap_rows, gap_fields)
    write_summary(args.out_dir / "experiment2_client_pool_summary.md", all_rows, gap_rows)
    print(f"Wrote client pool metrics to {client_csv}")
    print(f"Wrote gap metrics to {gap_csv}")
    print(f"Total seconds: {time.time() - start_all:.2f}")


if __name__ == "__main__":
    main()
