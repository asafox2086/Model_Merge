#!/usr/bin/env python3
"""Collect prediction-collapse diagnostics for merge methods and clients.

The formal accuracy tables are not enough for the collapse story. This script
recomputes selected merged checkpoints, evaluates individual clients, and
records balanced accuracy, macro F1, collapse ratio, prediction entropy, and
confusion-derived statistics. It also writes dataset-separated summaries and
figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.lamp_merge_analysis import LAMP_MERGE_ABLATION_MODES
from utils.lamp_merge_stats import default_prototype_root, validate_lamp_merge_stats_root

FORMAL_SMALL_DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
]

FORMAL_SMALL_MODELS = ["resnet", "convnext", "vit_t", "swin_tiny"]

DEFAULT_BASELINES = [
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
    "lamp_merge",
]

LAMP_METHODS = {"lamp_merge", "lamp_merge_analysis"}

METRIC_FIELDS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mean_precision",
    "loss",
    "collapse_ratio",
    "collapse_over_true_majority",
    "collapse_excess",
    "effective_predicted_classes",
    "pred_nonzero_classes",
    "pred_entropy_norm",
    "prob_entropy_norm",
    "mean_max_probability",
    "pred_true_tv",
    "prob_true_tv",
]

VECTOR_FIELDS = [
    "per_class_recall",
    "per_class_precision",
    "per_class_f1",
    "true_counts",
    "predicted_class_counts",
    "prob_mean",
]

PLOT_METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced Acc."),
    ("macro_f1", "Macro F1"),
    ("collapse_ratio", "Collapse Ratio"),
    ("effective_predicted_classes", "Effective Classes"),
    ("pred_true_tv", "Pred-True TV"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Collect dataset-separated prediction diagnostics.")
    parser.add_argument("--model-hub-root", type=Path, default=ROOT / "model_hub")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Med_data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "prediction_diagnostics")
    parser.add_argument("--metrics-csv", type=Path, default=ROOT / "My_merge_ret" / "reports" / "prediction_diagnostics.csv")
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "My_merge_ret" / "reports" / "prediction_diagnostics")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "My_merge_ret" / "figures" / "prediction_diagnostics")
    parser.add_argument("--task-type", choices=["small"], default="small")
    parser.add_argument("--datasets", nargs="*", default=FORMAL_SMALL_DATASETS)
    parser.add_argument("--small-models", nargs="*", default=["resnet"])
    parser.add_argument("--num-clients", nargs="*", type=int, default=[3])
    parser.add_argument("--betas", nargs="*", type=float, default=[0.01])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--methods", nargs="*", default=DEFAULT_BASELINES)
    parser.add_argument("--include-clients", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-weight-mode", choices=["sample", "equal"], default="equal")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--delete-merged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--summarize-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Read the existing metrics CSV and regenerate summaries/figures without evaluating checkpoints.",
    )
    parser.add_argument(
        "--lamp-merge-prototype-root",
        type=Path,
        default=default_prototype_root(ROOT),
    )
    parser.add_argument("--lamp-merge-proto-count-power", type=float, default=0.45)
    parser.add_argument("--lamp-merge-reference-head-scale", type=float, default=20.0)
    parser.add_argument("--lamp-merge-prevalence-threshold", type=float, default=0.5)
    parser.add_argument("--lamp-merge-reference-prior-threshold", type=float, default=2.5)
    parser.add_argument("--lamp-merge-reference-prior-max-tau", type=float, default=5.0)
    parser.add_argument("--lamp-merge-reference-prior-tau", type=float, default=None)
    parser.add_argument("--lamp-merge-ablation-mode", choices=sorted(LAMP_MERGE_ABLATION_MODES), default="full")
    parser.add_argument("--lamp-merge-ablation-modes", nargs="*", choices=sorted(LAMP_MERGE_ABLATION_MODES), default=[])
    parser.add_argument("--lamp-merge-ablation-seed", type=int, default=1701)
    parser.add_argument("--lamp-merge-prevalence-smoothing", type=float, default=1.0)
    return parser.parse_args()


def setup_environment() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_LOCAL_FILES_ONLY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def beta_key(value: object) -> str:
    return format(float(value), "g")


def beta_dirname(value: object) -> str:
    return "beta_" + beta_key(value).replace(".", "p")


def filter_manifest(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = [row for row in rows if row["task_type"] == args.task_type]
    if args.datasets:
        datasets = set(args.datasets)
        selected = [row for row in selected if row["dataset"] in datasets]
    if args.small_models:
        models = set(args.small_models)
        selected = [row for row in selected if row["model"] in models]
    if args.num_clients:
        clients = set(args.num_clients)
        selected = [row for row in selected if int(row["num_clients"]) in clients]
    if args.betas:
        betas = {beta_key(item) for item in args.betas}
        selected = [row for row in selected if beta_key(row["beta"]) in betas]
    if args.seeds:
        seeds = set(args.seeds)
        selected = [row for row in selected if int(row["seed"]) in seeds]
    selected.sort(
        key=lambda row: (
            row["dataset"],
            row["model"],
            int(row["num_clients"]),
            float(row["beta"]),
            int(row["seed"]),
        )
    )
    if args.skip > 0:
        selected = selected[args.skip :]
    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def lamp_ablation_modes(args: argparse.Namespace) -> list[str]:
    if args.lamp_merge_ablation_modes:
        return list(args.lamp_merge_ablation_modes)
    return [str(args.lamp_merge_ablation_mode)]


def method_label(method: str, ablation_mode: str, args: argparse.Namespace) -> str:
    if method not in LAMP_METHODS:
        return method
    modes = lamp_ablation_modes(args)
    if len(modes) == 1 and ablation_mode == "full":
        return method
    return f"{method}:{ablation_mode}"


def expand_method_specs(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []
    modes = lamp_ablation_modes(args)
    for method in args.methods:
        if method in LAMP_METHODS:
            for mode in modes:
                specs.append((method, mode, method_label(method, mode, args)))
        else:
            specs.append((method, "", method))
    return specs


def append_csv(path: Path, row: dict[str, object], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def read_existing_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "OK":
                continue
            keys.add(result_key(row))
    return keys


def result_key(row: dict[str, object]) -> tuple[str, ...]:
    return (
        str(row.get("task_type", "")),
        str(row.get("dataset", "")),
        str(row.get("model", "")),
        str(row.get("num_clients", "")),
        beta_key(row.get("beta", 0.0)),
        str(row.get("seed", "")),
        str(row.get("source", "")),
        str(row.get("method", "")),
        str(row.get("base_method", "")),
        str(row.get("ablation_mode", "")),
        str(row.get("client_id", "")),
        str(row.get("split", "")),
    )


def hub_dir_for_row(args: argparse.Namespace, row: dict[str, str]) -> Path:
    return args.model_hub_root / row["hub_dir"]


def load_meta_for_row(args: argparse.Namespace, row: dict[str, str]) -> dict[str, object]:
    from utils import load_json

    return load_json(args.model_hub_root / row["meta_path"])


def build_merge_cfg(
    row: dict[str, str],
    method: str,
    args: argparse.Namespace,
    ablation_mode: str = "",
) -> dict[str, object]:
    from merge import METHOD_DEFAULTS

    output_root = args.output_root
    if method in LAMP_METHODS and ablation_mode:
        output_root = args.output_root / f"{method}_{ablation_mode}"

    cfg: dict[str, object] = dict(METHOD_DEFAULTS)
    cfg.update(
        {
            "task_type": row["task_type"],
            "dataset": row["dataset"],
            "model": row["model"],
            "clip_model": row.get("clip_model", ""),
            "num_clients": int(row["num_clients"]),
            "beta": float(row["beta"]),
            "seed": int(row["seed"]),
            "method": method,
            "merge_weight_mode": args.merge_weight_mode,
            "model_hub_root": str(args.model_hub_root),
            "data_root": str(args.data_root),
            "output_root": str(output_root),
            "device": args.device,
            "num_workers": int(args.num_workers),
            "batch_size": int(args.batch_size),
            "split": args.split,
            "amp": str(args.device).startswith("cuda"),
            "lamp_merge_prototype_root": str(args.lamp_merge_prototype_root),
            "lamp_merge_proto_count_power": float(args.lamp_merge_proto_count_power),
            "lamp_merge_reference_head_scale": float(args.lamp_merge_reference_head_scale),
            "lamp_merge_prevalence_threshold": float(args.lamp_merge_prevalence_threshold),
            "lamp_merge_reference_prior_threshold": float(args.lamp_merge_reference_prior_threshold),
            "lamp_merge_reference_prior_max_tau": float(args.lamp_merge_reference_prior_max_tau),
            "lamp_merge_ablation_mode": ablation_mode or "full",
            "lamp_merge_ablation_seed": int(args.lamp_merge_ablation_seed),
            "lamp_merge_prevalence_smoothing": float(args.lamp_merge_prevalence_smoothing),
        }
    )
    if args.lamp_merge_reference_prior_tau is not None:
        cfg["lamp_merge_reference_prior_tau"] = float(args.lamp_merge_reference_prior_tau)
    return cfg


def entropy(dist: np.ndarray) -> float:
    nz = dist[dist > 0]
    if nz.size == 0:
        return 0.0
    return float(-(nz * np.log(nz)).sum())


def classification_metrics(
    confusion: np.ndarray,
    prob_sum: np.ndarray,
    total_loss: float,
    mean_max_prob_sum: float,
) -> dict[str, object]:
    total = int(confusion.sum())
    num_classes = int(confusion.shape[0])
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
    num_eval_classes = int(valid.sum())
    pred_total = max(int(predicted.sum()), 1)
    true_total = max(int(support.sum()), 1)
    pred_dist = predicted.astype(float) / float(pred_total)
    true_dist = support.astype(float) / float(true_total)
    prob_dist = prob_sum.astype(float) / max(float(prob_sum.sum()), 1.0)
    pred_entropy = entropy(pred_dist)
    prob_entropy = entropy(prob_dist)
    true_majority_ratio = float(true_dist.max()) if true_dist.size else 0.0
    pred_majority_ratio = float(pred_dist.max()) if pred_dist.size else 0.0
    prob_true_tv = float(0.5 * np.abs(prob_dist - true_dist).sum()) if prob_dist.size else 0.0
    pred_true_tv = float(0.5 * np.abs(pred_dist - true_dist).sum()) if pred_dist.size else 0.0
    return {
        "accuracy": float(true_positive.sum() / max(total, 1)),
        "balanced_accuracy": float(recalls[valid].mean()) if valid.any() else 0.0,
        "macro_f1": float(f1[valid].mean()) if valid.any() else 0.0,
        "mean_precision": float(precisions[valid].mean()) if valid.any() else 0.0,
        "loss": float(total_loss / max(total, 1)),
        "num_samples": int(total),
        "num_classes": int(num_classes),
        "num_total_classes": int(num_classes),
        "num_eval_classes": num_eval_classes,
        "true_majority_class": int(support.argmax()) if support.size else -1,
        "true_majority_ratio": true_majority_ratio,
        "pred_majority_class": int(predicted.argmax()) if predicted.size else -1,
        "collapse_ratio": pred_majority_ratio,
        "collapse_over_true_majority": float(pred_majority_ratio / max(true_majority_ratio, 1e-12)),
        "collapse_excess": float(pred_majority_ratio - true_majority_ratio),
        "effective_predicted_classes": float(math.exp(pred_entropy)),
        "pred_nonzero_classes": int((predicted > 0).sum()),
        "pred_entropy_norm": float(pred_entropy / math.log(num_classes)) if num_classes > 1 else 0.0,
        "prob_entropy_norm": float(prob_entropy / math.log(num_classes)) if num_classes > 1 else 0.0,
        "mean_max_probability": float(mean_max_prob_sum / max(total, 1)),
        "pred_true_tv": pred_true_tv,
        "prob_true_tv": prob_true_tv,
        "per_class_recall": " ".join(f"{x:.6f}" for x in recalls.tolist()),
        "per_class_precision": " ".join(f"{x:.6f}" for x in precisions.tolist()),
        "per_class_f1": " ".join(f"{x:.6f}" for x in f1.tolist()),
        "true_counts": " ".join(str(int(x)) for x in support.tolist()),
        "predicted_class_counts": " ".join(str(int(x)) for x in predicted.tolist()),
        "prob_mean": " ".join(f"{float(x):.6f}" for x in prob_dist.tolist()),
        "confusion_matrix_json": json.dumps(confusion.tolist(), separators=(",", ":")),
    }


def evaluate_checkpoint_predictions(
    meta: dict[str, object],
    checkpoint: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    from utils.runtime import build_runtime
    from utils.state_dict import extract_state_dict

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    runtime = build_runtime(
        meta=meta,
        data_root=str(args.data_root),
        split=args.split,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    model = runtime["model"]
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()
    forward_fn = runtime["forward_fn"]
    loader = runtime["loader"]
    num_classes = int(meta.get("num_classes") or len(meta.get("class_names", [])))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    prob_sum = np.zeros(num_classes, dtype=np.float64)
    total_loss = 0.0
    mean_max_prob_sum = 0.0
    amp_enabled = str(args.device).startswith("cuda") and device.type == "cuda"

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = forward_fn(model, x)
                    loss = F.cross_entropy(logits, y, reduction="sum")
            else:
                logits = forward_fn(model, x)
                loss = F.cross_entropy(logits, y, reduction="sum")
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            total_loss += float(loss.detach().cpu())
            prob_sum += probs.detach().cpu().double().sum(dim=0).numpy()
            mean_max_prob_sum += float(probs.max(dim=1).values.detach().cpu().sum())
            y_np = y.detach().cpu().numpy().astype(int)
            p_np = preds.detach().cpu().numpy().astype(int)
            for target, pred in zip(y_np, p_np):
                if 0 <= target < num_classes and 0 <= pred < num_classes:
                    confusion[target, pred] += 1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return classification_metrics(confusion, prob_sum, total_loss, mean_max_prob_sum)


def base_result_row(
    row: dict[str, str],
    source: str,
    method: str,
    client_id: str,
    args: argparse.Namespace,
    base_method: str = "",
    ablation_mode: str = "",
) -> OrderedDict[str, object]:
    return OrderedDict(
        {
            "status": "OK",
            "error": "",
            "seconds": "",
            "source": source,
            "task_type": row["task_type"],
            "dataset": row["dataset"],
            "model": row["model"],
            "num_clients": int(row["num_clients"]),
            "beta": beta_key(row["beta"]),
            "seed": int(row["seed"]),
            "method": method,
            "base_method": base_method or method,
            "ablation_mode": ablation_mode,
            "client_id": client_id,
            "split": args.split,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )


def evaluate_client_rows(
    row: dict[str, str],
    meta: dict[str, object],
    args: argparse.Namespace,
    fields: list[str],
    existing: set[tuple[str, ...]],
) -> None:
    from utils import load_checkpoint

    hub_dir = hub_dir_for_row(args, row)
    for client_idx, client in enumerate(meta.get("clients", [])):
        method = f"client_{client_idx}"
        result = base_result_row(row, "client", method, str(client_idx), args)
        key = result_key(result)
        if args.resume and key in existing:
            continue
        start = time.time()
        try:
            ckpt_path = hub_dir / str(client.get("checkpoint", f"client_{client_idx}.pt"))
            checkpoint = load_checkpoint(ckpt_path, device="cpu")
            metrics = evaluate_checkpoint_predictions(meta, checkpoint, args)
            result.update(metrics)
            result.update({"seconds": f"{time.time() - start:.2f}", "checkpoint_path": str(ckpt_path)})
            append_csv(args.metrics_csv, result, fields)
            existing.add(key)
        except Exception as exc:
            result.update({"status": "FAIL", "error": str(exc), "seconds": f"{time.time() - start:.2f}"})
            append_csv(args.metrics_csv, result, fields)


def evaluate_method_row(
    row: dict[str, str],
    method: str,
    ablation_mode: str,
    display_method: str,
    args: argparse.Namespace,
    fields: list[str],
    existing: set[tuple[str, ...]],
) -> None:
    from merge import run_merge
    from utils import load_checkpoint, load_json

    result = base_result_row(row, "merge", display_method, "", args, base_method=method, ablation_mode=ablation_mode)
    key = result_key(result)
    if args.resume and key in existing:
        return
    start = time.time()
    try:
        cfg = build_merge_cfg(row, method, args, ablation_mode=ablation_mode)
        merge_info = run_merge(cfg)
        checkpoint_path = Path(str(merge_info["merged_checkpoint"]))
        meta = load_json(Path(str(merge_info["meta_path"])))
        checkpoint = load_checkpoint(checkpoint_path, device="cpu")
        metrics = evaluate_checkpoint_predictions(meta, checkpoint, args)
        result.update(metrics)
        result.update(
            {
                "seconds": f"{time.time() - start:.2f}",
                "checkpoint_path": str(checkpoint_path),
                "merged_dir": str(merge_info.get("merged_dir", "")),
            }
        )
        append_csv(args.metrics_csv, result, fields)
        existing.add(key)
        if args.delete_merged and checkpoint_path.exists():
            checkpoint_path.unlink()
    except Exception as exc:
        result.update({"status": "FAIL", "error": str(exc), "seconds": f"{time.time() - start:.2f}"})
        append_csv(args.metrics_csv, result, fields)


def read_ok_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "OK"]


def aggregate_client_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_case: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("source") != "client":
            continue
        key = (
            row["task_type"],
            row["dataset"],
            row["model"],
            row["num_clients"],
            beta_key(row["beta"]),
            row["seed"],
            row["split"],
        )
        by_case[key].append(row)

    out: list[dict[str, object]] = []
    for key, items in sorted(by_case.items()):
        for aggregate_name, selector in [
            ("client_mean", None),
            ("client_best", "accuracy"),
        ]:
            if selector is None:
                base = dict(items[0])
                base["source"] = "client_aggregate"
                base["method"] = aggregate_name
                base["client_id"] = ""
                base["num_aggregated_clients"] = len(items)
                for metric in METRIC_FIELDS:
                    vals = [float(item[metric]) for item in items if item.get(metric) not in {"", None}]
                    base[metric] = mean(vals) if vals else ""
                for field in VECTOR_FIELDS:
                    vectors = [parse_vector(item.get(field, "")) for item in items]
                    vectors = [vec for vec in vectors if vec.size > 0]
                    if vectors:
                        width = max(vec.size for vec in vectors)
                        matrix = np.zeros((len(vectors), width), dtype=float)
                        for idx, vec in enumerate(vectors):
                            matrix[idx, : vec.size] = vec
                        base[field] = format_vector(matrix.mean(axis=0))
                matrices = [parse_confusion(item.get("confusion_matrix_json", "")) for item in items]
                matrices = [mat for mat in matrices if mat.size > 0]
                if matrices:
                    rows_n = max(mat.shape[0] for mat in matrices)
                    cols_n = max(mat.shape[1] for mat in matrices)
                    cube = np.zeros((len(matrices), rows_n, cols_n), dtype=float)
                    for idx, mat in enumerate(matrices):
                        cube[idx, : mat.shape[0], : mat.shape[1]] = mat
                    base["confusion_matrix_json"] = json.dumps(cube.mean(axis=0).round(6).tolist(), separators=(",", ":"))
                out.append(base)
            else:
                best = max(items, key=lambda item: float(item.get(selector, 0.0)))
                base = dict(best)
                base["source"] = "client_aggregate"
                base["method"] = aggregate_name
                base["client_id"] = best.get("client_id", "")
                base["num_aggregated_clients"] = len(items)
                out.append(base)
    return out


def write_csv_full(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def group_mean(rows: list[dict[str, object]], metric: str) -> float:
    vals = [float(row[metric]) for row in rows if row.get(metric) not in {"", None}]
    return mean(vals) if vals else float("nan")


def build_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("source") == "client":
            continue
        grouped[(str(row["dataset"]), str(row["method"]))].append(row)
    summary = []
    for (dataset, method), items in sorted(grouped.items()):
        out = OrderedDict({"dataset": dataset, "method": method, "cases": len(items)})
        for metric in METRIC_FIELDS:
            out[f"mean_{metric}"] = group_mean(items, metric)
        summary.append(out)
    return summary


def method_sort_key(method: str) -> tuple[int, int, str]:
    base, _, mode = method.partition(":")
    order = [
        "client_mean",
        "client_best",
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
        "lamp_merge",
    ]
    mode_order = [
        "",
        "full",
        "m1_only",
        "avg_m2",
        "prototype_head_agg",
        "global_feature_mean",
        "support_only",
        "prototype_shuffle",
        "uniform_client_weight",
        "binary_support",
        "global_client_size_weight",
        "no_prevalence",
        "uniform_prevalence",
        "smoothed_prevalence",
    ]
    return (
        order.index(base) if base in order else len(order),
        mode_order.index(mode) if mode in mode_order else len(mode_order),
        method,
    )


def parse_vector(value: object) -> np.ndarray:
    if value in {"", None}:
        return np.array([], dtype=float)
    try:
        return np.array([float(item) for item in str(value).replace(",", " ").split()], dtype=float)
    except ValueError:
        return np.array([], dtype=float)


def format_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.6f}" for value in values.tolist())


def parse_confusion(value: object) -> np.ndarray:
    if value in {"", None}:
        return np.array([], dtype=float)
    try:
        matrix = np.array(json.loads(str(value)), dtype=float)
    except (json.JSONDecodeError, ValueError, TypeError):
        return np.array([], dtype=float)
    return matrix if matrix.ndim == 2 else np.array([], dtype=float)


def fmt(value: object, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def write_markdown_summary(summary_rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_dataset[str(row["dataset"])].append(row)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Prediction Collapse Diagnostics by Dataset\n\n")
        handle.write("Each table averages over the evaluated model/client/beta cases within one dataset.\n\n")
        handle.write("`collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.\n\n")
        for dataset in sorted(by_dataset):
            handle.write(f"## {dataset}\n\n")
            handle.write("| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |\n")
            handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in sorted(by_dataset[dataset], key=lambda item: method_sort_key(str(item["method"]))):
                handle.write(
                    f"| {row['method']} | {row['cases']} | "
                    f"{fmt(row['mean_accuracy'])} | {fmt(row['mean_balanced_accuracy'])} | "
                    f"{fmt(row['mean_macro_f1'])} | {fmt(row['mean_collapse_ratio'])} | "
                    f"{fmt(row['mean_effective_predicted_classes'])} | {fmt(row['mean_pred_true_tv'])} |\n"
                )
            handle.write("\n")


def collect_distribution_matrix(rows: list[dict[str, object]], dataset: str) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    true_vectors: list[np.ndarray] = []
    for row in rows:
        if str(row.get("dataset", "")) != dataset:
            continue
        if row.get("source") == "client":
            continue
        pred_counts = parse_vector(row.get("predicted_class_counts", ""))
        if pred_counts.size > 0 and pred_counts.sum() > 0:
            grouped[str(row.get("method", ""))].append(pred_counts / pred_counts.sum())
        true_counts = parse_vector(row.get("true_counts", ""))
        if true_counts.size > 0 and true_counts.sum() > 0:
            true_vectors.append(true_counts / true_counts.sum())

    methods = sorted(grouped, key=method_sort_key)
    width = 0
    for vectors in grouped.values():
        width = max(width, *(vec.size for vec in vectors))
    if true_vectors:
        width = max(width, *(vec.size for vec in true_vectors))
    matrix = np.zeros((len(methods), width), dtype=float)
    for row_idx, method in enumerate(methods):
        vectors = grouped[method]
        padded = np.zeros((len(vectors), width), dtype=float)
        for vec_idx, vec in enumerate(vectors):
            padded[vec_idx, : vec.size] = vec
        matrix[row_idx] = padded.mean(axis=0)
    true_dist = None
    if true_vectors:
        padded = np.zeros((len(true_vectors), width), dtype=float)
        for vec_idx, vec in enumerate(true_vectors):
            padded[vec_idx, : vec.size] = vec
        true_dist = padded.mean(axis=0)
    return methods, matrix, true_dist


def plot_distribution_figures(rows: list[dict[str, object]], figure_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset")})
    for dataset in datasets:
        methods, pred_matrix, true_dist = collect_distribution_matrix(rows, dataset)
        if not methods or pred_matrix.size == 0:
            continue
        if true_dist is not None:
            labels = ["true_test"] + methods
            matrix = np.vstack([true_dist, pred_matrix])
        else:
            labels = methods
            matrix = pred_matrix
        fig_height = max(5.5, 0.35 * len(labels) + 2.0)
        fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
        image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=max(float(matrix.max()), 1e-6))
        ax.set_title(f"{dataset}: predicted class distribution")
        ax.set_xlabel("Class index")
        ax.set_ylabel("Method")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([str(idx) for idx in range(matrix.shape[1])])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        for label in ax.get_yticklabels():
            if label.get_text().startswith("lamp_merge"):
                label.set_color("#B23A48")
                label.set_fontweight("bold")
            elif label.get_text().startswith("client"):
                label.set_color("#1F6F54")
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="fraction")
        fig.savefig(figure_dir / f"{dataset}_prediction_distribution.png", dpi=180)
        plt.close(fig)


def plot_dataset_figures(summary_rows: list[dict[str, object]], all_rows: list[dict[str, object]], figure_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_dataset[str(row["dataset"])].append(row)

    for dataset, items in sorted(by_dataset.items()):
        items = sorted(items, key=lambda item: method_sort_key(str(item["method"])))
        methods = [str(item["method"]) for item in items]
        fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
        axes_flat = list(axes.reshape(-1))
        for ax, (metric, title) in zip(axes_flat, PLOT_METRICS):
            values = [float(item.get(f"mean_{metric}", float("nan"))) for item in items]
            colors = ["#5B6C8F" for _ in methods]
            for idx, method in enumerate(methods):
                if method.startswith("lamp_merge"):
                    colors[idx] = "#B23A48"
                elif method.startswith("client"):
                    colors[idx] = "#3B8C6E"
            ax.bar(methods, values, color=colors)
            ax.set_title(title)
            ax.tick_params(axis="x", labelrotation=55, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{dataset}: prediction diagnostics", fontsize=14)
        fig.savefig(figure_dir / f"{dataset}_prediction_diagnostics.png", dpi=180)
        plt.close(fig)
    plot_distribution_figures(all_rows, figure_dir)


def write_outputs(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = [dict(row) for row in read_ok_rows(args.metrics_csv)]
    rows.extend(aggregate_client_rows([dict(row) for row in rows]))
    aggregate_csv = args.summary_dir / "prediction_diagnostics_with_client_aggregates.csv"
    write_csv_full(aggregate_csv, rows)
    summary_rows = build_summary_rows(rows)
    write_csv_full(args.summary_dir / "prediction_diagnostics_dataset_summary.csv", summary_rows)
    write_markdown_summary(summary_rows, args.summary_dir / "prediction_diagnostics_dataset_summary.md")
    plot_dataset_figures(summary_rows, rows, args.figure_dir)


def main() -> None:
    setup_environment()
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        write_outputs(args)
        print(f"Wrote dataset summary: {args.summary_dir / 'prediction_diagnostics_dataset_summary.md'}")
        print(f"Wrote figures: {args.figure_dir}")
        return
    if any(method in LAMP_METHODS for method in args.methods):
        validate_lamp_merge_stats_root(args.lamp_merge_prototype_root, min_files=1)

    manifest = filter_manifest(load_manifest(args.model_hub_root / "manifest.csv"), args)
    if not manifest:
        raise SystemExit("No manifest rows matched the requested filters.")

    fields = [
        "status",
        "error",
        "seconds",
        "source",
        "task_type",
        "dataset",
        "model",
        "num_clients",
        "beta",
        "seed",
        "method",
        "base_method",
        "ablation_mode",
        "client_id",
        "split",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mean_precision",
        "loss",
        "num_samples",
        "num_classes",
        "num_total_classes",
        "num_eval_classes",
        "true_majority_class",
        "true_majority_ratio",
        "pred_majority_class",
        "collapse_ratio",
        "collapse_over_true_majority",
        "collapse_excess",
        "effective_predicted_classes",
        "pred_nonzero_classes",
        "pred_entropy_norm",
        "prob_entropy_norm",
        "mean_max_probability",
        "pred_true_tv",
        "prob_true_tv",
        "per_class_recall",
        "per_class_precision",
        "per_class_f1",
        "true_counts",
        "predicted_class_counts",
        "prob_mean",
        "confusion_matrix_json",
        "checkpoint_path",
        "merged_dir",
        "updated_at",
    ]
    existing = read_existing_keys(args.metrics_csv) if args.resume else set()
    method_specs = expand_method_specs(args)
    total = len(manifest) * len(method_specs) + (sum(int(row["num_clients"]) for row in manifest) if args.include_clients else 0)
    current = 0
    print(
        f"Selected cases={len(manifest)} methods={len(method_specs)} include_clients={args.include_clients} "
        f"total_evals={total}"
    )

    for row in manifest:
        meta = load_meta_for_row(args, row)
        label = f"{row['dataset']}:{row['model']}:c{row['num_clients']}:b{beta_key(row['beta'])}"
        if args.include_clients:
            before = len(existing)
            evaluate_client_rows(row, meta, args, fields, existing)
            current += int(row["num_clients"])
            print(f"[{current}/{total}] clients {label} | new={len(existing) - before}")
        for method, ablation_mode, display_method in method_specs:
            before = len(existing)
            evaluate_method_row(row, method, ablation_mode, display_method, args, fields, existing)
            current += 1
            status = "skip" if len(existing) == before else "done"
            print(f"[{current}/{total}] {status} {display_method} {label}")

    write_outputs(args)
    print(f"Wrote raw metrics: {args.metrics_csv}")
    print(f"Wrote dataset summary: {args.summary_dir / 'prediction_diagnostics_dataset_summary.md'}")
    print(f"Wrote figures: {args.figure_dir}")


if __name__ == "__main__":
    main()
