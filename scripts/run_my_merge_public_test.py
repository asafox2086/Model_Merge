#!/usr/bin/env python3
"""Run external public-data evaluation for my_merge.

The public data is used only as an external evaluation split. Merge-time
statistics still come from client-side aggregate uploads such as prototype
stats, not from public images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_LOCAL_FILES_ONLY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from medmerge_empirical_study.scripts.collect_prediction_metrics import evaluate_checkpoint_predictions
from merge import METHOD_DEFAULTS, run_merge
from utils import load_checkpoint, load_json


DATASETS = ["bloodmnist_224", "dermamnist_224"]
SMALL_MODELS = ["resnet", "convnext", "vit_t", "swin_tiny"]
VLM_MODELS = ["openai/clip-vit-base-patch32"]
CLIENTS = [3, 5, 7]
BETAS = [0.0, 0.01, 0.1]
SETTINGS = [
    (3, 0.0, "c3_b0"),
    (3, 0.01, "c3_b0.01"),
    (3, 0.1, "c3_b0.1"),
    (5, 0.0, "c5_b0"),
    (5, 0.01, "c5_b0.01"),
    (5, 0.1, "c5_b0.1"),
    (7, 0.0, "c7_b0"),
    (7, 0.01, "c7_b0.01"),
    (7, 0.1, "c7_b0.1"),
]


FIELDS = [
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
    "prototype_root",
    "merge_data_root",
    "eval_data_root",
    "updated_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run my_merge on an external public evaluation split.")
    parser.add_argument("--model-hub-root", type=Path, default=ROOT / "model_hub")
    parser.add_argument("--source-data-root", type=Path, default=ROOT / "Med_data")
    parser.add_argument("--public-data-root", type=Path, default=ROOT / "PublicMedLabeled_data")
    parser.add_argument("--prototype-root", type=Path, default=ROOT / "outputs/my_merge_reference_proto_stats_recall_full_table_20260705")
    parser.add_argument("--source-config", type=Path, default=ROOT / "outputs/my_merge_reference_proto_recall_full_table_20260705/reports/batch_config.json")
    parser.add_argument("--source-eval-summary", type=Path, default=ROOT / "outputs/my_merge_reference_proto_recall_full_table_20260705/reports/eval_summary.csv")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=ROOT / "My_merge_ret/public_test_summary.md")
    parser.add_argument("--summary-copy", type=Path, default=ROOT / "My_merge_ret/public_test_eval_summary.csv")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--small-models", nargs="*", default=SMALL_MODELS)
    parser.add_argument("--include-vlm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--methods", nargs="*", default=["my_merge"])
    parser.add_argument("--num-clients", nargs="*", type=int, default=CLIENTS)
    parser.add_argument("--betas", nargs="*", type=float, default=BETAS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--small-batch-size", type=int, default=128)
    parser.add_argument("--vlm-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-failures", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--delete-merged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def rewrite_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def norm_beta(raw: object) -> str:
    return format(float(raw), "g")


def public_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    model_name = row.get("model") if row.get("task_type") == "small" else row.get("clip_model")
    return (
        row.get("task_type", ""),
        row.get("dataset", ""),
        model_name or "",
        str(int(float(row.get("num_clients", 0)))),
        norm_beta(row.get("beta", 0)),
        str(int(float(row.get("seed", 0)))),
        row.get("method", ""),
    )


def existing_keys(path: Path, rerun_failures: bool) -> set[tuple[str, ...]]:
    keys = set()
    for row in read_csv(path):
        if rerun_failures and row.get("status") != "OK":
            continue
        keys.add(public_key(row))
    return keys


def load_source_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def selected_manifest(args: argparse.Namespace) -> list[dict[str, str]]:
    manifest_path = args.model_hub_root / "manifest.csv"
    rows = read_csv(manifest_path)
    datasets = set(args.datasets)
    small_models = set(args.small_models)
    clients = {str(int(x)) for x in args.num_clients}
    betas = {norm_beta(x) for x in args.betas}
    selected: list[dict[str, str]] = []
    for row in rows:
        if row["dataset"] not in datasets:
            continue
        if str(int(row["num_clients"])) not in clients:
            continue
        if norm_beta(row["beta"]) not in betas:
            continue
        if str(int(row["seed"])) != str(int(args.seed)):
            continue
        if row["task_type"] == "small" and row["model"] in small_models:
            selected.append(row)
        elif args.include_vlm and row["task_type"] == "vlm" and row["clip_model"] in VLM_MODELS:
            selected.append(row)
    selected.sort(
        key=lambda row: (
            row["task_type"],
            row["dataset"],
            row["model"] or row["clip_model"],
            int(row["num_clients"]),
            float(row["beta"]),
        )
    )
    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def build_cfg(row: dict[str, str], method: str, args: argparse.Namespace, source_cfg: dict[str, object]) -> dict[str, object]:
    cfg = dict(METHOD_DEFAULTS)
    cfg.update(source_cfg)
    cfg.update(
        {
            "task_type": row["task_type"],
            "dataset": row["dataset"],
            "model": row["model"],
            "clip_model": row["clip_model"],
            "num_clients": int(row["num_clients"]),
            "beta": float(row["beta"]),
            "seed": int(row["seed"]),
            "method": method,
            "merge_weight_mode": "equal",
            "model_hub_root": str(args.model_hub_root),
            "data_root": str(args.source_data_root),
            "output_root": str(args.output_root),
            "device": args.device,
            "num_workers": args.num_workers,
            "split": args.split,
            "amp": str(args.device).startswith("cuda"),
            "my_merge_prototype_root": str(args.prototype_root),
            "my_merge_client_prototype_root": str(args.prototype_root),
        }
    )
    if row["task_type"] == "small":
        cfg["batch_size"] = args.small_batch_size
    else:
        cfg["batch_size"] = args.vlm_batch_size
    return cfg


def f4(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.4f}"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def split_info(data_root: Path, dataset: str, split: str) -> dict[str, object]:
    data = np.load(data_root / f"{dataset}.npz")
    images = data[f"{split}_images"]
    labels = data[f"{split}_labels"].reshape(-1)
    counts = [(int(label), int((labels == label).sum())) for label in sorted(set(labels.tolist()))]
    return {
        "samples": int(images.shape[0]),
        "shape": "x".join(str(x) for x in images.shape[1:]),
        "num_classes": len(counts),
        "counts": ", ".join(f"{label}:{count}" for label, count in counts),
    }


def method_rows_for_table(rows: list[dict[str, str]]) -> list[str]:
    order = []
    for row in rows:
        method = row["method"]
        if method not in order:
            order.append(method)
    return order


def accuracy_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int, str, str], float]:
    out = {}
    for row in rows:
        if row.get("status") != "OK":
            continue
        model_name = row["model"] if row["task_type"] == "small" else row["clip_model"]
        out[(row["task_type"], row["dataset"], model_name, int(row["num_clients"]), norm_beta(row["beta"]), row["method"])] = float(row["accuracy"])
    return out


def source_lookup(source_summary: Path) -> dict[tuple[str, str, str, int, str, str], float]:
    out = {}
    for row in read_csv(source_summary):
        if row.get("method") != "my_merge":
            continue
        model_name = row["model"] if row["task_type"] == "small" else row["clip_model"]
        out[(row["task_type"], row["dataset"], model_name, int(float(row["num_clients"])), norm_beta(row["beta"]), row["method"])] = float(row["test_acc"])
    return out


def average_for(lookup: dict[tuple[str, str, str, int, str, str], float], task_type: str, dataset: str, model_name: str, clients: int, method: str) -> float | None:
    values = []
    for cand_clients, beta, _label in SETTINGS:
        if cand_clients != clients:
            continue
        value = lookup.get((task_type, dataset, model_name, clients, norm_beta(beta), method))
        if value is None:
            return None
        values.append(value)
    return sum(values) / len(values) if values else None


def emit_table(lines: list[str], rows: list[dict[str, str]], task_type: str, model_name: str, datasets: list[str], methods: list[str]) -> None:
    lookup = accuracy_lookup(rows)
    raw_headers = [label for _clients, _beta, label in SETTINGS]
    avg_headers = [f"c{clients}_avg" for clients in CLIENTS]
    lines.extend([f"### {model_name}", "", "#### Raw", ""])
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append('    <tr><th rowspan="2">method</th>' + "".join(f'<th colspan="{len(raw_headers)}">{dataset}</th>' for dataset in datasets) + "</tr>")
    lines.append("    <tr>" + "".join(f"<th>{header}</th>" for _dataset in datasets for header in raw_headers) + "</tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")
    for method in methods:
        cells = []
        for dataset in datasets:
            for clients, beta, _label in SETTINGS:
                cells.append(f4(lookup.get((task_type, dataset, model_name, clients, norm_beta(beta), method))))
        lines.append("    <tr><td>" + method + "</td>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    lines.append("  </tbody>")
    lines.append("</table>")
    lines.extend(["", "#### Client Average", ""])
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append('    <tr><th rowspan="2">method</th>' + "".join(f'<th colspan="{len(avg_headers)}">{dataset}</th>' for dataset in datasets) + "</tr>")
    lines.append("    <tr>" + "".join(f"<th>{header}</th>" for _dataset in datasets for header in avg_headers) + "</tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")
    for method in methods:
        cells = []
        for dataset in datasets:
            for clients in CLIENTS:
                cells.append(f4(average_for(lookup, task_type, dataset, model_name, clients, method)))
        lines.append("    <tr><td>" + method + "</td>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    lines.append("  </tbody>")
    lines.append("</table>")
    lines.append("")


def build_report(args: argparse.Namespace, metrics_path: Path) -> None:
    rows = read_csv(metrics_path)
    ok_rows = [row for row in rows if row.get("status") == "OK"]
    methods = method_rows_for_table(ok_rows) or args.methods
    src = source_lookup(args.source_eval_summary)
    public = accuracy_lookup(ok_rows)

    dataset_info = []
    for dataset in args.datasets:
        path = args.public_data_root / f"{dataset}.npz"
        if path.exists():
            info = split_info(args.public_data_root, dataset, args.split)
            dataset_info.append([dataset, args.split, info["samples"], info["shape"], info["num_classes"], info["counts"]])

    deltas = []
    for key, pub_acc in public.items():
        if key[-1] != "my_merge":
            continue
        src_acc = src.get(key)
        if src_acc is not None:
            deltas.append(pub_acc - src_acc)

    by_method: dict[str, list[float]] = defaultdict(list)
    for row in ok_rows:
        by_method[row["method"]].append(float(row["accuracy"]))
    summary_rows = [
        [method, len(values), f4(mean(values)), f4(min(values)), f4(max(values))]
        for method, values in sorted(by_method.items())
    ]

    lines = [
        "# public_test Summary",
        "",
        "This table evaluates the already-defined merge method on independent public labeled medical images.",
        "The public images are used only for final external evaluation. They are not used for merge-time selection, training, or candidate tuning.",
        "",
        "## Scope",
        "",
        md_table(
            ["Item", "Value"],
            [
                ["merge method", ", ".join(args.methods)],
                ["source merge data", f"`{args.source_data_root}`"],
                ["client aggregate stats", f"`{args.prototype_root}`"],
                ["public eval data", f"`{args.public_data_root}`"],
                ["public split", args.split],
                ["device", args.device],
                ["completed OK cases", len(ok_rows)],
                ["total recorded rows", len(rows)],
            ],
        ),
        "",
        "## Public Split",
        "",
        md_table(["Dataset", "Split", "Samples", "Shape", "Classes", "Label Counts"], dataset_info),
        "",
        "## Method Summary",
        "",
        md_table(["Method", "OK Cases", "Mean Acc", "Min Acc", "Max Acc"], summary_rows),
    ]
    if deltas:
        lines.extend(
            [
                "",
                "## Source-to-Public Shift",
                "",
                md_table(
                    ["Compared method", "Matched cases", "Mean public-source acc", "Min shift", "Max shift"],
                    [["my_merge", len(deltas), f"{mean(deltas):+.4f}", f"{min(deltas):+.4f}", f"{max(deltas):+.4f}"]],
                ),
            ]
        )

    lines.extend(["", "## Small", ""])
    for model in args.small_models:
        emit_table(lines, ok_rows, "small", model, args.datasets, methods)
    if args.include_vlm:
        lines.extend(["## VLM", ""])
        for model in VLM_MODELS:
            emit_table(lines, ok_rows, "vlm", model, args.datasets, methods)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    rewrite_csv(args.summary_copy, [dict(row) for row in rows])


def main() -> None:
    args = parse_args()
    metrics_path = args.output_root / "reports" / "public_test_eval_summary.csv"
    source_cfg = load_source_config(args.source_config)
    manifest = selected_manifest(args)
    if not manifest:
        raise SystemExit("No manifest rows matched public-test scope.")
    keys_done = existing_keys(metrics_path, rerun_failures=args.rerun_failures) if args.resume else set()
    total = len(manifest) * len(args.methods)
    print(f"public-test start | rows={len(manifest)} methods={len(args.methods)} total={total}")
    print(f"metrics={metrics_path}")
    done = 0
    for row in manifest:
        for method in args.methods:
            done += 1
            base = {
                "task_type": row["task_type"],
                "dataset": row["dataset"],
                "model": row["model"],
                "clip_model": row["clip_model"],
                "num_clients": int(row["num_clients"]),
                "beta": norm_beta(row["beta"]),
                "seed": int(row["seed"]),
                "method": method,
                "split": args.split,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            key = public_key({k: str(v) for k, v in base.items()})
            if key in keys_done:
                print(f"[{done}/{total}] skip existing {key}", flush=True)
                continue
            start = time.time()
            try:
                print(f"[{done}/{total}] merge+public-eval {key}", flush=True)
                cfg = build_cfg(row, method, args, source_cfg)
                merge_info = run_merge(cfg)
                checkpoint_path = Path(str(merge_info["merged_checkpoint"]))
                meta = load_json(Path(str(merge_info["meta_path"])))
                checkpoint = load_checkpoint(checkpoint_path, device="cpu")
                eval_cfg = dict(cfg)
                eval_cfg["data_root"] = str(args.public_data_root)
                eval_cfg["split"] = args.split
                metrics = evaluate_checkpoint_predictions(meta, checkpoint, eval_cfg)
                result = dict(base)
                result.update(metrics)
                result.update(
                    {
                        "status": "OK",
                        "error": "",
                        "seconds": f"{time.time() - start:.2f}",
                        "merged_checkpoint": str(checkpoint_path),
                        "prototype_root": str(args.prototype_root),
                        "merge_data_root": str(args.source_data_root),
                        "eval_data_root": str(args.public_data_root),
                    }
                )
                append_csv(metrics_path, result)
                keys_done.add(key)
                if args.delete_merged and checkpoint_path.exists():
                    checkpoint_path.unlink()
                build_report(args, metrics_path)
            except Exception as exc:
                result = dict(base)
                result.update(
                    {
                        "status": "FAIL",
                        "error": str(exc),
                        "seconds": f"{time.time() - start:.2f}",
                        "prototype_root": str(args.prototype_root),
                        "merge_data_root": str(args.source_data_root),
                        "eval_data_root": str(args.public_data_root),
                    }
                )
                append_csv(metrics_path, result)
                build_report(args, metrics_path)
                print(f"[{done}/{total}] FAIL {key}: {exc}", flush=True)
    build_report(args, metrics_path)
    print(f"public-test done | metrics={metrics_path} | report={args.report_path}")


if __name__ == "__main__":
    main()
