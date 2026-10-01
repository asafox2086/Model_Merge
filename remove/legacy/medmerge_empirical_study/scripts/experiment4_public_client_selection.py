#!/usr/bin/env python3
"""Select the best client using public validation data and test on source data."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 4: public validation client selection.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--model-hub-root", type=Path, default=None)
    parser.add_argument("--source-data-root", type=Path, default=None)
    parser.add_argument("--public-data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPO_ROOT / "medmerge_empirical_study/results/experiment4_public_client_selection")
    parser.add_argument("--task-type", choices=["small", "vlm"], default="small")
    parser.add_argument("--datasets", nargs="*", default=["bloodmnist", "dermamnist"])
    parser.add_argument("--small-models", nargs="*", default=["resnet"])
    parser.add_argument("--clip-models", nargs="*", default=None)
    parser.add_argument("--num-clients", nargs="*", type=int, default=[3])
    parser.add_argument("--betas", nargs="*", type=float, default=[0.0, 0.01, 0.1])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def setup_repo_imports(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    scripts_dir = repo_root / "medmerge_empirical_study" / "scripts"
    for path in [repo_root, scripts_dir]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
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


def setting_key(row: dict[str, object]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(row["task_type"]),
        str(row["dataset"]),
        str(row.get("model", "")),
        str(row.get("clip_model", "")),
        str(row["num_clients"]),
        format(float(row["beta"]), "g"),
        str(row["seed"]),
    )


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    return float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1])


def rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def evaluate_clients_for_split(meta: dict[str, object], checkpoint_paths: list[Path], cfg: dict[str, object]) -> list[dict[str, object]]:
    from experiment2_client_pool import evaluate_client_pool

    rows = evaluate_client_pool(meta, checkpoint_paths, cfg)
    return [row for row in rows if str(row["baseline"]).startswith("single_client_")]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def summarize_selection(selection_rows: list[dict[str, object]], score_rows: list[dict[str, object]], out_path: Path) -> None:
    by_rule: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selection_rows:
        by_rule[str(row["selection_rule"])].append(row)

    source_rule = "source_val_accuracy_best"
    public_rule = "public_val_accuracy_best"
    oracle_rule = "oracle_test_accuracy_best"
    paired_by_rule: dict[str, dict[tuple[str, str, str, str, str, str, str], dict[str, object]]] = defaultdict(dict)
    for row in selection_rows:
        paired_by_rule[str(row["selection_rule"])][setting_key(row)] = row

    public_regrets = []
    public_matches_source = 0
    public_matches_oracle = 0
    for key, public_row in paired_by_rule[public_rule].items():
        source_row = paired_by_rule[source_rule].get(key)
        oracle_row = paired_by_rule[oracle_rule].get(key)
        if source_row is not None:
            public_regrets.append(float(source_row["source_test_accuracy"]) - float(public_row["source_test_accuracy"]))
            public_matches_source += int(str(source_row["selected_client_id"]) == str(public_row["selected_client_id"]))
        if oracle_row is not None:
            public_matches_oracle += int(str(oracle_row["selected_client_id"]) == str(public_row["selected_client_id"]))

    correlations = [row for row in score_rows if row["row_type"] == "setting_correlation"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# 实验四：公开验证集选择最强客户端\n\n")
        handle.write("## 工作流核对\n\n")
        handle.write("固定已训练客户端 checkpoint，不做权重融合。对每个 setting，先用 public validation set 给所有客户端打分，选择分数最高的客户端，再在源域 test set 上评估被选中的客户端。\n\n")
        handle.write("## 选择规则摘要\n\n")
        handle.write("| selection rule | cases | mean source-test acc | mean source-test bacc | mean source-test macro F1 | mean selected public acc | mean selected source-val acc |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for rule in sorted(by_rule):
            rows = by_rule[rule]
            handle.write(
                f"| {rule} | {len(rows)} | "
                f"{mean(float(row['source_test_accuracy']) for row in rows):.4f} | "
                f"{mean(float(row['source_test_balanced_accuracy']) for row in rows):.4f} | "
                f"{mean(float(row['source_test_macro_f1']) for row in rows):.4f} | "
                f"{mean(float(row['public_val_accuracy']) for row in rows):.4f} | "
                f"{mean(float(row['source_val_accuracy']) for row in rows):.4f} |\n"
            )

        handle.write("\n## Public 选择相对 Source-Val 选择的 regret\n\n")
        if public_regrets:
            handle.write(f"- cases: {len(public_regrets)}\n")
            handle.write(f"- mean regret: {mean(public_regrets):+.4f}\n")
            handle.write(f"- max regret: {max(public_regrets):+.4f}\n")
            handle.write(f"- public 与 source-val 选中同一客户端: {public_matches_source}/{len(public_regrets)}\n")
            handle.write(f"- public 与 oracle-test 选中同一客户端: {public_matches_oracle}/{len(public_regrets)}\n")
        else:
            handle.write("没有可计算的 paired setting。\n")

        handle.write("\n## 排名相关性\n\n")
        if correlations:
            handle.write("| dataset | model | clients | beta | public acc vs test acc | source-val acc vs test acc |\n")
            handle.write("|---|---|---:|---:|---:|---:|\n")
            for row in correlations:
                handle.write(
                    f"| {row['dataset']} | {row['model']} | {row['num_clients']} | {row['beta']} | "
                    f"{float(row['spearman_public_acc_source_test_acc']):.4f} | "
                    f"{float(row['spearman_source_val_acc_source_test_acc']):.4f} |\n"
                )

        handle.write("\n## 结论读取方式\n\n")
        handle.write("如果 `public_val_accuracy_best` 的 source-test accuracy 接近 `source_val_accuracy_best`，说明公开数据可以作为客户端选择代理；如果 regret 明显为正，说明 public 排名不能稳定替代源域验证集。\n")


def main() -> None:
    args = parse_args()
    setup_repo_imports(args.repo_root)

    from utils import ensure_checkpoint_files, ensure_task_matches_config, find_hub_experiment_dir, load_hub_meta

    args.model_hub_root = args.model_hub_root or args.repo_root / "model_hub"
    args.source_data_root = args.source_data_root or args.repo_root / "Med_data"
    args.public_data_root = args.public_data_root or args.repo_root / "PublicMedLabeled_data"

    manifest = filter_manifest(load_manifest(args.model_hub_root / "manifest.csv"), args)
    available_public = {path.stem for path in args.public_data_root.glob("*.npz")}
    manifest = [row for row in manifest if row["dataset"] in available_public]
    if not manifest:
        raise SystemExit("No manifest rows matched the requested filters and public datasets.")

    selection_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    start_all = time.time()
    for idx, row in enumerate(manifest, start=1):
        cfg_base: dict[str, object] = {
            "task_type": row["task_type"],
            "dataset": row["dataset"],
            "model": row.get("model", ""),
            "clip_model": row.get("clip_model", ""),
            "num_clients": int(row["num_clients"]),
            "beta": float(row["beta"]),
            "seed": int(row["seed"]),
            "model_hub_root": str(args.model_hub_root),
            "device": args.device,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
        print(f"[{idx}/{len(manifest)}] public client selection {setting_key(cfg_base)}")
        setting_start = time.time()
        exp_dir = find_hub_experiment_dir(args.model_hub_root, cfg_base)
        meta = load_hub_meta(exp_dir)
        ensure_task_matches_config(meta, cfg_base)
        checkpoint_paths = ensure_checkpoint_files(exp_dir, meta)

        split_specs = {
            "public_val": (args.public_data_root, "val"),
            "source_val": (args.source_data_root, "val"),
            "source_test": (args.source_data_root, "test"),
        }
        split_rows: dict[str, dict[int, dict[str, object]]] = {}
        for split_name, (data_root, split) in split_specs.items():
            cfg = dict(cfg_base)
            cfg.update({"data_root": str(data_root), "split": split})
            rows = evaluate_clients_for_split(meta, checkpoint_paths, cfg)
            split_rows[split_name] = {int(row["client_id"]): row for row in rows}

        client_ids = sorted(split_rows["source_test"])
        for client_id in client_ids:
            base = {
                "row_type": "client_score",
                "task_type": row["task_type"],
                "dataset": row["dataset"],
                "model": row.get("model", ""),
                "clip_model": row.get("clip_model", ""),
                "num_clients": int(row["num_clients"]),
                "beta": format(float(row["beta"]), "g"),
                "seed": int(row["seed"]),
                "client_id": client_id,
            }
            for split_name in ["public_val", "source_val", "source_test"]:
                metrics = split_rows[split_name][client_id]
                base[f"{split_name}_accuracy"] = float(metrics["accuracy"])
                base[f"{split_name}_balanced_accuracy"] = float(metrics["balanced_accuracy"])
                base[f"{split_name}_macro_f1"] = float(metrics["macro_f1"])
                base[f"{split_name}_loss"] = float(metrics["loss"])
            score_rows.append(base)

        public_acc = [float(split_rows["public_val"][client_id]["accuracy"]) for client_id in client_ids]
        source_val_acc = [float(split_rows["source_val"][client_id]["accuracy"]) for client_id in client_ids]
        source_test_acc = [float(split_rows["source_test"][client_id]["accuracy"]) for client_id in client_ids]
        score_rows.append(
            {
                "row_type": "setting_correlation",
                "task_type": row["task_type"],
                "dataset": row["dataset"],
                "model": row.get("model", ""),
                "clip_model": row.get("clip_model", ""),
                "num_clients": int(row["num_clients"]),
                "beta": format(float(row["beta"]), "g"),
                "seed": int(row["seed"]),
                "spearman_public_acc_source_test_acc": spearman(public_acc, source_test_acc),
                "spearman_source_val_acc_source_test_acc": spearman(source_val_acc, source_test_acc),
            }
        )

        selection_rules = [
            ("public_val_accuracy_best", "public_val", "accuracy"),
            ("public_val_balanced_accuracy_best", "public_val", "balanced_accuracy"),
            ("source_val_accuracy_best", "source_val", "accuracy"),
            ("source_val_balanced_accuracy_best", "source_val", "balanced_accuracy"),
            ("oracle_test_accuracy_best", "source_test", "accuracy"),
        ]
        for rule, split_name, metric_name in selection_rules:
            selected_id = max(client_ids, key=lambda cid: float(split_rows[split_name][cid][metric_name]))
            source_test = split_rows["source_test"][selected_id]
            public_val = split_rows["public_val"][selected_id]
            source_val = split_rows["source_val"][selected_id]
            selection_rows.append(
                {
                    "task_type": row["task_type"],
                    "dataset": row["dataset"],
                    "model": row.get("model", ""),
                    "clip_model": row.get("clip_model", ""),
                    "num_clients": int(row["num_clients"]),
                    "beta": format(float(row["beta"]), "g"),
                    "seed": int(row["seed"]),
                    "selection_rule": rule,
                    "selected_client_id": selected_id,
                    "selection_split": split_name,
                    "selection_metric": metric_name,
                    "selection_score": float(split_rows[split_name][selected_id][metric_name]),
                    "public_val_accuracy": float(public_val["accuracy"]),
                    "public_val_balanced_accuracy": float(public_val["balanced_accuracy"]),
                    "public_val_macro_f1": float(public_val["macro_f1"]),
                    "source_val_accuracy": float(source_val["accuracy"]),
                    "source_val_balanced_accuracy": float(source_val["balanced_accuracy"]),
                    "source_val_macro_f1": float(source_val["macro_f1"]),
                    "source_test_accuracy": float(source_test["accuracy"]),
                    "source_test_balanced_accuracy": float(source_test["balanced_accuracy"]),
                    "source_test_macro_f1": float(source_test["macro_f1"]),
                    "source_test_loss": float(source_test["loss"]),
                    "seconds": f"{time.time() - setting_start:.2f}",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

    selection_fields = [
        "task_type",
        "dataset",
        "model",
        "clip_model",
        "num_clients",
        "beta",
        "seed",
        "selection_rule",
        "selected_client_id",
        "selection_split",
        "selection_metric",
        "selection_score",
        "public_val_accuracy",
        "public_val_balanced_accuracy",
        "public_val_macro_f1",
        "source_val_accuracy",
        "source_val_balanced_accuracy",
        "source_val_macro_f1",
        "source_test_accuracy",
        "source_test_balanced_accuracy",
        "source_test_macro_f1",
        "source_test_loss",
        "seconds",
        "updated_at",
    ]
    score_fields = [
        "row_type",
        "task_type",
        "dataset",
        "model",
        "clip_model",
        "num_clients",
        "beta",
        "seed",
        "client_id",
        "public_val_accuracy",
        "public_val_balanced_accuracy",
        "public_val_macro_f1",
        "public_val_loss",
        "source_val_accuracy",
        "source_val_balanced_accuracy",
        "source_val_macro_f1",
        "source_val_loss",
        "source_test_accuracy",
        "source_test_balanced_accuracy",
        "source_test_macro_f1",
        "source_test_loss",
        "spearman_public_acc_source_test_acc",
        "spearman_source_val_acc_source_test_acc",
    ]
    write_csv(args.out_dir / "client_selection_rows.csv", selection_rows, selection_fields)
    write_csv(args.out_dir / "client_score_rows.csv", score_rows, score_fields)
    summarize_selection(selection_rows, score_rows, args.out_dir / "public_client_selection_summary.md")
    print(f"Wrote selection rows to {args.out_dir / 'client_selection_rows.csv'}")
    print(f"Wrote score rows to {args.out_dir / 'client_score_rows.csv'}")
    print(f"Wrote summary to {args.out_dir / 'public_client_selection_summary.md'}")
    print(f"Total seconds: {time.time() - start_all:.2f}")


if __name__ == "__main__":
    main()
