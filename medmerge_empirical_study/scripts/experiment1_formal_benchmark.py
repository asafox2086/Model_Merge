#!/usr/bin/env python3
"""Summarize formal MedMNISTMerge result tables for experiment 1.

This script intentionally uses only the already published markdown result
tables. Those tables contain accuracy but not per-sample predictions, so the
outputs are accuracy-level stability diagnostics plus class-prior proxies.
Prediction-level metrics are produced by collect_prediction_metrics.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np


SETTING_RE = re.compile(r"^c(?P<num_clients>\d+)_b(?P<beta>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize formal benchmark markdown tables.")
    parser.add_argument("--repo-root", type=Path, default=Path("program/MedMNISTMerge"))
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("medmerge_empirical_study/results/experiment1"))
    return parser.parse_args()


def beta_to_float(raw: str) -> float:
    return float(raw.replace("p", "."))


def read_tables(result_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(result_dir.glob("*.md")):
        if path.name == "all_results.md":
            continue
        dataset = path.stem.replace("(1)", "")
        section = ""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("### "):
                section = line[4:].strip()
            if not line.startswith("| method |"):
                i += 1
                continue

            header = [cell.strip() for cell in line.strip().strip("|").split("|")]
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                cells = [cell.strip() for cell in lines[j].strip().strip("|").split("|")]
                if len(cells) == len(header):
                    method = cells[0]
                    for setting, value in zip(header[1:], cells[1:]):
                        match = SETTING_RE.match(setting)
                        if not match:
                            continue
                        rows.append(
                            {
                                "dataset": dataset,
                                "task_type": "vlm" if "clip" in section.lower() else "small",
                                "backbone": section,
                                "method": method,
                                "setting": setting,
                                "num_clients": int(match.group("num_clients")),
                                "beta": beta_to_float(match.group("beta")),
                                "accuracy": float(value),
                                "source_file": str(path),
                            }
                        )
                j += 1
            i = j
    return rows


def load_class_priors(data_root: Path, datasets: set[str]) -> dict[str, dict[str, object]]:
    priors: dict[str, dict[str, object]] = {}
    for dataset in sorted(datasets):
        candidates = [data_root / f"{dataset}.npz", data_root / f"{dataset}_224.npz"]
        npz_path = next((item for item in candidates if item.exists()), None)
        if npz_path is None:
            continue
        arr = np.load(npz_path)
        labels = arr["test_labels"].reshape(-1).astype(int)
        counts = np.bincount(labels)
        priors[dataset] = {
            "dataset": dataset,
            "npz_path": str(npz_path),
            "num_classes": int(len(counts)),
            "num_test_samples": int(labels.size),
            "majority_class": int(counts.argmax()),
            "majority_prior": float(counts.max() / max(labels.size, 1)),
            "random_prior": float(1.0 / max(len(counts), 1)),
            "class_counts": " ".join(str(int(x)) for x in counts.tolist()),
        }
    return priors


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize_methods(rows: list[dict[str, object]], priors: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    setting_keys = sorted({(row["dataset"], row["task_type"], row["backbone"], row["setting"]) for row in rows})
    by_key_method = {
        (row["dataset"], row["task_type"], row["backbone"], row["setting"], row["method"]): float(row["accuracy"])
        for row in rows
    }
    wins: Counter[str] = Counter()
    top3: Counter[str] = Counter()
    ranks: dict[str, list[int]] = defaultdict(list)
    deltas_vs_avg: dict[str, list[float]] = defaultdict(list)
    better_than_avg: Counter[str] = Counter()

    for key in setting_keys:
        candidates = [
            row
            for row in rows
            if (row["dataset"], row["task_type"], row["backbone"], row["setting"]) == key
        ]
        candidates.sort(key=lambda item: float(item["accuracy"]), reverse=True)
        if not candidates:
            continue
        best = float(candidates[0]["accuracy"])
        for rank, item in enumerate(candidates, start=1):
            method = str(item["method"])
            ranks[method].append(rank)
            if rank <= 3:
                top3[method] += 1
            if abs(float(item["accuracy"]) - best) <= 1e-12:
                wins[method] += 1
        avg_acc = by_key_method.get((*key, "avg"))
        if avg_acc is not None:
            for item in candidates:
                method = str(item["method"])
                delta = float(item["accuracy"]) - avg_acc
                deltas_vs_avg[method].append(delta)
                if delta > 1e-12:
                    better_than_avg[method] += 1

    summary: list[dict[str, object]] = []
    for method, items in sorted(by_method.items()):
        accs = [float(item["accuracy"]) for item in items]
        near_majority = 0
        low_performance = 0
        for item in items:
            prior = priors.get(str(item["dataset"]))
            if not prior:
                continue
            acc = float(item["accuracy"])
            if abs(acc - float(prior["majority_prior"])) <= 0.003:
                near_majority += 1
            if acc <= float(prior["random_prior"]) + 0.02:
                low_performance += 1
        deltas = deltas_vs_avg.get(method, [])
        summary.append(
            {
                "method": method,
                "num_cases": len(items),
                "mean_accuracy": f"{mean(accs):.6f}",
                "median_accuracy": f"{median(accs):.6f}",
                "p25_accuracy": f"{percentile(accs, 25):.6f}",
                "p75_accuracy": f"{percentile(accs, 75):.6f}",
                "win_count": wins[method],
                "top3_count": top3[method],
                "average_rank": f"{mean(ranks[method]):.6f}" if ranks[method] else "",
                "near_majority_prior_count": near_majority,
                "near_majority_prior_rate": f"{near_majority / max(len(items), 1):.6f}",
                "low_performance_count": low_performance,
                "low_performance_rate": f"{low_performance / max(len(items), 1):.6f}",
                "mean_delta_vs_avg": f"{mean(deltas):.6f}" if deltas else "",
                "median_delta_vs_avg": f"{median(deltas):.6f}" if deltas else "",
                "better_than_avg_count": better_than_avg[method],
                "better_than_avg_rate": f"{better_than_avg[method] / max(len(deltas), 1):.6f}" if deltas else "",
            }
        )
    summary.sort(key=lambda row: float(row["mean_accuracy"]), reverse=True)
    return summary


def summarize_groups(rows: list[dict[str, object]], priors: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    groups = {
        "weight_geometry": {
            "avg",
            "ties",
            "dare_linear",
            "dare_ties",
            "breadcrumbs",
            "model_stock",
            "from",
            "iso_c",
            "free_merge",
            "robustmerge",
        },
        "image_statistics": {"fisher", "regmean"},
    }
    out: list[dict[str, object]] = []
    for group_name, methods in groups.items():
        items = [row for row in rows if row["method"] in methods]
        accs = [float(row["accuracy"]) for row in items]
        near_majority = 0
        low_performance = 0
        for row in items:
            prior = priors.get(str(row["dataset"]))
            if not prior:
                continue
            acc = float(row["accuracy"])
            near_majority += int(abs(acc - float(prior["majority_prior"])) <= 0.003)
            low_performance += int(acc <= float(prior["random_prior"]) + 0.02)
        out.append(
            {
                "group": group_name,
                "methods": " ".join(sorted(methods)),
                "num_cases": len(items),
                "mean_accuracy": f"{mean(accs):.6f}",
                "median_accuracy": f"{median(accs):.6f}",
                "near_majority_prior_rate": f"{near_majority / max(len(items), 1):.6f}",
                "low_performance_rate": f"{low_performance / max(len(items), 1):.6f}",
            }
        )
    return out


def summarize_by_factor(rows: list[dict[str, object]], factor: str) -> list[dict[str, object]]:
    grouped: dict[tuple[object, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row[factor], str(row["method"]))].append(float(row["accuracy"]))
    out = []
    for (factor_value, method), values in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        out.append(
            {
                factor: factor_value,
                "method": method,
                "num_cases": len(values),
                "mean_accuracy": f"{mean(values):.6f}",
                "median_accuracy": f"{median(values):.6f}",
            }
        )
    return out


def write_markdown_summary(
    path: Path,
    rows: list[dict[str, object]],
    priors: dict[str, dict[str, object]],
    method_summary: list[dict[str, object]],
    group_summary: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = sorted({str(row["dataset"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    settings = {(row["dataset"], row["task_type"], row["backbone"], row["setting"]) for row in rows}
    top_methods = method_summary[:5]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 实验一：已有融合方法正式结果诊断\n\n")
        handle.write("## 工作流核对\n\n")
        handle.write("- 已使用已有训练好客户端对应的正式结果表。\n")
        handle.write("- 已覆盖 accuracy、average rank、win count、delta vs Avg、类别先验相关低效诊断。\n")
        handle.write("- 现有正式表不含逐样本预测，因此 balanced accuracy、macro F1、真实 collapse rate 需要通过 `collect_prediction_metrics.py` 重新评估生成。\n\n")
        handle.write("## 覆盖范围\n\n")
        handle.write(f"- 数据集数：{len(datasets)}，数据集：{', '.join(datasets)}\n")
        handle.write(f"- 方法数：{len(methods)}，方法：{', '.join(methods)}\n")
        handle.write(f"- setting 数：{len(settings)}\n")
        handle.write(f"- accuracy 记录数：{len(rows)}\n\n")
        handle.write("## 类别先验\n\n")
        handle.write("| dataset | classes | test n | majority prior | majority class |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for dataset in datasets:
            prior = priors.get(dataset)
            if not prior:
                continue
            handle.write(
                f"| {dataset} | {prior['num_classes']} | {prior['num_test_samples']} | "
                f"{float(prior['majority_prior']):.4f} | {prior['majority_class']} |\n"
            )
        handle.write("\n## 方法排名摘要\n\n")
        handle.write("| method | mean acc | median acc | wins | top3 | avg rank | low-performance rate | near-majority-prior rate | mean delta vs avg |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in method_summary:
            handle.write(
                f"| {row['method']} | {float(row['mean_accuracy']):.4f} | {float(row['median_accuracy']):.4f} | "
                f"{row['win_count']} | {row['top3_count']} | {float(row['average_rank']):.2f} | "
                f"{float(row['low_performance_rate']):.1%} | {float(row['near_majority_prior_rate']):.1%} | "
                f"{float(row['mean_delta_vs_avg'] or 0.0):+.4f} |\n"
            )
        handle.write("\n## 方法组摘要\n\n")
        handle.write("| group | cases | mean acc | median acc | low-performance rate | near-majority-prior rate |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in group_summary:
            handle.write(
                f"| {row['group']} | {row['num_cases']} | {float(row['mean_accuracy']):.4f} | "
                f"{float(row['median_accuracy']):.4f} | {float(row['low_performance_rate']):.1%} | "
                f"{float(row['near_majority_prior_rate']):.1%} |\n"
            )
        handle.write("\n## 初步结论\n\n")
        handle.write("1. 从正式 accuracy 表看，没有单一方法跨全部 setting 稳定领先。\n")
        handle.write("2. 多个方法的 median accuracy 接近数据集随机或多数类先验，提示需要逐样本预测诊断。\n")
        handle.write("3. DermaMNIST 等类别不均衡数据集的多数类先验很高，单看 accuracy 容易把多数类预测误判为有效融合。\n")
        handle.write("4. 下一步必须根据工作流运行逐样本评估，补充 balanced accuracy、macro F1、per-class recall 和真实 collapse rate。\n\n")
        handle.write("## Top methods by mean accuracy\n\n")
        for index, row in enumerate(top_methods, start=1):
            handle.write(f"{index}. {row['method']}: mean accuracy {float(row['mean_accuracy']):.4f}\n")


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir or args.repo_root / "result"
    data_root = args.data_root or args.repo_root / "Med_data"
    rows = read_tables(result_dir)
    if not rows:
        raise SystemExit(f"No result rows parsed from {result_dir}")
    priors = load_class_priors(data_root, {str(row["dataset"]) for row in rows})

    method_summary = summarize_methods(rows, priors)
    group_summary = summarize_groups(rows, priors)
    by_dataset = summarize_by_factor(rows, "dataset")
    by_beta = summarize_by_factor(rows, "beta")
    by_clients = summarize_by_factor(rows, "num_clients")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "formal_accuracy_rows.csv", rows)
    write_csv(args.out_dir / "class_priors.csv", list(priors.values()))
    write_csv(args.out_dir / "method_summary_accuracy.csv", method_summary)
    write_csv(args.out_dir / "method_group_summary_accuracy.csv", group_summary)
    write_csv(args.out_dir / "by_dataset_accuracy.csv", by_dataset)
    write_csv(args.out_dir / "by_beta_accuracy.csv", by_beta)
    write_csv(args.out_dir / "by_num_clients_accuracy.csv", by_clients)
    write_markdown_summary(
        args.out_dir / "experiment1_formal_accuracy_summary.md",
        rows,
        priors,
        method_summary,
        group_summary,
    )
    print(f"Wrote experiment 1 formal benchmark summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
