#!/usr/bin/env python3
"""Build expanded merge-vs-best-client comparison tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "article"
RESULTS = ROOT / "results"


def find_repo_root(study_root: Path) -> Path:
    for parent in [study_root.parent, *study_root.parents]:
        if (parent / "model_hub" / "manifest.csv").exists():
            return parent
    legacy = study_root.parent / "program" / "MedMNISTMerge"
    if (legacy / "model_hub" / "manifest.csv").exists():
        return legacy
    raise FileNotFoundError("Could not locate model_hub/manifest.csv")


REPO = find_repo_root(ROOT)
MODEL_HUB = REPO / "model_hub"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def f4(x: float) -> str:
    return f"{x:.4f}"


def sf4(x: float) -> str:
    return f"{x:+.4f}"


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def normalize_beta(raw: str) -> str:
    return format(float(raw), "g")


def dataset_no_suffix(name: str) -> str:
    return name.replace("_224", "")


def build_best_client_lookup() -> dict[tuple[str, str, str, str, str], dict[str, object]]:
    rows = read_csv(MODEL_HUB / "manifest.csv")
    lookup: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for row in rows:
        dataset = dataset_no_suffix(row["dataset"])
        task_type = row["task_type"]
        backbone = row["model"] if task_type == "small" else row["clip_model"]
        key = (dataset, task_type, backbone, row["num_clients"], normalize_beta(row["beta"]))
        meta = json.loads((MODEL_HUB / row["meta_path"]).read_text(encoding="utf-8"))
        clients = meta.get("clients", [])
        if not clients:
            continue
        best = max(clients, key=lambda item: float(item.get("test_acc", float("-inf"))))
        lookup[key] = {
            "best_client_id": best.get("client_id", ""),
            "best_client_test_acc": float(best["test_acc"]),
            "best_client_val_acc": float(best.get("best_val_acc", "nan")),
        }
    return lookup


def build_full_proxy_rows() -> list[dict[str, object]]:
    formal_rows = read_csv(RESULTS / "experiment1" / "formal_accuracy_rows.csv")
    best_lookup = build_best_client_lookup()
    out: list[dict[str, object]] = []
    missing = 0
    for row in formal_rows:
        key = (
            row["dataset"],
            row["task_type"],
            row["backbone"],
            row["num_clients"],
            normalize_beta(row["beta"]),
        )
        best = best_lookup.get(key)
        if best is None:
            missing += 1
            continue
        merged_acc = float(row["accuracy"])
        best_acc = float(best["best_client_test_acc"])
        out.append(
            {
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "backbone": row["backbone"],
                "method": row["method"],
                "num_clients": row["num_clients"],
                "beta": normalize_beta(row["beta"]),
                "merged_accuracy": f"{merged_acc:.6f}",
                "best_client_accuracy_proxy": f"{best_acc:.6f}",
                "delta_accuracy": f"{merged_acc - best_acc:.6f}",
                "best_client_id": best["best_client_id"],
                "best_client_val_acc": f"{float(best['best_client_val_acc']):.6f}",
                "source": "formal_accuracy_rows + model_hub_meta_test_acc",
            }
        )
    if missing:
        print(f"Missing best-client proxy for {missing} formal rows")
    return out


def summarize_full_proxy(rows: list[dict[str, object]]) -> str:
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    table_rows = []
    for method, items in sorted(by_method.items(), key=lambda kv: mean(float(r["delta_accuracy"]) for r in kv[1]), reverse=True):
        merged = [float(r["merged_accuracy"]) for r in items]
        best = [float(r["best_client_accuracy_proxy"]) for r in items]
        delta = [float(r["delta_accuracy"]) for r in items]
        better = sum(x > 0 for x in delta)
        near = sum(abs(x) <= 0.02 for x in delta)
        worse_2 = sum(x < -0.02 for x in delta)
        worse_5 = sum(x < -0.05 for x in delta)
        table_rows.append(
            [
                method,
                len(items),
                f4(mean(merged)),
                f4(median(merged)),
                f4(mean(best)),
                sf4(mean(delta)),
                sf4(median(delta)),
                sf4(percentile(delta, 25)),
                sf4(percentile(delta, 75)),
                f"{better}/{len(items)}",
                pct(better / len(items)),
                pct(near / len(items)),
                pct(worse_2 / len(items)),
                pct(worse_5 / len(items)),
            ]
        )
    return md_table(
        [
            "方法",
            "Cases",
            "Mean Merged Acc",
            "Median Merged Acc",
            "Mean Best Acc",
            "Mean Delta",
            "Median Delta",
            "P25 Delta",
            "P75 Delta",
            "Better",
            "Better Rate",
            "Near ±0.02",
            "Worse >0.02",
            "Worse >0.05",
        ],
        table_rows,
    )


def summarize_full_proxy_by(rows: list[dict[str, object]], field: str) -> str:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row[field]), str(row["method"]))].append(row)

    table_rows = []
    for (value, method), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        delta = [float(r["delta_accuracy"]) for r in items]
        merged = [float(r["merged_accuracy"]) for r in items]
        best = [float(r["best_client_accuracy_proxy"]) for r in items]
        better = sum(x > 0 for x in delta)
        table_rows.append(
            [
                value,
                method,
                len(items),
                f4(mean(merged)),
                f4(mean(best)),
                sf4(mean(delta)),
                pct(better / len(items)),
            ]
        )
    return md_table(
        [field, "方法", "Cases", "Mean Merged Acc", "Mean Best Acc", "Mean Delta", "Better Rate"],
        table_rows,
    )


def build_unified_subset_rows() -> list[dict[str, object]]:
    merged_rows = read_csv(RESULTS / "experiment2_all_methods" / "prediction_metrics.csv")
    client_rows = read_csv(RESULTS / "experiment2" / "client_pool_metrics.csv")
    best_by_beta = {
        normalize_beta(r["beta"]): r
        for r in client_rows
        if r["baseline"] == "best_single_client"
    }
    out: list[dict[str, object]] = []
    for row in merged_rows:
        if row.get("status", "OK") != "OK":
            continue
        beta = normalize_beta(row["beta"])
        best = best_by_beta[beta]
        merged_acc = float(row["accuracy"])
        best_acc = float(best["accuracy"])
        merged_bacc = float(row["balanced_accuracy"])
        best_bacc = float(best["balanced_accuracy"])
        merged_f1 = float(row["macro_f1"])
        best_f1 = float(best["macro_f1"])
        out.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "num_clients": row["num_clients"],
                "beta": beta,
                "method": row["method"],
                "merged_accuracy": f"{merged_acc:.6f}",
                "best_accuracy": f"{best_acc:.6f}",
                "delta_accuracy": f"{merged_acc - best_acc:.6f}",
                "merged_balanced_accuracy": f"{merged_bacc:.6f}",
                "best_balanced_accuracy": f"{best_bacc:.6f}",
                "delta_balanced_accuracy": f"{merged_bacc - best_bacc:.6f}",
                "merged_macro_f1": f"{merged_f1:.6f}",
                "best_macro_f1": f"{best_f1:.6f}",
                "delta_macro_f1": f"{merged_f1 - best_f1:.6f}",
                "merged_collapse": row["majority_prediction_ratio"],
                "best_collapse": best["majority_prediction_ratio"],
            }
        )
    return out


def summarize_unified_subset(rows: list[dict[str, object]]) -> str:
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    table_rows = []
    for method, items in sorted(by_method.items(), key=lambda kv: mean(float(r["delta_balanced_accuracy"]) for r in kv[1]), reverse=True):
        d_acc = [float(r["delta_accuracy"]) for r in items]
        d_bacc = [float(r["delta_balanced_accuracy"]) for r in items]
        d_f1 = [float(r["delta_macro_f1"]) for r in items]
        better_bacc = sum(x > 0 for x in d_bacc)
        better_f1 = sum(x > 0 for x in d_f1)
        table_rows.append(
            [
                method,
                len(items),
                f4(mean(float(r["merged_accuracy"]) for r in items)),
                f4(mean(float(r["best_accuracy"]) for r in items)),
                sf4(mean(d_acc)),
                f4(mean(float(r["merged_balanced_accuracy"]) for r in items)),
                f4(mean(float(r["best_balanced_accuracy"]) for r in items)),
                sf4(mean(d_bacc)),
                f4(mean(float(r["merged_macro_f1"]) for r in items)),
                f4(mean(float(r["best_macro_f1"]) for r in items)),
                sf4(mean(d_f1)),
                f"{better_bacc}/{len(items)}",
                f"{better_f1}/{len(items)}",
            ]
        )
    return md_table(
        [
            "方法",
            "Cases",
            "Merged Acc",
            "Best Acc",
            "Delta Acc",
            "Merged BAcc",
            "Best BAcc",
            "Delta BAcc",
            "Merged F1",
            "Best F1",
            "Delta F1",
            "BAcc Better",
            "F1 Better",
        ],
        table_rows,
    )


def detail_unified_subset(rows: list[dict[str, object]]) -> str:
    table_rows = []
    for row in sorted(rows, key=lambda r: (float(r["beta"]), str(r["method"]))):
        table_rows.append(
            [
                row["beta"],
                row["method"],
                f4(float(row["merged_accuracy"])),
                f4(float(row["best_accuracy"])),
                sf4(float(row["delta_accuracy"])),
                f4(float(row["merged_balanced_accuracy"])),
                f4(float(row["best_balanced_accuracy"])),
                sf4(float(row["delta_balanced_accuracy"])),
                f4(float(row["merged_macro_f1"])),
                f4(float(row["best_macro_f1"])),
                sf4(float(row["delta_macro_f1"])),
            ]
        )
    return md_table(
        [
            "Beta",
            "方法",
            "Merged Acc",
            "Best Acc",
            "Delta Acc",
            "Merged BAcc",
            "Best BAcc",
            "Delta BAcc",
            "Merged F1",
            "Best F1",
            "Delta F1",
        ],
        table_rows,
    )


def write_markdown(full_rows: list[dict[str, object]], unified_rows: list[dict[str, object]]) -> None:
    full_settings = {
        (
            row["dataset"],
            row["task_type"],
            row["backbone"],
            row["num_clients"],
            row["beta"],
        )
        for row in full_rows
    }
    full_summary = summarize_full_proxy(full_rows)
    full_by_dataset = summarize_full_proxy_by(full_rows, "dataset")
    full_by_clients = summarize_full_proxy_by(full_rows, "num_clients")
    full_by_beta = summarize_full_proxy_by(full_rows, "beta")
    unified_summary = summarize_unified_subset(unified_rows)
    unified_detail = detail_unified_subset(unified_rows)

    text = f"""# 融合模型与最优单客户端扩展对比

这个文件专门回答一个问题：已有模型融合 baseline 是否真正超过最优单客户端。

本文给出两个口径：

1. **统一重评子集**：DermaMNIST / ResNet / 3 clients / beta = 0, 0.01, 0.1，重新跑 12 个融合方法，记录 accuracy、balanced accuracy、macro F1 和 collapse。这个口径最严格，但覆盖面较小。
2. **全量 accuracy proxy**：使用 `experiment1/formal_accuracy_rows.csv` 中 12 方法、567 个 setting 的融合 accuracy，并用 `model_hub/meta.json` 中每个 setting 的客户端 `test_acc` 最大值作为 best-client proxy。这个口径覆盖面大，但 `model_hub` 里的客户端 test acc 不完全等同于当前统一重评脚本口径，因此只作为全量趋势诊断。

`Delta` 均表示：

```text
Delta = merged_method - best_single_client
```

## 1. 统一重评子集：12 方法 vs 最优客户端

覆盖范围：

- Dataset：DermaMNIST
- Backbone：ResNet
- Clients：3
- Beta：0, 0.01, 0.1
- Methods：12
- Cases：36

### 1.1 方法平均结果

{unified_summary}

观察：按 balanced accuracy 和 macro F1 看，绝大多数方法低于最优单客户端。少数方法在某些 beta 上能超过 best client，但平均并没有形成稳定优势。

### 1.2 分 beta 细表

{unified_detail}

## 2. 全量 accuracy proxy：12 方法 vs best-client proxy

覆盖范围：

- Datasets：7
- Model/backbone：8 个 small backbone + 1 个 CLIP VLM
- Clients：3, 5, 7
- Beta：0, 0.01, 0.1
- Fusion methods：12
- Matched settings：{len(full_settings)}
- Matched accuracy rows：{len(full_rows)}

注意：这里的 `Best Acc` 来自 `model_hub/meta.json` 中客户端训练日志记录的 `test_acc` 最大值。它能用于观察大范围趋势，但不能替代统一重评得到的 best-client 指标。正式融合表共有 567 个 setting / 6804 条 accuracy；其中 VLM 的 OrganAMNIST 和 PathMNIST 共 18 个 setting 在当前 `model_hub` 中没有可对齐的 best-client proxy，因此本节只汇总成功匹配的 setting。

### 2.1 方法平均结果

{full_summary}

### 2.2 按数据集分组

{full_by_dataset}

### 2.3 按客户端数量分组

{full_by_clients}

### 2.4 按 beta 分组

{full_by_beta}

## 3. 结论写法

更稳妥的论文表述是：

> Existing post-hoc merging methods exhibit near-best-client behavior rather than reliable complementary knowledge integration. In the unified DermaMNIST/ResNet/c3 subset, most methods fail to outperform the best single client under balanced accuracy and macro F1. The full accuracy-level proxy further suggests that, across many settings, merged models often remain close to or below the best-client proxy. This supports using best single client as a mandatory baseline for medical model merging.

中文表述：

> 现有后训练融合方法更多表现为“接近最优客户端”的行为，而不是稳定产生多客户端互补收益。在 DermaMNIST/ResNet/c3 的统一重评子集上，大多数方法在 balanced accuracy 和 macro F1 上不能超过最优单客户端；全量 accuracy proxy 也显示，融合模型通常停留在 best-client proxy 附近或以下。因此，医学模型融合实验必须报告 best single client baseline。
"""
    ARTICLE.mkdir(parents=True, exist_ok=True)
    (ARTICLE / "merge_vs_best_client_expanded.md").write_text(text, encoding="utf-8")


def main() -> None:
    full_rows = build_full_proxy_rows()
    unified_rows = build_unified_subset_rows()
    write_csv(RESULTS / "experiment2_all_methods" / "merge_vs_best_full_proxy_rows.csv", full_rows)
    write_csv(RESULTS / "experiment2_all_methods" / "merge_vs_best_unified_subset_rows.csv", unified_rows)
    write_markdown(full_rows, unified_rows)
    print(f"Wrote {len(full_rows)} full proxy rows")
    print(f"Wrote {len(unified_rows)} unified subset rows")
    print(f"Wrote {ARTICLE / 'merge_vs_best_client_expanded.md'}")


if __name__ == "__main__":
    main()
