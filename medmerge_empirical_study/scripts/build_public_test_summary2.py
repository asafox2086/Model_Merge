#!/usr/bin/env python3
"""Build the public-test Summary Table 2 for the medical merge study."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parent
RESULTS = STUDY_ROOT / "results"
OUT_DIR = RESULTS / "experiment5_public_test_summary2"
PUBLIC_METRICS = OUT_DIR / "prediction_metrics.csv"
SOURCE_FORMAL = RESULTS / "experiment1" / "formal_accuracy_rows.csv"
LEAKAGE_METRICS = RESULTS / "experiment5_public_test_summary2_public_stats_leakage" / "prediction_metrics.csv"
SUMMARY_MD = OUT_DIR / "public_test_summary2.md"
FIG_DIR = OUT_DIR / "figures"
SOURCE_DATA_ROOT = REPO_ROOT / "Med_data"
PUBLIC_DATA_ROOT = REPO_ROOT / "PublicMedFingerprint_data"

METHOD_ORDER = [
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def f4(value: float) -> str:
    return f"{value:.4f}"


def sf4(value: float) -> str:
    return f"{value:+.4f}"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def norm_beta(raw: str) -> str:
    return format(float(raw), "g")


def public_dataset_name(raw: str) -> str:
    return raw.replace("_224", "")


def public_setting_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        public_dataset_name(row["dataset"]),
        row["task_type"],
        row["model"] or row["clip_model"],
        str(row["num_clients"]),
        norm_beta(row["beta"]),
        str(row["seed"]),
    )


def source_setting_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row["dataset"],
        row["task_type"],
        row["backbone"],
        str(row["num_clients"]),
        norm_beta(row["beta"]),
        "42",
    )


def method_key(row: dict[str, str]) -> tuple[tuple[str, str, str, str, str, str], str]:
    return public_setting_key(row), row["method"]


def average_rank_map(items: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(ordered[j][1], ordered[i][1], rel_tol=0.0, abs_tol=1e-12):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    return ranks


def ranks_by_setting(rows: list[dict[str, str]], setting_fn, accuracy_key: str = "accuracy") -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, ...], list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[setting_fn(row)].append((row["method"], f(row, accuracy_key)))
    out: dict[tuple[str, str], float] = {}
    for setting, items in grouped.items():
        ranks = average_rank_map(items)
        for method, rank in ranks.items():
            out[(setting, method)] = rank
    return out


def win_counts(rows: list[dict[str, str]], setting_fn, accuracy_key: str = "accuracy") -> dict[str, int]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[setting_fn(row)].append(row)
    wins: dict[str, int] = defaultdict(int)
    for items in grouped.values():
        best = max(f(row, accuracy_key) for row in items)
        for row in items:
            if math.isclose(f(row, accuracy_key), best, rel_tol=0.0, abs_tol=1e-12):
                wins[row["method"]] += 1
    return wins


def source_rows_for_scope() -> list[dict[str, str]]:
    rows = read_csv(SOURCE_FORMAL)
    methods = set(METHOD_ORDER)
    return [
        row
        for row in rows
        if row["dataset"] in {"bloodmnist", "dermamnist"}
        and row["task_type"] == "small"
        and row["backbone"] == "resnet"
        and row["num_clients"] == "3"
        and norm_beta(row["beta"]) in {"0", "0.01", "0.1"}
        and row["method"] in methods
    ]


def split_info(data_root: Path, dataset: str) -> dict[str, object]:
    data = np.load(data_root / f"{dataset}.npz")
    images = data["test_images"]
    labels = data["test_labels"].reshape(-1)
    counts = [(int(label), int((labels == label).sum())) for label in sorted(set(labels.tolist()))]
    return {
        "samples": int(images.shape[0]),
        "shape": "x".join(str(x) for x in images.shape[1:]),
        "num_classes": len(counts),
        "counts": counts,
    }


def counts_text(counts: list[tuple[int, int]]) -> str:
    return ", ".join(f"{label}:{count}" for label, count in counts)


def test_set_comparison_rows() -> list[list[object]]:
    rows = []
    for dataset in ["bloodmnist_224", "dermamnist_224"]:
        source = split_info(SOURCE_DATA_ROOT, dataset)
        public = split_info(PUBLIC_DATA_ROOT, dataset)
        rows.append(
            [
                dataset,
                source["samples"],
                public["samples"],
                source["num_classes"],
                public["num_classes"],
                source["shape"],
                public["shape"],
                counts_text(source["counts"]),
                counts_text(public["counts"]),
            ]
        )
    return rows


def summarize_public(public_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    ranks = ranks_by_setting(public_rows, public_setting_key)
    wins = win_counts(public_rows, public_setting_key)
    avg_by_setting = {public_setting_key(row): f(row, "accuracy") for row in public_rows if row["method"] == "avg"}
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in public_rows:
        by_method[row["method"]].append(row)

    out: list[dict[str, object]] = []
    for method, items in sorted(by_method.items(), key=lambda kv: mean(f(row, "accuracy") for row in kv[1]), reverse=True):
        low_count = 0
        deltas = []
        rank_values = []
        for row in items:
            support = [int(x) for x in row["class_support"].split()]
            random_prior = 1.0 / max(len(support), 1)
            if f(row, "accuracy") <= random_prior + 0.02:
                low_count += 1
            deltas.append(f(row, "accuracy") - avg_by_setting[public_setting_key(row)])
            rank_values.append(ranks[(public_setting_key(row), method)])
        out.append(
            {
                "method": method,
                "cases": len(items),
                "mean_accuracy": mean(f(row, "accuracy") for row in items),
                "median_accuracy": median(f(row, "accuracy") for row in items),
                "mean_balanced_accuracy": mean(f(row, "balanced_accuracy") for row in items),
                "mean_macro_f1": mean(f(row, "macro_f1") for row in items),
                "wins": wins.get(method, 0),
                "average_rank": mean(rank_values),
                "low_performance_rate": low_count / len(items),
                "mean_delta_vs_avg": mean(deltas),
                "mean_collapse": mean(f(row, "majority_prediction_ratio") for row in items),
                "median_effective_classes": median(f(row, "effective_predicted_classes") for row in items),
            }
        )
    return out


def public_summary_table(rows: list[dict[str, object]]) -> str:
    table = []
    for row in rows:
        table.append(
            [
                row["method"],
                row["cases"],
                f4(float(row["mean_accuracy"])),
                f4(float(row["median_accuracy"])),
                f4(float(row["mean_balanced_accuracy"])),
                f4(float(row["mean_macro_f1"])),
                row["wins"],
                f"{float(row['average_rank']):.2f}",
                pct(float(row["low_performance_rate"])),
                sf4(float(row["mean_delta_vs_avg"])),
                f4(float(row["mean_collapse"])),
                f"{float(row['median_effective_classes']):.2f}",
            ]
        )
    return md_table(
        [
            "方法",
            "Cases",
            "Mean Acc",
            "Median Acc",
            "Mean BAcc",
            "Mean Macro F1",
            "Wins",
            "Avg Rank",
            "Low Perf.",
            "Delta vs Avg",
            "Mean Collapse",
            "Median Eff. Classes",
        ],
        table,
    )


def build_source_public_rows(public_rows: list[dict[str, str]], source_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    public_lookup = {method_key(row): row for row in public_rows}
    out: list[dict[str, object]] = []
    for source in source_rows:
        setting = source_setting_key(source)
        public = public_lookup.get((setting, source["method"]))
        if public is None:
            continue
        source_acc = f(source, "accuracy")
        public_acc = f(public, "accuracy")
        out.append(
            {
                "dataset": setting[0],
                "task_type": setting[1],
                "backbone": setting[2],
                "num_clients": setting[3],
                "beta": setting[4],
                "seed": setting[5],
                "method": source["method"],
                "source_accuracy": source_acc,
                "public_accuracy": public_acc,
                "delta_public_minus_source": public_acc - source_acc,
                "abs_delta": abs(public_acc - source_acc),
            }
        )
    return out


def source_public_summary_table(joined: list[dict[str, object]], source_rows: list[dict[str, str]], public_rows: list[dict[str, str]]) -> str:
    source_ranks = ranks_by_setting(source_rows, source_setting_key)
    public_ranks = ranks_by_setting(public_rows, public_setting_key)
    source_wins = win_counts(source_rows, source_setting_key)
    public_wins = win_counts(public_rows, public_setting_key)
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in joined:
        by_method[str(row["method"])].append(row)

    table = []
    for method, items in sorted(by_method.items(), key=lambda kv: mean(float(row["delta_public_minus_source"]) for row in kv[1])):
        source_rank_values = []
        public_rank_values = []
        for row in items:
            setting = (row["dataset"], row["task_type"], row["backbone"], row["num_clients"], row["beta"], row["seed"])
            source_rank_values.append(source_ranks[(setting, method)])
            public_rank_values.append(public_ranks[(setting, method)])
        source_rank = mean(source_rank_values)
        public_rank = mean(public_rank_values)
        table.append(
            [
                method,
                len(items),
                f4(mean(float(row["source_accuracy"]) for row in items)),
                f4(mean(float(row["public_accuracy"]) for row in items)),
                sf4(mean(float(row["delta_public_minus_source"]) for row in items)),
                f4(mean(float(row["abs_delta"]) for row in items)),
                source_wins.get(method, 0),
                public_wins.get(method, 0),
                f"{source_rank:.2f}",
                f"{public_rank:.2f}",
                sf4(public_rank - source_rank),
            ]
        )
    return md_table(
        [
            "方法",
            "Cases",
            "Source Mean Acc",
            "Public Mean Acc",
            "Public-Source",
            "Mean Abs Shift",
            "Source Wins",
            "Public Wins",
            "Source Avg Rank",
            "Public Avg Rank",
            "Rank Shift",
        ],
        table,
    )


def spearman_from_ranks(a: dict[str, float], b: dict[str, float]) -> float:
    methods = sorted(set(a) & set(b))
    if len(methods) < 2:
        return float("nan")
    av = np.asarray([a[m] for m in methods], dtype=float)
    bv = np.asarray([b[m] for m in methods], dtype=float)
    if np.std(av) == 0.0 or np.std(bv) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def setting_switch_rows(source_rows: list[dict[str, str]], public_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    source_group: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    public_group: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        source_group[source_setting_key(row)].append(row)
    for row in public_rows:
        public_group[public_setting_key(row)].append(row)

    out: list[dict[str, object]] = []
    for setting in sorted(source_group):
        if setting not in public_group:
            continue
        source_items = source_group[setting]
        public_items = public_group[setting]
        source_ranks = average_rank_map([(row["method"], f(row, "accuracy")) for row in source_items])
        public_ranks = average_rank_map([(row["method"], f(row, "accuracy")) for row in public_items])
        source_best_acc = max(f(row, "accuracy") for row in source_items)
        public_best_acc = max(f(row, "accuracy") for row in public_items)
        source_best = [row["method"] for row in source_items if math.isclose(f(row, "accuracy"), source_best_acc, abs_tol=1e-12)]
        public_best = [row["method"] for row in public_items if math.isclose(f(row, "accuracy"), public_best_acc, abs_tol=1e-12)]
        source_mean = mean(f(row, "accuracy") for row in source_items)
        public_mean = mean(f(row, "accuracy") for row in public_items)
        out.append(
            {
                "dataset": setting[0],
                "beta": setting[4],
                "source_best_method": ",".join(sorted(source_best)),
                "source_best_accuracy": source_best_acc,
                "public_best_method": ",".join(sorted(public_best)),
                "public_best_accuracy": public_best_acc,
                "best_method_changed": ",".join(sorted(source_best)) != ",".join(sorted(public_best)),
                "source_mean_accuracy": source_mean,
                "public_mean_accuracy": public_mean,
                "mean_accuracy_shift": public_mean - source_mean,
                "rank_spearman": spearman_from_ranks(source_ranks, public_ranks),
            }
        )
    return out


def setting_switch_table(rows: list[dict[str, object]]) -> str:
    table = []
    for row in rows:
        table.append(
            [
                row["dataset"],
                row["beta"],
                row["source_best_method"],
                f4(float(row["source_best_accuracy"])),
                row["public_best_method"],
                f4(float(row["public_best_accuracy"])),
                "yes" if row["best_method_changed"] else "no",
                sf4(float(row["mean_accuracy_shift"])),
                f"{float(row['rank_spearman']):.3f}",
            ]
        )
    return md_table(
        [
            "Dataset",
            "Beta",
            "Source Best",
            "Source Best Acc",
            "Public Best",
            "Public Best Acc",
            "Best Changed",
            "Mean Acc Shift",
            "Rank Spearman",
        ],
        table,
    )


def leakage_table(strict_rows: list[dict[str, str]]) -> str:
    if not LEAKAGE_METRICS.exists():
        return "未找到 public-root 参与统计的辅助结果。"
    leakage_rows = [row for row in read_csv(LEAKAGE_METRICS) if row.get("status") == "OK"]
    strict_lookup = {method_key(row): row for row in strict_rows}
    leakage_lookup = {method_key(row): row for row in leakage_rows}
    rows = []
    for method in ["regmean", "fisher"]:
        diffs = []
        strict_acc = []
        leakage_acc = []
        for key, strict in strict_lookup.items():
            if key[1] != method or key not in leakage_lookup:
                continue
            leak = leakage_lookup[key]
            strict_acc.append(f(strict, "accuracy"))
            leakage_acc.append(f(leak, "accuracy"))
            diffs.append(f(leak, "accuracy") - f(strict, "accuracy"))
        if diffs:
            rows.append([method, len(diffs), f4(mean(strict_acc)), f4(mean(leakage_acc)), sf4(mean(diffs)), f4(max(abs(x) for x in diffs))])
    return md_table(["方法", "Cases", "Strict Public-Test Acc", "Public-Stats Acc", "PublicStats-Strict", "Max Abs Diff"], rows)


def build_figure(joined: list[dict[str, object]]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in joined:
        by_method[str(row["method"])].append(row)
    methods = [method for method in METHOD_ORDER if method in by_method]
    source = [mean(float(row["source_accuracy"]) for row in by_method[method]) for method in methods]
    public = [mean(float(row["public_accuracy"]) for row in by_method[method]) for method in methods]
    delta = [public[i] - source[i] for i in range(len(methods))]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    x = np.arange(len(methods))
    width = 0.38
    axes[0].bar(x - width / 2, source, width, label="Source test", color="#4C78A8")
    axes[0].bar(x + width / 2, public, width, label="Public test", color="#F58518")
    axes[0].set_xticks(x, methods, rotation=45, ha="right")
    axes[0].set_ylabel("Mean accuracy")
    axes[0].set_title("Source vs public test accuracy")
    axes[0].legend()

    colors = ["#54A24B" if value >= 0 else "#E45756" for value in delta]
    axes[1].bar(x, delta, color=colors)
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(x, methods, rotation=45, ha="right")
    axes[1].set_ylabel("Public - source accuracy")
    axes[1].set_title("Accuracy shift")
    fig.tight_layout()
    path = FIG_DIR / "fig_public_test_summary2_shift.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    public_rows = [row for row in read_csv(PUBLIC_METRICS) if row.get("status") == "OK"]
    source_rows = source_rows_for_scope()
    public_summary = summarize_public(public_rows)
    joined = build_source_public_rows(public_rows, source_rows)
    switch_rows = setting_switch_rows(source_rows, public_rows)
    figure_path = build_figure(joined)

    write_csv(OUT_DIR / "public_test_method_summary.csv", public_summary)
    write_csv(OUT_DIR / "source_public_accuracy_shift.csv", joined)
    write_csv(OUT_DIR / "source_public_setting_switches.csv", switch_rows)

    overall_source = mean(float(row["source_accuracy"]) for row in joined)
    overall_public = mean(float(row["public_accuracy"]) for row in joined)
    overall_abs_shift = mean(float(row["abs_delta"]) for row in joined)
    changed = sum(1 for row in switch_rows if row["best_method_changed"])
    mean_spearman = mean(float(row["rank_spearman"]) for row in switch_rows)

    public_sample_rows = [
        [dataset, samples]
        for dataset, samples in sorted({(row["dataset"], row["num_samples"]) for row in public_rows})
    ]

    md = f"""# 汇总表2：公开数据集作为 Test Set

本表使用 `PublicMedFingerprint_data/test` 作为公开测试集。融合/统计数据仍使用源域 `Med_data`，因此这是严格的 public-test evaluation，不让公开 test 图像参与融合统计。

## 实验范围

{md_table(
        ["项目", "设置"],
        [
            ["模型", "small / resnet"],
            ["数据集", "bloodmnist_224, dermamnist_224"],
            ["客户端数", "3"],
            ["Beta", "0, 0.01, 0.1"],
            ["Seed", "42"],
            ["方法数", str(len(METHOD_ORDER))],
            ["总 case", str(len(public_rows))],
            ["融合/统计数据", "`Med_data`"],
            ["评估数据", "`PublicMedFingerprint_data/test`"],
        ],
    )}

公开 test 样本数：

{md_table(["Dataset", "Public Test Samples"], public_sample_rows)}

## 新旧 Test Set 的异同

这里的“旧 test”指源域 `Med_data/test`，也就是训练好客户端和既有 formal accuracy 表使用的 in-domain test set；“新 test”指本次新增的 `PublicMedFingerprint_data/test`，用于模拟同类型公开医学数据上的外部测试。

相同点：

{md_table(
        ["维度", "相同点"],
        [
            ["任务类型", "二者都对应同一个 MedMNIST 子任务：BloodMNIST 或 DermaMNIST。"],
            ["类别空间", "同一数据集的新旧 test 类别 ID 一致；BloodMNIST 都是 8 类，DermaMNIST 都是 7 类。"],
            ["输入格式", "二者都是 `224x224x3` 图像，可直接用同一批已训练客户端和融合模型评估。"],
            ["评估对象", "二者评估的是同一组 `resnet + 3 clients + beta in {0,0.01,0.1} + seed 42 + 12 methods`。"],
        ],
    )}

不同点：

{md_table(
        ["维度", "旧 test: Med_data/test", "新 test: PublicMedFingerprint_data/test"],
        [
            ["域属性", "源域 in-domain test，和客户端训练/验证数据来自同一数据资产。", "外部公开 test，同任务同类别，但不保证采集来源、类别先验、图像风格与源域一致。"],
            ["实验角色", "用来衡量模型在原始源域任务上的性能。", "用来观察同类型公开数据上是否仍保持相同 accuracy 和方法排名。"],
            ["是否参与融合统计", "本实验中 Fisher/RegMean 等统计型融合仍使用 `Med_data`。", "严格主实验中不参与融合统计，只参与最终 evaluation。"],
            ["类别比例", "保留源域 test 的原始类别比例，DermaMNIST 明显偏向 class 5。", "BloodMNIST 为均衡 8 类；DermaMNIST 比源域更均衡，class 5 不再占绝对多数。"],
            ["可解释含义", "高分说明模型适配源域。", "分数和排名大幅变化说明源域适配不等价于外部公开数据鲁棒性。"],
        ],
    )}

样本数和类别分布：

{md_table(
        [
            "Dataset",
            "Old Samples",
            "New Samples",
            "Old Classes",
            "New Classes",
            "Old Shape",
            "New Shape",
            "Old Test Label Counts",
            "New Test Label Counts",
        ],
        test_set_comparison_rows(),
    )}

## 指标定义

{md_table(
        ["指标", "含义"],
        [
            ["Mean Acc", "该方法在所有 public-test cases 上的 accuracy 平均值。"],
            ["Median Acc", "该方法在所有 public-test cases 上的 accuracy 中位数。"],
            ["Mean BAcc", "Balanced accuracy，先算每类 recall 再对类别平均。"],
            ["Mean Macro F1", "Macro F1，先算每类 F1 再对类别平均。"],
            ["Wins", "每个 setting 内 accuracy 第一的次数；并列第一都计入。"],
            ["Avg Rank", "每个 setting 内按 accuracy 排名后的平均名次，越小越好；并列使用平均名次。"],
            ["Low Perf.", "`accuracy <= 1 / 类别数 + 0.02` 的 case 比例。"],
            ["Delta vs Avg", "同一 setting 下 `method_accuracy - avg_accuracy` 的平均值。"],
            ["Mean Collapse", "预测最多类别占所有预测的比例，越高说明越容易塌缩到少数类别。"],
            ["Median Eff. Classes", "由预测分布熵换算出的有效预测类别数，中位数越低说明预测类别越单一。"],
        ],
    )}

## Public-Test 主汇总表

{public_summary_table(public_summary)}

## 与源域 Test 的同设置 Accuracy 对照

源域列来自 `results/experiment1/formal_accuracy_rows.csv` 中同样的 `bloodmnist/dermamnist + resnet + c3 + beta in {{0,0.01,0.1}} + 12 methods`。这里比较的是 accuracy 和 rank；BAcc/Macro F1 只有本次逐样本 public-test 评估中有。

整体平均 accuracy 从源域 test 的 `{overall_source:.4f}` 变为 public test 的 `{overall_public:.4f}`，平均绝对变化为 `{overall_abs_shift:.4f}`。6 个 dataset/beta setting 中，top method 切换了 `{changed}/6` 个；平均 rank Spearman 为 `{mean_spearman:.3f}`。

{source_public_summary_table(joined, source_rows, public_rows)}

## 每个 Setting 的 Top Method 切换

{setting_switch_table(switch_rows)}

## 辅助检查：Public 图像参与统计会怎样

我还保留了第一版结果在 `results/experiment5_public_test_summary2_public_stats_leakage/`。那一版 `--data-root` 直接指向 public root，导致 Fisher/RegMean 的统计也看到了 public 数据，因此不作为主结论，只用来检查数据依赖方法对统计来源的敏感性。

{leakage_table(public_rows)}

## 图

![Source vs Public Accuracy Shift](figures/{figure_path.name})
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text(md, encoding="utf-8")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
