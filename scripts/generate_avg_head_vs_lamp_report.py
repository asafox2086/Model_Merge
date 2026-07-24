#!/usr/bin/env python3
"""Generate an ACC/F1 report for the averaged classifier-head baseline."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
]
MODELS = ["resnet", "convnext", "vit_t", "swin_tiny"]
CLIENTS = [3, 5, 7]
BETAS = ["0", "0.01", "0.1"]


def beta_key(value: object) -> str:
    return format(float(value), "g")


def read_method(path: Path, method: str) -> dict[tuple[str, str, int, str], tuple[float, float]]:
    values: dict[tuple[str, str, int, str], tuple[float, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "OK":
                continue
            if row.get("source") != "merge":
                continue
            if row.get("method") != method:
                continue
            key = (
                row["dataset"],
                row["model"],
                int(row["num_clients"]),
                beta_key(row["beta"]),
            )
            values[key] = (float(row["accuracy"]), float(row["macro_f1"]))
    return values


def fmt_pair(value: tuple[float, float] | None) -> str:
    if value is None:
        return "-"
    return f"{value[0]:.4f} / {value[1]:.4f}"


def fmt_delta(value: tuple[float, float] | None) -> str:
    if value is None:
        return "-"
    return f"{value[0]:+.4f} / {value[1]:+.4f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    aligns = ["---"] + ["---:" for _ in headers[1:]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def raw_headers() -> list[str]:
    headers = ["method"]
    for dataset in DATASETS:
        for client in CLIENTS:
            for beta in BETAS:
                headers.append(f"{dataset}:c{client}_b{beta}")
    return headers


def raw_rows(
    model: str,
    avg_head: dict[tuple[str, str, int, str], tuple[float, float]],
    lamp: dict[tuple[str, str, int, str], tuple[float, float]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for name, source in [("LAMP-Merge", lamp), ("avg_head", avg_head)]:
        row = [name]
        for dataset in DATASETS:
            for client in CLIENTS:
                for beta in BETAS:
                    row.append(fmt_pair(source.get((dataset, model, client, beta))))
        rows.append(row)
    delta = ["avg_head - LAMP-Merge"]
    for dataset in DATASETS:
        for client in CLIENTS:
            for beta in BETAS:
                a = avg_head.get((dataset, model, client, beta))
                l = lamp.get((dataset, model, client, beta))
                delta.append(fmt_delta((a[0] - l[0], a[1] - l[1])) if a and l else "-")
    rows.append(delta)
    return rows


def average(values: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not values:
        return None
    return (
        sum(item[0] for item in values) / len(values),
        sum(item[1] for item in values) / len(values),
    )


def client_average_rows(
    model: str,
    avg_head: dict[tuple[str, str, int, str], tuple[float, float]],
    lamp: dict[tuple[str, str, int, str], tuple[float, float]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for name, source in [("LAMP-Merge", lamp), ("avg_head", avg_head)]:
        row = [name]
        for dataset in DATASETS:
            vals = [
                source[(dataset, model, client, beta)]
                for client in CLIENTS
                for beta in BETAS
                if (dataset, model, client, beta) in source
            ]
            row.append(fmt_pair(average(vals)))
        all_vals = [
            source[(dataset, model, client, beta)]
            for dataset in DATASETS
            for client in CLIENTS
            for beta in BETAS
            if (dataset, model, client, beta) in source
        ]
        row.append(fmt_pair(average(all_vals)))
        rows.append(row)

    delta_row = ["avg_head - LAMP-Merge"]
    for dataset in DATASETS:
        a = average(
            [
                avg_head[(dataset, model, client, beta)]
                for client in CLIENTS
                for beta in BETAS
                if (dataset, model, client, beta) in avg_head
            ]
        )
        l = average(
            [
                lamp[(dataset, model, client, beta)]
                for client in CLIENTS
                for beta in BETAS
                if (dataset, model, client, beta) in lamp
            ]
        )
        delta_row.append(fmt_delta((a[0] - l[0], a[1] - l[1])) if a and l else "-")
    a_all = average(list(v for k, v in avg_head.items() if k[1] == model))
    l_all = average(list(v for k, v in lamp.items() if k[1] == model))
    delta_row.append(fmt_delta((a_all[0] - l_all[0], a_all[1] - l_all[1])) if a_all and l_all else "-")
    rows.append(delta_row)
    return rows


def coverage(values: dict[tuple[str, str, int, str], tuple[float, float]]) -> dict[str, int]:
    by_model: dict[str, int] = defaultdict(int)
    for _, model, _, _ in values:
        by_model[model] += 1
    return dict(by_model)


def main() -> None:
    avg_path = ROOT / "My_merge_ret" / "reports" / "avg_head_full_20260724c.csv"
    lamp_path = ROOT / "My_merge_ret" / "reports" / "prediction_diagnostics_full.csv"
    out_path = ROOT / "My_merge_ret" / "avg_head_vs_LAMP-Merge汇总表.md"

    avg_head = read_method(avg_path, "avg_head")
    lamp = read_method(lamp_path, "lamp_merge:full")
    expected = len(DATASETS) * len(MODELS) * len(CLIENTS) * len(BETAS)
    if len(avg_head) != expected:
        raise RuntimeError(f"avg_head has {len(avg_head)} cells, expected {expected}")
    if len(lamp) != expected:
        raise RuntimeError(f"LAMP-Merge has {len(lamp)} cells, expected {expected}")

    lines: list[str] = []
    lines.append("# avg_head 与 LAMP-Merge 汇总表")
    lines.append("")
    lines.append("- 实验配置与正式 LAMP-Merge 完全一致：5 个医学数据集、4 个 backbone、clients = 3/5/7、Dirichlet beta = 0/0.01/0.1、seed = 42、test split、equal merge weight。")
    lines.append("- 单元格格式为 `ACC / macro-F1`。差值行表示 `avg_head - LAMP-Merge`。")
    lines.append("- `avg_head` 使用共享 reference backbone，并将客户端训练后的 classifier head 参数逐元素平均；LAMP-Merge 使用 reference-space diagnostic prototype reconstruction 与 long-tail prevalence calibration 构造诊断分类头。")
    lines.append("")

    all_avg = average(list(avg_head.values()))
    all_lamp = average(list(lamp.values()))
    lines.append("## Overall")
    lines.append("")
    lines.append(
        table(
            ["method", "cells", "mean ACC / F1"],
            [
                ["LAMP-Merge", str(len(lamp)), fmt_pair(all_lamp)],
                ["avg_head", str(len(avg_head)), fmt_pair(all_avg)],
                [
                    "avg_head - LAMP-Merge",
                    str(len(avg_head)),
                    fmt_delta((all_avg[0] - all_lamp[0], all_avg[1] - all_lamp[1])),
                ],
            ],
        )
    )
    lines.append("")

    for model in MODELS:
        lines.append(f"## {model}")
        lines.append("")
        lines.append("### Raw")
        lines.append("")
        lines.append(table(raw_headers(), raw_rows(model, avg_head, lamp)))
        lines.append("")
        lines.append("### Client Average")
        lines.append("")
        lines.append(table(["method", *DATASETS, "Avg"], client_average_rows(model, avg_head, lamp)))
        lines.append("")

    lines.append("## Completeness Check")
    lines.append("")
    lines.append(table(["method", *MODELS, "total"], [
        ["LAMP-Merge", *[str(coverage(lamp).get(model, 0)) for model in MODELS], str(len(lamp))],
        ["avg_head", *[str(coverage(avg_head).get(model, 0)) for model in MODELS], str(len(avg_head))],
    ]))
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
