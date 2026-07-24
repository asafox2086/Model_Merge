#!/usr/bin/env python3
"""Replace only numeric values in the four main result tables of v5.tex.

The script keeps the existing LaTeX table layout, row names, colors, macros,
and ordering. It only rewrites ACC / Macro-F1 cell values and the LAMP-Merge
margin numbers inside the existing table rows.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
]

MODEL_TO_LABEL = {
    "swin_tiny": "tab:client-avg-swin-tiny",
    "convnext": "tab:client-avg-convnext",
    "resnet": "tab:client-avg-resnet",
    "vit_t": "tab:client-avg-vit-tiny",
}

TABLE_ROW_TO_METHOD = {
    "Weight Avg.": "head_avg",
    "TIES": "head_ties",
    "DARE-Linear": "head_dare_linear",
    "DARE-TIES": "head_dare_ties",
    "RegMean": "head_regmean",
    "Fisher": "head_fisher",
    "Breadcrumbs": "head_breadcrumbs",
    "Model Stock": "head_model_stock",
    "Iso-C": "head_iso_c",
    "Free-Merging": "head_free_merge",
    "RobustMerge": "head_robustmerge",
    "FROM": "head_from",
    "LAMP-Merge": "lamp_merge",
}

BASELINE_METHODS = [m for m in TABLE_ROW_TO_METHOD.values() if m != "lamp_merge"]
CLIENTS = [3, 5, 7]
BETA_VALUES = [0.0, 0.01, 0.1]


def beta_float(value: str) -> float:
    return float(value)


def read_metrics(paths: list[Path]) -> dict[tuple[str, str, str, int, float], tuple[float, float]]:
    lookup: dict[tuple[str, str, str, int, float], tuple[float, float]] = {}
    duplicate_count = defaultdict(int)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "OK" or row.get("source") != "merge":
                    continue
                method = row.get("method", "")
                if method == "lamp_merge:full":
                    method = "lamp_merge"
                dataset = row.get("dataset", "")
                model = row.get("model", "")
                if dataset not in DATASETS or model not in MODEL_TO_LABEL:
                    continue
                key = (
                    method,
                    dataset,
                    model,
                    int(float(row["num_clients"])),
                    beta_float(row["beta"]),
                )
                if not row.get("accuracy") or not row.get("macro_f1"):
                    continue
                lookup[key] = (float(row["accuracy"]) * 100.0, float(row["macro_f1"]) * 100.0)
                duplicate_count[key] += 1
    return lookup


def client_average(
    lookup: dict[tuple[str, str, str, int, float], tuple[float, float]],
    method: str,
    dataset: str,
    model: str,
    clients: int,
    allow_missing: bool = False,
) -> tuple[float, float]:
    values = []
    missing = []
    for beta in BETA_VALUES:
        key = (method, dataset, model, clients, beta)
        if key not in lookup:
            missing.append(key)
        else:
            values.append(lookup[key])
    if missing:
        if allow_missing:
            return (float("nan"), float("nan"))
        formatted = "\n".join(map(str, missing[:20]))
        raise RuntimeError(f"Missing ACC/F1 rows for {len(missing)} beta cases:\n{formatted}")
    return (
        sum(v[0] for v in values) / len(values),
        sum(v[1] for v in values) / len(values),
    )


def table_values(
    lookup: dict[tuple[str, str, str, int, float], tuple[float, float]],
    method: str,
    model: str,
    allow_missing: bool = False,
) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for dataset in DATASETS:
        dataset_values = []
        for clients in CLIENTS:
            value = client_average(lookup, method, dataset, model, clients, allow_missing=allow_missing)
            values.append(value)
            dataset_values.append(value)
        valid_values = [v for v in dataset_values if not (v[0] != v[0] or v[1] != v[1])]
        if len(valid_values) == len(dataset_values):
            values.append(
                (
                    sum(v[0] for v in dataset_values) / len(dataset_values),
                    sum(v[1] for v in dataset_values) / len(dataset_values),
                )
            )
        else:
            values.append((float("nan"), float("nan")))
    return values


def lamp_margins(
    lookup: dict[tuple[str, str, str, int, float], tuple[float, float]],
    model: str,
    lamp_values: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    margins = []
    idx = 0
    for dataset in DATASETS:
        for clients in CLIENTS:
            baseline_values = [
                client_average(lookup, method, dataset, model, clients)
                for method in BASELINE_METHODS
            ]
            best_acc = max(v[0] for v in baseline_values)
            best_f1 = max(v[1] for v in baseline_values)
            margins.append((lamp_values[idx][0] - best_acc, lamp_values[idx][1] - best_f1))
            idx += 1
        baseline_avg_values = []
        for method in BASELINE_METHODS:
            vals = [client_average(lookup, method, dataset, model, clients) for clients in CLIENTS]
            baseline_avg_values.append(
                (
                    sum(v[0] for v in vals) / len(vals),
                    sum(v[1] for v in vals) / len(vals),
                )
            )
        best_avg_acc = max(v[0] for v in baseline_avg_values)
        best_avg_f1 = max(v[1] for v in baseline_avg_values)
        margins.append((lamp_values[idx][0] - best_avg_acc, lamp_values[idx][1] - best_avg_f1))
        idx += 1
    return margins


def fmt_pair(value: tuple[float, float]) -> str:
    return f"{value[0]:.2f} / {value[1]:.2f}"


def fmt_margin(value: tuple[float, float]) -> str:
    return f"{value[0]:+.2f} / {value[1]:+.2f}"


def replace_row_numbers(row: str, new_values: list[tuple[float, float]]) -> str:
    pattern = re.compile(r"\d+\.\d+\s*/\s*\d+\.\d+")
    old = pattern.findall(row)
    if len(old) != len(new_values):
        raise RuntimeError(f"Expected {len(new_values)} numeric cells, found {len(old)} in row:\n{row[:240]}")
    values_iter = iter(new_values)

    def repl(match: re.Match[str]) -> str:
        value = next(values_iter)
        if value[0] != value[0] or value[1] != value[1]:
            return match.group(0)
        return fmt_pair(value)

    return pattern.sub(repl, row)


def replace_lamp_row_numbers(
    row: str,
    new_values: list[tuple[float, float]],
    new_margins: list[tuple[float, float]],
) -> str:
    cell_pattern = re.compile(
        r"\\tworowbestreshl\{\\textbf\{(?P<value>[-+]?\d+\.\d+\s*/\s*[-+]?\d+\.\d+)\}\}"
        r"\{(?P<margin>[-+]?\d+\.\d+\s*/\s*[-+]?\d+\.\d+)\}"
    )
    matches = list(cell_pattern.finditer(row))
    if len(matches) != len(new_values):
        raise RuntimeError(f"Expected {len(new_values)} LAMP cells, found {len(matches)} in row:\n{row[:240]}")
    idx = 0

    def repl(_match: re.Match[str]) -> str:
        nonlocal idx
        text = rf"\tworowbestreshl{{\textbf{{{fmt_pair(new_values[idx])}}}}}{{{fmt_margin(new_margins[idx])}}}"
        idx += 1
        return text

    return cell_pattern.sub(repl, row)


def replace_lamp_row_values_only(row: str, new_values: list[tuple[float, float]]) -> str:
    cell_pattern = re.compile(
        r"\\tworowbestreshl\{\\textbf\{(?P<value>[-+]?\d+\.\d+\s*/\s*[-+]?\d+\.\d+)\}\}"
        r"\{(?P<margin>[-+]?\d+\.\d+\s*/\s*[-+]?\d+\.\d+)\}"
    )
    matches = list(cell_pattern.finditer(row))
    if len(matches) != len(new_values):
        raise RuntimeError(f"Expected {len(new_values)} LAMP cells, found {len(matches)} in row:\n{row[:240]}")
    idx = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx
        value = new_values[idx]
        idx += 1
        if value[0] != value[0] or value[1] != value[1]:
            return match.group(0)
        return rf"\tworowbestreshl{{\textbf{{{fmt_pair(value)}}}}}{{{match.group('margin')}}}"

    return cell_pattern.sub(repl, row)


def replace_table_rows(
    tex: str,
    lookup: dict[tuple[str, str, str, int, float], tuple[float, float]],
    allow_missing: bool = False,
) -> str:
    updated = tex
    for model, label in MODEL_TO_LABEL.items():
        label_pos = updated.find(rf"\label{{{label}}}")
        if label_pos < 0:
            raise RuntimeError(f"Cannot find table label {label}")
        table_start = updated.rfind(r"\begin{table*}", 0, label_pos)
        table_end = updated.find(r"\end{table*}", label_pos)
        if table_start < 0 or table_end < 0:
            raise RuntimeError(f"Cannot locate table environment for {label}")
        table_end += len(r"\end{table*}")
        table = updated[table_start:table_end]

        for row_label, method in TABLE_ROW_TO_METHOD.items():
            row_pattern = re.compile(rf"(?m)^.*{re.escape(row_label)}.*\\\\$")
            match = row_pattern.search(table)
            if not match:
                raise RuntimeError(f"Cannot find row {row_label} in {label}")
            values = table_values(lookup, method, model, allow_missing=allow_missing)
            if method == "lamp_merge":
                if allow_missing:
                    new_row = replace_lamp_row_values_only(match.group(0), values)
                else:
                    new_row = replace_lamp_row_numbers(match.group(0), values, lamp_margins(lookup, model, values))
            else:
                new_row = replace_row_numbers(match.group(0), values)
            table = table[: match.start()] + new_row + table[match.end() :]

        updated = updated[:table_start] + table + updated[table_end:]
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, default=Path("v5.tex"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lookup = read_metrics(args.metrics)
    required = [
        (method, dataset, model, clients, beta)
        for method in TABLE_ROW_TO_METHOD.values()
        for dataset in DATASETS
        for model in MODEL_TO_LABEL
        for clients in CLIENTS
        for beta in BETA_VALUES
    ]
    missing = [key for key in required if key not in lookup]
    if missing and not args.allow_missing:
        print(f"Missing required ACC/F1 cells: {len(missing)}")
        for key in missing[:80]:
            print(key)
        raise SystemExit(2)
    if missing and args.allow_missing:
        print(f"Missing required ACC/F1 cells left unchanged: {len(missing)}")
        for key in missing[:80]:
            print(key)

    tex = args.tex.read_text(encoding="utf-8")
    updated = replace_table_rows(tex, lookup, allow_missing=args.allow_missing)
    if args.check_only:
        print("All required cells are present; table replacement dry-run passed.")
        return
    dest = args.out or args.tex
    dest.write_text(updated, encoding="utf-8")
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
