#!/usr/bin/env python3
"""Prepare the latest manuscript and paper-aligned experiment CSV files."""

from __future__ import annotations

import argparse
import csv
import html
import re
import shutil
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "My_merge_ret" / "reports"
M2_CLIENT_AVERAGE_REPORT = REPORT_DIR / "lamp_merge_m2_internal_ablation_full_client_average.csv"
FORMAL_ROOT = (
    ROOT
    / "outputs"
    / "lamp_merge_internal_ablation_full_20260718_formal_gamma055_s18p75_tau2p5_lambda4p25"
)
BASELINE_2X2_ROOT = (
    ROOT
    / "outputs"
    / "lamp_merge_baseline_2x2_20260719_followup_formal_baseline_2x2"
)
LPC_HPARAM_3X10_ROOT = (
    ROOT / "outputs" / "lamp_merge_hparam_3x10_20260719_followup_formal_hparam_3x10"
)
DPR_HPARAM_3X10_ROOT = ROOT / "outputs" / "lamp_merge_hparam_3x10_20260719_full_dpr"

DATASETS = [
    ("bloodmnist_224", "Blood"),
    ("dermamnist_224", "Derma"),
    ("organcmnist_224", "Organ-C"),
    ("organsmnist_224", "Organ-S"),
    ("chaoshengmnist_224", "Ultrasound"),
]
BACKBONES = [
    ("resnet", "ResNet", "tab:client-avg-resnet"),
    ("convnext", "ConvNeXt", "tab:client-avg-convnext"),
    ("vit_t", "ViT-Tiny", "tab:client-avg-vit-t"),
    ("swin_tiny", "Swin-Tiny", "tab:client-avg-swin-tiny"),
]

MAIN_METHOD_NAMES = {
    "avg": "Weight Averaging",
    "ties": "TIES-Merging",
    "dare_linear": "DARE-Linear",
    "dare_ties": "DARE-TIES",
    "regmean": "RegMean",
    "fisher": "Fisher",
    "breadcrumbs": "Breadcrumbs",
    "model_stock": "Model Stock",
    "from": "FROM",
    "iso_c": "Iso-C",
    "free_merge": "Free-Merging",
    "robustmerge": "RobustMerge",
    "LAMP-Merge": "LAMP-Merge",
}

MODE_COLUMNS = {
    "full": "LAMP-Merge",
    "avg_m2": "Baseline + LPC",
    "m1_only": "DPR only",
    "prototype_head_agg": "Classifier-head aggregation",
    "global_feature_mean": "Global-feature mean",
    "support_only": "Support-only synthetic head",
    "prototype_shuffle": "Shuffled-label prototype",
    "uniform_client_weight": "Uniform client weight",
    "binary_support": "Binary support only",
    "global_client_size_weight": "Global client-size weight",
    "no_prevalence": "No prevalence calibration",
    "uniform_prevalence": "Uniform prevalence prior",
    "always_on_calibration": "Always-on calibration",
    "client_balanced_prevalence": "Client-balanced prior",
}

BASELINE_2X2_CONFIGURATIONS = [
    ("TIES-Merging", "TIES-Merging", "No", "reference_ties_head"),
    ("TIES-Merging + LPC", "TIES-Merging", "Yes", "reference_ties_head_lpc"),
    ("DARE-Linear", "DARE-Linear", "No", "reference_dare_head"),
    ("DARE-Linear + LPC", "DARE-Linear", "Yes", "reference_dare_head_lpc"),
]

DATASET_STATS = [
    {
        "dataset": "Blood",
        "classes": 8,
        "train": 11959,
        "validation": 1712,
        "test": 3421,
        "total": 17092,
        "sample": "figures/sample_blood.png",
    },
    {
        "dataset": "Derma",
        "classes": 7,
        "train": 7007,
        "validation": 1003,
        "test": 2005,
        "total": 10015,
        "sample": "figures/sample_derma.png",
    },
    {
        "dataset": "Organ-C",
        "classes": 11,
        "train": 12975,
        "validation": 2392,
        "test": 8216,
        "total": 23583,
        "sample": "figures/sample_organ_c.png",
    },
    {
        "dataset": "Organ-S",
        "classes": 11,
        "train": 13932,
        "validation": 2452,
        "test": 8827,
        "total": 25211,
        "sample": "figures/sample_organ_s.png",
    },
    {
        "dataset": "Ultrasound",
        "classes": 8,
        "train": 3869,
        "validation": 549,
        "test": 1113,
        "total": 5531,
        "sample": "figures/sample_ultrasound.png",
    },
]


class TableBodyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag == "td" and self.current_row is not None:
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.current_cell is not None and self.current_row is not None:
            value = html.unescape("".join(self.current_cell)).strip()
            self.current_row.append(value)
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_annotated_csv(
    path: Path,
    description: str,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["说明", description])
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([row.get(field, "") for field in fieldnames])


def extract_master_client_average(backbone: str) -> list[list[str]]:
    text = (ROOT / "My_merge_ret" / "汇总表.md").read_text(encoding="utf-8")
    section = text.split(f"### {backbone}", 1)[1]
    if "\n### " in section:
        section = section.split("\n### ", 1)[0]
    section = section.split("#### Client Average", 1)[1]
    table = section[section.index("<table>") : section.index("</table>") + len("</table>")]
    parser = TableBodyParser()
    parser.feed(table)
    rows = [row for row in parser.rows if len(row) == 16]
    if len(rows) != 13:
        raise RuntimeError(f"Expected 13 methods for {backbone}, got {len(rows)}")
    return rows


def load_client_average_rows() -> list[dict[str, str]]:
    rows = read_csv(REPORT_DIR / "lamp_merge_internal_ablation_full_client_average.csv")
    m2_rows = read_csv(M2_CLIENT_AVERAGE_REPORT)
    m2_by_key = {
        (row["task_type"], row["dataset"], row["model"], int(row["num_clients"])): row
        for row in m2_rows
    }
    for row in rows:
        key = (row["task_type"], row["dataset"], row["model"], int(row["num_clients"]))
        if key not in m2_by_key:
            raise RuntimeError(f"Missing M2 internal-ablation cell: {key}")
        source = m2_by_key[key]
        for field in ("Always-on calibration", "Client-balanced prior"):
            row[field] = source[field]
    if len(m2_by_key) != 60:
        raise RuntimeError(f"Expected 60 M2 client-average cells, got {len(m2_by_key)}")
    return rows


def load_formal_lamp_values(client_rows: list[dict[str, str]]) -> dict[tuple[str, str, int], float]:
    values: dict[tuple[str, str, int], float] = {}
    for row in client_rows:
        key = (row["model"], row["dataset"], int(row["num_clients"]))
        values[key] = float(row["LAMP-Merge"])
    if len(values) != 60:
        raise RuntimeError(f"Expected 60 formal client-average cells, got {len(values)}")
    return values


def build_main_csvs(csv_dir: Path, formal_values: dict[tuple[str, str, int], float]) -> None:
    for backbone, backbone_label, _ in BACKBONES:
        source_rows = extract_master_client_average(backbone)
        output_rows: list[dict[str, object]] = []
        fields = ["Method"]
        for _, dataset_label in DATASETS:
            fields.extend(
                [
                    f"{dataset_label} K=3 ACC (%)",
                    f"{dataset_label} K=5 ACC (%)",
                    f"{dataset_label} K=7 ACC (%)",
                    f"{dataset_label} Avg ACC (%)",
                ]
            )

        for source in source_rows:
            source_method = source[0]
            method = MAIN_METHOD_NAMES.get(source_method, source_method)
            row: dict[str, object] = {"Method": method}
            values = [float(value) for value in source[1:]]
            for dataset_index, (dataset_key, dataset_label) in enumerate(DATASETS):
                if source_method == "LAMP-Merge":
                    dataset_values = [
                        formal_values[(backbone, dataset_key, num_clients)] for num_clients in (3, 5, 7)
                    ]
                else:
                    start = dataset_index * 3
                    dataset_values = values[start : start + 3]
                for num_clients, value in zip((3, 5, 7), dataset_values):
                    row[f"{dataset_label} K={num_clients} ACC (%)"] = f"{100 * value:.2f}"
                row[f"{dataset_label} Avg ACC (%)"] = f"{100 * sum(dataset_values) / 3:.2f}"
            output_rows.append(row)

        write_annotated_csv(
            csv_dir / f"主实验_{backbone_label}.csv",
            f"本表对应论文 {backbone_label} 主实验表；每个 K 数值对三个 Dirichlet beta 设置取平均，Avg 再对 K=3/5/7 取平均，单位为 ACC 百分比。",
            fields,
            output_rows,
        )


def aggregate_module_ablation(client_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    configurations = [
        ("Weight-Averaging Baseline", "Baseline", "avg"),
        ("Baseline + DPR", "Baseline + DPR", "M1 only"),
        ("LAMP-Merge (Baseline + DPR + LPC)", "LAMP-Merge", "LAMP-Merge"),
    ]
    by_dataset: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[str, list[float]] = defaultdict(list)
    for row in client_rows:
        dataset_label = dict(DATASETS)[row["dataset"]]
        for _, _, source_column in configurations:
            value = float(row[source_column])
            by_dataset[dataset_label][source_column].append(value)
            overall[source_column].append(value)

    output_rows: list[dict[str, object]] = []
    for setting, short_label, source_column in configurations:
        row: dict[str, object] = {
            "Setting": setting,
            "Paper label": short_label,
            "Overall ACC (%)": f"{100 * sum(overall[source_column]) / len(overall[source_column]):.2f}",
        }
        for _, dataset_label in DATASETS:
            values = by_dataset[dataset_label][source_column]
            row[f"{dataset_label} ACC (%)"] = f"{100 * sum(values) / len(values):.2f}"
        output_rows.append(row)
    return output_rows


def build_baseline_2x2_rows(client_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    baseline_values: dict[tuple[str, str], float] = {}
    for setting, baseline, lpc, mode in BASELINE_2X2_CONFIGURATIONS:
        rows = read_csv(BASELINE_2X2_ROOT / mode / "reports" / "batch_status.csv")
        if len(rows) != 180:
            raise RuntimeError(f"Expected 180 rows for {mode}, got {len(rows)}")
        failed = [row for row in rows if row["status"] != "OK"]
        if failed:
            raise RuntimeError(f"Found {len(failed)} failed rows for {mode}")

        cells: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        for row in rows:
            key = (row["dataset"], row["model"], int(row["num_clients"]))
            cells[key].append(float(row["test_acc"]))
        if len(cells) != 60 or any(len(values) != 3 for values in cells.values()):
            raise RuntimeError(f"Expected 60 three-beta client-average cells for {mode}")

        client_averages = {key: sum(values) / len(values) for key, values in cells.items()}
        baseline_values[(baseline, lpc)] = sum(client_averages.values()) / len(client_averages)

    dpr_only_values = [float(row["M1 only"]) for row in client_rows]
    full_values = [float(row["LAMP-Merge"]) for row in client_rows]
    if len(dpr_only_values) != 60 or len(full_values) != 60:
        raise RuntimeError("Expected 60 canonical client-average cells for DPR ablations")
    shared_values = {
        "Baseline + DPR": sum(dpr_only_values) / len(dpr_only_values),
        "Baseline + DPR + LPC": sum(full_values) / len(full_values),
    }

    output_rows = []
    configurations = [
        ("Baseline", "No", "No"),
        ("Baseline + DPR", "Yes", "No"),
        ("Baseline + LPC", "No", "Yes"),
        ("Baseline + DPR + LPC", "Yes", "Yes"),
    ]
    for baseline in ("TIES-Merging", "DARE-Linear"):
        for setting, dpr, lpc in configurations:
            if setting == "Baseline":
                value = baseline_values[(baseline, "No")]
            elif setting == "Baseline + LPC":
                value = baseline_values[(baseline, "Yes")]
            else:
                value = shared_values[setting]
            output_rows.append(
                {
                    "Baseline": baseline,
                    "Setting": setting,
                    "DPR": dpr,
                    "LPC": lpc,
                    "Client-average cells": 60,
                    "Overall ACC (%)": f"{100 * value:.2f}",
                }
            )
    return output_rows


def internal_ablation_rows(
    client_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    diagnostic_modes = [
        ("Classifier-head aggregation", "prototype_head_agg", "Classifier-head aggregation"),
        ("Global-feature mean", "global_feature_mean", "Global-feature mean"),
        ("Support-only synthetic head", "support_only", "Support-only synthetic head"),
        ("Shuffled-label prototype", "prototype_shuffle", "Shuffled-label prototype"),
        ("Uniform client weight", "uniform_client_weight", "Uniform client weight"),
        ("Binary support only", "binary_support", "Binary support only"),
        ("Global client-size weight", "global_client_size_weight", "Global client-size weight"),
        ("LAMP-Merge", "full", "LAMP-Merge"),
    ]
    prevalence_modes = [
        ("Uniform prevalence prior", "uniform_prevalence", "Uniform prevalence prior"),
        ("Always-on calibration", "always_on_calibration", "Always-on calibration"),
        ("Client-balanced prior", "client_balanced_prevalence", "Client-balanced prior"),
        ("LAMP-Merge", "full", "LAMP-Merge"),
    ]

    def build(entries: list[tuple[str, str, str]]) -> tuple[list[dict[str, object]], dict[str, float]]:
        output = []
        overall_values: dict[str, float] = {}
        for label, mode, source_column in entries:
            result: dict[str, object] = {"Setting": label, "Mode": mode}
            all_values = []
            for dataset_key, dataset_label in DATASETS:
                values = [
                    float(row[source_column]) for row in client_rows if row["dataset"] == dataset_key
                ]
                if len(values) != 12:
                    raise RuntimeError(f"Expected 12 {mode} cells for {dataset_key}, got {len(values)}")
                dataset_mean = sum(values) / len(values)
                result[f"{dataset_label} ACC (%)"] = f"{100 * dataset_mean:.2f}"
                all_values.extend(values)
            overall = sum(all_values) / len(all_values)
            result["Avg ACC (%)"] = f"{100 * overall:.2f}"
            overall_values[mode] = overall
            output.append(result)
        return output, overall_values

    diagnostic_rows, diagnostic_values = build(diagnostic_modes)
    prevalence_rows, prevalence_values = build(prevalence_modes)
    return diagnostic_rows, prevalence_rows, {**diagnostic_values, **prevalence_values}


def build_diagnostic_table() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows = read_csv(REPORT_DIR / "prediction_diagnostics_full.csv")
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "OK" and row["source"] == "merge":
            by_method[row["method"]].append(row)
    if len(by_method["lamp_merge:full"]) != 180:
        raise RuntimeError("Formal LAMP diagnostic coverage is incomplete")

    def mean(method: str, field: str, dataset: str | None = None) -> float:
        values = [
            float(row[field])
            for row in by_method[method]
            if row[field] and (dataset is None or row["dataset"] == dataset)
        ]
        return sum(values) / len(values)

    generic_methods = [method for method in by_method if not method.startswith("lamp_merge:")]
    metric_defs = [
        ("Collapse ratio", "lower", "collapse_ratio", 100.0),
        ("Effective predicted classes", "higher", "effective_predicted_classes", 1.0),
        ("Predicted-true TV", "lower", "pred_true_tv", 100.0),
    ]
    output_rows = []
    summary: dict[str, dict[str, object]] = {}
    for metric, direction, field, scale in metric_defs:
        lamp_row: dict[str, object] = {"Metric": metric, "Direction": direction, "Series": "LAMP-Merge"}
        reference_row: dict[str, object] = {
            "Metric": metric,
            "Direction": direction,
            "Series": "Best generic reference",
        }
        dataset_summary: dict[str, object] = {}
        reference_methods = []
        for dataset_key, dataset_label in DATASETS:
            lamp_value = mean("lamp_merge:full", field, dataset_key)
            reference_values = {
                method: mean(method, field, dataset_key) for method in generic_methods
            }
            chooser = min if direction == "lower" else max
            reference_method = chooser(reference_values, key=reference_values.get)
            reference_value = reference_values[reference_method]
            lamp_row[f"{dataset_label} Avg"] = f"{scale * lamp_value:.2f}"
            reference_row[f"{dataset_label} Avg"] = f"{scale * reference_value:.2f}"
            reference_methods.append(
                f"{dataset_label}: {MAIN_METHOD_NAMES.get(reference_method, reference_method)}"
            )
            dataset_summary[dataset_label] = {
                "lamp": lamp_value,
                "reference": reference_value,
                "reference_method": reference_method,
            }

        lamp_value = mean("lamp_merge:full", field)
        reference_values = {method: mean(method, field) for method in generic_methods}
        chooser = min if direction == "lower" else max
        reference_method = chooser(reference_values, key=reference_values.get)
        reference_value = reference_values[reference_method]
        lamp_row["Overall Avg"] = f"{scale * lamp_value:.2f}"
        reference_row["Overall Avg"] = f"{scale * reference_value:.2f}"
        lamp_row["Reference methods"] = "--"
        reference_row["Reference methods"] = "; ".join(
            reference_methods
            + [f"Overall: {MAIN_METHOD_NAMES.get(reference_method, reference_method)}"]
        )
        output_rows.extend([reference_row, lamp_row])
        summary[field] = {
            "metric": metric,
            "direction": direction,
            "scale": scale,
            "datasets": dataset_summary,
            "overall": {
                "lamp": lamp_value,
                "reference": reference_value,
                "reference_method": reference_method,
            },
        }
    return output_rows, summary


def parse_count_vector(value: str) -> list[float]:
    return [float(item) for item in value.strip().split()]


def build_ultrasound_distribution() -> list[dict[str, object]]:
    rows = read_csv(REPORT_DIR / "prediction_diagnostics_full.csv")
    methods = [
        ("lamp_merge:full", "LAMP-Merge"),
        ("ties", "TIES-Merging"),
        ("dare_linear", "DARE-Linear"),
        ("breadcrumbs", "Breadcrumbs"),
    ]
    sums = defaultdict(lambda: [0.0] * 8)
    counts = defaultdict(int)
    true_sums = [0.0] * 8
    true_count = 0
    for row in rows:
        if row["status"] != "OK" or row["source"] != "merge" or row["dataset"] != "chaoshengmnist_224":
            continue
        method = row["method"]
        if method in dict(methods):
            values = parse_count_vector(row["predicted_class_counts"])
            total = sum(values)
            for index, value in enumerate(values):
                sums[method][index] += value / total
            counts[method] += 1
        if method == "lamp_merge:full":
            values = parse_count_vector(row["true_counts"])
            total = sum(values)
            for index, value in enumerate(values):
                true_sums[index] += value / total
            true_count += 1

    true_distribution = [value / true_count for value in true_sums]
    output_rows = []
    for method, label in methods:
        distribution = [value / counts[method] for value in sums[method]]
        row: dict[str, object] = {"Series": label}
        for index, value in enumerate(distribution):
            absolute_difference = 100 * abs(value - true_distribution[index])
            row[f"Class {index} absolute difference vs. true (pp)"] = f"{absolute_difference:.4f}"
        output_rows.append(row)
    return output_rows


def build_hparam_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    expected_keys = {
        (dataset, model, num_clients, beta)
        for dataset, _ in DATASETS
        for model, _, _ in BACKBONES
        for num_clients in (3, 5, 7)
        for beta in (0.0, 0.01, 0.1)
    }
    dpr_pattern = re.compile(r"dpr_gamma_([0-9p]+)_s_([0-9p]+)$")
    lpc_pattern = re.compile(r"lpc_tau_([0-9p]+)_lambda_([0-9p]+)$")
    diagnostic: list[dict[str, object]] = []
    prevalence: list[dict[str, object]] = []

    def summarize_batch_status(path: Path) -> dict[str, object]:
        rows = read_csv(path / "reports" / "batch_status.csv")
        if len(rows) != 180 or any(row["status"] != "OK" for row in rows):
            raise RuntimeError(f"Incomplete hyperparameter run: {path}")
        if any(row["lamp_merge_ablation_mode"] != "full" for row in rows):
            raise RuntimeError(f"Hyperparameter run is not full LAMP-Merge: {path}")
        values_by_key: dict[tuple[str, str, int, float], float] = {}
        for row in rows:
            key = (row["dataset"], row["model"], int(row["num_clients"]), float(row["beta"]))
            values_by_key[key] = float(row["test_acc"])
        if set(values_by_key) != expected_keys:
            raise RuntimeError(f"Unexpected hyperparameter coverage: {path}")
        client_values = [
            sum(values_by_key[(dataset, model, num_clients, beta)] for beta in (0.0, 0.01, 0.1)) / 3
            for dataset, _ in DATASETS
            for model, _, _ in BACKBONES
            for num_clients in (3, 5, 7)
        ]
        return {
            "Raw cells": len(values_by_key),
            "Client-average cells": len(client_values),
            "Mean ACC (%)": f"{100 * sum(client_values) / len(client_values):.4f}",
        }

    def is_complete_batch(path: Path) -> bool:
        status_path = path / "reports" / "batch_status.csv"
        if not status_path.exists():
            return False
        rows = read_csv(status_path)
        return len(rows) == 180 and all(row["status"] == "OK" for row in rows)

    for child in sorted(DPR_HPARAM_3X10_ROOT.iterdir()):
        if not child.is_dir():
            continue
        match = dpr_pattern.match(child.name)
        if match is None or not is_complete_batch(child):
            continue
        diagnostic.append(
            {
                "Source": "Full configuration grid",
                "Evidence exponent gamma": float(match.group(1).replace("p", ".")),
                "Prototype-head scale s": float(match.group(2).replace("p", ".")),
                **summarize_batch_status(child),
            }
        )
    for child in sorted(LPC_HPARAM_3X10_ROOT.iterdir()):
        if not child.is_dir():
            continue
        match = lpc_pattern.match(child.name)
        if match is None or not is_complete_batch(child):
            continue
        prevalence.append(
            {
                "Source": "Full configuration grid",
                "Activation threshold tau": float(match.group(1).replace("p", ".")),
                "Calibration strength lambda": float(match.group(2).replace("p", ".")),
                **summarize_batch_status(child),
            }
        )
    if diagnostic and len(diagnostic) != 30:
        raise RuntimeError("Expected a complete full-configuration 3x10 DPR grid")
    if len(prevalence) != 30:
        raise RuntimeError("Expected a complete full-configuration 3x10 LPC grid")
    return diagnostic, prevalence


def crop_dataset_samples(paper_dir: Path) -> None:
    from PIL import Image

    intro_source = paper_dir / "figures" / "4a3123dfc3ae4ea3ce76e62b4cbc7d60.png"
    intro_target = paper_dir / "figures" / "intro.png"
    if intro_source.exists() and not intro_target.exists():
        shutil.copy2(intro_source, intro_target)

    names = ["blood", "ultrasound", "derma", "organ_c", "organ_s"]
    sample_paths = [paper_dir / "figures" / f"sample_{name}.png" for name in names]
    if all(path.exists() for path in sample_paths):
        return
    source = Image.open(paper_dir / "figures" / "dataset_examples.png")
    width, height = source.size
    centers = [0.096, 0.298, 0.500, 0.701, 0.903]
    crop_width = int(width * 0.114)
    top = int(height * 0.205)
    bottom = int(height * 0.49)
    for center, name in zip(centers, names):
        center_x = int(width * center)
        left = max(0, center_x - crop_width // 2)
        right = min(width, left + crop_width)
        sample = source.crop((left, top, right, bottom))
        sample.save(paper_dir / "figures" / f"sample_{name}.png")


def replace_table_block(tex: str, label: str, replacement: str) -> str:
    label_position = tex.rindex(rf"\label{{{label}}}")
    start = tex.rfind(r"\begin{table", 0, label_position)
    end_marker = r"\end{table"
    end = tex.index(end_marker, label_position)
    end = tex.index("}", end) + 1
    return tex[:start] + replacement + tex[end:]


def replace_figure_block(tex: str, label: str, replacement: str) -> str:
    label_position = tex.rindex(rf"\label{{{label}}}")
    start = tex.rfind(r"\begin{figure", 0, label_position)
    end = tex.index(r"\end{figure", label_position)
    end = tex.index("}", end) + 1
    return tex[:start] + replacement + tex[end:]


def remove_figure_block(tex: str, label: str) -> str:
    return replace_figure_block(tex, label, "")


def replace_lamp_row(
    tex: str,
    label: str,
    backbone: str,
    formal_values: dict[tuple[str, str, int], float],
) -> str:
    label_position = tex.rindex(rf"\label{{{label}}}")
    table_end = tex.index(r"\end{table*}", label_position)
    block = tex[label_position:table_end]
    lamp_position = block.rindex(r"\textbf{LAMP-Merge}")
    row_start = block.rfind(r"\rowcolor[HTML]{FFF9C4}", 0, lamp_position)
    if row_start < 0:
        raise RuntimeError(f"LAMP-Merge result row not found for {label}")
    row_end = block.index(r"\\", lamp_position) + 2
    values = []
    for dataset_key, _ in DATASETS:
        dataset_values = [formal_values[(backbone, dataset_key, clients)] for clients in (3, 5, 7)]
        values.extend(dataset_values)
        values.append(sum(dataset_values) / 3)
    cells = " &\n".join(rf"\textbf{{{100 * value:.2f}}}" for value in values)
    replacement = (
        r"\rowcolor[HTML]{FFF9C4}" + "\n"
        + r"\textbf{LAMP-Merge} &" + "\n"
        + cells
        + r" \\"
    )
    updated_block = block[:row_start] + replacement + block[row_end:]
    return tex[:label_position] + updated_block + tex[table_end:]


INTERNAL_SHORT_LABELS = {
    "Classifier-head aggregation": "CHA",
    "Global-feature mean": "GFM",
    "Support-only synthetic head": "SSH",
    "Shuffled-label prototype": "SLP",
    "Uniform client weight": "UCW",
    "Binary support only": "BSO",
    "Global client-size weight": "GCSW",
    "No prevalence calibration": "No LPC",
    "Uniform prevalence prior": "UPP",
    "Always-on calibration": "AOC",
    "Client-balanced prior": "CBP",
    "LAMP-Merge": "LAMP",
}


def build_internal_dataset_table(
    rows: list[dict[str, object]],
    caption: str,
    label: str,
    divider_mode: str | None = None,
    dpr_categories: bool = False,
) -> str:
    dataset_fields = [f"{dataset_label} ACC (%)" for _, dataset_label in DATASETS]
    body = []
    for index, row in enumerate(rows):
        mode = str(row["Mode"])
        if divider_mode and mode == divider_mode:
            body.append(r"\hdashline")
        setting = INTERNAL_SHORT_LABELS[str(row["Setting"])]
        values = [str(row[field]) for field in dataset_fields] + [str(row["Avg ACC (%)"])]
        category = ""
        if dpr_categories and mode == "prototype_head_agg":
            category = r"\multirow{4}{*}{\shortstack{Prototype\\semantics}}"
        elif dpr_categories and mode == "uniform_client_weight":
            category = r"\multirow{3}{*}{\shortstack{Client-support\\weighting}}"
        if mode == "full":
            body.append(r"\hline \hline")
            setting = rf"\textbf{{{setting}}}"
            values = [rf"\textbf{{{value}}}" for value in values]
            setting = rf"\cellcolor[HTML]{{FFF9C4}}{setting}"
            values = [rf"\cellcolor[HTML]{{FFF9C4}}{value}" for value in values]
        elif index % 2 and row["Setting"] != "Always-on calibration":
            setting = rf"\cellcolor{{gray!10}}{setting}"
            values = [rf"\cellcolor{{gray!10}}{value}" for value in values]
        if dpr_categories:
            setting = category + " & " + setting
        body.append(setting + " & " + " & ".join(values) + " \\\\")

    return "\n".join(
        [
            r"\begin{table}[!b]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\scriptsize",
            r"\renewcommand{\arraystretch}{1.20}",
            r"\setlength{\tabcolsep}{2.4pt}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{clccccc:c}" if dpr_categories else r"\begin{tabular}{lccccc:c}",
            r"\noalign{\hrule height 1.25pt}",
            r"\rowcolor[HTML]{F2F2F2}",
            (r"\textbf{Category} & \textbf{Setting} & \textbf{Blood} & \textbf{Derma} & \textbf{Organ-C} & \textbf{Organ-S} & \textbf{US} & \textbf{Avg} \\") if dpr_categories else r"\textbf{Setting} & \textbf{Blood} & \textbf{Derma} & \textbf{Organ-C} & \textbf{Organ-S} & \textbf{US} & \textbf{Avg} \\",
            r"\hline \hline",
            *body,
            r"\noalign{\hrule height 1.25pt}",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )


def build_collapse_dataset_table(diagnostic_summary: dict[str, dict[str, object]]) -> str:
    metric_labels = {
        "collapse_ratio": r"Collapse $\downarrow$",
        "effective_predicted_classes": r"Eff. classes $\uparrow$",
        "pred_true_tv": r"Pred-true TV $\downarrow$",
    }
    body = []
    for metric_index, (field, metric_label) in enumerate(metric_labels.items()):
        item = diagnostic_summary[field]
        scale = float(item["scale"])
        lamp_values = [
            scale * float(item["datasets"][dataset_label]["lamp"])
            for _, dataset_label in DATASETS
        ] + [scale * float(item["overall"]["lamp"])]
        reference_values = [
            scale * float(item["datasets"][dataset_label]["reference"])
            for _, dataset_label in DATASETS
        ] + [scale * float(item["overall"]["reference"])]
        if metric_index:
            body.append(r"\hdashline")
        body.extend(
            [
                r" & \cellcolor{gray!10}Best ref. & "
                + " & ".join(rf"\cellcolor{{gray!10}}{value:.2f}" for value in reference_values)
                + " \\\\",
                rf"\multirow{{-2}}{{*}}{{{metric_label}}}"
                + r" & \cellcolor[HTML]{FFF9C4}\textbf{LAMP} & "
                + " & ".join(rf"\cellcolor[HTML]{{FFF9C4}}\textbf{{{value:.2f}}}" for value in lamp_values)
                + " \\\\",
            ]
        )

    return "\n".join(
        [
            r"\begin{table}[!b]",
            r"\centering",
            r"\caption{Prediction-collapse diagnostics by dataset. Best ref. is selected separately for each metric and dataset. Ratio metrics use percentage scale; Avg averages the five datasets.}",
            r"\label{tab:collapse-key}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.2pt}",
            r"\renewcommand{\arraystretch}{1.16}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{clccccc:c}",
            r"\noalign{\hrule height 1.25pt}",
            r"\rowcolor[HTML]{F2F2F2}",
            r"\textbf{Metric} & \textbf{Series} & \textbf{Blood} & \textbf{Derma} & \textbf{Organ-C} & \textbf{Organ-S} & \textbf{US} & \textbf{Avg} \\",
            r"\hline \hline",
            *body,
            r"\noalign{\hrule height 1.25pt}",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )


def build_dataset_statistics_table() -> str:
    body = []
    for index, item in enumerate(DATASET_STATS):
        if index % 2:
            body.append(r"\rowcolor{gray!10}")
        sample = (
            r"\makebox[1.05cm][c]{\raisebox{-0.16cm}{\includegraphics"
            rf"[width=0.95cm,height=0.52cm,keepaspectratio]{{{item['sample']}}}}}}}"
        )
        body.append(
            f"{item['dataset']} & {item['classes']} & {item['train']:,} & {item['validation']:,} & "
            f"{item['test']:,} & {item['total']:,} & {sample} " + "\\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{Dataset statistics and representative samples for the five medical image benchmarks. Counts are taken from the exact NPZ files used in our experiments.}",
            r"\label{tab:dataset-statistics}",
            r"\scriptsize",
            r"\renewcommand{\arraystretch}{1.25}",
            r"\setlength{\tabcolsep}{2.2pt}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{lrrrrr:c}",
            r"\noalign{\hrule height 1.25pt}",
            r"\rowcolor[HTML]{F2F2F2}",
            r"\textbf{Dataset} & \textbf{Classes} & \textbf{Train} & \textbf{Val} & \textbf{Test} & \textbf{Total} & \textbf{Sample} \\",
            r"\hline \hline",
            *body,
            r"\noalign{\hrule height 1.25pt}",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )


def update_tex(
    paper_dir: Path,
    formal_values: dict[tuple[str, str, int], float],
    baseline_2x2_rows: list[dict[str, object]],
    diagnostic_rows: list[dict[str, object]],
    prevalence_rows: list[dict[str, object]],
    diagnostic_summary: dict[str, dict[str, object]],
    has_dpr_sensitivity: bool,
) -> None:
    path = paper_dir / "v4_3.tex"
    tex = path.read_text(encoding="utf-8")
    if r"\usepackage{placeins}" not in tex:
        tex = tex.replace(
            r"\usepackage{arydshln}",
            "\\usepackage{arydshln}\n\\usepackage{placeins}",
        )
    tex = tex.replace(r"where $\gamma=0.45$ is the default evidence exponent", r"where $\gamma=0.55$ is the default evidence exponent")
    tex = tex.replace(r"where $s=20$ is the default head scale", r"where $s=18.75$ is the default head scale")
    tex = tex.replace(r"with default maximum calibration strength $\lambda=5.0$", r"with default maximum calibration strength $\lambda=4.25$")
    baseline_by_configuration = {
        (str(row["Baseline"]), str(row["Setting"])): row for row in baseline_2x2_rows
    }
    ties_baseline = float(
        baseline_by_configuration[("TIES-Merging", "Baseline")]["Overall ACC (%)"]
    )
    ties_with_lpc = float(
        baseline_by_configuration[("TIES-Merging", "Baseline + LPC")]["Overall ACC (%)"]
    )
    dare_baseline = float(
        baseline_by_configuration[("DARE-Linear", "Baseline")]["Overall ACC (%)"]
    )
    dare_with_lpc = float(
        baseline_by_configuration[("DARE-Linear", "Baseline + LPC")]["Overall ACC (%)"]
    )
    dpr_only = float(
        baseline_by_configuration[("TIES-Merging", "Baseline + DPR")]["Overall ACC (%)"]
    )
    dpr_with_lpc = float(
        baseline_by_configuration[("TIES-Merging", "Baseline + DPR + LPC")]["Overall ACC (%)"]
    )
    prevalence_by_setting = {
        str(row["Setting"]): float(row["Avg ACC (%)"])
        for row in prevalence_rows
    }
    uniform_prior = prevalence_by_setting["Uniform prevalence prior"]
    always_on = prevalence_by_setting["Always-on calibration"]
    client_balanced = prevalence_by_setting["Client-balanced prior"]
    full_prevalence = prevalence_by_setting["LAMP-Merge"]
    module_ablation_intro = (
        "Module-level ablations use exactly the same experimental setup as the main experiments. "
        "The Weight-Averaging Baseline, Baseline + DPR, and full LAMP-Merge achieve overall "
        r"client-average ACC values of 22.04\%, 58.91\%, and 62.21\%, respectively. We additionally "
        "measure the complete DPR/LPC factorial effect through the strict $2\\times2$ "
        "TIES/DARE controls in "
        "Fig.~\\ref{fig:baseline-2x2-ablation}."
    )
    tex = re.sub(
        r"Module-level ablations use exactly the same experimental setup as the main experiments\..*",
        lambda _: module_ablation_intro,
        tex,
        count=1,
    )
    tex = re.sub(
        r"(Fig\.~\\ref\{fig:hparam-analysis\} )evaluates*\b",
        r"\1evaluates",
        tex,
    )
    tex = tex.replace(
        r"The default $s=20$ and $\lambda=5$ lie in broad stable regions, indicating that the gain is not produced by narrow hyperparameter tuning.",
        r"The fixed method uses $\gamma=0.55$, $s=18.75$, $\tau=2.5$, and $\lambda=4.25$; these values lie in broad stable regions, indicating that the gain is not produced by narrow hyperparameter tuning.",
    )
    tex = tex.replace(
        r"Fig.~\ref{fig:hparam-analysis} evaluates the prototype-head scale $s$ and calibration strength $\lambda$ over 180 raw cells and 60 client-average cells.",
        r"Fig.~\ref{fig:hparam-analysis} evaluates DPR and LPC sensitivities under the full LAMP-Merge configuration; every curve point uses 180 raw cells and 60 client-average cells.",
    )
    tex = tex.replace(
        "generic merge concentrates predictions on one class, whereas LAMP-Merge recovers a broader class allocation aligned with the test distribution.",
        "representative generic merges concentrate predictions on a few classes, whereas LAMP-Merge recovers a broader class allocation aligned with the test distribution.",
    )
    tex = tex.replace(
        "we compare the predicted class allocation induced by LAMP-Merge with that of the strongest generic merge. The representative generic merges",
        "we compare the predicted class allocation induced by LAMP-Merge with those of representative generic merges. These generic merges",
    )
    tex = tex.replace(
        "The generic merge concentrates predictions on one class, while LAMP-Merge recovers a broader class allocation aligned with the test distribution.",
        "Generic merges concentrate predictions on a few classes, while LAMP-Merge recovers a broader class allocation aligned with the test distribution.",
    )
    tex = tex.replace(
        "To visualize the prediction-distribution changes measured by the diagnostic metrics, we plot the per-class difference between each method's predicted allocation and the true test distribution. Values closer to zero indicate better prevalence alignment; positive and negative values respectively indicate over- and under-prediction. LAMP-Merge exhibits smaller deviations than representative generic merges across most classes.",
        "To visualize the prediction-distribution changes measured by the diagnostic metrics, we plot the absolute per-class gap between each method's predicted allocation and the true test distribution. Values closer to zero indicate better prevalence alignment. LAMP-Merge exhibits smaller deviations than representative generic merges across most classes.",
    )
    tex = tex.replace(
        "Per-class prediction deviations on Ultrasound, averaged over the four visual backbones, $K\\in\\{3,5,7\\}$, and the three Dirichlet skew levels. Values are predicted minus true class proportions in percentage points; zero denotes exact distributional agreement.",
        "Absolute per-class prediction gaps on Ultrasound, averaged over the four visual backbones, $K\\in\\{3,5,7\\}$, and the three Dirichlet skew levels. Values are absolute predicted--true class-proportion differences in percentage points; zero denotes exact distributional agreement.",
    )

    for backbone, _, label in BACKBONES:
        tex = replace_lamp_row(tex, label, backbone, formal_values)

    tex = re.sub(r"(?m)^Avg\s*&", "Weight Avg. &", tex)
    tex = re.sub(r"(?ms)^% \\begin\{table\*\}\[!p\].*?^% \\end\{table\*\}\s*", "", tex)

    diagnostic_table = build_internal_dataset_table(
        diagnostic_rows,
        "DPR internal ablation by dataset. Each entry averages all backbones, client counts, and Dirichlet settings; Avg averages the five datasets.",
        "tab:diagnostic-internal-ablation",
        divider_mode="uniform_client_weight",
        dpr_categories=True,
    )
    tex = replace_table_block(tex, "tab:diagnostic-internal-ablation", diagnostic_table)

    prevalence_table = build_internal_dataset_table(
        prevalence_rows,
        "LPC internal ablation by dataset. UPP uses a uniform prior, AOC removes the activation gate, and CBP averages client-normalized priors. Each entry averages all backbones, client counts, and Dirichlet settings; Avg averages the five datasets.",
        "tab:prevalence-internal-ablation",
    )
    tex = replace_table_block(tex, "tab:prevalence-internal-ablation", prevalence_table)

    collapse_table = build_collapse_dataset_table(diagnostic_summary)
    tex = replace_table_block(tex, "tab:collapse-key", collapse_table)

    dataset_table = build_dataset_statistics_table()
    tex = replace_table_block(tex, "tab:dataset-statistics", dataset_table)

    tex = tex.replace(
        r"Fig.~\ref{fig:dataset-ablation-acc} and Table~\ref{tab:diagnostic-internal-ablation} show that neither an arbitrary head nor a class-agnostic statistic explains the gain.",
        r"Table~\ref{tab:diagnostic-internal-ablation} shows that neither an arbitrary head nor a class-agnostic statistic explains the gain.",
    )
    baseline_2x2_text = (
        "We further conduct three full-scope LPC internal controls while retaining DPR. "
        "Uniform prevalence prior (UPP) sets \\(\\pi_c=1/C\\), removing empirical long-tail "
        "prevalence. Always-on calibration (AOC) replaces the thresholded activation "
        "\\(\\mathbf{1}[r>\\tau]\\) with 1 while retaining the empirical prior. Client-balanced "
        "prior (CBP) first normalizes each client's prevalence counts and then averages clients, "
        "rather than weighting the global prior by raw client counts. "
        "Table~\\ref{tab:prevalence-internal-ablation} reports overall mean ACC values of "
        f"{uniform_prior:.2f}\\% (UPP), {always_on:.2f}\\% (AOC), and {client_balanced:.2f}\\% (CBP), "
        f"compared with {full_prevalence:.2f}\\% for LAMP-Merge. Thus, UPP, AOC, and CBP are lower "
        f"by {full_prevalence - uniform_prior:.2f}, {full_prevalence - always_on:.2f}, and "
        f"{full_prevalence - client_balanced:.2f} percentage points, respectively. The small AOC "
        "gap indicates that always applying the empirical prior is close but still inferior, whereas "
        "UPP and CBP show that both the empirical long-tail prior and its client-size-aware aggregation "
        "are needed. The strict controls in "
        "Fig.~\\ref{fig:baseline-2x2-ablation} cross DPR "
        "and LPC for each classifier-head baseline. Without DPR, LPC raises TIES from "
        f"{ties_baseline:.2f}\\% to {ties_with_lpc:.2f}\\% and DARE from "
        f"{dare_baseline:.2f}\\% to {dare_with_lpc:.2f}\\%. DPR raises the two baselines to "
        f"{dpr_only:.2f}\\%, and adding LPC reaches {dpr_with_lpc:.2f}\\%. Thus, DPR provides "
        "the dominant improvement, while LPC supplies a complementary gain in the complete method."
    )
    tex = re.sub(
        r"We further replace the prior term inside long-tail prevalence calibration\..*?(?=\n\n\\begin\{figure\})",
        lambda _: baseline_2x2_text,
        tex,
        count=1,
    )
    baseline_2x2_figure = "\n".join(
        [
            r"\begin{figure}[!b]",
            r"\centering",
            r"\includegraphics[width=\columnwidth]{figures/03_baseline_2x2_ablation.pdf}",
            r"\caption{Strict $2\times2$ DPR/LPC ablations for TIES-Merging and DARE-Linear. \textbf{All experimental datasets are included.}}",
            r"\label{fig:baseline-2x2-ablation}",
            r"\end{figure}",
        ]
    )
    if r"\label{fig:dataset-ablation-acc}" in tex:
        tex = replace_figure_block(tex, "fig:dataset-ablation-acc", baseline_2x2_figure)
    else:
        tex = replace_figure_block(tex, "fig:baseline-2x2-ablation", baseline_2x2_figure)

    if has_dpr_sensitivity:
        hparam_intro = (
            r"Fig.~\ref{fig:hparam-analysis} evaluates DPR and LPC sensitivities under the full "
            r"LAMP-Merge configuration. Each curve point uses 180 raw cells and 60 client-average cells. "
            r"The fixed operating point uses $\gamma=0.55$, $s=18.75$, $\tau=2.5$, and $\lambda=4.25$; "
            r"the sweeps assess deviations around these values while holding the other module fixed."
        )
        hparam_caption = (
            r"DPR and LPC sensitivity under the full LAMP-Merge configuration. Each line varies one module "
            r"while fixing the other at its selected operating point. Each point averages 60 client-average cells."
        )
        hparam_reproducibility = (
            r"The sensitivity figure uses full-configuration sweeps only. The DPR grid uses "
            r"\(\gamma\in\{0.50,0.55,0.60\}\) and ten prototype-head scales "
            r"\(s\in\{13.75,15.00,\ldots,25.00\}\), while holding \(\tau=2.5\) and \(\lambda=4.25\) fixed. "
            r"The LPC grid uses \(\tau\in\{2.0,2.5,3.0\}\) and ten calibration strengths "
            r"\(\lambda\in\{3.00,3.25,\ldots,5.25\}\), while holding \(\gamma=0.55\) and \(s=18.75\) fixed. "
            r"At every grid point, we evaluate the five datasets, four backbone families, three client counts, "
            r"and three Dirichlet skew levels, yielding 180 raw result cells and 60 client-average cells after "
            r"averaging the three skew levels for each dataset--backbone--client-count combination."
        )
    else:
        hparam_intro = (
            r"Fig.~\ref{fig:hparam-analysis} evaluates LPC sensitivity under the full LAMP-Merge configuration. "
            r"Each curve point uses 180 raw cells and 60 client-average cells. The sweep fixes DPR at "
            r"$\gamma=0.55$ and $s=18.75$ while varying $\tau$ and $\lambda$ around the selected operating point."
        )
        hparam_caption = (
            r"DPR and LPC sensitivity under the full LAMP-Merge configuration. Completed curve points average "
            r"60 client-average cells."
        )
        hparam_reproducibility = (
            r"The sensitivity figure uses the completed full-configuration LPC sweep only. It uses "
            r"\(\tau\in\{2.0,2.5,3.0\}\) and ten calibration strengths "
            r"\(\lambda\in\{3.00,3.25,\ldots,5.25\}\), while holding \(\gamma=0.55\) and \(s=18.75\) fixed. "
            r"At every grid point, we evaluate the five datasets, four backbone families, three client counts, "
            r"and three Dirichlet skew levels, yielding 180 raw result cells and 60 client-average cells after "
            r"averaging the three skew levels for each dataset--backbone--client-count combination."
        )
    tex = re.sub(
        r"Fig\.~\\ref\{fig:hparam-analysis\}.*?(?=\n\n\\begin\{figure\}(?:\[[^\]]+\])?)",
        lambda _: hparam_intro,
        tex,
        count=1,
    )
    tex = re.sub(
        r"The sensitivity figure uses full-configuration sweeps only\..*?(?=\n\nFor reuse,)",
        lambda _: hparam_reproducibility,
        tex,
        count=1,
    )
    combined_hparam_figure = "\n".join(
        [
            r"\begin{figure}[!htbp]",
            r"\centering",
            r"\includegraphics[width=\columnwidth]{figures/05_hyperparameter_sensitivity.pdf}",
            rf"\caption{{{hparam_caption}}}",
            r"\label{fig:hparam-analysis}",
            r"\end{figure}",
        ]
    )
    if r"\label{fig:hparam-analysis-dpr}" in tex:
        tex = replace_figure_block(tex, "fig:hparam-analysis-dpr", combined_hparam_figure)
    else:
        tex = replace_figure_block(tex, "fig:hparam-analysis", combined_hparam_figure)
    if r"\label{fig:hparam-analysis-lpc}" in tex:
        tex = remove_figure_block(tex, "fig:hparam-analysis-lpc")
    tex = re.sub(
        r"\n\n\\subsection\{t-SNE Visualization Experiment\}.*?(?=\n\n\\FloatBarrier)",
        "",
        tex,
        count=1,
        flags=re.DOTALL,
    )
    for figure_name in [
        "01_module_ablation_accuracy",
        "02_internal_module_ablations",
        "03_baseline_2x2_ablation",
        "04_ultrasound_class_distribution",
        "05_hyperparameter_sensitivity",
        "07_tsne_output_probability",
    ]:
        tex = tex.replace(f"figures/{figure_name}", f"figures/new/{figure_name}")
    tex = tex.replace(
        r"\begin{figure}[t]"
        "\n\\centering"
        "\n\\includegraphics[width=\\linewidth]{figures/new/04_ultrasound_class_distribution.pdf}",
        r"\begin{figure}[!htbp]"
        "\n\\centering"
        "\n\\includegraphics[width=\\linewidth]{figures/new/04_ultrasound_class_distribution.pdf}",
    )
    if r"\label{fig:dataset-examples}" in tex:
        tex = remove_figure_block(tex, "fig:dataset-examples")
    tex = tex.replace(
        r"Fig.~\ref{fig:hparam-analysis-dpr} and Fig.~\ref{fig:hparam-analysis-lpc}",
        r"Fig.~\ref{fig:hparam-analysis}",
    )
    tex = re.sub(
        r"(Fig\.~\\ref\{fig:hparam-analysis\} )evaluates*\b",
        r"\1evaluates",
        tex,
    )
    tex = tex.replace("DPRM", "DPR")
    tex = tex.replace("\\FloatBarrier\n\\section{Conclusion}", r"\section{Conclusion}")
    path.write_text(tex, encoding="utf-8")


def write_index(csv_dir: Path) -> None:
    rows = [
        ("Tables 1--4", "Main client-average ACC", "主实验_ResNet.csv; 主实验_ConvNeXt.csv; 主实验_ViT-Tiny.csv; 主实验_Swin-Tiny.csv"),
        ("Ablation text", "Incremental DPR/LPC module ablation", "消融.csv"),
        ("Figure 4", "Complete TIES/DARE 2x2 DPR/LPC ablation", "基线2x2消融.csv"),
        ("Table 5 / Figure 3 left", "DPR internal ablation", "诊断原型重建内部消融.csv"),
        ("Table 6 / Figure 3 right", "LPC internal ablation", "长尾患病率校准内部消融.csv"),
        ("Table 7", "Prediction-collapse diagnostics", "预测坍缩诊断.csv"),
        ("Figure 5", "Ultrasound predicted distribution", "超声预测类别分布.csv"),
        ("Hyperparameter figure", "Combined DPR/LPC sensitivity", "超参数分析_诊断原型重建.csv; 超参数分析_长尾患病率校准.csv"),
        ("Appendix Table", "Dataset statistics and samples", "数据集统计.csv"),
    ]
    write_annotated_csv(
        csv_dir / "图表与CSV对应关系.csv",
        "本表给出论文图表与整理后 CSV 的一一对应关系；所有方法名称均使用论文口径。",
        ["Paper item", "Experiment", "CSV file"],
        [dict(zip(["Paper item", "Experiment", "CSV file"], row)) for row in rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, default=ROOT / "论文实验数据")
    args = parser.parse_args()

    paper_dir = args.paper_dir.resolve()
    csv_dir = args.csv_dir.resolve()
    if not (paper_dir / "v4_3.tex").exists():
        raise FileNotFoundError(paper_dir / "v4_3.tex")
    csv_dir.mkdir(parents=True, exist_ok=True)

    client_rows = load_client_average_rows()
    formal_values = load_formal_lamp_values(client_rows)
    build_main_csvs(csv_dir, formal_values)

    module_rows = aggregate_module_ablation(client_rows)
    module_fields = ["Setting", "Paper label", "Overall ACC (%)"] + [
        f"{label} ACC (%)" for _, label in DATASETS
    ]
    write_annotated_csv(
        csv_dir / "消融.csv",
        "本表对应模块级消融实验；DPR 表示 Diagnostic Prototype Reconstruction，LPC 表示 Long-tail Prevalence Calibration，单位为 client-average ACC 百分比。图中仅报告 Baseline、Baseline + DPR 和 LAMP-Merge。",
        module_fields,
        module_rows,
    )

    baseline_2x2_rows = build_baseline_2x2_rows(client_rows)
    baseline_2x2_fields = [
        "Baseline",
        "Setting",
        "DPR",
        "LPC",
        "Client-average cells",
        "Overall ACC (%)",
    ]
    write_annotated_csv(
        csv_dir / "基线2x2消融.csv",
        "本表对应 TIES/DARE 的完整 DPR×LPC 2x2 消融；每个 baseline 均包含 Baseline、+DPR、+LPC 和 +DPR+LPC 四种设置。数值为五个数据集、四个 backbone 和 K=3/5/7 共 60 个 client-average cells 的总体均值，单位为 ACC 百分比，不分数据集展示。",
        baseline_2x2_fields,
        baseline_2x2_rows,
    )

    diagnostic_rows, prevalence_rows, _ = internal_ablation_rows(client_rows)
    internal_fields = ["Setting", "Mode"] + [
        f"{label} ACC (%)" for _, label in DATASETS
    ] + ["Avg ACC (%)"]
    write_annotated_csv(
        csv_dir / "诊断原型重建内部消融.csv",
        "本表对应 DPR 内部消融；各设置保留 LPC，仅替换诊断原型语义或类别支持权重。每个数据集数值对四个 backbone、K=3/5/7 和三个 beta 设置取平均，单位为 ACC 百分比。",
        internal_fields,
        diagnostic_rows,
    )
    write_annotated_csv(
        csv_dir / "长尾患病率校准内部消融.csv",
        "本表对应 LPC 内部消融；各设置保留 DPR，仅替换患病率先验估计或长尾激活门控。每个数据集数值对四个 backbone、K=3/5/7 和三个 beta 设置取平均，单位为 ACC 百分比。",
        internal_fields,
        prevalence_rows,
    )

    diagnostic_table, diagnostic_summary = build_diagnostic_table()
    write_annotated_csv(
        csv_dir / "预测坍缩诊断.csv",
        "本表对应预测坍缩诊断实验；每个数据集数值对四个 backbone、K=3/5/7 和三个 beta 设置取平均。Collapse ratio 和 Predicted-true TV 使用百分比，Effective predicted classes 使用类别数；Best generic reference 按数据集和指标分别选择最优通用模型合并基线。",
        ["Metric", "Direction", "Series"]
        + [f"{label} Avg" for _, label in DATASETS]
        + ["Overall Avg", "Reference methods"],
        diagnostic_table,
    )

    distribution_rows = build_ultrasound_distribution()
    distribution_fields = ["Series"] + [
        f"Class {index} absolute difference vs. true (pp)" for index in range(8)
    ]
    write_annotated_csv(
        csv_dir / "超声预测类别分布.csv",
        "本表对应 Ultrasound 预测分布相对真实分布的逐类绝对差值图；每个方法对四个 backbone、K=3/5/7 和三个 beta 设置取平均，差值为预测比例与真实比例之差的绝对值，单位为百分点。",
        distribution_fields,
        distribution_rows,
    )

    diagnostic_hparams, prevalence_hparams = build_hparam_rows()
    write_annotated_csv(
        csv_dir / "超参数分析_诊断原型重建.csv",
        "本表记录完整 LAMP-Merge 配置下的 DPR 3x10 网格；每点覆盖五个数据集、四个 backbone、K=3/5/7 和三个 beta，ACC 使用百分比。",
        ["Source", "Evidence exponent gamma", "Prototype-head scale s", "Raw cells", "Client-average cells", "Mean ACC (%)"],
        diagnostic_hparams,
    )
    write_annotated_csv(
        csv_dir / "超参数分析_长尾患病率校准.csv",
        "本表记录完整 LAMP-Merge 配置下的 LPC 3x10 网格；每点覆盖五个数据集、四个 backbone、K=3/5/7 和三个 beta，ACC 使用百分比。",
        ["Source", "Activation threshold tau", "Calibration strength lambda", "Raw cells", "Client-average cells", "Mean ACC (%)"],
        prevalence_hparams,
    )

    write_annotated_csv(
        csv_dir / "数据集统计.csv",
        "本表对应附录数据集统计表；Train、Validation、Test 规模来自本仓库实际使用的五个 NPZ 文件，Sample file 指向论文压缩包内示例图。",
        ["Dataset", "Classes", "Train", "Validation", "Test", "Total", "Sample file"],
        [
            {
                "Dataset": item["dataset"],
                "Classes": item["classes"],
                "Train": item["train"],
                "Validation": item["validation"],
                "Test": item["test"],
                "Total": item["total"],
                "Sample file": item["sample"],
            }
            for item in DATASET_STATS
        ],
    )

    crop_dataset_samples(paper_dir)
    update_tex(
        paper_dir,
        formal_values,
        baseline_2x2_rows,
        diagnostic_rows,
        prevalence_rows,
        diagnostic_summary,
        bool(diagnostic_hparams),
    )
    write_index(csv_dir)
    print(f"Prepared manuscript in {paper_dir}")
    print(f"Wrote CSV files to {csv_dir}")


if __name__ == "__main__":
    main()
