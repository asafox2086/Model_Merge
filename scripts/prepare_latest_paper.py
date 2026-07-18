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
FORMAL_ROOT = (
    ROOT
    / "outputs"
    / "lamp_merge_internal_ablation_full_20260718_formal_gamma055_s18p75_tau2p5_lambda4p25"
)

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
    "avg_m2": "avg+M2",
    "m1_only": "M1 only",
    "prototype_head_agg": "Classifier-head aggregation",
    "global_feature_mean": "Global-feature mean",
    "support_only": "Support-only synthetic head",
    "prototype_shuffle": "Shuffled-label prototype",
    "uniform_client_weight": "Uniform client weight",
    "binary_support": "Binary support only",
    "global_client_size_weight": "Global client-size weight",
    "no_prevalence": "No prevalence calibration",
    "uniform_prevalence": "Uniform prevalence prior",
}

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
    return read_csv(REPORT_DIR / "lamp_merge_internal_ablation_full_client_average.csv")


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
        ("Baseline + LPC", "Baseline + LPC", "avg+M2"),
        ("Baseline + DPRM", "Baseline + DPRM", "M1 only"),
        ("LAMP-Merge (Baseline + DPRM + LPC)", "LAMP-Merge", "LAMP-Merge"),
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


def internal_ablation_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    rows = read_csv(REPORT_DIR / "lamp_merge_internal_ablation_full.csv")
    values = {row["mode"]: float(row["client_average_mean_acc"]) for row in rows}
    full = values["full"]
    diagnostic_modes = [
        ("LAMP-Merge", "full"),
        ("Classifier-head aggregation", "prototype_head_agg"),
        ("Global-feature mean", "global_feature_mean"),
        ("Support-only synthetic head", "support_only"),
        ("Shuffled-label prototype", "prototype_shuffle"),
        ("Uniform client weight", "uniform_client_weight"),
        ("Binary support only", "binary_support"),
        ("Global client-size weight", "global_client_size_weight"),
    ]
    prevalence_modes = [
        ("LAMP-Merge", "full"),
        ("No prevalence calibration", "no_prevalence"),
        ("Uniform prevalence prior", "uniform_prevalence"),
    ]

    def build(entries: list[tuple[str, str]]) -> list[dict[str, object]]:
        output = []
        for label, mode in entries:
            value = values[mode]
            output.append(
                {
                    "Setting": label,
                    "Mean ACC (%)": f"{100 * value:.2f}",
                    "Drop vs. LAMP-Merge (pp)": "--" if mode == "full" else f"{100 * (full - value):.2f}",
                }
            )
        return output

    return build(diagnostic_modes), build(prevalence_modes), values


def build_diagnostic_table() -> tuple[list[dict[str, object]], dict[str, dict[str, float | str]]]:
    rows = read_csv(REPORT_DIR / "prediction_diagnostics_full.csv")
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "OK" and row["source"] == "merge":
            by_method[row["method"]].append(row)
    if len(by_method["lamp_merge:full"]) != 180:
        raise RuntimeError("Formal LAMP diagnostic coverage is incomplete")

    def mean(method: str, field: str) -> float:
        values = [float(row[field]) for row in by_method[method] if row[field]]
        return sum(values) / len(values)

    generic_methods = [method for method in by_method if not method.startswith("lamp_merge:")]
    metric_defs = [
        ("Collapse ratio", "lower", "collapse_ratio", 100.0),
        ("Effective predicted classes", "higher", "effective_predicted_classes", 1.0),
        ("Predicted-true TV", "lower", "pred_true_tv", 100.0),
    ]
    output_rows = []
    summary: dict[str, dict[str, float | str]] = {}
    for metric, direction, field, scale in metric_defs:
        lamp_value = mean("lamp_merge:full", field)
        reference_values = {method: mean(method, field) for method in generic_methods}
        if direction == "lower":
            reference_method = min(reference_values, key=reference_values.get)
        else:
            reference_method = max(reference_values, key=reference_values.get)
        reference_value = reference_values[reference_method]
        output_rows.append(
            {
                "Metric": metric,
                "Direction": direction,
                "LAMP-Merge": f"{scale * lamp_value:.2f}",
                "Best generic reference": f"{scale * reference_value:.2f}",
                "Reference method": MAIN_METHOD_NAMES.get(reference_method, reference_method),
            }
        )
        summary[field] = {
            "lamp": lamp_value,
            "reference": reference_value,
            "reference_method": reference_method,
        }
    return output_rows, summary


def parse_count_vector(value: str) -> list[float]:
    return [float(item) for item in value.strip().split()]


def build_ultrasound_distribution() -> list[dict[str, object]]:
    rows = read_csv(REPORT_DIR / "prediction_diagnostics_full.csv")
    methods = [
        ("lamp_merge:full", "LAMP-Merge"),
        ("avg", "Weight Averaging"),
        ("ties", "TIES-Merging"),
        ("dare_linear", "DARE-Linear"),
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
        total_variation = 0.5 * sum(abs(a - b) for a, b in zip(distribution, true_distribution))
        row: dict[str, object] = {"Series": label, "TV": f"{total_variation:.4f}"}
        for index, value in enumerate(distribution):
            row[f"Class {index} proportion (%)"] = f"{100 * value:.4f}"
        output_rows.append(row)
    true_row: dict[str, object] = {"Series": "True distribution", "TV": "0.0000"}
    for index, value in enumerate(true_distribution):
        true_row[f"Class {index} proportion (%)"] = f"{100 * value:.4f}"
    output_rows.append(true_row)
    return output_rows


def build_hparam_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = read_csv(REPORT_DIR / "lamp_merge_hparam_interaction_full.csv")
    diagnostic = []
    prevalence = []
    for row in rows:
        item = {
            "Raw cells": row["raw_cells"],
            "Client-average cells": row["client_average_cells"],
            "Mean ACC (%)": f"{100 * float(row['client_average_mean_acc']):.4f}",
        }
        if row["module"] == "diagnostic prototype reconstruction":
            item = {
                "Evidence exponent gamma": row["curve_value"],
                "Prototype-head scale s": row["x_value"],
                **item,
            }
            diagnostic.append(item)
        elif row["module"] == "long-tail prevalence calibration":
            item = {
                "Activation threshold tau": row["curve_value"],
                "Calibration strength lambda": row["x_value"],
                **item,
            }
            prevalence.append(item)
    return diagnostic, prevalence


def crop_dataset_samples(paper_dir: Path) -> None:
    from PIL import Image

    intro_source = paper_dir / "figures" / "4a3123dfc3ae4ea3ce76e62b4cbc7d60.png"
    intro_target = paper_dir / "figures" / "intro.png"
    if intro_source.exists() and not intro_target.exists():
        shutil.copy2(intro_source, intro_target)

    source = Image.open(paper_dir / "figures" / "dataset_examples.png")
    width, height = source.size
    centers = [0.096, 0.298, 0.500, 0.701, 0.903]
    names = ["blood", "ultrasound", "derma", "organ_c", "organ_s"]
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


def update_tex(
    paper_dir: Path,
    formal_values: dict[tuple[str, str, int], float],
    internal_values: dict[str, float],
    diagnostic_summary: dict[str, dict[str, float | str]],
) -> None:
    path = paper_dir / "v4_3.tex"
    tex = path.read_text(encoding="utf-8")
    tex = tex.replace(r"where $\gamma=0.45$ is the default evidence exponent", r"where $\gamma=0.55$ is the default evidence exponent")
    tex = tex.replace(r"where $s=20$ is the default head scale", r"where $s=18.75$ is the default head scale")
    tex = tex.replace(r"with default maximum calibration strength $\lambda=5.0$", r"with default maximum calibration strength $\lambda=4.25$")
    tex = tex.replace(
        "Module-level ablations use exactly the same experimental setup as the main experiments. We compare three configurations: full LAMP-Merge, diagnostic prototype reconstruction only, and weight averaging augmented only with long-tail prevalence calibration. Both components provide substantial performance improvements.",
        "Module-level ablations use exactly the same experimental setup as the main experiments. We compare the Weight-Averaging Baseline, Baseline + LPC, Baseline + DPRM, and full LAMP-Merge (Baseline + DPRM + LPC).",
    )
    tex = tex.replace(
        "Module-level ablations use exactly the same experimental setup as the main experiments. We compare the Weight-Averaging Baseline, Baseline + LPC, Baseline + DPRM, and full LAMP-Merge (Baseline + DPRM + LPC). The paper names are used consistently instead of implementation labels such as M1, M2, or avg+M1.",
        "Module-level ablations use exactly the same experimental setup as the main experiments. We compare the Weight-Averaging Baseline, Baseline + LPC, Baseline + DPRM, and full LAMP-Merge (Baseline + DPRM + LPC).",
    )
    tex = tex.replace(
        r"The default $s=20$ and $\lambda=5$ lie in broad stable regions, indicating that the gain is not produced by narrow hyperparameter tuning.",
        r"The fixed method uses $\gamma=0.55$, $s=18.75$, $\tau=2.5$, and $\lambda=4.25$; these values lie in broad stable regions, indicating that the gain is not produced by narrow hyperparameter tuning.",
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

    for backbone, _, label in BACKBONES:
        tex = replace_lamp_row(tex, label, backbone, formal_values)

    tex = re.sub(r"(?m)^Avg\s*&", "Weight Avg. &", tex)
    tex = re.sub(r"(?ms)^% \\begin\{table\*\}\[!p\].*?^% \\end\{table\*\}\s*", "", tex)

    diagnostic_table = rf"""\begin{{table}}[t]
\centering
\caption{{Internal ablation of diagnostic prototype reconstruction. \textbf{{All reported values are averaged over all backbones}}.}}
\label{{tab:diagnostic-internal-ablation}}
\scriptsize
\renewcommand{{\arraystretch}}{{1.35}}
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{llr}}
\noalign{{\hrule height 1.25pt}}
\rowcolor[HTML]{{F2F2F2}}
\textbf{{Type}} & \textbf{{Setting}} & \textbf{{Avg}} \\
\hline \hline
Prototype & Cls-head agg. & {100 * internal_values['prototype_head_agg']:.2f} \\
\rowcolor{{gray!10}}
& Global feat. mean & {100 * internal_values['global_feature_mean']:.2f} \\
& Support-only & {100 * internal_values['support_only']:.2f} \\
\rowcolor{{gray!10}}
& Shuffled proto. & {100 * internal_values['prototype_shuffle']:.2f} \\
\hdashline
Weight & Uniform client & {100 * internal_values['uniform_client_weight']:.2f} \\
\rowcolor{{gray!10}}
& Binary support & {100 * internal_values['binary_support']:.2f} \\
& Client-size weight & {100 * internal_values['global_client_size_weight']:.2f} \\
\hline \hline
\rowcolor[HTML]{{FFF9C4}}
\textbf{{Full}} & \textbf{{LAMP-Merge}} & \textbf{{{100 * internal_values['full']:.2f}}} \\
\noalign{{\hrule height 1.25pt}}
\end{{tabular}}
\end{{table}}"""
    tex = replace_table_block(tex, "tab:diagnostic-internal-ablation", diagnostic_table)

    prevalence_table = rf"""\begin{{table}}[t]
\centering
\caption{{Internal ablation of long-tail prevalence calibration. \textbf{{All reported values are averaged over all backbones}}.}}
\label{{tab:prevalence-internal-ablation}}
\scriptsize
\renewcommand{{\arraystretch}}{{1.35}}
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lr}}
\noalign{{\hrule height 1.25pt}}
\rowcolor[HTML]{{F2F2F2}}
\textbf{{Setting}} & \textbf{{Avg}} \\
\hline \hline
No LPC & {100 * internal_values['no_prevalence']:.2f} \\
\rowcolor{{gray!10}}
Uniform prior & {100 * internal_values['uniform_prevalence']:.2f} \\
\hline \hline
\rowcolor[HTML]{{FFF9C4}}
\textbf{{LAMP-Merge}} & \textbf{{{100 * internal_values['full']:.2f}}} \\
\noalign{{\hrule height 1.25pt}}
\end{{tabular}}
\end{{table}}"""
    tex = replace_table_block(tex, "tab:prevalence-internal-ablation", prevalence_table)

    collapse = diagnostic_summary["collapse_ratio"]
    effective = diagnostic_summary["effective_predicted_classes"]
    total_variation = diagnostic_summary["pred_true_tv"]
    collapse_table = rf"""\begin{{table}}[t]
\centering
\caption{{Prediction-collapse diagnostics averaged over five medical datasets and four backbones per dataset. Ratio metrics are reported in percentage scale.}}
\label{{tab:collapse-key}}
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\renewcommand{{\arraystretch}}{{1.35}}
\begin{{tabular}}{{lcc}}
\noalign{{\hrule height 1.25pt}}
\rowcolor[HTML]{{F2F2F2}}
\textbf{{Metric}} & \textbf{{LAMP-Merge}} & \textbf{{Best reference}} \\
\hline \hline
Collapse ratio $\downarrow$ & {100 * float(collapse['lamp']):.2f} & {100 * float(collapse['reference']):.2f} \\
\rowcolor{{gray!10}}
Effective classes $\uparrow$ & {float(effective['lamp']):.2f} & {float(effective['reference']):.2f} \\
Pred-true TV $\downarrow$ & {100 * float(total_variation['lamp']):.2f} & {100 * float(total_variation['reference']):.2f} \\
\noalign{{\hrule height 1.25pt}}
\end{{tabular}}
\end{{table}}"""
    tex = replace_table_block(tex, "tab:collapse-key", collapse_table)

    dataset_rows = []
    for item in DATASET_STATS:
        dataset_rows.append(
            f"{item['dataset']} & {item['classes']} & {item['train']:,} & {item['validation']:,} & "
            f"{item['test']:,} & {item['total']:,} & "
            rf"\includegraphics[width=1.28cm,height=0.66cm,keepaspectratio]{{{item['sample']}}} \\"
        )
    dataset_table = r"""\begin{table*}[t]
\centering
\caption{Dataset statistics and representative samples for the five medical image benchmarks. Counts are taken from the exact NPZ files used in our experiments.}
\label{tab:dataset-statistics}
\scriptsize
\renewcommand{\arraystretch}{1.35}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrc}
\noalign{\hrule height 1.25pt}
\rowcolor[HTML]{F2F2F2}
\textbf{Dataset} & \textbf{Classes} & \textbf{Train} & \textbf{Val} & \textbf{Test} & \textbf{Total} & \textbf{Sample} \\
\hline \hline
""" + "\n".join(dataset_rows) + r"""
\noalign{\hrule height 1.25pt}
\end{tabular}
\end{table*}"""
    tex = replace_table_block(tex, "tab:dataset-statistics", dataset_table)

    tex = tex.replace(
        r"\caption{Internal ablation of both Module}",
        r"\caption{Internal ablations of the Diagnostic Prototype Reconstruction Module (DPRM) and Long-tail Prevalence Calibration (LPC).}",
    )
    path.write_text(tex, encoding="utf-8")


def write_index(csv_dir: Path) -> None:
    rows = [
        ("Tables 1--4", "Main client-average ACC", "主实验_ResNet.csv; 主实验_ConvNeXt.csv; 主实验_ViT-Tiny.csv; 主实验_Swin-Tiny.csv"),
        ("Figure 3", "Module-level ablation", "消融.csv"),
        ("Table 5 / Figure 4 left", "DPRM internal ablation", "诊断原型重建内部消融.csv"),
        ("Table 6 / Figure 4 right", "LPC internal ablation", "长尾患病率校准内部消融.csv"),
        ("Table 7", "Prediction-collapse diagnostics", "预测坍缩诊断.csv"),
        ("Figure 5", "Ultrasound predicted distribution", "超声预测类别分布.csv"),
        ("Figures 6--7", "Hyperparameter sensitivity", "超参数分析_诊断原型重建.csv; 超参数分析_长尾患病率校准.csv"),
        ("Figure 8", "Blood/ResNet output-probability t-SNE", "tSNE输出概率坐标.csv"),
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
    if csv_dir.exists():
        shutil.rmtree(csv_dir)
    csv_dir.mkdir(parents=True)

    client_rows = load_client_average_rows()
    formal_values = load_formal_lamp_values(client_rows)
    build_main_csvs(csv_dir, formal_values)

    module_rows = aggregate_module_ablation(client_rows)
    module_fields = ["Setting", "Paper label", "Overall ACC (%)"] + [
        f"{label} ACC (%)" for _, label in DATASETS
    ]
    write_annotated_csv(
        csv_dir / "消融.csv",
        "本表对应模块级消融实验；DPRM 表示 Diagnostic Prototype Reconstruction Module，LPC 表示 Long-tail Prevalence Calibration，单位为 client-average ACC 百分比。",
        module_fields,
        module_rows,
    )

    diagnostic_rows, prevalence_rows, internal_values = internal_ablation_rows()
    internal_fields = ["Setting", "Mean ACC (%)", "Drop vs. LAMP-Merge (pp)"]
    write_annotated_csv(
        csv_dir / "诊断原型重建内部消融.csv",
        "本表对应 DPRM 内部消融；各设置保留 LPC，仅替换诊断原型语义或类别支持权重，数值为 180 个正式实验单元的平均 ACC 百分比。",
        internal_fields,
        diagnostic_rows,
    )
    write_annotated_csv(
        csv_dir / "长尾患病率校准内部消融.csv",
        "本表对应 LPC 内部消融；各设置保留 DPRM，仅移除或替换患病率先验，数值为 180 个正式实验单元的平均 ACC 百分比。",
        internal_fields,
        prevalence_rows,
    )

    diagnostic_table, diagnostic_summary = build_diagnostic_table()
    write_annotated_csv(
        csv_dir / "预测坍缩诊断.csv",
        "本表对应预测坍缩诊断实验；Collapse ratio 和 Predicted-true TV 使用百分比，Effective predicted classes 使用类别数。Best generic reference 对每个指标分别选取最优通用模型合并基线。",
        ["Metric", "Direction", "LAMP-Merge", "Best generic reference", "Reference method"],
        diagnostic_table,
    )

    distribution_rows = build_ultrasound_distribution()
    distribution_fields = ["Series", "TV"] + [f"Class {index} proportion (%)" for index in range(8)]
    write_annotated_csv(
        csv_dir / "超声预测类别分布.csv",
        "本表对应 Ultrasound 预测类别分布图；每个方法对四个 backbone、K=3/5/7 和三个 beta 设置取平均，类别比例使用百分比。",
        distribution_fields,
        distribution_rows,
    )

    diagnostic_hparams, prevalence_hparams = build_hparam_rows()
    write_annotated_csv(
        csv_dir / "超参数分析_诊断原型重建.csv",
        "本表对应 DPRM 超参数分析，记录 evidence exponent gamma 与 prototype-head scale s 的 5x10 完整网格，ACC 使用百分比。",
        ["Evidence exponent gamma", "Prototype-head scale s", "Raw cells", "Client-average cells", "Mean ACC (%)"],
        diagnostic_hparams,
    )
    write_annotated_csv(
        csv_dir / "超参数分析_长尾患病率校准.csv",
        "本表对应 LPC 超参数分析，记录 activation threshold tau 与 calibration strength lambda 的 5x10 完整网格，ACC 使用百分比。",
        ["Activation threshold tau", "Calibration strength lambda", "Raw cells", "Client-average cells", "Mean ACC (%)"],
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
    update_tex(paper_dir, formal_values, internal_values, diagnostic_summary)
    write_index(csv_dir)
    print(f"Prepared manuscript in {paper_dir}")
    print(f"Wrote CSV files to {csv_dir}")


if __name__ == "__main__":
    main()
