#!/usr/bin/env python3
import argparse
import csv
import re
from copy import deepcopy
from pathlib import Path

from generate_combined_results_table import highlight_rows, parse_tables, render_table
from generate_my_merge_master_table import CLIENT_AVG_GROUPS, SETTINGS


ABLATION_LABELS = {
    "full": "my_merge full (none missing)",
    "avg_only": "my_merge avg_only (-M1,-M2)",
    "no_diagnostic_evidence": "my_merge -M1 diagnostic_evidence",
    "no_client_information": "my_merge -M1 client_info",
    "no_diagnostic_information": "my_merge -M1 client_info",
    "no_fusion_selection": "my_merge -M2 conflict_resolution",
    "no_medical_fusion_selection": "my_merge -M2 conflict_resolution",
    "no_adaptive_candidates": "my_merge -M2 conflict_resolution",
    "no_adaptive_candidate_generation": "my_merge -M2 conflict_resolution",
    "no_m3": "my_merge -M2 conflict_resolution",
    "no_calibration": "my_merge -M2 calibration",
}


def parse_args():
    p = argparse.ArgumentParser("Generate combined baseline + my_merge ablation master table")
    p.add_argument("--base", default="result/all_results.md")
    p.add_argument("--grid-root", required=True, help="Ablation grid root containing one subdir per ablation")
    p.add_argument("--dest", required=True)
    return p.parse_args()


def ablation_name(grid_root, csv_path):
    rel = csv_path.parent.parent.relative_to(grid_root)
    return rel.parts[0] if rel.parts else "unknown"


def discover_eval_csvs(grid_root):
    grid_root = Path(grid_root)
    paths = set(grid_root.glob("*/reports/eval_summary.csv"))
    paths.update(grid_root.glob("*/*/reports/eval_summary.csv"))
    return sorted(path for path in paths if path.is_file())


def load_lookup(grid_root):
    lookup = {}
    ablations = []
    for csv_path in discover_eval_csvs(grid_root):
        ablation = ablation_name(Path(grid_root), csv_path)
        if ablation not in ablations:
            ablations.append(ablation)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("method") != "my_merge":
                    continue
                task_type = row["task_type"]
                model_name = row["model"] if task_type == "small" else row["clip_model"]
                key = (
                    ablation,
                    task_type,
                    row["dataset"],
                    model_name,
                    int(float(row["num_clients"])),
                    float(row["beta"]),
                )
                lookup[key] = float(row["test_acc"])
    if "avg_only" not in ablations:
        ablations.append("avg_only")
    return lookup, sorted(ablations, key=ablation_sort_key)


def ablation_sort_key(name):
    order = [
        "full",
        "no_client_information",
        "no_fusion_selection",
        "no_adaptive_candidates",
        "avg_only",
        "no_diagnostic_evidence",
        "no_calibration",
    ]
    return (order.index(name) if name in order else len(order), name)


def fmt(value):
    return "-" if value is None else f"{value:.4f}"


DATASET_TH_RE = re.compile(r'<th\s+colspan="\d+">(.*?)</th>')


def table_datasets(base_table):
    datasets = []
    for line in base_table["header_lines"]:
        match = DATASET_TH_RE.search(line)
        if match:
            datasets.append(match.group(1))
    return datasets


def task_type_for_section(section):
    return "vlm" if "vlm" in section.lower() else "small"


def raw_values(lookup, ablation, task_type, model_name, datasets):
    values = []
    for dataset in datasets:
        for num_clients, beta, _label in SETTINGS:
            values.append(lookup.get((ablation, task_type, dataset, model_name, num_clients, beta)))
    return [fmt(value) for value in values]


def client_average_values(lookup, ablation, task_type, model_name, datasets):
    values = []
    for dataset in datasets:
        for num_clients, _label in CLIENT_AVG_GROUPS:
            group = []
            for cand_clients, beta, _raw_label in SETTINGS:
                if cand_clients != num_clients:
                    continue
                value = lookup.get((ablation, task_type, dataset, model_name, num_clients, beta))
                if value is not None:
                    group.append(value)
            values.append(sum(group) / len(group) if len(group) == 3 else None)
    return [fmt(value) for value in values]


def baseline_avg_values(base_table):
    for row in base_table["rows"]:
        if row.get("method") == "avg":
            return list(row["values"])
    return []


def ablation_rows_for_table(base_table, lookup, ablations, section, model_name, kind):
    task_type = task_type_for_section(section)
    datasets = table_datasets(base_table)
    rows = []
    for ablation in ablations:
        if kind == "Raw":
            values = raw_values(lookup, ablation, task_type, model_name, datasets)
        else:
            values = client_average_values(lookup, ablation, task_type, model_name, datasets)
        if ablation == "avg_only" and all(value == "-" for value in values):
            values = baseline_avg_values(base_table)
        if all(value == "-" for value in values):
            continue
        rows.append(
            {
                "method": ABLATION_LABELS.get(ablation, f"my_merge {ablation}"),
                "values": values,
            }
        )
    return rows


def merge_tables_with_ablations(base_tables, lookup, ablations):
    merged = {}
    for key, base_table in base_tables.items():
        section, model_name, kind = key
        rows = deepcopy(base_table["rows"])
        rows.extend(ablation_rows_for_table(base_table, lookup, ablations, section, model_name, kind))
        merged[key] = {
            "header_lines": base_table["header_lines"],
            "footer_lines": base_table["footer_lines"],
            "rows": highlight_rows(rows),
        }
    return merged


def build_output(merged_tables, grid_root, ablations):
    lines = [
        "# Experiment Master Tables With my_merge Ablations",
        "",
        f"- Base table: `result/all_results.md`.",
        f"- my_merge ablation source: `{grid_root}`.",
        "- Row labels state which module is missing. `M1` = Medical Evidence Client Weighting, `M2` = Conflict-Aware Delta and Specialist Preservation.",
        "- Highlight rule: highest value is bold, second-highest distinct value is underlined.",
        "",
        "## Ablation Rows",
        "",
        "| row | missing module |",
        "| --- | --- |",
    ]
    for ablation in ablations:
        label = ABLATION_LABELS.get(ablation, f"my_merge {ablation}")
        missing = label.split(" ", 2)[-1]
        lines.append(f"| `{label}` | `{missing}` |")
    lines.append("")

    current_section = None
    current_model = None
    for section, model_name, kind in merged_tables.keys():
        if section != current_section:
            lines.append(f"## {section}")
            lines.append("")
            current_section = section
            current_model = None
        if model_name != current_model:
            lines.append(f"### {model_name}")
            lines.append("")
            current_model = model_name
        lines.append(f"#### {kind}")
        lines.append("")
        lines.extend(render_table(merged_tables[(section, model_name, kind)]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    _intro, base_tables = parse_tables(Path(args.base))
    grid_root = Path(args.grid_root)
    lookup, ablations = load_lookup(grid_root)
    merged = merge_tables_with_ablations(base_tables, lookup, ablations)
    content = build_output(merged, args.grid_root, ablations)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
