#!/usr/bin/env python3
import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

from generate_combined_results_table import highlight_rows, parse_tables, render_table
from generate_my_merge_master_table import CLIENT_AVG_GROUPS, SETTINGS, SMALL_DATASETS, SMALL_MODELS, VLM_MODELS


ABLATION_LABELS = {
    "full": "my_merge full (none missing)",
    "avg_only": "my_merge avg_only (-M1,-M2)",
    "no_diagnostic_evidence": "my_merge -M1 diagnostic_evidence",
    "no_domain_preprocess": "my_merge -M1 diagnostic_info",
    "no_modality_features": "my_merge -M1 diagnostic_info",
    "no_medical_preprocess": "my_merge -M1 diagnostic_info",
    "no_medical_prior": "my_merge -M1 diagnostic_info",
    "no_client_information": "my_merge -M1 client_info",
    "no_diagnostic_information": "my_merge -M1 client_info",
    "no_rarity": "my_merge -M1 class_rarity",
    "no_focal": "my_merge -M1 focal_weight",
    "no_domain_focus": "my_merge -M1 domain_focus",
    "no_fusion_selection": "my_merge -M2 fusion_select",
    "no_medical_fusion_selection": "my_merge -M2 fusion_select",
    "no_balanced_selection": "my_merge -M2 balanced_select",
    "no_layerwise": "my_merge -M2 layerwise",
    "no_residual": "my_merge -M2 residual",
    "no_candidate_bank": "my_merge -M2 candidate_bank",
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


def discover_merge_jsons(grid_root):
    grid_root = Path(grid_root)
    return sorted(path for path in grid_root.glob("*/*/merged/**/merge_result.json") if path.is_file())


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
    return lookup, sorted(ablations, key=ablation_sort_key)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_weight(value):
    value = safe_float(value)
    return "-" if value is None else f"{value:.4f}"


def load_weight_rows(grid_root):
    rows = []
    for json_path in discover_merge_jsons(Path(grid_root)):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("method") != "my_merge":
            continue
        try:
            ablation = json_path.relative_to(grid_root).parts[0]
        except Exception:
            ablation = "unknown"
        method_info = payload.get("method_info", {})
        client_info = method_info.get("client_diagnostic_information", [])
        base = {
            "ablation": ablation,
            "task_type": payload.get("task_type", ""),
            "dataset": payload.get("dataset", ""),
            "model": payload.get("model") or payload.get("clip_model", ""),
            "num_clients": int(float(payload.get("num_clients", 0) or 0)),
            "beta": float(payload.get("beta", 0.0) or 0.0),
            "seed": payload.get("seed", ""),
            "selected_candidate": method_info.get("selected_candidate", ""),
        }
        if client_info:
            for item in client_info:
                row = dict(base)
                row.update(
                    {
                        "client_index": int(item.get("client_index", 0) or 0),
                        "client_name": item.get("client_name", ""),
                        "prior_weight_pi": safe_float(item.get("base_weight")),
                        "diagnostic_weight_alpha_all": safe_float(item.get("overall_weight")),
                        "medical_weight_alpha_morph": safe_float(item.get("morphology_weight")),
                    }
                )
                rows.append(row)
        else:
            weights = method_info.get("base_weights", method_info.get("normalized_weights", []))
            source_clients = payload.get("source_clients", [])
            for client_idx, weight in enumerate(weights):
                row = dict(base)
                row.update(
                    {
                        "client_index": int(client_idx),
                        "client_name": source_clients[client_idx] if client_idx < len(source_clients) else f"client_{client_idx}.pt",
                        "prior_weight_pi": safe_float(weight),
                        "diagnostic_weight_alpha_all": None,
                        "medical_weight_alpha_morph": None,
                    }
                )
                rows.append(row)
    return rows


def case_sort_key(row):
    return (
        ablation_sort_key(row.get("ablation", "")),
        row.get("task_type", ""),
        row.get("dataset", ""),
        row.get("model", ""),
        int(row.get("num_clients", 0) or 0),
        float(row.get("beta", 0.0) or 0.0),
        int(row.get("client_index", 0) or 0),
    )


def summarize_weight_rows(weight_rows):
    grouped = {}
    for row in weight_rows:
        grouped.setdefault(row["ablation"], []).append(row)
    summaries = []
    for ablation in sorted(grouped, key=ablation_sort_key):
        group = grouped[ablation]
        pi_values = [row["prior_weight_pi"] for row in group if row["prior_weight_pi"] is not None]
        alpha_all_values = [row["diagnostic_weight_alpha_all"] for row in group if row["diagnostic_weight_alpha_all"] is not None]
        alpha_morph_values = [row["medical_weight_alpha_morph"] for row in group if row["medical_weight_alpha_morph"] is not None]
        delta_all = [
            abs(row["diagnostic_weight_alpha_all"] - row["prior_weight_pi"])
            for row in group
            if row["diagnostic_weight_alpha_all"] is not None and row["prior_weight_pi"] is not None
        ]
        delta_morph = [
            abs(row["medical_weight_alpha_morph"] - row["prior_weight_pi"])
            for row in group
            if row["medical_weight_alpha_morph"] is not None and row["prior_weight_pi"] is not None
        ]
        summaries.append(
            {
                "ablation": ablation,
                "rows": len(group),
                "mean_pi": sum(pi_values) / len(pi_values) if pi_values else None,
                "mean_alpha_all": sum(alpha_all_values) / len(alpha_all_values) if alpha_all_values else None,
                "mean_alpha_morph": sum(alpha_morph_values) / len(alpha_morph_values) if alpha_morph_values else None,
                "mean_abs_delta_all": sum(delta_all) / len(delta_all) if delta_all else None,
                "mean_abs_delta_morph": sum(delta_morph) / len(delta_morph) if delta_morph else None,
            }
        )
    return summaries


def build_weight_sections(weight_rows):
    if not weight_rows:
        return [
            "## Weight Analysis",
            "",
            "- No `merge_result.json` weight records were found. Re-run my_merge with diagnostics enabled.",
            "",
        ]

    lines = [
        "## Weight Analysis",
        "",
        "- Weight details are generated from `merge_result.json`; they are not manually filled.",
        "- Full per-client weights are kept outside this master table to preserve readability.",
        "- Deep analysis: `reports/client_weight_analysis.md`.",
        "- Raw detail CSV: `reports/client_weight_detail.csv`.",
        "",
        "### Average Weight Change",
        "",
        "| ablation | rows | mean_pi | mean_alpha_all | mean_alpha_morph | mean_abs_delta_all | mean_abs_delta_morph |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summarize_weight_rows(weight_rows):
        lines.append(
            f"| `{item['ablation']}` | {item['rows']} | {fmt_weight(item['mean_pi'])} | {fmt_weight(item['mean_alpha_all'])} | {fmt_weight(item['mean_alpha_morph'])} | {fmt_weight(item['mean_abs_delta_all'])} | {fmt_weight(item['mean_abs_delta_morph'])} |"
        )
    lines.append("")
    return lines


def ablation_sort_key(name):
    order = [
        "full",
        "no_client_information",
        "no_fusion_selection",
        "avg_only",
        "no_diagnostic_evidence",
        "no_medical_prior",
        "no_domain_preprocess",
        "no_rarity",
        "no_focal",
        "no_domain_focus",
        "no_layerwise",
        "no_residual",
        "no_candidate_bank",
        "no_balanced_selection",
        "no_calibration",
    ]
    return (order.index(name) if name in order else len(order), name)


def fmt(value):
    return "-" if value is None else f"{value:.4f}"


def raw_values(lookup, ablation, task_type, model_name):
    values = []
    for dataset in SMALL_DATASETS:
        for num_clients, beta, _label in SETTINGS:
            values.append(lookup.get((ablation, task_type, dataset, model_name, num_clients, beta)))
    return [fmt(value) for value in values]


def client_average_values(lookup, ablation, task_type, model_name):
    values = []
    for dataset in SMALL_DATASETS:
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


def ablation_rows_for_table(lookup, ablations, section, model_name, kind):
    task_type = "small" if section == "Small" else "vlm"
    rows = []
    for ablation in ablations:
        if kind == "Raw":
            values = raw_values(lookup, ablation, task_type, model_name)
        else:
            values = client_average_values(lookup, ablation, task_type, model_name)
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
        rows.extend(ablation_rows_for_table(lookup, ablations, section, model_name, kind))
        merged[key] = {
            "header_lines": base_table["header_lines"],
            "footer_lines": base_table["footer_lines"],
            "rows": highlight_rows(rows),
        }
    return merged


def build_output(merged_tables, grid_root, ablations, weight_rows):
    lines = [
        "# Experiment Master Tables With my_merge Ablations",
        "",
        f"- Base table: `result/all_results.md`.",
        f"- my_merge ablation source: `{grid_root}`.",
        "- Row labels state which module is missing. `M1` = Diagnostic Evidence Client Information Estimation, `M2` = Medical Evidence Guided Fusion and Selection.",
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
    lines.extend(build_weight_sections(weight_rows))

    current_section = None
    current_model = None
    order = []
    for section in ["Small", "VLM"]:
        models = SMALL_MODELS if section == "Small" else VLM_MODELS
        for model_name in models:
            for kind in ["Raw", "Client Average"]:
                key = (section, model_name, kind)
                if key in merged_tables:
                    order.append(key)

    for section, model_name, kind in order:
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
    weight_rows = load_weight_rows(grid_root)
    merged = merge_tables_with_ablations(base_tables, lookup, ablations)
    content = build_output(merged, args.grid_root, ablations, weight_rows)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
