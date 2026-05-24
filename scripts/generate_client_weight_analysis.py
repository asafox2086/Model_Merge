#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from generate_ablation_combined_results_table import (
    ablation_sort_key,
    case_sort_key,
    discover_merge_jsons,
    fmt,
    fmt_weight,
    load_lookup,
    load_weight_rows,
)


def parse_args():
    p = argparse.ArgumentParser("Generate detailed client/model weight analysis for my_merge")
    p.add_argument("--grid-root", required=True)
    p.add_argument("--dest-dir", required=True)
    return p.parse_args()


def case_id(row):
    return (
        row["ablation"],
        row["task_type"],
        row["dataset"],
        row["model"],
        row["num_clients"],
        row["beta"],
        row["seed"],
    )


def case_key_without_ablation(row):
    return (
        row["task_type"],
        row["dataset"],
        row["model"],
        row["num_clients"],
        row["beta"],
    )


def load_selected_lookup(grid_root):
    selected = {}
    for json_path in discover_merge_jsons(Path(grid_root)):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("method") != "my_merge":
            continue
        rel = json_path.relative_to(grid_root)
        ablation = rel.parts[0] if rel.parts else "unknown"
        method_info = payload.get("method_info", {})
        key = (
            ablation,
            payload.get("task_type", ""),
            payload.get("dataset", ""),
            payload.get("model") or payload.get("clip_model", ""),
            int(float(payload.get("num_clients", 0) or 0)),
            float(payload.get("beta", 0.0) or 0.0),
        )
        selected[key] = method_info.get("selected_candidate", "")
    return selected


def expand_lookup_model_aliases(lookup):
    expanded = dict(lookup)
    for key, value in list(lookup.items()):
        ablation, task_type, dataset, model, num_clients, beta = key
        if "/" in model:
            expanded[(ablation, task_type, dataset, model.split("/")[-1], num_clients, beta)] = value
        elif task_type == "vlm":
            expanded[(ablation, task_type, dataset, f"openai/{model}", num_clients, beta)] = value
    return expanded


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def stdev(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def top_share(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return max(values) if values else None


def entropy_norm(rows, field):
    values = [row[field] for row in rows if row[field] is not None and row[field] > 0]
    if not values:
        return None
    total = sum(values)
    if total <= 0:
        return None
    probs = [v / total for v in values]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(probs)) if len(probs) > 1 else 0.0


def summarize_cases(weight_rows, acc_lookup, selected_lookup):
    grouped = defaultdict(list)
    for row in weight_rows:
        grouped[case_id(row)].append(row)

    out = []
    for key, rows in grouped.items():
        ablation, task_type, dataset, model, num_clients, beta, seed = key
        acc_key = (ablation, task_type, dataset, model, num_clients, beta)
        deltas_all = [
            abs(row["diagnostic_weight_alpha_all"] - row["prior_weight_pi"])
            for row in rows
            if row["diagnostic_weight_alpha_all"] is not None and row["prior_weight_pi"] is not None
        ]
        deltas_morph = [
            abs(row["medical_weight_alpha_morph"] - row["prior_weight_pi"])
            for row in rows
            if row["medical_weight_alpha_morph"] is not None and row["prior_weight_pi"] is not None
        ]
        out.append(
            {
                "ablation": ablation,
                "task_type": task_type,
                "dataset": dataset,
                "model": model,
                "num_clients": num_clients,
                "beta": beta,
                "seed": seed,
                "selected_candidate": selected_lookup.get(acc_key, rows[0].get("selected_candidate", "")),
                "test_acc": acc_lookup.get(acc_key),
                "mean_abs_delta_all": mean(deltas_all),
                "mean_abs_delta_morph": mean(deltas_morph),
                "max_alpha_all": top_share(rows, "diagnostic_weight_alpha_all"),
                "max_alpha_morph": top_share(rows, "medical_weight_alpha_morph"),
                "entropy_alpha_all": entropy_norm(rows, "diagnostic_weight_alpha_all"),
                "entropy_alpha_morph": entropy_norm(rows, "medical_weight_alpha_morph"),
            }
        )
    return sorted(out, key=lambda r: (ablation_sort_key(r["ablation"]), r["task_type"], r["dataset"], r["model"], r["num_clients"], r["beta"]))


def summarize_groups(case_rows, group_fields):
    grouped = defaultdict(list)
    for row in case_rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    out = []
    for key, rows in grouped.items():
        base = dict(zip(group_fields, key))
        base.update(
            {
                "cases": len(rows),
                "mean_acc": mean(row["test_acc"] for row in rows),
                "mean_abs_delta_all": mean(row["mean_abs_delta_all"] for row in rows),
                "mean_abs_delta_morph": mean(row["mean_abs_delta_morph"] for row in rows),
                "mean_max_alpha_all": mean(row["max_alpha_all"] for row in rows),
                "mean_max_alpha_morph": mean(row["max_alpha_morph"] for row in rows),
                "mean_entropy_alpha_all": mean(row["entropy_alpha_all"] for row in rows),
                "mean_entropy_alpha_morph": mean(row["entropy_alpha_morph"] for row in rows),
            }
        )
        out.append(base)
    return sorted(out, key=lambda r: tuple(str(r.get(field, "")) for field in group_fields))


def add_full_gains(case_rows):
    lookup = {case_key_without_ablation(row): row for row in case_rows if row["ablation"] == "full"}
    for row in case_rows:
        full = lookup.get(case_key_without_ablation(row))
        row["delta_acc_vs_full"] = None
        if full and row["test_acc"] is not None and full["test_acc"] is not None:
            row["delta_acc_vs_full"] = row["test_acc"] - full["test_acc"]
    return case_rows


def render_table(lines, headers, rows, limit=None):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows[:limit]:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                value = fmt(value)
            values.append(str(value) if value is not None else "-")
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")


def render_markdown(path, weight_rows, case_rows, by_ablation, by_dataset, by_model, top_changed, top_concentrated):
    lines = [
        "# Client Weight Analysis",
        "",
        "- Source: generated from `merge_result.json`; no manual table filling.",
        "- `pi` is the prior merge weight.",
        "- `alpha_all` is the M1 overall diagnostic weight.",
        "- `alpha_morph` is the M2 morphology/medical-evidence weight.",
        "- CSV files in this folder contain the full detail for filtering and plotting.",
        "",
        "## Files",
        "",
        "- `client_weight_detail.csv`: one row per ablation/case/client.",
        "- `client_weight_case_summary.csv`: one row per ablation/case.",
        "- `client_weight_by_ablation.csv`: whether each ablation really changes weights.",
        "- `client_weight_by_dataset.csv`: which datasets create larger medical reweighting.",
        "- `client_weight_by_model.csv`: which backbone families are more sensitive.",
        "",
        "## Ablation Summary",
        "",
    ]
    render_table(
        lines,
        [
            "ablation",
            "cases",
            "mean_acc",
            "mean_abs_delta_all",
            "mean_abs_delta_morph",
            "mean_max_alpha_all",
            "mean_entropy_alpha_all",
        ],
        by_ablation,
    )

    lines.extend(["## Dataset Summary", ""])
    render_table(
        lines,
        ["ablation", "dataset", "cases", "mean_acc", "mean_abs_delta_all", "mean_abs_delta_morph", "mean_max_alpha_all"],
        by_dataset,
    )

    lines.extend(["## Model Summary", ""])
    render_table(
        lines,
        ["ablation", "task_type", "model", "cases", "mean_acc", "mean_abs_delta_all", "mean_abs_delta_morph", "mean_entropy_alpha_all"],
        by_model,
    )

    lines.extend(["## Cases With Largest Weight Shift", ""])
    render_table(
        lines,
        [
            "ablation",
            "task_type",
            "dataset",
            "model",
            "num_clients",
            "beta",
            "selected_candidate",
            "test_acc",
            "mean_abs_delta_all",
            "mean_abs_delta_morph",
        ],
        top_changed,
        limit=30,
    )

    lines.extend(["## Cases With Most Concentrated Diagnostic Weight", ""])
    render_table(
        lines,
        [
            "ablation",
            "task_type",
            "dataset",
            "model",
            "num_clients",
            "beta",
            "selected_candidate",
            "test_acc",
            "max_alpha_all",
            "entropy_alpha_all",
        ],
        top_concentrated,
        limit=30,
    )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    args = parse_args()
    grid_root = Path(args.grid_root)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    acc_lookup, _ablations = load_lookup(grid_root)
    acc_lookup = expand_lookup_model_aliases(acc_lookup)
    selected_lookup = load_selected_lookup(grid_root)
    selected_lookup = expand_lookup_model_aliases(selected_lookup)
    weight_rows = load_weight_rows(grid_root)
    case_rows = add_full_gains(summarize_cases(weight_rows, acc_lookup, selected_lookup))

    detail_fields = [
        "ablation",
        "task_type",
        "dataset",
        "model",
        "num_clients",
        "beta",
        "seed",
        "selected_candidate",
        "client_index",
        "client_name",
        "prior_weight_pi",
        "diagnostic_weight_alpha_all",
        "medical_weight_alpha_morph",
    ]
    write_csv(dest_dir / "client_weight_detail.csv", sorted(weight_rows, key=case_sort_key), detail_fields)

    case_fields = [
        "ablation",
        "task_type",
        "dataset",
        "model",
        "num_clients",
        "beta",
        "seed",
        "selected_candidate",
        "test_acc",
        "delta_acc_vs_full",
        "mean_abs_delta_all",
        "mean_abs_delta_morph",
        "max_alpha_all",
        "max_alpha_morph",
        "entropy_alpha_all",
        "entropy_alpha_morph",
    ]
    write_csv(dest_dir / "client_weight_case_summary.csv", case_rows, case_fields)

    by_ablation = summarize_groups(case_rows, ["ablation"])
    by_dataset = summarize_groups(case_rows, ["ablation", "dataset"])
    by_model = summarize_groups(case_rows, ["ablation", "task_type", "model"])
    write_csv(dest_dir / "client_weight_by_ablation.csv", by_ablation, list(by_ablation[0].keys()))
    write_csv(dest_dir / "client_weight_by_dataset.csv", by_dataset, list(by_dataset[0].keys()))
    write_csv(dest_dir / "client_weight_by_model.csv", by_model, list(by_model[0].keys()))

    top_changed = sorted(
        case_rows,
        key=lambda r: (r["mean_abs_delta_morph"] is not None, r["mean_abs_delta_morph"] or -1, r["mean_abs_delta_all"] or -1),
        reverse=True,
    )
    top_concentrated = sorted(
        case_rows,
        key=lambda r: (r["max_alpha_all"] is not None, r["max_alpha_all"] or -1),
        reverse=True,
    )
    render_markdown(
        dest_dir / "client_weight_analysis.md",
        weight_rows,
        case_rows,
        by_ablation,
        by_dataset,
        by_model,
        top_changed,
        top_concentrated,
    )
    print(f"wrote {dest_dir / 'client_weight_analysis.md'}")


if __name__ == "__main__":
    main()
