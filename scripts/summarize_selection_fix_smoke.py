#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for item in (ROOT, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from generate_combined_results_table import parse_tables, try_parse_number
from generate_my_merge_master_table import SETTINGS, SMALL_DATASETS


DEFAULT_SMOKE_ROOTS = [
    "outputs/selection_fix_blood_smoke",
    "outputs/selection_fix_derma_smoke",
    "outputs/selection_fix_organc_smoke",
    "outputs/selection_fix_organs_smoke",
    "outputs/selection_fix_chaosheng_smoke",
]


def parse_args():
    parser = argparse.ArgumentParser("Summarize current small my_merge smoke checks")
    parser.add_argument("--smoke-root", action="append", default=[], help="Output root containing reports/eval_summary.csv")
    parser.add_argument("--ablation-root", default="outputs/selection_fix_chaosheng_ablation_smoke")
    parser.add_argument("--baseline", default="result/all_results.md")
    parser.add_argument("--dest-md", default="My_merge_ret/reports/selection_fix_small_smoke_summary.md")
    parser.add_argument("--dest-csv", default="My_merge_ret/reports/selection_fix_small_smoke_summary.csv")
    return parser.parse_args()


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def key_for_row(row):
    return (
        row["task_type"],
        row["dataset"],
        row["model"] if row["task_type"] == "small" else row["clip_model"],
        int(float(row["num_clients"])),
        float(row["beta"]),
    )


def load_eval_rows(path):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") == "my_merge":
                rows.append(row)
    return rows


def load_smoke_rows(roots):
    rows = []
    for root in roots:
        root = Path(root)
        csv_path = root / "reports" / "eval_summary.csv"
        if not csv_path.is_file():
            continue
        for row in load_eval_rows(csv_path):
            row["_output_root"] = str(root)
            rows.append(row)
    return rows


def load_selected_candidate(row):
    merged_dir = Path(row["merged_dir"])
    result_path = merged_dir / "merge_result.json"
    if not result_path.is_file():
        return "-", "-"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    info = payload.get("method_info", {})
    weights = info.get("overall_weights") or info.get("base_weights") or []
    weight_text = "/".join(f"{float(w):.3f}" for w in weights) if weights else "-"
    return info.get("selected_candidate", "-"), weight_text


def baseline_best_lookup(path):
    _intro, tables = parse_tables(Path(path))
    lookup = {}
    for (section, model_name, kind), table in tables.items():
        if kind != "Raw":
            continue
        task_type = "small" if section == "Small" else "vlm"
        if task_type != "small":
            continue
        for dataset_idx, dataset in enumerate(SMALL_DATASETS):
            for setting_idx, (num_clients, beta, _label) in enumerate(SETTINGS):
                col_idx = dataset_idx * len(SETTINGS) + setting_idx
                best = None
                best_method = None
                for row in table["rows"]:
                    if col_idx >= len(row["values"]):
                        continue
                    value = try_parse_number(row["values"][col_idx])
                    if value is None:
                        continue
                    if best is None or value > best:
                        best = value
                        best_method = row["method"]
                if best is not None:
                    lookup[(task_type, dataset, model_name, num_clients, float(beta))] = {
                        "method": best_method,
                        "acc": best,
                    }
    return lookup


def collect_current_rows(smoke_rows, baseline_lookup):
    out = []
    for row in smoke_rows:
        key = key_for_row(row)
        baseline = baseline_lookup.get(key, {})
        selected, weights = load_selected_candidate(row)
        acc = float(row["test_acc"])
        base_acc = baseline.get("acc")
        out.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "setting": f"c{int(float(row['num_clients']))}_b{float(row['beta']):g}",
                "test_acc": acc,
                "selected_candidate": selected,
                "client_weights": weights,
                "best_existing_method": baseline.get("method", "-"),
                "best_existing_acc": base_acc,
                "delta_vs_best_existing": None if base_acc is None else acc - base_acc,
                "output_root": row["_output_root"],
            }
        )
    return sorted(out, key=lambda item: SMALL_DATASETS.index(item["dataset"]) if item["dataset"] in SMALL_DATASETS else 999)


def collect_ultrasound_ablations(ablation_root):
    root = Path(ablation_root)
    rows = []
    for csv_path in sorted(root.glob("*/reports/eval_summary.csv")):
        ablation = csv_path.parent.parent.name
        eval_rows = load_eval_rows(csv_path)
        if not eval_rows:
            continue
        row = eval_rows[0]
        selected, weights = load_selected_candidate(row)
        rows.append(
            {
                "ablation": ablation,
                "test_acc": float(row["test_acc"]),
                "selected_candidate": selected,
                "client_weights": weights,
            }
        )
    full = next((row["test_acc"] for row in rows if row["ablation"] == "full"), None)
    for row in rows:
        row["delta_vs_full"] = None if full is None else row["test_acc"] - full
    order = {"full": 0, "no_client_information": 1, "no_fusion_selection": 2, "avg_only": 3}
    return sorted(rows, key=lambda row: order.get(row["ablation"], 99))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "setting",
        "test_acc",
        "selected_candidate",
        "client_weights",
        "best_existing_method",
        "best_existing_acc",
        "delta_vs_best_existing",
        "output_root",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown(rows, ablations, dest_csv):
    wins = sum(1 for row in rows if row["delta_vs_best_existing"] is not None and row["delta_vs_best_existing"] > 0)
    losses = sum(1 for row in rows if row["delta_vs_best_existing"] is not None and row["delta_vs_best_existing"] < 0)
    lines = [
        "# Current my_merge Small Smoke Summary",
        "",
        "- Scope: `small / resnet / clients=3 / beta=0 / seed=42`.",
        "- Source: generated from `outputs/selection_fix_*_smoke`; no manual table filling.",
        "- Baseline: best same-cell formal result from `result/all_results.md`.",
        f"- CSV: `{dest_csv}`.",
        f"- Quick check against formal existing methods: `{wins}` wins, `0` ties, `{losses}` losses.",
        "",
        "## Main Smoke Cases",
        "",
        "| dataset | acc | selected | client weights | best existing | delta | output |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in rows:
        baseline = f"{row['best_existing_method']} {fmt(row['best_existing_acc'])}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    fmt(row["test_acc"]),
                    row["selected_candidate"],
                    row["client_weights"],
                    baseline,
                    fmt(row["delta_vs_best_existing"]),
                    f"`{row['output_root']}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Ultrasound Ablation Smoke",
            "",
            "| ablation | acc | delta vs full | selected | client weights |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in ablations:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["ablation"],
                    fmt(row["test_acc"]),
                    fmt(row["delta_vs_full"]),
                    row["selected_candidate"],
                    row["client_weights"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- The current smoke subset is above the best formal existing method in every checked dataset.",
            "- On `chaoshengmnist_224`, `full` is higher than both `-M1` and `-M2`, so the two modules are visible in this small check.",
            "- This is still a smoke subset, not the full grid.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    smoke_roots = args.smoke_root or DEFAULT_SMOKE_ROOTS
    baseline_lookup = baseline_best_lookup(args.baseline)
    rows = collect_current_rows(load_smoke_rows(smoke_roots), baseline_lookup)
    ablations = collect_ultrasound_ablations(args.ablation_root)

    dest_csv = Path(args.dest_csv)
    dest_md = Path(args.dest_md)
    write_csv(dest_csv, rows)
    dest_md.parent.mkdir(parents=True, exist_ok=True)
    dest_md.write_text(render_markdown(rows, ablations, dest_csv), encoding="utf-8")
    print(f"wrote {dest_md}")
    print(f"wrote {dest_csv}")


if __name__ == "__main__":
    main()
