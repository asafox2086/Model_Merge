#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


SETTINGS = [
    (3, 0.0, "c3_b0"),
    (3, 0.01, "c3_b0.01"),
    (3, 0.1, "c3_b0.1"),
    (5, 0.0, "c5_b0"),
    (5, 0.01, "c5_b0.01"),
    (5, 0.1, "c5_b0.1"),
    (7, 0.0, "c7_b0"),
    (7, 0.01, "c7_b0.01"),
    (7, 0.1, "c7_b0.1"),
]

CLIENT_AVG_GROUPS = [
    (3, "c3_avg"),
    (5, "c5_avg"),
    (7, "c7_avg"),
]

SMALL_DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
]

SMALL_MODELS = [
    "resnet",
    "convnext",
    "vit_t",
    "swin_tiny",
]

VLM_MODELS = [
    "openai/clip-vit-base-patch32",
]


def parse_args():
    p = argparse.ArgumentParser("Generate my_merge master markdown table from eval summaries")
    p.add_argument(
        "--output-root",
        required=True,
        nargs="+",
        help="One or more batch output roots, e.g. outputs/custom_methods_<run_tag>",
    )
    p.add_argument("--dest", required=True, help="Destination markdown file")
    return p.parse_args()


def load_eval_rows(output_roots):
    rows = []
    for output_root in output_roots:
        csv_paths = []
        direct_path = output_root / "reports" / "eval_summary.csv"
        if direct_path.exists():
            csv_paths.append(direct_path)
        csv_paths.extend(sorted(output_root.glob("*/reports/eval_summary.csv")))
        for csv_path in csv_paths:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows.extend(list(csv.DictReader(f)))
    return rows


def key_for_row(row):
    task_type = row["task_type"]
    dataset = row["dataset"]
    model_name = row["model"] if task_type == "small" else row["clip_model"]
    num_clients = int(float(row["num_clients"]))
    beta = float(row["beta"])
    return task_type, dataset, model_name, num_clients, beta


def build_lookup(rows):
    lookup = {}
    for row in rows:
        if row.get("method") != "my_merge":
            continue
        lookup[key_for_row(row)] = float(row["test_acc"])
    return lookup


def fmt(value):
    return "-" if value is None else f"{value:.4f}"


def get_raw_value(lookup, task_type, dataset, model_name, num_clients, beta):
    return lookup.get((task_type, dataset, model_name, num_clients, beta))


def get_client_average(lookup, task_type, dataset, model_name, num_clients):
    values = []
    for cand_clients, beta, _ in SETTINGS:
        if cand_clients != num_clients:
            continue
        value = get_raw_value(lookup, task_type, dataset, model_name, num_clients, beta)
        if value is None:
            return None
        values.append(value)
    return sum(values) / len(values) if values else None


def emit_header(lines, dataset_names, subheaders):
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append("    <tr>")
    lines.append('      <th rowspan="2">method</th>')
    for dataset in dataset_names:
        lines.append(f'      <th colspan="{len(subheaders)}">{dataset}</th>')
    lines.append("    </tr>")
    lines.append("    <tr>")
    for _dataset in dataset_names:
        for header in subheaders:
            lines.append(f"      <th>{header}</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")


def emit_single_method_row(lines, values):
    lines.append("    <tr>")
    lines.append("      <td>my_merge</td>")
    for value in values:
        lines.append(f"      <td>{fmt(value)}</td>")
    lines.append("    </tr>")
    lines.append("  </tbody>")
    lines.append("</table>")


def build_model_section(lines, lookup, *, task_type, model_name, dataset_names, raw_headers, avg_headers):
    lines.append(f"### {model_name}")
    lines.append("")
    lines.append("#### Raw")
    lines.append("")
    emit_header(lines, dataset_names, raw_headers)
    raw_values = []
    for dataset in dataset_names:
        for num_clients, beta, _label in SETTINGS:
            raw_values.append(get_raw_value(lookup, task_type, dataset, model_name, num_clients, beta))
    emit_single_method_row(lines, raw_values)
    lines.append("")
    lines.append("#### Client Average")
    lines.append("")
    emit_header(lines, dataset_names, avg_headers)
    avg_values = []
    for dataset in dataset_names:
        for num_clients, _label in CLIENT_AVG_GROUPS:
            avg_values.append(get_client_average(lookup, task_type, dataset, model_name, num_clients))
    emit_single_method_row(lines, avg_values)
    lines.append("")


def build_markdown(output_roots):
    rows = load_eval_rows(output_roots)
    lookup = build_lookup(rows)
    roots_label = ", ".join(str(root) for root in output_roots)
    lines = [
        "# Experiment Master Tables",
        "",
        "- Layout: aligned with `result/all_results.md`.",
        f"- Source output root: `{roots_label}`.",
        "- Values are filled from real `eval_summary.csv` results for `my_merge`; missing combinations are shown as `-`.",
        "",
        "## Small",
        "",
    ]

    raw_headers = [label for _c, _b, label in SETTINGS]
    avg_headers = [label for _c, label in CLIENT_AVG_GROUPS]

    for model_name in SMALL_MODELS:
        build_model_section(
            lines,
            lookup,
            task_type="small",
            model_name=model_name,
            dataset_names=SMALL_DATASETS,
            raw_headers=raw_headers,
            avg_headers=avg_headers,
        )

    lines.extend([
        "## VLM",
        "",
    ])
    for model_name in VLM_MODELS:
        build_model_section(
            lines,
            lookup,
            task_type="vlm",
            model_name=model_name,
            dataset_names=SMALL_DATASETS,
            raw_headers=raw_headers,
            avg_headers=avg_headers,
        )

    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    output_roots = [Path(item) for item in args.output_root]
    dest = Path(args.dest)
    content = build_markdown(output_roots)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
