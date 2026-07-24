#!/usr/bin/env python3
import argparse
import csv
from collections import OrderedDict
from pathlib import Path

from generate_lamp_merge_master_table import CLIENT_AVG_GROUPS, SETTINGS, emit_header, fmt


DEFAULT_METHOD_ORDER = [
    "avg",
    "avg_head",
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


def parse_args():
    p = argparse.ArgumentParser("Generate a formal-method master table from eval_summary.csv files")
    p.add_argument("--output-root", required=True, nargs="+")
    p.add_argument("--dest", required=True)
    p.add_argument("--section-title", default="Small")
    p.add_argument("--task-type", default="small", choices=["small", "vlm"])
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--small-models", nargs="+", required=True)
    p.add_argument("--methods", nargs="*", default=None)
    return p.parse_args()


def load_rows(output_roots):
    rows = []
    for root in output_roots:
        for csv_path in sorted(Path(root).rglob("eval_summary.csv")):
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows.extend(csv.DictReader(f))
    return rows


def row_key(row):
    task_type = row["task_type"]
    model_name = row["model"] if task_type == "small" else row["clip_model"]
    return (
        row["method"],
        task_type,
        row["dataset"],
        model_name,
        int(float(row["num_clients"])),
        float(row["beta"]),
    )


def build_lookup(rows):
    lookup = {}
    discovered = OrderedDict()
    for row in rows:
        method = row.get("method")
        if not method:
            continue
        discovered.setdefault(method, None)
        lookup[row_key(row)] = float(row["test_acc"])
    return lookup, list(discovered.keys())


def method_order(requested, discovered):
    methods = requested if requested else DEFAULT_METHOD_ORDER
    ordered = [method for method in methods if method in discovered]
    ordered.extend(method for method in discovered if method not in ordered)
    return ordered


def value_for(lookup, method, task_type, dataset, model_name, num_clients, beta):
    return lookup.get((method, task_type, dataset, model_name, num_clients, beta))


def client_average(lookup, method, task_type, dataset, model_name, num_clients):
    values = []
    for cand_clients, beta, _label in SETTINGS:
        if cand_clients != num_clients:
            continue
        value = value_for(lookup, method, task_type, dataset, model_name, num_clients, beta)
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if len(values) == 3 else None


def emit_method_rows(lines, rows):
    for method, values in rows:
        lines.append("    <tr>")
        lines.append(f"      <td>{method}</td>")
        for value in values:
            lines.append(f"      <td>{fmt(value)}</td>")
        lines.append("    </tr>")
    lines.append("  </tbody>")
    lines.append("</table>")


def build_model_section(lines, lookup, methods, *, task_type, model_name, datasets):
    raw_headers = [label for _clients, _beta, label in SETTINGS]
    avg_headers = [label for _clients, label in CLIENT_AVG_GROUPS]

    lines.append(f"### {model_name}")
    lines.append("")
    lines.append("#### Raw")
    lines.append("")
    emit_header(lines, datasets, raw_headers)
    raw_rows = []
    for method in methods:
        values = []
        for dataset in datasets:
            for num_clients, beta, _label in SETTINGS:
                values.append(value_for(lookup, method, task_type, dataset, model_name, num_clients, beta))
        if any(value is not None for value in values):
            raw_rows.append((method, values))
    emit_method_rows(lines, raw_rows)
    lines.append("")

    lines.append("#### Client Average")
    lines.append("")
    emit_header(lines, datasets, avg_headers)
    avg_rows = []
    for method in methods:
        values = []
        for dataset in datasets:
            for num_clients, _label in CLIENT_AVG_GROUPS:
                values.append(client_average(lookup, method, task_type, dataset, model_name, num_clients))
        if any(value is not None for value in values):
            avg_rows.append((method, values))
    emit_method_rows(lines, avg_rows)
    lines.append("")


def build_markdown(args, lookup, methods):
    lines = [
        "# Experiment Master Tables",
        "",
        "- Layout: one model per table; rows are methods; columns are grouped by dataset.",
        f"- Source output root: `{', '.join(args.output_root)}`.",
        "",
        f"## {args.section_title}",
        "",
    ]
    for model_name in args.small_models:
        build_model_section(
            lines,
            lookup,
            methods,
            task_type=args.task_type,
            model_name=model_name,
            datasets=args.datasets,
        )
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    rows = load_rows(args.output_root)
    lookup, discovered = build_lookup(rows)
    methods = method_order(args.methods, discovered)
    content = build_markdown(args, lookup, methods)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
