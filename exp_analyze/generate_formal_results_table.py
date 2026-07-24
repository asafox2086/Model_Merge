#!/usr/bin/env python3
import argparse
import csv
from collections import OrderedDict
from pathlib import Path

from generate_lamp_merge_master_table import CLIENT_AVG_GROUPS, SETTINGS, emit_header, fmt


DEFAULT_METHOD_ORDER = [
    "head_avg",
    "head_ties",
    "head_dare_linear",
    "head_dare_ties",
    "head_regmean",
    "head_fisher",
    "head_breadcrumbs",
    "head_model_stock",
    "head_from",
    "head_iso_c",
    "head_free_merge",
    "head_robustmerge",
    "LAMP-Merge",
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
    p.add_argument("--metric-format", choices=["acc", "acc_f1"], default="acc")
    p.add_argument("--diagnostics-csv", type=Path, default=None)
    p.add_argument("--lamp-diagnostics-csv", type=Path, default=None)
    return p.parse_args()


def load_rows(output_roots):
    rows = []
    for root in output_roots:
        for csv_path in sorted(Path(root).rglob("eval_summary.csv")):
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows.extend(csv.DictReader(f))
    return rows


def load_diagnostics_rows(paths):
    rows = []
    for path in paths:
        if not path:
            continue
        path = Path(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
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


def diagnostics_key(row):
    task_type = row["task_type"]
    model_name = row["model"] if task_type == "small" else row.get("clip_model", "")
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


def build_diagnostics_lookup(rows):
    lookup = {}
    discovered = OrderedDict()
    for row in rows:
        if row.get("status") != "OK" or row.get("source") != "merge":
            continue
        method = row.get("method")
        if method in {"lamp_merge", "lamp_merge:full"}:
            method = "LAMP-Merge"
        if not method:
            continue
        discovered.setdefault(method, None)
        item = dict(row)
        item["method"] = method
        lookup[diagnostics_key(item)] = (float(row["accuracy"]), float(row["macro_f1"]))
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
    if len(values) != 3:
        return None
    if isinstance(values[0], tuple):
        return (
            sum(value[0] for value in values) / len(values),
            sum(value[1] for value in values) / len(values),
        )
    return sum(values) / len(values)


def fmt_value(value, metric_format):
    if value is None:
        return "-"
    if isinstance(value, tuple):
        return f"{value[0]:.4f} / {value[1]:.4f}"
    return f"{value:.4f}"


def emit_method_rows(lines, rows, metric_format):
    for method, values in rows:
        lines.append("    <tr>")
        lines.append(f"      <td>{method}</td>")
        for value in values:
            lines.append(f"      <td>{fmt_value(value, metric_format)}</td>")
        lines.append("    </tr>")
    lines.append("  </tbody>")
    lines.append("</table>")


def build_model_section(lines, lookup, methods, *, task_type, model_name, datasets, metric_format):
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
        raw_rows.append((method, values))
    emit_method_rows(lines, raw_rows, metric_format)
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
        avg_rows.append((method, values))
    emit_method_rows(lines, avg_rows, metric_format)
    lines.append("")


def build_markdown(args, lookup, methods):
    lines = [
        "# Experiment Master Tables",
        "",
        "- Layout: same as `汇总表.md`: one model per table; rows are methods; columns are grouped by dataset.",
        f"- Source output root: `{', '.join(args.output_root)}`.",
        "- Baseline rows are classifier-head-only versions: each method keeps the original baseline rule but applies it only to the classifier head, then evaluates it on the shared reference backbone.",
        "- Cell format: `ACC / macro-F1`." if args.metric_format == "acc_f1" else "- Cell format: `ACC`.",
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
            metric_format=args.metric_format,
        )
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    if args.metric_format == "acc_f1":
        diag_paths = []
        if args.diagnostics_csv:
            diag_paths.append(args.diagnostics_csv)
        if args.lamp_diagnostics_csv:
            diag_paths.append(args.lamp_diagnostics_csv)
        rows = load_diagnostics_rows(diag_paths)
        lookup, discovered = build_diagnostics_lookup(rows)
    else:
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
