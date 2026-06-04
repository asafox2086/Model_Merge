#!/usr/bin/env python3
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_combined_results_table import parse_tables, try_parse_number
from generate_my_merge_master_table import SETTINGS, SMALL_DATASETS


def read_eval(root):
    path = Path(root) / "reports" / "eval_summary.csv"
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        model = row["model"] if row["task_type"] == "small" else (row.get("clip_model") or row.get("model"))
        key = (row["task_type"], row["dataset"], model, int(float(row["num_clients"])), float(row["beta"]))
        out[key] = float(row["test_acc"])
    return out


def formal_best(keys, table_path):
    _intro, tables = parse_tables(Path(table_path))
    lookup = defaultdict(dict)
    for (section, model_name, kind), table in tables.items():
        if kind != "Raw":
            continue
        task_type = "small" if section == "Small" else "vlm"
        for row in table["rows"]:
            method = row["method"]
            for dataset_idx, dataset in enumerate(SMALL_DATASETS):
                for setting_idx, (clients, beta, _label) in enumerate(SETTINGS):
                    col_idx = dataset_idx * len(SETTINGS) + setting_idx
                    if col_idx >= len(row["values"]):
                        continue
                    value = try_parse_number(row["values"][col_idx])
                    if value is None:
                        continue
                    lookup[(task_type, dataset, model_name, int(clients), float(beta))][method] = value
    return {key: max(lookup[key].values()) for key in keys if key in lookup and lookup[key]}


def wtl(actual, baseline, eps=1e-12):
    common = sorted(set(actual) & set(baseline))
    wins = ties = losses = 0
    rows = []
    for key in common:
        delta = actual[key] - baseline[key]
        if delta > eps:
            wins += 1
        elif delta < -eps:
            losses += 1
        else:
            ties += 1
        rows.append((key, baseline[key], actual[key], delta))
    mean_delta = sum(row[3] for row in rows) / len(rows) if rows else 0.0
    return {"n": len(rows), "wins": wins, "ties": ties, "losses": losses, "mean_delta": mean_delta, "rows": rows}


def selected_counts(root):
    path = Path(root) / "reports" / "my_merge_client_diagnostic_weights.csv"
    if not path.exists():
        return Counter()
    seen = set()
    selected = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["task_type"],
                row["dataset"],
                row["model"],
                row["clip_model"],
                row["num_clients"],
                row["beta"],
                row["seed"],
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(row.get("selected_candidate", ""))
    return Counter(selected)


def fmt(value):
    return f"{value:.6f}"


def summary_line(label, summary):
    return (
        f"{label}: n={summary['n']} W/T/L="
        f"{summary['wins']}/{summary['ties']}/{summary['losses']}, "
        f"mean_delta={fmt(summary['mean_delta'])}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--formal-table", default="result/all_results.md")
    parser.add_argument("--dest-md", required=True)
    parser.add_argument("--dest-csv", required=True)
    parser.add_argument("--append-progress", default="")
    parser.add_argument("--title", default="Ultrasound Run Summary")
    args = parser.parse_args()

    candidate = read_eval(args.candidate_root)
    baseline = read_eval(args.baseline_root)
    best = formal_best(set(candidate), args.formal_table)
    vs_baseline = wtl(candidate, baseline)
    vs_best = wtl(candidate, best)

    by_model = defaultdict(list)
    for key, base_value, actual_value, delta in vs_baseline["rows"]:
        by_model[key[2]].append(delta)

    lines = [
        f"# {args.title}",
        "",
        f"- candidate: `{args.candidate_root}`",
        f"- baseline: `{args.baseline_root}`",
        f"- {summary_line('vs ' + args.baseline_label, vs_baseline)}",
        f"- {summary_line('vs formal best', vs_best)}",
        "",
        "## By Model vs Baseline",
        "",
        "| model | rows | W/T/L | mean delta | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, deltas in sorted(by_model.items()):
        wins = sum(delta > 1e-12 for delta in deltas)
        losses = sum(delta < -1e-12 for delta in deltas)
        ties = len(deltas) - wins - losses
        lines.append(
            f"| {model} | {len(deltas)} | {wins}/{ties}/{losses} | "
            f"{fmt(sum(deltas) / len(deltas))} | {fmt(min(deltas))} | {fmt(max(deltas))} |"
        )

    counts = selected_counts(args.candidate_root)
    lines.extend(["", "## Selected Candidates", "", "| candidate | count |", "| --- | ---: |"])
    for candidate_name, count in counts.most_common():
        lines.append(f"| {candidate_name} | {count} |")

    lines.extend(
        [
            "",
            "## Largest Changes vs Baseline",
            "",
            "| task | dataset | model | clients | beta | baseline | candidate | delta |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, base_value, actual_value, delta in sorted(vs_baseline["rows"], key=lambda item: abs(item[3]), reverse=True)[:30]:
        task_type, dataset, model, clients, beta = key
        lines.append(
            f"| {task_type} | {dataset} | {model} | {clients} | {beta:g} | "
            f"{fmt(base_value)} | {fmt(actual_value)} | {fmt(delta)} |"
        )

    dest_md = Path(args.dest_md)
    dest_md.parent.mkdir(parents=True, exist_ok=True)
    dest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dest_csv = Path(args.dest_csv)
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with dest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_type", "dataset", "model", "num_clients", "beta", "baseline", "candidate", "delta"])
        for key, base_value, actual_value, delta in vs_baseline["rows"]:
            writer.writerow([*key, base_value, actual_value, delta])

    if args.append_progress:
        progress = Path(args.append_progress)
        with progress.open("a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f"## 2026-06-04 {args.title}\n\n")
            f.write(f"- candidate: `{args.candidate_root}`。\n")
            f.write(f"- {summary_line('vs ' + args.baseline_label, vs_baseline)}。\n")
            f.write(f"- {summary_line('vs formal best', vs_best)}。\n")
            f.write(f"- 汇总表：`{args.dest_md}`，明细：`{args.dest_csv}`。\n")


if __name__ == "__main__":
    main()
