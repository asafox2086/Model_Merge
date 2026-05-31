#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


KEY_FIELDS = ("task_type", "dataset", "model", "clip_model", "num_clients", "beta", "seed", "merge_weight_mode")


def read_eval_rows(root):
    root = Path(root)
    rows = []
    for path in sorted(root.glob("**/reports/eval_summary.csv")):
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        direct = root / "reports" / "eval_summary.csv"
        if direct.exists():
            with direct.open("r", encoding="utf-8", newline="") as f:
                rows.extend(csv.DictReader(f))
    return rows


def key_of(row):
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)


def index_rows(rows):
    indexed = {}
    for row in rows:
        indexed[key_of(row)] = row
    return indexed


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group_key(row):
    return (row.get("task_type", ""), row.get("dataset", ""), row.get("model") or row.get("clip_model", ""))


def fmt(value):
    return "-" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser("Compare full my_merge and my_merge_extra results")
    parser.add_argument("--my-merge-root", required=True)
    parser.add_argument("--extra-root", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--csv-dest", default="")
    args = parser.parse_args()

    my_rows = read_eval_rows(args.my_merge_root)
    extra_rows = read_eval_rows(args.extra_root)
    my_idx = index_rows(my_rows)
    extra_idx = index_rows(extra_rows)
    common = sorted(set(my_idx) & set(extra_idx))

    detail_rows = []
    exact_same = 0
    for key in common:
        my_row = my_idx[key]
        extra_row = extra_idx[key]
        my_acc = as_float(my_row.get("test_acc"))
        extra_acc = as_float(extra_row.get("test_acc"))
        diff = None if my_acc is None or extra_acc is None else extra_acc - my_acc
        same = diff is not None and abs(diff) <= 1e-12
        exact_same += int(same)
        detail_rows.append(
            {
                **{field: my_row.get(field, "") for field in KEY_FIELDS},
                "my_merge_acc": my_acc,
                "my_merge_extra_acc": extra_acc,
                "extra_minus_my_merge": diff,
                "exact_same": same,
            }
        )

    groups = {}
    for row in detail_rows:
        groups.setdefault((row["task_type"], row["dataset"], row["model"] or row["clip_model"]), []).append(row)

    lines = [
        "# my_merge vs my_merge_extra Full Comparison",
        "",
        f"- my_merge root: `{args.my_merge_root}`",
        f"- my_merge_extra root: `{args.extra_root}`",
        f"- my_merge rows: {len(my_idx)}",
        f"- my_merge_extra rows: {len(extra_idx)}",
        f"- matched rows: {len(common)}",
        f"- exactly same accuracy rows: {exact_same}/{len(common)}",
        "",
        "## Group Summary",
        "",
        "| task | dataset | model | rows | mean my_merge | mean extra | mean extra-my | same rows |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (task, dataset, model), rows in sorted(groups.items()):
        my_vals = [row["my_merge_acc"] for row in rows if row["my_merge_acc"] is not None]
        extra_vals = [row["my_merge_extra_acc"] for row in rows if row["my_merge_extra_acc"] is not None]
        diff_vals = [row["extra_minus_my_merge"] for row in rows if row["extra_minus_my_merge"] is not None]
        same_count = sum(1 for row in rows if row["exact_same"])
        lines.append(
            f"| {task} | {dataset} | {model} | {len(rows)} | "
            f"{fmt(sum(my_vals) / len(my_vals) if my_vals else None)} | "
            f"{fmt(sum(extra_vals) / len(extra_vals) if extra_vals else None)} | "
            f"{fmt(sum(diff_vals) / len(diff_vals) if diff_vals else None)} | {same_count} |"
        )

    lines.extend(
        [
            "",
            "## Largest Absolute Differences",
            "",
            "| task | dataset | model | clients | beta | my_merge | extra | extra-my | same |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    by_abs = sorted(
        [row for row in detail_rows if row["extra_minus_my_merge"] is not None],
        key=lambda row: abs(row["extra_minus_my_merge"]),
        reverse=True,
    )
    for row in by_abs[:30]:
        lines.append(
            f"| {row['task_type']} | {row['dataset']} | {row['model'] or row['clip_model']} | "
            f"{row['num_clients']} | {row['beta']} | {fmt(row['my_merge_acc'])} | "
            f"{fmt(row['my_merge_extra_acc'])} | {fmt(row['extra_minus_my_merge'])} | {row['exact_same']} |"
        )

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.csv_dest:
        csv_dest = Path(args.csv_dest)
        csv_dest.parent.mkdir(parents=True, exist_ok=True)
        with csv_dest.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(KEY_FIELDS) + ["my_merge_acc", "my_merge_extra_acc", "extra_minus_my_merge", "exact_same"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)


if __name__ == "__main__":
    main()
