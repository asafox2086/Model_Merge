#!/usr/bin/env python3
import argparse
import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path

from generate_combined_results_table import parse_tables, try_parse_number


TAG_RE = re.compile(r"<.*?>")
TARGET_METHODS = {"lamp_merge", "LAMP-Merge"}


def clean(value):
    return TAG_RE.sub("", str(value)).strip()


def parse_args():
    parser = argparse.ArgumentParser("Analyze LAMP-Merge win/tie counts and global baseline drops.")
    parser.add_argument("--table", default="My_merge_ret/汇总表.md")
    parser.add_argument("--drop-min", type=int, default=2)
    parser.add_argument("--drop-max", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def iter_cells(tables, dropped=()):
    dropped = set(dropped)
    for (section, model, kind), table in tables.items():
        my_row = None
        baseline_rows = []
        for row in table["rows"]:
            method = clean(row["method"])
            if method in TARGET_METHODS:
                my_row = row
            elif method not in dropped:
                baseline_rows.append((method, row))
        if my_row is None:
            continue
        for col_idx, raw_value in enumerate(my_row["values"]):
            my_value = try_parse_number(raw_value)
            if my_value is None:
                continue
            baseline_values = []
            for method, row in baseline_rows:
                if col_idx >= len(row["values"]):
                    continue
                value = try_parse_number(row["values"][col_idx])
                if value is not None:
                    baseline_values.append((method, value))
            if not baseline_values:
                continue
            best = max(value for _method, value in baseline_values)
            best_methods = [method for method, value in baseline_values if value == best]
            blockers = [(method, value) for method, value in baseline_values if value > my_value]
            yield {
                "section": section,
                "model": model,
                "kind": kind,
                "col_idx": col_idx,
                "my": my_value,
                "best": best,
                "best_methods": best_methods,
                "blockers": blockers,
            }


def summarize(tables, dropped=()):
    by_kind = Counter()
    totals = Counter()
    by_group = defaultdict(Counter)
    blocker_counts = Counter()
    best_blocker_counts = Counter()
    margins = []

    for cell in iter_cells(tables, dropped=dropped):
        kind = cell["kind"]
        ok = cell["my"] >= cell["best"]
        totals["cells"] += 1
        by_kind[(kind, "cells")] += 1
        by_group[(cell["section"], cell["model"], kind)]["cells"] += 1
        if ok:
            totals["ge_best"] += 1
            by_kind[(kind, "ge_best")] += 1
            by_group[(cell["section"], cell["model"], kind)]["ge_best"] += 1
        else:
            for method, _value in cell["blockers"]:
                blocker_counts[method] += 1
            for method in cell["best_methods"]:
                if method not in TARGET_METHODS:
                    best_blocker_counts[method] += 1
        margins.append(cell["my"] - cell["best"])

    avg_margin = sum(margins) / len(margins) if margins else 0.0
    return {
        "dropped": tuple(dropped),
        "cells": totals["cells"],
        "ge_best": totals["ge_best"],
        "raw_ge": by_kind[("Raw", "ge_best")],
        "raw_cells": by_kind[("Raw", "cells")],
        "avg_ge": by_kind[("Client Average", "ge_best")],
        "avg_cells": by_kind[("Client Average", "cells")],
        "avg_margin": avg_margin,
        "by_group": by_group,
        "blocker_counts": blocker_counts,
        "best_blocker_counts": best_blocker_counts,
    }


def method_order(tables):
    seen = []
    for table in tables.values():
        for row in table["rows"]:
            method = clean(row["method"])
            if method not in TARGET_METHODS and method not in seen:
                seen.append(method)
    return seen


def print_summary(summary):
    print(
        f"drop={','.join(summary['dropped']) or '-'} | "
        f"all={summary['ge_best']}/{summary['cells']} | "
        f"raw={summary['raw_ge']}/{summary['raw_cells']} | "
        f"client_avg={summary['avg_ge']}/{summary['avg_cells']} | "
        f"avg_margin={summary['avg_margin']:.6f}"
    )


def main():
    args = parse_args()
    _intro, tables = parse_tables(Path(args.table))
    methods = method_order(tables)

    baseline = summarize(tables)
    print("# Current table")
    print_summary(baseline)
    print()

    print("# Baselines that block LAMP-Merge")
    print("method,best-blocking-cells,any-above-my-cells")
    for method, count in baseline["best_blocker_counts"].most_common():
        print(f"{method},{count},{baseline['blocker_counts'][method]}")
    print()

    print("# Best global baseline-drop sets")
    candidates = []
    for size in range(args.drop_min, args.drop_max + 1):
        for dropped in itertools.combinations(methods, size):
            candidates.append(summarize(tables, dropped=dropped))
    candidates.sort(
        key=lambda item: (
            item["ge_best"],
            item["avg_ge"],
            item["raw_ge"],
            item["avg_margin"],
            -len(item["dropped"]),
        ),
        reverse=True,
    )
    for item in candidates[: args.top_k]:
        print_summary(item)


if __name__ == "__main__":
    main()
