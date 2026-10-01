#!/usr/bin/env python3
"""Validate prediction metrics and aggregate beta/K means for the paper."""

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("accuracy", "macro_f1")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, members in sorted(groups.items()):
        entry = dict(zip(fields, key))
        entry["averaged_cells"] = len(members)
        entry.update({metric: mean(float(row[metric]) for row in members) for metric in METRICS})
        result.append(entry)
    return result


def main():
    spec = json.loads((ROOT / "experiments/paper.json").read_text())
    suites = {name: value for name, value in spec["suites"].items() if value["kind"] == "diagnostics"}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--suite", choices=suites, default="main")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", nargs="+", default=spec["datasets"])
    parser.add_argument("--models", nargs="+", default=spec["models"])
    parser.add_argument("--clients", type=int, nargs="+", default=spec["clients"])
    parser.add_argument("--betas", type=float, nargs="+", default=spec["betas"])
    parser.add_argument("--seed", type=int, default=spec["seed"])
    parser.add_argument("--methods", nargs="+", help="Expected display method names for a deliberately reduced run")
    args = parser.parse_args()
    suite = suites[args.suite]
    modes = suite["modes"]
    lamp_labels = ["lamp_merge"] if modes == ["full"] else [f"lamp_merge:{mode}" for mode in modes]
    methods = args.methods or (spec["baselines"] if suite.get("baselines") else []) + lamp_labels
    latest = {}
    for path in args.metrics:
        local = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("source") != "merge" or row.get("split") != "test" or int(row["seed"]) != args.seed:
                    continue
                key = (row["dataset"], row["model"], int(row["num_clients"]), float(row["beta"]), row["method"])
                local[key] = row
        for key, row in local.items():
            if key in latest and any(row.get(field) != latest[key].get(field) for field in (*METRICS, "status")):
                raise SystemExit(f"Conflicting duplicate case across CSVs: {key}")
            latest[key] = row
    expected = set(itertools.product(args.datasets, args.models, args.clients, args.betas, methods))
    missing = expected - latest.keys()
    if missing:
        raise SystemExit(f"Incomplete run: missing {len(missing)} cases; example: {sorted(missing)[0]}")
    rows = []
    for key in sorted(expected):
        row = latest[key]
        if row["status"] != "OK":
            raise SystemExit(f"Failed case: {key}: {row.get('error', '')}")
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise SystemExit(f"Invalid {metric} for {key}: {value}")
        rows.append(row)
    client_average = aggregate(rows, ("dataset", "model", "method", "num_clients"))
    dataset_average = aggregate(client_average, ("dataset", "model", "method"))
    overall = aggregate(dataset_average, ("method",))
    output = args.output_dir or args.metrics[0].parent
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [("client_average", client_average), ("dataset_average", dataset_average), ("overall", overall)]:
        write_csv(output / f"paper_{name}.csv", values)
    provenance = {"suite": args.suite, "sources": [str(path.resolve()) for path in args.metrics],
                  "datasets": args.datasets, "models": args.models, "clients": args.clients,
                  "betas": args.betas, "seed": args.seed, "methods": methods,
                  "raw_cells": len(rows), "unit": "fraction (0-1)"}
    (output / "paper_summary.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(f"Validated {len(rows)} raw cells; summaries: {output}")


if __name__ == "__main__":
    main()
