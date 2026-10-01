#!/usr/bin/env python3
"""Plan and run isolated experiments matching the final manuscript."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "experiments/paper.json"


def read_spec():
    return json.loads(SPEC_PATH.read_text())


def read_grid(suite):
    with (ROOT / suite["csv"]).open(encoding="utf-8", newline="") as handle:
        next(handle)
        return [
            {flag: float(row[column]) for column, flag in suite["parameters"].items()}
            for row in csv.DictReader(handle)
        ]


def parse_args():
    spec = read_spec()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=spec["suites"], nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run commands; otherwise only print the plan")
    parser.add_argument("--check", action="store_true", help="Validate input files without running experiments")
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-hub-root", type=Path, default=ROOT / "model_hub")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Med_data")
    parser.add_argument("--prototype-root", type=Path, default=ROOT / "outputs/lamp_merge_client_local_proto_stats")
    parser.add_argument("--datasets", nargs="+", choices=spec["datasets"], default=spec["datasets"])
    parser.add_argument("--models", nargs="+", choices=spec["models"], default=spec["models"])
    parser.add_argument("--clients", nargs="+", type=int, default=spec["clients"])
    parser.add_argument("--betas", nargs="+", type=float, default=spec["betas"])
    parser.add_argument("--seed", type=int, default=spec["seed"])
    parser.add_argument("--methods", nargs="+", choices=spec["baselines"] + ["lamp_merge"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--keep-merged", action="store_true")
    parser.add_argument("--point", type=int, help="Run one zero-based sweep point")
    args = parser.parse_args()
    if not args.list and not args.suite:
        parser.error("provide a suite or --list")
    if args.batch_size <= 0 or args.num_workers < 0 or any(count <= 0 for count in args.clients):
        parser.error("batch size and client counts must be positive; workers must be nonnegative")
    if Path(args.tag).name != args.tag or args.tag in {"", ".", ".."}:
        parser.error("tag must be a single directory name")
    if args.suite:
        kind = spec["suites"][args.suite]["kind"]
        if args.point is not None and kind != "sweep":
            parser.error("--point is only supported for hyperparameter sweeps")
        if args.methods and kind == "sweep":
            parser.error("hyperparameter sweeps always use lamp_merge")
    return args


def option(flag, values):
    return ["--" + flag, *map(str, values)]


def build_jobs(args, spec):
    suite = spec["suites"][args.suite]
    output = (args.output_root or ROOT / "outputs/paper" / args.tag / args.suite).resolve()
    common = [
        *option("model-hub-root", [args.model_hub_root.resolve()]),
        *option("data-root", [args.data_root.resolve()]),
        "--task-type", "small", *option("datasets", args.datasets),
        *option("small-models", args.models), *option("num-clients", args.clients),
        *option("betas", args.betas), *option("seeds", [args.seed]),
        "--device", args.device, "--num-workers", str(args.num_workers),
        "--merge-weight-mode", "equal", "--resume",
        "--no-delete-merged" if args.keep_merged else "--delete-merged",
        "--lamp-merge-prototype-root", str(args.prototype_root.resolve()),
    ]
    methods = args.methods or (spec["baselines"] if suite.get("baselines") else []) + ["lamp_merge"]
    if suite["kind"] == "diagnostics":
        report = output / "reports/metrics.csv"
        command = [sys.executable, str(ROOT / "exp_analyze/collect_prediction_diagnostics.py"), *common,
                   "--output-root", str(output / "checkpoints"),
                   "--metrics-csv", str(report), "--summary-dir", str(output / "reports"),
                   "--figure-dir", str(output / "figures"),
                   "--batch-size", str(args.batch_size), *option("methods", methods),
                   *option("lamp-merge-ablation-modes", suite["modes"]),
                   "--include-clients" if suite.get("include_clients") else "--no-include-clients"]
        for flag, value in spec["hyperparameters"].items():
            command.extend(option(flag, [value]))
        return output, [{"name": args.suite, "command": command, "report": str(report)}]
    grid = read_grid(suite)
    if args.point is not None and not 0 <= args.point < len(grid):
        raise ValueError(f"point must be in [0, {len(grid) - 1}]")
    jobs = []
    for index, values in enumerate(grid):
        if args.point is not None and args.point != index:
            continue
        point_root = output / f"point_{index:03d}"
        command = [sys.executable, str(ROOT / "scripts/run_all_avg_eval.py"), *common,
                   "--output-root", str(point_root), "--method", "lamp_merge",
                   "--small-batch-size", str(args.batch_size)]
        for flag, value in {**spec["hyperparameters"], **values}.items():
            command.extend(option(flag, [value]))
        jobs.append({"name": f"point_{index:03d}", "parameters": values, "command": command,
                     "report": str(point_root / "reports/batch_status.csv")})
    return output, jobs


def preflight(args, spec):
    manifest_path = args.model_hub_root / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {}
    for row in rows:
        if row["task_type"] == "small" and int(row["seed"]) == args.seed:
            key = (row["dataset"], row["model"], int(row["num_clients"]), float(row["beta"]))
            if key in indexed:
                raise ValueError(f"Duplicate manifest case: {key}")
            indexed[key] = row
    errors = []
    count = 0
    needs_prototypes = not args.methods or "lamp_merge" in args.methods
    for dataset, model, clients, beta in itertools.product(args.datasets, args.models, args.clients, args.betas):
        key = (dataset, model, clients, beta)
        row = indexed.get(key)
        if row is None:
            errors.append(f"Missing manifest case: {key}")
            continue
        meta_path = args.model_hub_root / row["meta_path"]
        if not meta_path.is_file():
            errors.append(f"Missing metadata: {meta_path}")
            continue
        meta = json.loads(meta_path.read_text())
        paths = [args.data_root / f"{dataset}.npz"]
        paths.extend(meta_path.parent / item["checkpoint"] for item in meta["clients"])
        reference = f"{model}__cls{meta['num_classes']}__in{meta.get('in_channels', 3)}__pretrained{int(bool(meta.get('pretrained', False)))}__seed{meta.get('seed', 0)}.pt"
        paths.append(ROOT / "reference_cache/small" / reference)
        if needs_prototypes:
            paths.append(args.prototype_root / Path(row["meta_path"]).parent / "prototype_stats.pt")
        errors.extend(f"Missing input: {path}" for path in paths if not path.is_file())
        count += 1
    if errors:
        raise ValueError("\n".join(sorted(set(errors))))
    print(f"Input paths OK: {count} cases; prototype payloads are validated by the runtime.")
    return count


def validate_diagnostics(path, args, spec):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    latest = {}
    fields = ("task_type", "dataset", "model", "num_clients", "beta", "seed", "source", "method", "client_id", "split")
    for row in rows:
        key = tuple(format(float(row[field]), "g") if field == "beta" else row.get(field, "") for field in fields)
        latest[key] = row
    suite = spec["suites"][args.suite]
    methods = args.methods or (spec["baselines"] if suite.get("baselines") else []) + ["lamp_merge"]
    labels = []
    for method in methods:
        if method != "lamp_merge":
            labels.append(method)
        elif suite["modes"] == ["full"]:
            labels.append(method)
        else:
            labels.extend(f"{method}:{mode}" for mode in suite["modes"])
    expected = set()
    for dataset, model, clients, beta in itertools.product(args.datasets, args.models, args.clients, args.betas):
        prefix = ("small", dataset, model, str(clients), format(float(beta), "g"), str(args.seed))
        expected.update((*prefix, "merge", label, "", "test") for label in labels)
        if suite.get("include_clients"):
            expected.update((*prefix, "client", f"client_{index}", str(index), "test") for index in range(clients))
    missing = expected - latest.keys()
    failed = [latest[key] for key in expected & latest.keys() if latest[key]["status"] != "OK"]
    if failed or missing:
        raise RuntimeError(f"Incomplete diagnostics: {len(missing)} missing cases, {len(failed)} failures; see {path}")


def main():
    args = parse_args()
    spec = read_spec()
    if args.list:
        for name, suite in spec["suites"].items():
            print(f"{name:16} {suite['description']}")
        return
    output, jobs = build_jobs(args, spec)
    print(f"Suite: {args.suite}; output: {output}; jobs: {len(jobs)}")
    for job in jobs:
        print(shlex.join(job["command"]))
    if args.check or args.execute:
        preflight(args, spec)
    if not args.execute:
        return
    plan = {"spec_sha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(), "suite": args.suite, "jobs": jobs}
    plan_path = output / (f"plan_point_{args.point:03d}.json" if args.point is not None else "plan.json")
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError(f"Existing plan differs; choose a new --tag or --output-root: {plan_path}")
    output.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    env = os.environ.copy()
    for key in ("HF_LOCAL_FILES_ONLY", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(key, "1")
    for job in jobs:
        log_path = output / f"{job['name']}.log"
        print(f"Running {job['name']}; log: {log_path}", flush=True)
        with log_path.open("a") as log:
            subprocess.run(job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if spec["suites"][args.suite]["kind"] == "diagnostics":
            validate_diagnostics(Path(job["report"]), args, spec)
    print(f"Completed {args.suite}: {output}")


if __name__ == "__main__":
    main()
