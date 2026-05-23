#!/usr/bin/env python3
import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path


ABLATIONS = [
    "full",
    "no_client_information",
    "no_fusion_selection",
    "avg_only",
]


def parse_args():
    p = argparse.ArgumentParser("Append my_merge ablation progress snapshots to a log")
    p.add_argument("--root", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--interval", type=int, default=600)
    return p.parse_args()


def load_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def progress_lines(root):
    lines = [datetime.now().strftime("%F %T %Z")]
    root = Path(root)
    for name in ABLATIONS:
        rows = load_rows(root / name / "reports" / "batch_status.csv")
        counts = {}
        for row in rows:
            status = row.get("status", "")
            counts[status] = counts.get(status, 0) + 1
        last = rows[-1] if rows else {}
        lines.append(
            (
                f"{name}: {len(rows)}/225 {counts} "
                f"last={last.get('updated_at', '')} "
                f"{last.get('task_type', '')} {last.get('dataset', '')} "
                f"{last.get('model') or last.get('clip_model', '')} "
                f"c{last.get('num_clients', '')} b{last.get('beta', '')} "
                f"sec={last.get('seconds', '')} err={last.get('error', '')}"
            )
        )
    return lines


def active_process_lines():
    try:
        proc = subprocess.run(
            [
                "pgrep",
                "-f",
                "^/data/liyapeng_grp/program/MedMNISTMerge/.gpuenv/bin/python scripts/run_all_avg_eval.py",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        pids = [pid for pid in proc.stdout.splitlines() if pid.strip()]
        if not pids:
            return ["no active run_all_avg_eval python"]
        ps = subprocess.run(
            ["ps", "-o", "pid,stat,etime,%cpu,%mem", "-p", ",".join(pids)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return ps.stdout.rstrip().splitlines()
    except Exception as exc:
        return [f"process check failed: {exc}"]


def gpu_lines():
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return proc.stdout.rstrip().splitlines()
    except Exception as exc:
        return [f"gpu check failed: {exc}"]


def write_snapshot(root, log_path):
    lines = []
    lines.extend(progress_lines(root))
    lines.extend(active_process_lines())
    lines.extend(gpu_lines())
    lines.append("---")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    args = parse_args()
    log_path = Path(args.log)
    while True:
        write_snapshot(args.root, log_path)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
