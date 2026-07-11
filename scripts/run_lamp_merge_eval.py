#!/usr/bin/env python3
"""Run LAMP-Merge and evaluation over entries in model_hub/manifest.csv."""

import argparse
import csv
import gc
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import run_evaluate
from merge import METHOD_DEFAULTS, run_merge
from utils import make_eval_output_dir, save_csv
from utils.lamp_merge_stats import default_prototype_root


STATUS_FIELDS = [
    "status",
    "task_type",
    "dataset",
    "model",
    "clip_model",
    "num_clients",
    "beta",
    "seed",
    "eval_json",
    "merged_deleted",
    "seconds",
    "test_acc",
    "test_loss",
    "updated_at",
    "error",
]


def parse_args():
    p = argparse.ArgumentParser("Run the public LAMP-Merge pipeline over a model_hub manifest.")
    p.add_argument("--model-hub-root", type=str, default=str(ROOT / "model_hub"))
    p.add_argument("--data-root", type=str, default=str(ROOT / "Med_data"))
    p.add_argument("--output-root", type=str, default=str(ROOT / "outputs" / "lamp_merge_eval"))
    p.add_argument("--lamp-merge-prototype-root", type=str, default="")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--task-type", choices=["all", "small", "vlm"], default="small")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--small-models", nargs="*", default=None)
    p.add_argument("--clip-models", nargs="*", default=None)
    p.add_argument("--num-clients", nargs="*", type=int, default=None)
    p.add_argument("--betas", nargs="*", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--delete-merged", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lamp-merge-proto-count-power", type=float, default=METHOD_DEFAULTS["lamp_merge_proto_count_power"])
    p.add_argument("--lamp-merge-reference-head-scale", type=float, default=METHOD_DEFAULTS["lamp_merge_reference_head_scale"])
    p.add_argument("--lamp-merge-reference-prior-threshold", type=float, default=METHOD_DEFAULTS["lamp_merge_reference_prior_threshold"])
    p.add_argument("--lamp-merge-reference-prior-max-tau", type=float, default=METHOD_DEFAULTS["lamp_merge_reference_prior_max_tau"])
    return p.parse_args()


def load_manifest(model_hub_root):
    path = Path(model_hub_root) / "manifest.csv"
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def selected_rows(rows, args):
    out = rows
    if args.task_type != "all":
        out = [row for row in out if row["task_type"] == args.task_type]
    if args.datasets:
        allowed = set(args.datasets)
        out = [row for row in out if row["dataset"] in allowed]
    if args.small_models:
        allowed = set(args.small_models)
        out = [row for row in out if row["task_type"] != "small" or row["model"] in allowed]
    if args.clip_models:
        allowed = {item.split("/")[-1] for item in args.clip_models}
        out = [row for row in out if row["task_type"] != "vlm" or row["clip_model"].split("/")[-1] in allowed]
    if args.num_clients:
        allowed = {int(item) for item in args.num_clients}
        out = [row for row in out if int(row["num_clients"]) in allowed]
    if args.betas:
        allowed = {format(float(item), "g") for item in args.betas}
        out = [row for row in out if format(float(row["beta"]), "g") in allowed]
    if args.seed is not None:
        out = [row for row in out if int(row["seed"]) == int(args.seed)]
    return out


def row_cfg(row, args):
    cfg = {
        "task_type": row["task_type"],
        "dataset": row["dataset"],
        "model": row.get("model", ""),
        "clip_model": row.get("clip_model", ""),
        "num_clients": int(row["num_clients"]),
        "beta": float(row["beta"]),
        "seed": int(row["seed"]),
        "method": "lamp_merge",
        "merge_weight_mode": "sample",
        "model_hub_root": args.model_hub_root,
        "data_root": args.data_root,
        "output_root": args.output_root,
        "lamp_merge_prototype_root": args.lamp_merge_prototype_root
        or str(default_prototype_root(ROOT)),
        "lamp_merge_proto_count_power": float(args.lamp_merge_proto_count_power),
        "lamp_merge_reference_head_scale": float(args.lamp_merge_reference_head_scale),
        "lamp_merge_reference_prior_threshold": float(args.lamp_merge_reference_prior_threshold),
        "lamp_merge_reference_prior_max_tau": float(args.lamp_merge_reference_prior_max_tau),
        "split": "test",
        "device": args.device,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
    }
    return cfg


def status_row(cfg, **extra):
    row = {
        "status": extra.get("status", ""),
        "task_type": cfg["task_type"],
        "dataset": cfg["dataset"],
        "model": cfg.get("model", ""),
        "clip_model": cfg.get("clip_model", ""),
        "num_clients": cfg["num_clients"],
        "beta": cfg["beta"],
        "seed": cfg["seed"],
        "eval_json": extra.get("eval_json", ""),
        "merged_deleted": extra.get("merged_deleted", ""),
        "seconds": extra.get("seconds", ""),
        "test_acc": extra.get("test_acc", ""),
        "test_loss": extra.get("test_loss", ""),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "error": extra.get("error", ""),
    }
    return row


def main():
    args = parse_args()
    rows = selected_rows(load_manifest(args.model_hub_root), args)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    status_path = Path(args.output_root) / "lamp_merge_status.csv"
    statuses = []

    print(f"LAMP-Merge run start | tasks={len(rows)} | output_root={args.output_root}")
    for idx, row in enumerate(rows, start=1):
        cfg = row_cfg(row, args)
        eval_dir = make_eval_output_dir(cfg["output_root"], cfg)
        eval_json = eval_dir / "eval.json"
        if args.resume and eval_json.exists():
            statuses.append(status_row(cfg, status="skip", eval_json=str(eval_json), merged_deleted=""))
            print(f"skip ({idx}/{len(rows)}) {cfg['dataset']} {cfg.get('model') or cfg.get('clip_model')}")
            continue

        start = time.time()
        try:
            merge_result = run_merge(cfg)
            eval_payload, eval_path = run_evaluate(cfg, merged_dir=merge_result["merged_dir"])
            deleted = False
            if args.delete_merged:
                merged_path = Path(merge_result["merged_checkpoint"])
                if merged_path.exists():
                    merged_path.unlink()
                    deleted = True
            statuses.append(
                status_row(
                    cfg,
                    status="ok",
                    eval_json=str(eval_path),
                    merged_deleted=str(deleted),
                    seconds=f"{time.time() - start:.2f}",
                    test_acc=f"{float(eval_payload['test_acc']):.6f}",
                    test_loss=f"{float(eval_payload['test_loss']):.6f}",
                )
            )
            print(
                f"ok ({idx}/{len(rows)}) {cfg['dataset']} "
                f"{cfg.get('model') or cfg.get('clip_model')} acc={float(eval_payload['test_acc']):.4f}"
            )
        except Exception as exc:  # noqa: BLE001 - batch runner records failures and continues.
            statuses.append(
                status_row(
                    cfg,
                    status="error",
                    seconds=f"{time.time() - start:.2f}",
                    error=repr(exc),
                )
            )
            print(f"error ({idx}/{len(rows)}) {cfg['dataset']}: {exc}", file=sys.stderr)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            save_csv(status_path, statuses)

    print(f"LAMP-Merge run done | status={status_path}")


if __name__ == "__main__":
    main()
