#!/usr/bin/env python3
"""Evaluate deployable new_lamp_merge checkpoints after each client arrival."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluators import evaluate_small_checkpoint
from methods.new_lamp_merge import add_async_client, initialize_async_lamp_state, synthesize_async_lamp_state
from utils import find_hub_experiment_dir, load_config, load_hub_meta, save_json, set_seed


def _client_upload_path(cfg, client_id):
    upload_root = cfg.get("async_client_upload_root")
    if not upload_root:
        raise ValueError("async experiment requires async_client_upload_root with one file per client.")
    path = Path(upload_root) / f"client_{client_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"client upload not found: {path}")
    return path


def _load_client_upload(cfg, client_id):
    path = _client_upload_path(cfg, client_id)
    upload = torch.load(path, map_location="cpu", weights_only=False)
    if upload.get("format") != "new_lamp_merge_client_upload_v1":
        raise ValueError(f"Invalid async client upload format: {path}")
    if int(upload.get("client_id", -1)) != int(client_id):
        raise ValueError(f"Client upload id does not match its arrival slot: {path}")
    if upload.get("feature_space") != "reference_model":
        raise ValueError(f"Client upload is not a reference-model prototype: {path}")
    client = upload.get("client")
    if not isinstance(client, dict):
        raise ValueError(f"Client upload has no client statistics: {path}")
    return client, path


def _evaluation_amp(meta, cfg):
    if meta.get("dataset") == "bloodmnist_224":
        return False
    return bool(cfg.get("amp", False))


def _save_step(output_root, meta, merged_state, state, trace, result, step):
    step_dir = Path(output_root) / "async_steps" / f"k_{step:02d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    step_meta = dict(meta)
    step_meta["num_clients"] = int(step)
    torch.save({"state_dict": merged_state, "meta": step_meta}, step_dir / "merged.pt")
    torch.save(state, step_dir / "async_state.pt")
    save_json(step_dir / "meta.json", step_meta)
    save_json(step_dir / "merge_trace.json", trace)
    save_json(step_dir / "eval.json", result)
    return step_dir


def run_experiment(cfg):
    set_seed(int(cfg.get("seed", 42)))
    exp_dir = find_hub_experiment_dir(cfg["model_hub_root"], cfg)
    meta = load_hub_meta(exp_dir)
    if meta.get("task_type") != "small":
        raise ValueError("The asynchronous experiment script currently supports task_type=small only.")

    order = [int(client_id) for client_id in cfg.get("async_client_order", [])]
    if not order or len(order) != len(set(order)):
        raise ValueError("async_client_order must be a non-empty list of distinct client ids.")

    state = initialize_async_lamp_state(meta, cfg)
    device = torch.device(cfg.get("device", "cpu"))
    amp_enabled = _evaluation_amp(meta, cfg)
    rows = []
    for step, client_id in enumerate(order, start=1):
        client, upload_path = _load_client_upload(cfg, client_id)
        add_async_client(state, client, client_id, meta, cfg)
        merged_state, trace = synthesize_async_lamp_state(state, meta, cfg)
        result = evaluate_small_checkpoint(
            meta=meta,
            checkpoint={"state_dict": merged_state},
            data_root=cfg["data_root"],
            split=cfg.get("split", "test"),
            device=device,
            batch_size=int(cfg.get("batch_size", 128)),
            num_workers=int(cfg.get("num_workers", 4)),
            amp=amp_enabled,
        )
        result = {
            "k": step,
            "new_client_id": client_id,
            "received_client_ids": trace["received_client_ids"],
            "acc": float(result["acc"]),
            "macro_f1": float(result["macro_f1"]),
            "loss": float(result["loss"]),
            "num_samples": int(result["num_samples"]),
            "amp_enabled": amp_enabled,
            "client_upload_path": str(upload_path),
        }
        step_dir = _save_step(cfg["output_root"], meta, merged_state, state, trace, result, step)
        result["checkpoint_dir"] = str(step_dir)
        rows.append(result)
        print(f"k={step} client={client_id} acc={result['acc']:.4f} macro_f1={result['macro_f1']:.4f}", flush=True)

    report_dir = Path(cfg["output_root"]) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "new_lamp_merge_async_k1_to_k7.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "method": "new_lamp_merge",
        "dataset": meta["dataset"],
        "backbone": meta["model"],
        "split": cfg.get("split", "test"),
        "amp_enabled": amp_enabled,
        "delivery_mode": "strict_per_client_upload",
        "client_upload_root": str(cfg["async_client_upload_root"]),
        "rows": rows,
    }
    save_json(report_dir / "new_lamp_merge_async_k1_to_k7.json", report)
    return report


def main():
    parser = argparse.ArgumentParser("Run a k=1..K asynchronous new_lamp_merge experiment.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = run_experiment(load_config(args.config))
    print(json.dumps(report["rows"], indent=2))


if __name__ == "__main__":
    main()
