#!/usr/bin/env python3
"""Run strict asynchronous new_lamp_merge experiments over a dataset/model grid."""

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_new_lamp_merge_async import run_experiment


DEFAULT_DATASETS = [
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
]
DEFAULT_MODELS = ["resnet", "convnext", "vit_t", "swin_tiny"]


def _output_root(root, dataset, model):
    return Path(root) / f"new_lamp_merge_async_{dataset.removesuffix('_224')}_{model}_k7"


def _prepare_uploads(prototype_path, upload_root):
    upload_root.mkdir(parents=True, exist_ok=True)
    payload = torch.load(prototype_path, map_location="cpu", weights_only=False)
    if payload.get("feature_space") != "reference_model":
        raise ValueError(f"Expected reference_model statistics: {prototype_path}")
    for client_id, client in enumerate(payload.get("clients", [])):
        torch.save(
            {
                "format": "new_lamp_merge_client_upload_v1",
                "client_id": client_id,
                "feature_space": payload["feature_space"],
                "prevalence_source": payload.get("prevalence_source", ""),
                "client": client,
            },
            upload_root / f"client_{client_id}.pt",
        )
    return len(payload.get("clients", []))


def _cfg(args, dataset, model):
    output_root = _output_root(args.output_root, dataset, model)
    prototype_path = (
        Path(args.prototype_root)
        / "small"
        / dataset
        / model
        / "clients_7"
        / "beta_0"
        / "seed_42"
        / "prototype_stats.pt"
    )
    client_count = _prepare_uploads(prototype_path, output_root / "client_uploads")
    if client_count != 7:
        raise ValueError(f"Expected seven clients for {dataset}/{model}, got {client_count}.")
    return {
        "task_type": "small",
        "dataset": dataset,
        "model": model,
        "num_clients": 7,
        "beta": 0.0,
        "seed": 42,
        "method": "new_lamp_merge",
        "model_hub_root": args.model_hub_root,
        "data_root": args.data_root,
        "output_root": str(output_root),
        "async_client_upload_root": str(output_root / "client_uploads"),
        "lamp_merge_proto_count_power": 0.55,
        "lamp_merge_reference_head_scale": 18.75,
        "lamp_merge_reference_prior_threshold": 2.5,
        "lamp_merge_reference_prior_max_tau": 4.25,
        "async_client_order": list(range(7)),
        "split": "test",
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": True,
    }


def main():
    parser = argparse.ArgumentParser("Run the full strict asynchronous new_lamp_merge grid.")
    parser.add_argument("--model-hub-root", default="/data2/liyapeng_grp/program/MedMNISTMerge/model_hub")
    parser.add_argument("--data-root", default="/data2/liyapeng_grp/program/MedMNISTMerge/Med_data")
    parser.add_argument("--prototype-root", default="/data2/liyapeng_grp/program/MedMNISTMerge/outputs/lamp_merge_client_local_proto_stats")
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        for model in args.models:
            print(f"running dataset={dataset} model={model}", flush=True)
            report = run_experiment(_cfg(args, dataset, model))
            final = report["rows"][-1]
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": model,
                    "k": final["k"],
                    "acc": final["acc"],
                    "macro_f1": final["macro_f1"],
                    "amp_enabled": report["amp_enabled"],
                    "report_path": str(
                        _output_root(args.output_root, dataset, model)
                        / "reports"
                        / "new_lamp_merge_async_k1_to_k7.csv"
                    ),
                }
            )

    summary = Path(args.output_root) / "reports" / "new_lamp_merge_async_grid_k7.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"grid summary saved to: {summary}")


if __name__ == "__main__":
    main()
