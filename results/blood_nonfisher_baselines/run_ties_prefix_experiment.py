#!/usr/bin/env python3
"""Evaluate TIES after each prefix of the BloodMNIST client-arrival order."""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import torch


RESULT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, required=True)
    parser.add_argument("--model-hub-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    framework_root = args.framework_root.resolve()
    if str(framework_root) not in sys.path:
        sys.path.insert(0, str(framework_root))

    from evaluators import evaluate_small_checkpoint
    from merge import METHOD_DEFAULTS, merge_with_method
    from utils import ensure_state_dicts_compatible, extract_state_dict, load_checkpoint, load_hub_meta, set_seed

    exp_dir = (
        args.model_hub_root
        / "small"
        / "bloodmnist_224"
        / "resnet"
        / "clients_7"
        / "beta_0"
        / "seed_42"
    )
    meta = load_hub_meta(exp_dir)
    checkpoints = [load_checkpoint(exp_dir / client["checkpoint"], device="cpu") for client in meta["clients"]]
    state_dicts = [extract_state_dict(checkpoint) for checkpoint in checkpoints]
    ensure_state_dicts_compatible(state_dicts)
    cfg = {**METHOD_DEFAULTS, "method": "ties", "merge_weight_mode": "equal", "seed": 42}
    device = torch.device(args.device)
    rows = []

    for k in range(1, 8):
        step_meta = copy.deepcopy(meta)
        step_meta["num_clients"] = k
        step_meta["clients"] = step_meta["clients"][:k]
        set_seed(42)
        merged_state, _ = merge_with_method(
            "ties",
            state_dicts[:k],
            [1] * k,
            step_meta,
            checkpoints[:k],
            cfg,
        )
        result = evaluate_small_checkpoint(
            meta=step_meta,
            checkpoint={"state_dict": merged_state},
            data_root=str(args.data_root),
            split="test",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=False,
        )
        rows.append(
            {
                "k": k,
                "received_client_ids": list(range(k)),
                "acc": float(result["acc"]),
                "loss": float(result["loss"]),
                "num_samples": int(result["num_samples"]),
                "amp_enabled": False,
            }
        )
        print(f"k={k} acc={result['acc']:.6f}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.result_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.result_root / "ties_k1_to_k7.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "method": "ties",
        "dataset": "bloodmnist_224",
        "backbone": "resnet",
        "num_clients": 7,
        "beta": 0.0,
        "seed": 42,
        "delivery_order": list(range(7)),
        "amp_enabled": False,
        "rows": rows,
    }
    (args.result_root / "ties_k1_to_k7.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
