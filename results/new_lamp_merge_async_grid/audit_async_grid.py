#!/usr/bin/env python3
"""Audit strict asynchronous state and final one-shot equivalence."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.lamp_merge import merge_lamp_merge
from methods.new_lamp_merge import merge_new_lamp_merge
from utils import find_hub_experiment_dir, load_hub_meta, set_seed

from plot_async_grid import DATASETS, MODELS, report_path


STATE_KEYS = {
    "version",
    "base_state",
    "weight_key",
    "bias_key",
    "prototype_numerator",
    "evidence_sum",
    "prototype_class_counts",
    "prevalence_counts",
    "received_client_ids",
}


def config(args, dataset, model):
    return {
        "task_type": "small",
        "dataset": dataset,
        "model": model,
        "num_clients": 7,
        "beta": 0.0,
        "seed": 42,
        "method": "lamp_merge",
        "model_hub_root": str(args.model_hub_root),
        "lamp_merge_prototype_root": str(args.prototype_root),
        "lamp_merge_proto_count_power": 0.55,
        "lamp_merge_reference_head_scale": 18.75,
        "lamp_merge_reference_prior_threshold": 2.5,
        "lamp_merge_reference_prior_max_tau": 4.25,
    }


def compare_state_dicts(left, right):
    if list(left) != list(right):
        raise AssertionError("state-dict keys differ")
    max_abs_error = 0.0
    for key in left:
        if left[key].shape != right[key].shape:
            raise AssertionError(f"shape differs for {key}")
        error = float((left[key].float() - right[key].float()).abs().max().item())
        max_abs_error = max(max_abs_error, error)
        if not torch.allclose(left[key], right[key], rtol=0.0, atol=1e-6):
            raise AssertionError(f"state differs for {key}; max_abs_error={error}")
    return max_abs_error


def audit_steps(outputs_root, dataset, model):
    path = report_path(outputs_root, dataset, model)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("delivery_mode") != "strict_per_client_upload":
        raise AssertionError(f"unexpected delivery mode: {path}")
    rows = report.get("rows", [])
    if [row.get("k") for row in rows] != list(range(1, 8)):
        raise AssertionError(f"missing async steps: {path}")
    for row in rows:
        step = int(row["k"])
        state_path = Path(row["checkpoint_dir"]) / "async_state.pt"
        trace_path = Path(row["checkpoint_dir"]) / "merge_trace.json"
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if set(state) != STATE_KEYS:
            raise AssertionError(f"unexpected persisted-state keys at k={step}: {set(state)}")
        expected_ids = list(range(step))
        if state["received_client_ids"] != expected_ids or trace["received_client_ids"] != expected_ids:
            raise AssertionError(f"incorrect received-client sequence at k={step}")
        if row["received_client_ids"] != expected_ids:
            raise AssertionError(f"incorrect report sequence at k={step}")
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--model-hub-root", type=Path, default=Path("/data2/liyapeng_grp/program/MedMNISTMerge/model_hub"))
    parser.add_argument("--prototype-root", type=Path, default=Path("/data2/liyapeng_grp/program/MedMNISTMerge/outputs/lamp_merge_client_local_proto_stats"))
    parser.add_argument("--result-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    checks = []
    for dataset in DATASETS:
        for model in MODELS:
            cfg = config(args, dataset, model)
            exp_dir = find_hub_experiment_dir(str(args.model_hub_root), cfg)
            meta = load_hub_meta(exp_dir)
            report = audit_steps(args.outputs_root, dataset, model)
            set_seed(42)
            lamp_state, _ = merge_lamp_merge(None, None, meta=meta, cfg=cfg)
            set_seed(42)
            new_state, _ = merge_new_lamp_merge(None, None, meta=meta, cfg=cfg)
            max_abs_error = compare_state_dicts(lamp_state, new_state)
            checks.append(
                {
                    "dataset": dataset,
                    "backbone": model,
                    "strict_async_steps": True,
                    "final_state_matches_lamp_merge": True,
                    "max_abs_error": max_abs_error,
                    "report_path": report,
                }
            )

    result = {
        "strict_async": True,
        "final_state_equivalent_to_lamp_merge": True,
        "checks": checks,
    }
    args.result_root.mkdir(parents=True, exist_ok=True)
    (args.result_root / "strict_async_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
