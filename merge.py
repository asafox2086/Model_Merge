#!/usr/bin/env python3
"""One-shot server-side LAMP-Merge entry point."""

import argparse
from pathlib import Path

import torch

from methods import merge_lamp_merge, merge_new_lamp_merge, normalize_method_name
from utils import (
    append_summary_row,
    ensure_checkpoint_files,
    ensure_state_dicts_compatible,
    ensure_task_matches_config,
    extract_state_dict,
    find_hub_experiment_dir,
    load_checkpoint,
    load_config,
    load_hub_meta,
    make_merge_output_dir,
    save_json,
    set_seed,
)


METHOD_DEFAULTS = {
    "method": "lamp_merge",
    "merge_weight_mode": "sample",
    "lamp_merge_ablation_mode": "full",
    "lamp_merge_proto_count_power": 0.55,
    "lamp_merge_reference_head_scale": 18.75,
    "lamp_merge_reference_prior_threshold": 2.5,
    "lamp_merge_reference_prior_max_tau": 4.25,
}


def resolve_client_weights(meta, cfg):
    mode = cfg.get("merge_weight_mode", "sample")
    if mode == "sample":
        return [int(item["num_samples"]) for item in meta["clients"]]
    if mode == "equal":
        return [1 for _ in meta["clients"]]
    raise ValueError(f"Unsupported merge_weight_mode: {mode}")


def run_merge(cfg):
    set_seed(int(cfg.get("seed", 42)))
    method = normalize_method_name(cfg.get("method", "lamp_merge"))
    cfg = {**METHOD_DEFAULTS, **cfg, "method": method}

    exp_dir = find_hub_experiment_dir(cfg["model_hub_root"], cfg)
    meta = load_hub_meta(exp_dir)
    ensure_task_matches_config(meta, cfg)
    ckpt_paths = ensure_checkpoint_files(exp_dir, meta)

    checkpoints = [load_checkpoint(path, device="cpu") for path in ckpt_paths]
    state_dicts = [extract_state_dict(obj) for obj in checkpoints]
    ensure_state_dicts_compatible(state_dicts)

    weights = resolve_client_weights(meta, cfg)
    merge_fn = merge_new_lamp_merge if method == "new_lamp_merge" else merge_lamp_merge
    merged_state_dict, method_info = merge_fn(
        state_dicts,
        weights,
        meta=meta,
        checkpoints=checkpoints,
        cfg=cfg,
    )

    out_dir = make_merge_output_dir(cfg["output_root"], cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_ckpt_path = out_dir / "merged.pt"
    merged_meta_path = out_dir / "meta.json"
    merge_result_path = out_dir / "merge_result.json"

    merged_checkpoint = {
        "state_dict": merged_state_dict,
        "meta": {
            "task_type": meta["task_type"],
            "dataset": meta["dataset"],
            "model": meta["model"],
            "clip_model": meta.get("clip_model"),
            "num_clients": meta["num_clients"],
            "beta": meta["beta"],
            "seed": meta["seed"],
            "method": cfg["method"],
            "merge_weight_mode": cfg["merge_weight_mode"],
        },
    }
    torch.save(merged_checkpoint, merged_ckpt_path)
    save_json(merged_meta_path, meta)

    merge_result = {
        "task_type": meta["task_type"],
        "dataset": meta["dataset"],
        "model": meta["model"],
        "clip_model": meta.get("clip_model", ""),
        "num_clients": meta["num_clients"],
        "beta": meta["beta"],
        "seed": meta["seed"],
        "method": cfg["method"],
        "merge_weight_mode": cfg["merge_weight_mode"],
        "method_info": method_info,
        "source_clients": [item["checkpoint"] for item in meta["clients"]],
        "merged_checkpoint": str(merged_ckpt_path),
        "meta_path": str(merged_meta_path),
    }
    save_json(merge_result_path, merge_result)
    append_summary_row(
        Path(cfg["output_root"]) / "reports" / "merge_summary.csv",
        {
            "task_type": merge_result["task_type"],
            "dataset": merge_result["dataset"],
            "model": merge_result["model"],
            "clip_model": merge_result["clip_model"],
            "num_clients": merge_result["num_clients"],
            "beta": merge_result["beta"],
            "seed": merge_result["seed"],
            "method": merge_result["method"],
            "merge_weight_mode": merge_result["merge_weight_mode"],
            "merged_checkpoint": merge_result["merged_checkpoint"],
            "meta_path": merge_result["meta_path"],
        },
    )
    return {
        "merged_dir": str(out_dir),
        "merged_checkpoint": str(merged_ckpt_path),
        "meta_path": str(merged_meta_path),
        "merge_result_path": str(merge_result_path),
        "meta": meta,
    }


def parse_args():
    p = argparse.ArgumentParser("Run the one-shot LAMP-Merge server-side fusion step.")
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    result = run_merge(load_config(args.config))
    print(f"merged checkpoint saved to: {result['merged_checkpoint']}")
    print(f"merge result saved to: {result['merge_result_path']}")


if __name__ == "__main__":
    main()
