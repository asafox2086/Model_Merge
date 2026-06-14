#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_combined_results_table import parse_tables, try_parse_number
from generate_my_merge_master_table import SETTINGS, SMALL_DATASETS
from methods.my_merge import _is_classifier_tensor, _param_group
from utils import extract_state_dict, load_checkpoint, load_hub_meta
from utils.runtime import build_reference_bundle


EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser("Observe my_merge layer-wise parameter conflicts.")
    p.add_argument("--run-root", required=True, help="A small_resnet__my_merge output directory.")
    p.add_argument("--baseline", default="result/all_results.md")
    p.add_argument("--model-hub-root", default="model_hub")
    p.add_argument("--dest", required=True)
    return p.parse_args()


def key_for_row(row):
    return (
        row["task_type"],
        row["dataset"],
        row["model"] if row["task_type"] == "small" else row["clip_model"],
        int(float(row["num_clients"])),
        float(row["beta"]),
    )


def baseline_best_lookup(path):
    _intro, tables = parse_tables(Path(path))
    lookup = {}
    labels = [label for _clients, _beta, label in SETTINGS]
    for (section, model_name, kind), table in tables.items():
        if kind != "Raw":
            continue
        task_type = "small" if section == "Small" else "vlm"
        for dataset_idx, dataset in enumerate(SMALL_DATASETS):
            for setting_idx, (num_clients, beta, _label) in enumerate(SETTINGS):
                col_idx = dataset_idx * len(labels) + setting_idx
                values = []
                for row in table["rows"]:
                    if col_idx < len(row["values"]):
                        value = try_parse_number(row["values"][col_idx])
                        if value is not None:
                            values.append(value)
                if values:
                    lookup[(task_type, dataset, model_name, num_clients, float(beta))] = max(values)
    return lookup


def beta_label(beta):
    return format(float(beta), "g").replace(".", "p")


def hub_meta_path(model_hub_root, row):
    beta = beta_label(row["beta"])
    return (
        Path(model_hub_root)
        / row["task_type"]
        / row["dataset"]
        / row["model"]
        / f"clients_{int(float(row['num_clients']))}"
        / f"beta_{beta}"
        / f"seed_{int(float(row['seed']))}"
        / "meta.json"
    )


def load_state_dicts(model_hub_root, row):
    meta_path = hub_meta_path(model_hub_root, row)
    meta = load_hub_meta(meta_path.parent)
    state_dicts = []
    for client in meta["clients"]:
        path = meta_path.parent / client["checkpoint"]
        state_dicts.append(extract_state_dict(load_checkpoint(path, device="cpu")))
    reference, _param_names = build_reference_bundle(meta, device="cpu")
    return meta, state_dicts, reference


def group_keys(state_dict, reference, meta):
    groups = {"early": [], "mid": [], "late": [], "classifier": []}
    num_classes = int(meta["num_classes"])
    for key, value in state_dict.items():
        ref = reference.get(key)
        if ref is None or not torch.is_floating_point(value) or not torch.is_floating_point(ref):
            continue
        if tuple(value.shape) != tuple(ref.shape):
            continue
        if _is_classifier_tensor(key, value, num_classes):
            groups["classifier"].append(key)
        else:
            groups[_param_group(key, meta)].append(key)
    return groups


def flatten_delta(state_dict, reference, keys):
    if not keys:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat([(state_dict[key].float() - reference[key].float()).reshape(-1) for key in keys])


def conflict_for_group(state_dicts, reference, keys):
    if not keys:
        return {
            "params": 0,
            "sign_conflict": 0.0,
            "direction_conflict": 0.0,
            "norm_dispersion": 0.0,
            "mean_norm": 0.0,
            "conflict_score": 0.0,
        }
    task = torch.stack([flatten_delta(sd, reference, keys) for sd in state_dicts], dim=0)
    abs_mass = task.abs().sum(dim=0)
    active = abs_mass > EPS
    if torch.any(active):
        sign_agree = task.sum(dim=0).abs() / (abs_mass + EPS)
        sign_conflict = 1.0 - float((sign_agree[active] * abs_mass[active]).sum().item() / (abs_mass[active].sum().item() + EPS))
    else:
        sign_conflict = 0.0
    norms = task.norm(dim=1).clamp_min(EPS)
    if task.shape[0] > 1:
        unit = task / norms.view(-1, 1)
        cos = unit @ unit.t()
        mask = ~torch.eye(task.shape[0], dtype=torch.bool)
        direction_conflict = float(torch.clamp((1.0 - cos[mask].mean()) * 0.5, min=0.0, max=1.0).item())
    else:
        direction_conflict = 0.0
    norm_dispersion = float(torch.clamp(norms.std(unbiased=False) / (norms.mean() + EPS), min=0.0, max=1.0).item())
    conflict = max(0.0, min(1.0, 0.45 * sign_conflict + 0.35 * direction_conflict + 0.20 * norm_dispersion))
    return {
        "params": int(task.shape[1]),
        "sign_conflict": sign_conflict,
        "direction_conflict": direction_conflict,
        "norm_dispersion": norm_dispersion,
        "mean_norm": float(norms.mean().item()),
        "conflict_score": conflict,
    }


def hhi(values):
    return sum(float(x) * float(x) for x in values)


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    best_lookup = baseline_best_lookup(args.baseline)
    rows = []
    with (run_root / "reports" / "eval_summary.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") != "my_merge":
                continue
            merge_path = Path(row["merged_dir"]) / "merge_result.json"
            payload = json.loads(merge_path.read_text(encoding="utf-8"))["method_info"]
            meta, state_dicts, reference = load_state_dicts(args.model_hub_root, row)
            groups = group_keys(state_dicts[0], reference, meta)
            group_stats = {name: conflict_for_group(state_dicts, reference, keys) for name, keys in groups.items()}
            key = key_for_row(row)
            acc = float(row["test_acc"])
            best = best_lookup.get(key)
            routing = payload.get("routing_summary", {})
            out = {
                "dataset": row["dataset"],
                "model": row["model"],
                "num_clients": int(float(row["num_clients"])),
                "beta": float(row["beta"]),
                "test_acc": acc,
                "best_original": "" if best is None else best,
                "delta_vs_best": "" if best is None else acc - best,
                "fusion_hhi": hhi(payload.get("fusion_weights", [])),
                "fusion_max": max(payload.get("fusion_weights", [0.0])),
                "evidence_reliability": payload.get("evidence_reliability", 0.0),
                "class_morphology_separation": payload.get("class_morphology_separation", 0.0),
                "delta_blend_weight": routing.get("delta_blend_weight", 0.0),
                "specialist_anchor_weight": routing.get("specialist_anchor_weight", 0.0),
                "delta_agreement_mean_shift": routing.get("delta_agreement_mean_shift", 0.0),
            }
            for group, stats in group_stats.items():
                for name, value in stats.items():
                    out[f"{group}_{name}"] = value
            rows.append(out)

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
