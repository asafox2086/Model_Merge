#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from merge import METHOD_DEFAULTS, resolve_client_weights
from methods.common import (
    build_task_matrix,
    disjoint_merge,
    elect_sign,
    flatten_param_vector,
    mask_smallest_magnitude,
)
from methods.my_merge import (
    _consensus,
    _medical_weighted_merge,
    _specialist_scores,
    _task_conflict_stats,
)
from scripts.run_all_avg_eval import build_cfg, load_manifest
from scripts.summarize_my_merge_ablations import baseline_best_lookup, key_for_row
from utils import (
    ensure_checkpoint_files,
    ensure_state_dicts_compatible,
    ensure_task_matches_config,
    extract_state_dict,
    find_hub_experiment_dir,
    load_checkpoint,
    load_hub_meta,
)
from utils.runtime import build_reference_bundle
from utils.state_dict import average_state_dicts


TARGET_DATASETS = ("bloodmnist_224", "chaoshengmnist_224", "organcmnist_224")
TARGET_BETAS = (0.0, 0.01, 0.1)
TARGET_CLIENTS = (3, 5, 7)
DELTA_LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.0)
EPS = 1e-8


class _ArgsForCfg:
    def __init__(self, args, row):
        self.model_hub_root = args.model_hub_root
        self.data_root = args.data_root
        self.output_root = str(args.output_root)
        self.device = args.device
        self.small_batch_size = args.small_batch_size
        self.vlm_batch_size = args.vlm_batch_size
        self.num_workers = args.num_workers
        self.method = "my_merge"
        self.merge_weight_mode = args.merge_weight_mode
        self.density = METHOD_DEFAULTS["density"]
        self.dare_seed = METHOD_DEFAULTS["dare_seed"]
        self.fisher_eps = METHOD_DEFAULTS["fisher_eps"]
        self.fisher_normalize_weight = True
        self.fisher_minimal_weight = METHOD_DEFAULTS["fisher_minimal_weight"]
        self.regmean_eps = METHOD_DEFAULTS["regmean_eps"]
        self.regmean_reduce_non_diagonal_ratio = METHOD_DEFAULTS["regmean_reduce_non_diagonal_ratio"]
        self.breadcrumbs_top_k_keep = METHOD_DEFAULTS["breadcrumbs_top_k_keep"]
        self.breadcrumbs_top_k_remove = METHOD_DEFAULTS["breadcrumbs_top_k_remove"]
        self.breadcrumbs_alpha = METHOD_DEFAULTS["breadcrumbs_alpha"]
        self.model_stock_k = METHOD_DEFAULTS["model_stock_k"]
        self.adamerging_epochs = METHOD_DEFAULTS["adamerging_epochs"]
        self.adamerging_lr = METHOD_DEFAULTS["adamerging_lr"]
        self.adamerging_prior = METHOD_DEFAULTS["adamerging_prior"]
        self.adamerging_max_batches = METHOD_DEFAULTS["adamerging_max_batches"]
        self.from_k = METHOD_DEFAULTS["from_k"]
        self.iso_common_space_fraction = METHOD_DEFAULTS["iso_common_space_fraction"]
        self.free_filter_ratio = METHOD_DEFAULTS["free_filter_ratio"]
        self.free_scaling = METHOD_DEFAULTS["free_scaling"]
        self.free_include_all_2d = METHOD_DEFAULTS["free_include_all_2d"]
        self.robustmerge_mask_ratio = METHOD_DEFAULTS["robustmerge_mask_ratio"]
        self.robustmerge_att_ratio = METHOD_DEFAULTS["robustmerge_att_ratio"]
        self.robustmerge_fuse_weight = METHOD_DEFAULTS["robustmerge_fuse_weight"]
        self.robustmerge_include_all_2d = METHOD_DEFAULTS["robustmerge_include_all_2d"]
        self.stats_split = args.stats_split
        self.stats_batch_size = args.stats_batch_size
        self.stats_num_workers = args.stats_num_workers
        self.fisher_max_batches = METHOD_DEFAULTS["fisher_max_batches"]
        self.regmean_max_batches = METHOD_DEFAULTS["regmean_max_batches"]
        self.regmean_max_dim = METHOD_DEFAULTS["regmean_max_dim"]
        self.my_merge_stats_max_batches = args.my_merge_stats_max_batches
        self.my_merge_eval_max_batches = -1
        self.my_merge_bn_batches = -1
        self.my_merge_ablation = "full"
        self.my_merge_disable = ""
        self.my_merge_domain_control = False
        self.task_type = row["task_type"]
        self.datasets = None
        self.small_models = None
        self.clip_models = None


def parse_args():
    p = argparse.ArgumentParser("Compute model-space diagnostics for my_merge candidate shapes")
    p.add_argument("--model-hub-root", default=str(ROOT / "model_hub"))
    p.add_argument("--data-root", default=str(ROOT / "Med_data"))
    p.add_argument("--history-root", default=str(ROOT / "outputs/codex_restored_good_medical_full_20260611_1838/my_merge_ablation_grid/full/small_resnet__my_merge"))
    p.add_argument("--baseline", default=str(ROOT / "result/all_results.md"))
    p.add_argument("--output-csv", default=str(ROOT / "docs/my_merge_user_docs/candidate_space_metrics_20260612.csv"))
    p.add_argument("--output-md", default=str(ROOT / "docs/my_merge_user_docs/candidate_space_metrics_20260612.md"))
    p.add_argument("--datasets", nargs="*", default=list(TARGET_DATASETS))
    p.add_argument("--small-model", default="resnet")
    p.add_argument("--device", default="cpu")
    p.add_argument("--small-batch-size", type=int, default=64)
    p.add_argument("--vlm-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--stats-split", default="val")
    p.add_argument("--stats-batch-size", type=int, default=32)
    p.add_argument("--stats-num-workers", type=int, default=0)
    p.add_argument("--my-merge-stats-max-batches", type=int, default=16)
    p.add_argument("--merge-weight-mode", choices=["sample", "equal"], default="equal")
    p.add_argument("--output-root", default=str(ROOT / "outputs/candidate_space_diagnostics"))
    return p.parse_args()


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm(vector):
    return float(torch.linalg.vector_norm(vector.float()).item())


def _cos(left, right):
    left = left.float()
    right = right.float()
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom.item()) <= EPS:
        return 0.0
    return float(torch.clamp(torch.dot(left, right) / (denom + EPS), -1.0, 1.0).item())


def _ratio(num, den):
    return float(num) / float(den + EPS)


def _fmt(value):
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _candidate_name_for_lambda(value):
    return f"delta_{value:.2f}".replace(".", "p").replace("p00", "p00").replace("p50", "p50")


def _beta_dir(beta):
    return f"beta_{format(float(beta), 'g').replace('.', 'p')}"


def _history_merge_path(history_root, meta):
    return (
        Path(history_root)
        / "merged"
        / "small"
        / meta["dataset"]
        / meta["model"]
        / f"clients_{int(meta['num_clients'])}"
        / _beta_dir(meta["beta"])
        / f"seed_{int(meta['seed'])}"
        / "my_merge"
        / "merge_result.json"
    )


def _history_eval_lookup(history_root):
    path = Path(history_root) / "reports" / "eval_summary.csv"
    lookup = {}
    if not path.exists():
        return lookup
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lookup[key_for_row(row)] = float(row["test_acc"])
    return lookup


def _selected_metric(metrics, selected, key):
    item = (metrics or {}).get(selected) or {}
    return item.get(key)


def _load_case(row, args):
    cfg = build_cfg(row, _ArgsForCfg(args, row))
    cfg = {**METHOD_DEFAULTS, **cfg}
    exp_dir = find_hub_experiment_dir(cfg["model_hub_root"], cfg)
    meta = load_hub_meta(exp_dir)
    ensure_task_matches_config(meta, cfg)
    ckpt_paths = ensure_checkpoint_files(exp_dir, meta)
    checkpoints = [load_checkpoint(path, device="cpu") for path in ckpt_paths]
    state_dicts = [extract_state_dict(obj) for obj in checkpoints]
    ensure_state_dicts_compatible(state_dicts)
    weights = resolve_client_weights(meta, cfg, "my_merge")
    base_merged, base_weights = average_state_dicts(state_dicts, weights)
    reference, param_names = build_reference_bundle(meta, device="cpu")
    return cfg, meta, checkpoints, state_dicts, base_merged, torch.tensor(base_weights, dtype=torch.float32), reference, param_names


def _m1_from_history(history_payload, base_weights, num_classes):
    method_info = history_payload["method_info"]
    overall = torch.tensor(method_info["overall_weights"], dtype=torch.float32)
    morph = torch.tensor(method_info["morphology_weights"], dtype=torch.float32)
    class_weights = torch.tensor(method_info["class_weights"], dtype=torch.float32)
    if class_weights.ndim != 2 or class_weights.shape[0] != int(num_classes):
        class_weights = torch.stack([morph for _ in range(int(num_classes))], dim=0)
    consensus = _consensus(overall, morph, base_weights)
    return method_info, overall, morph, class_weights, consensus


def _build_case_metrics(meta, state_dicts, base_merged, base_weights, reference, param_names, method_info):
    overall = torch.tensor(method_info["overall_weights"], dtype=torch.float32)
    morph = torch.tensor(method_info["morphology_weights"], dtype=torch.float32)
    class_weights = torch.tensor(method_info["class_weights"], dtype=torch.float32)
    consensus = _consensus(overall, morph, base_weights)
    task_matrix, base_vector = build_task_matrix(state_dicts, reference, param_names)
    client_norms = torch.linalg.vector_norm(task_matrix.float(), dim=1).clamp_min(EPS)
    avg_delta = (task_matrix * base_weights.view(-1, 1)).sum(dim=0)
    consensus_delta = (task_matrix * consensus.view(-1, 1)).sum(dim=0)
    trimmed = mask_smallest_magnitude(task_matrix, preserve_density=0.5)
    sign_delta = disjoint_merge(trimmed, elect_sign(trimmed), weights=consensus.tolist())
    medical_state, _medical_trace = _medical_weighted_merge(
        state_dicts,
        base_merged,
        reference,
        base_weights,
        overall,
        morph,
        class_weights,
        int(meta["num_classes"]),
        meta,
    )
    medical_delta = flatten_param_vector(medical_state, param_names) - base_vector
    scores = _specialist_scores(method_info["client_diagnostic_information"], meta)
    score_order = torch.argsort(scores, descending=True)
    specialist_idx = int(score_order[0].item())
    specialist_second = float(scores[int(score_order[1].item())].item()) if scores.numel() > 1 else 0.0
    specialist_gap = float(scores[specialist_idx].item()) - specialist_second
    specialist_delta = task_matrix[specialist_idx]

    candidates = {}
    for lam in DELTA_LAMBDAS:
        candidates[_candidate_name_for_lambda(lam)] = (1.0 - lam) * avg_delta + lam * sign_delta
    candidates["medical_weighted_fusion"] = medical_delta
    candidates["specialist_client"] = specialist_delta

    conflict = _task_conflict_stats(state_dicts, reference, param_names)
    avg_norm = _norm(avg_delta)
    consensus_norm = _norm(consensus_delta)
    sign_norm = _norm(sign_delta)
    medical_norm = _norm(medical_delta)
    specialist_norm = _norm(specialist_delta)
    base_mass = float((base_weights * client_norms).sum().item())
    consensus_mass = float((consensus * client_norms).sum().item())
    case_metrics = {
        "sign_conflict": conflict["sign_conflict"],
        "direction_conflict": conflict["direction_conflict"],
        "norm_dispersion": conflict["norm_dispersion"],
        "conflict_score": conflict["conflict_score"],
        "base_cancellation": _ratio(avg_norm, base_mass),
        "consensus_cancellation": _ratio(consensus_norm, consensus_mass),
        "consensus_vs_avg_cos": _cos(consensus_delta, avg_delta),
        "sign_vs_avg_cos": _cos(sign_delta, avg_delta),
        "sign_vs_consensus_cos": _cos(sign_delta, consensus_delta),
        "medical_vs_consensus_cos": _cos(medical_delta, consensus_delta),
        "specialist_vs_consensus_cos": _cos(specialist_delta, consensus_delta),
        "avg_norm": avg_norm,
        "consensus_norm": consensus_norm,
        "sign_norm": sign_norm,
        "medical_norm": medical_norm,
        "specialist_norm": specialist_norm,
        "sign_norm_ratio_to_avg": _ratio(sign_norm, avg_norm),
        "medical_norm_ratio_to_avg": _ratio(medical_norm, avg_norm),
        "specialist_norm_ratio_to_avg": _ratio(specialist_norm, avg_norm),
        "weight_shift_l1": float(torch.abs(consensus - base_weights).sum().item()),
        "weight_concentration": float((consensus.max() - consensus.mean()).item()),
        "specialist_idx": specialist_idx,
        "specialist_score_gap": specialist_gap,
        "specialist_score_gap_ratio": _ratio(specialist_gap, float(scores[specialist_idx].item())),
        "sign_retained_l1_ratio": _ratio(float(trimmed.abs().sum().item()), float(task_matrix.abs().sum().item())),
    }
    anchors = {
        "avg": avg_delta,
        "consensus": consensus_delta,
        "sign": sign_delta,
        "medical": medical_delta,
        "specialist": specialist_delta,
    }
    return case_metrics, candidates, anchors


def _candidate_metrics(candidate_delta, anchors):
    cand_norm = _norm(candidate_delta)
    avg_norm = _norm(anchors["avg"])
    consensus_norm = _norm(anchors["consensus"])
    sign_norm = _norm(anchors["sign"])
    return {
        "candidate_norm": cand_norm,
        "candidate_norm_ratio_to_avg": _ratio(cand_norm, avg_norm),
        "candidate_norm_ratio_to_consensus": _ratio(cand_norm, consensus_norm),
        "distance_from_avg_ratio": _ratio(_norm(candidate_delta - anchors["avg"]), avg_norm),
        "distance_from_sign_ratio": _ratio(_norm(candidate_delta - anchors["sign"]), sign_norm),
        "cos_to_avg": _cos(candidate_delta, anchors["avg"]),
        "cos_to_consensus": _cos(candidate_delta, anchors["consensus"]),
        "cos_to_sign": _cos(candidate_delta, anchors["sign"]),
        "cos_to_medical": _cos(candidate_delta, anchors["medical"]),
        "cos_to_specialist": _cos(candidate_delta, anchors["specialist"]),
    }


def _manifest_rows(args):
    rows = []
    datasets = set(args.datasets)
    for row in load_manifest(Path(args.model_hub_root) / "manifest.csv"):
        if row["task_type"] != "small":
            continue
        if row["dataset"] not in datasets:
            continue
        if row["model"] != args.small_model:
            continue
        if int(row["num_clients"]) not in TARGET_CLIENTS:
            continue
        if float(row["beta"]) not in TARGET_BETAS:
            continue
        rows.append(row)
    return sorted(rows, key=lambda r: (r["dataset"], int(r["num_clients"]), float(r["beta"])))


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "model",
        "num_clients",
        "beta",
        "candidate",
        "historical_selected",
        "is_historical_selected",
        "historical_acc",
        "best_baseline",
        "historical_delta_vs_best",
        "historical_val_acc",
        "historical_val_medical_acc",
        "historical_selection_score",
        "conflict_score",
        "sign_conflict",
        "direction_conflict",
        "norm_dispersion",
        "base_cancellation",
        "consensus_cancellation",
        "weight_shift_l1",
        "weight_concentration",
        "specialist_idx",
        "specialist_score_gap_ratio",
        "candidate_norm_ratio_to_avg",
        "candidate_norm_ratio_to_consensus",
        "distance_from_avg_ratio",
        "distance_from_sign_ratio",
        "cos_to_avg",
        "cos_to_consensus",
        "cos_to_sign",
        "cos_to_medical",
        "cos_to_specialist",
        "sign_norm_ratio_to_avg",
        "medical_norm_ratio_to_avg",
        "specialist_norm_ratio_to_avg",
        "sign_retained_l1_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _mean(rows, key):
    vals = [_float(row.get(key), None) for row in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _write_md(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_rows = [row for row in rows if row["is_historical_selected"] == "1"]
    by_selected = defaultdict(list)
    for row in selected_rows:
        by_selected[row["historical_selected"]].append(row)

    lines = [
        "# my_merge 候选空间诊断（2026-06-12）",
        "",
        "用途：这份表只用于观察候选形态的参数空间现象，不作为最终结果表。`historical_selected` 来自历史好版本的验证集选择，只作为显微镜标签，不能作为最终算法输入。",
        "",
        "## 历史选中候选的空间画像",
        "",
        "| historical_selected | n | hist_delta | conflict | base_cancel | cons_cancel | weight_shift | dist_avg | cos_cons | cos_sign | norm/avg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(by_selected):
        group = by_selected[name]
        lines.append(
            "| {name} | {n} | {hist_delta} | {conflict} | {base_cancel} | {cons_cancel} | {weight_shift} | {dist_avg} | {cos_cons} | {cos_sign} | {norm_avg} |".format(
                name=name,
                n=len(group),
                hist_delta=_fmt(_mean(group, "historical_delta_vs_best")),
                conflict=_fmt(_mean(group, "conflict_score")),
                base_cancel=_fmt(_mean(group, "base_cancellation")),
                cons_cancel=_fmt(_mean(group, "consensus_cancellation")),
                weight_shift=_fmt(_mean(group, "weight_shift_l1")),
                dist_avg=_fmt(_mean(group, "distance_from_avg_ratio")),
                cos_cons=_fmt(_mean(group, "cos_to_consensus")),
                cos_sign=_fmt(_mean(group, "cos_to_sign")),
                norm_avg=_fmt(_mean(group, "candidate_norm_ratio_to_avg")),
            )
        )

    lines.extend([
        "",
        "## 27 个 ResNet 医学格子的历史标签与选中候选指标",
        "",
        "| dataset | setting | hist_selected | hist_delta | conflict | base_cancel | weight_shift | selected dist_avg | selected cos_cons | selected cos_sign | selected norm/avg |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(selected_rows, key=lambda r: (r["dataset"], int(r["num_clients"]), float(r["beta"]))):
        dataset = row["dataset"].replace("mnist_224", "").replace("chaosheng", "chaosheng")
        setting = f"c{int(row['num_clients'])}_b{float(row['beta']):g}"
        lines.append(
            "| {dataset} | {setting} | `{selected}` | {hist_delta} | {conflict} | {base_cancel} | {weight_shift} | {dist_avg} | {cos_cons} | {cos_sign} | {norm_avg} |".format(
                dataset=dataset,
                setting=setting,
                selected=row["historical_selected"],
                hist_delta=_fmt(row["historical_delta_vs_best"]),
                conflict=_fmt(row["conflict_score"]),
                base_cancel=_fmt(row["base_cancellation"]),
                weight_shift=_fmt(row["weight_shift_l1"]),
                dist_avg=_fmt(row["distance_from_avg_ratio"]),
                cos_cons=_fmt(row["cos_to_consensus"]),
                cos_sign=_fmt(row["cos_to_sign"]),
                norm_avg=_fmt(row["candidate_norm_ratio_to_avg"]),
            )
        )

    lines.extend([
        "",
        "## 当前观察",
        "",
        "- `delta_0p00` 到 `delta_1p00` 是一条连续的冲突处理路径，空间距离主要由 `distance_from_avg_ratio` 和 `candidate_norm_ratio_to_avg` 控制。",
        "- `specialist_client` 和 `medical_weighted_fusion` 不在这条连续路径上，应重点看它们与 `consensus`、`sign` 的方向一致性，而不是只看范数大小。",
        "- 如果同一历史候选组内部的 `conflict/base_cancel/weight_shift` 分布重叠很大，说明不能继续写单阈值路由，需要再找更稳定的中间现象。",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    history_eval = _history_eval_lookup(args.history_root)
    best_lookup = baseline_best_lookup(args.baseline)
    rows = []
    for row in _manifest_rows(args):
        cfg, meta, checkpoints, state_dicts, base_merged, base_weights, reference, param_names = _load_case(row, args)
        history_path = _history_merge_path(args.history_root, meta)
        if not history_path.exists():
            print(f"[WARN] missing history merge_result: {history_path}")
            continue
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
        method_info, _overall, _morph, _class_weights, _consensus = _m1_from_history(history_payload, base_weights, meta["num_classes"])
        case_metrics, candidates, anchors = _build_case_metrics(meta, state_dicts, base_merged, base_weights, reference, param_names, method_info)
        selected = method_info.get("selected_candidate", "")
        hist_key = ("small", meta["dataset"], meta["model"], int(meta["num_clients"]), float(meta["beta"]))
        historical_acc = history_eval.get(hist_key)
        best = best_lookup.get(hist_key)
        historical_delta = None if historical_acc is None or best is None else historical_acc - best
        historical_candidate_metrics = method_info.get("candidate_metrics") or {}

        for candidate, delta in candidates.items():
            metrics = _candidate_metrics(delta, anchors)
            hist_val = historical_candidate_metrics.get(candidate) or {}
            out = {
                "dataset": meta["dataset"],
                "model": meta["model"],
                "num_clients": int(meta["num_clients"]),
                "beta": float(meta["beta"]),
                "candidate": candidate,
                "historical_selected": selected,
                "is_historical_selected": "1" if candidate == selected else "0",
                "historical_acc": "" if historical_acc is None else historical_acc,
                "best_baseline": "" if best is None else best,
                "historical_delta_vs_best": "" if historical_delta is None else historical_delta,
                "historical_val_acc": hist_val.get("val_acc", ""),
                "historical_val_medical_acc": hist_val.get("val_medical_acc", ""),
                "historical_selection_score": hist_val.get("selection_score", ""),
                **case_metrics,
                **metrics,
            }
            rows.append(out)

    _write_csv(args.output_csv, rows)
    _write_md(args.output_md, rows)
    selected = [row for row in rows if row["is_historical_selected"] == "1"]
    print(f"wrote {len(rows)} candidate rows, {len(selected)} selected-case rows")
    print(f"csv: {args.output_csv}")
    print(f"md: {args.output_md}")


if __name__ == "__main__":
    main()
