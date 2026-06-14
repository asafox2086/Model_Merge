#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate import run_evaluate
from methods.common import (
    build_task_matrix,
    disjoint_merge,
    elect_sign,
    mask_smallest_magnitude,
    overlay_param_dict,
    vector_to_param_dict,
)
from methods.my_merge import _consensus, _medical_weighted_merge, _specialist_scores
from scripts.analyze_my_merge_candidate_space import (
    DELTA_LAMBDAS,
    TARGET_BETAS,
    TARGET_CLIENTS,
    TARGET_DATASETS,
    _history_eval_lookup,
    _history_merge_path,
    _load_case,
    _manifest_rows,
    _m1_from_history,
)
from scripts.summarize_my_merge_ablations import baseline_best_lookup, key_for_row
from utils import load_json, make_merge_output_dir, save_csv, save_json


def parse_args():
    p = argparse.ArgumentParser("Evaluate fixed my_merge candidate shapes for observation only")
    p.add_argument("--model-hub-root", default=str(ROOT / "model_hub"))
    p.add_argument("--data-root", default=str(ROOT / "Med_data"))
    p.add_argument("--history-root", default=str(ROOT / "outputs/codex_restored_good_medical_full_20260611_1838/my_merge_ablation_grid/full/small_resnet__my_merge"))
    p.add_argument("--baseline", default=str(ROOT / "result/all_results.md"))
    p.add_argument("--output-root", default=str(ROOT / "outputs/codex_candidate_family_probe_20260612"))
    p.add_argument("--output-md", default=str(ROOT / "docs/my_merge_user_docs/candidate_family_probe_20260612.md"))
    p.add_argument("--datasets", nargs="*", default=list(TARGET_DATASETS))
    p.add_argument("--small-model", default="resnet")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--small-batch-size", type=int, default=64)
    p.add_argument("--vlm-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--stats-split", default="val")
    p.add_argument("--stats-batch-size", type=int, default=32)
    p.add_argument("--stats-num-workers", type=int, default=0)
    p.add_argument("--my-merge-stats-max-batches", type=int, default=16)
    p.add_argument("--merge-weight-mode", choices=["sample", "equal"], default="equal")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--delete-merged", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _candidate_name_for_lambda(value):
    return f"delta_{float(value):.2f}".replace(".", "p")


def _candidate_method(candidate):
    return f"my_merge_probe_{candidate}"


def _candidate_label(method):
    prefix = "my_merge_probe_"
    return method[len(prefix):] if method.startswith(prefix) else method


def _clone_state(state):
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in state.items())


def _build_candidate_states(meta, state_dicts, base_merged, base_weights, reference, param_names, history_payload):
    method_info, overall, morph, class_weights, consensus = _m1_from_history(
        history_payload,
        base_weights,
        meta["num_classes"],
    )
    task_matrix, base_vector = build_task_matrix(state_dicts, reference, param_names)
    avg_delta = (task_matrix * base_weights.view(-1, 1)).sum(dim=0)
    trimmed = mask_smallest_magnitude(task_matrix, preserve_density=0.5)
    sign_delta = disjoint_merge(trimmed, elect_sign(trimmed), weights=consensus.tolist())

    candidates = OrderedDict()
    for lam in DELTA_LAMBDAS:
        name = _candidate_name_for_lambda(lam)
        delta = (1.0 - lam) * avg_delta + lam * sign_delta
        param_state = vector_to_param_dict(base_vector + delta, reference, param_names)
        candidates[name] = overlay_param_dict(base_merged, param_state)

    medical_state, _ = _medical_weighted_merge(
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
    candidates["medical_weighted_fusion"] = medical_state

    scores = _specialist_scores(method_info["client_diagnostic_information"], meta)
    specialist_idx = int(torch.argsort(scores, descending=True)[0].item())
    candidates["specialist_client"] = _clone_state(state_dicts[specialist_idx])
    return candidates


def _write_checkpoint(out_dir, meta, method, state):
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": state,
        "meta": {
            "task_type": meta["task_type"],
            "dataset": meta["dataset"],
            "model": meta["model"],
            "clip_model": meta.get("clip_model"),
            "num_clients": meta["num_clients"],
            "beta": meta["beta"],
            "seed": meta["seed"],
            "method": method,
            "merge_weight_mode": "equal",
        },
    }
    ckpt_path = out_dir / "merged.pt"
    torch.save(checkpoint, ckpt_path)
    save_json(out_dir / "meta.json", meta)
    return ckpt_path


def _existing_eval(output_root, cfg):
    path = Path(output_root) / "eval" / cfg["task_type"] / cfg["dataset"] / cfg["model"] / f"clients_{int(cfg['num_clients'])}"
    beta_dir = f"beta_{format(float(cfg['beta']), 'g').replace('.', 'p')}"
    eval_path = path / beta_dir / f"seed_{int(cfg['seed'])}" / cfg["method"] / "eval.json"
    if not eval_path.exists():
        return None
    payload = load_json(eval_path)
    if payload.get("method") != cfg["method"]:
        return None
    return payload


def _case_key(meta):
    return ("small", meta["dataset"], meta["model"], int(meta["num_clients"]), float(meta["beta"]))


def _collect_eval_rows(output_root):
    summary = Path(output_root) / "reports" / "eval_summary.csv"
    if not summary.exists():
        return []
    with summary.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_summary_md(path, rows):
    by_candidate = defaultdict(list)
    by_dataset = defaultdict(lambda: defaultdict(list))
    for row in rows:
        candidate = _candidate_label(row["method"])
        if not candidate.startswith("delta_") and candidate not in {"medical_weighted_fusion", "specialist_client"}:
            continue
        row = dict(row)
        row["candidate"] = candidate
        row["test_acc"] = float(row["test_acc"])
        row["delta_vs_best"] = float(row["delta_vs_best"])
        row["delta_vs_history"] = float(row["delta_vs_history"])
        by_candidate[candidate].append(row)
        by_dataset[row["dataset"]][candidate].append(row)

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    lines = [
        "# my_merge 固定候选族测试探针（2026-06-12）",
        "",
        "用途：这不是最终方法，也不写入正式汇总表。它只用于验证候选形态在测试集上的真实表现，帮助后续从现象设计规则。",
        "",
        "## Overall",
        "",
        "| candidate | n | mean_acc | mean_delta_vs_best | mean_delta_vs_history | W/T/L vs best |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in sorted(by_candidate):
        group = by_candidate[candidate]
        deltas = [r["delta_vs_best"] for r in group]
        lines.append(
            "| {candidate} | {n} | {acc:.4f} | {delta_best:+.4f} | {delta_hist:+.4f} | {w}/{t}/{l} |".format(
                candidate=candidate,
                n=len(group),
                acc=mean([r["test_acc"] for r in group]),
                delta_best=mean(deltas),
                delta_hist=mean([r["delta_vs_history"] for r in group]),
                w=sum(x > 5e-5 for x in deltas),
                t=sum(abs(x) <= 5e-5 for x in deltas),
                l=sum(x < -5e-5 for x in deltas),
            )
        )

    lines.extend(["", "## By Dataset", ""])
    for dataset in sorted(by_dataset):
        lines.extend([
            f"### {dataset}",
            "",
            "| candidate | n | mean_acc | mean_delta_vs_best | mean_delta_vs_history | W/T/L vs best |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for candidate in sorted(by_dataset[dataset]):
            group = by_dataset[dataset][candidate]
            deltas = [r["delta_vs_best"] for r in group]
            lines.append(
                "| {candidate} | {n} | {acc:.4f} | {delta_best:+.4f} | {delta_hist:+.4f} | {w}/{t}/{l} |".format(
                    candidate=candidate,
                    n=len(group),
                    acc=mean([r["test_acc"] for r in group]),
                    delta_best=mean(deltas),
                    delta_hist=mean([r["delta_vs_history"] for r in group]),
                    w=sum(x > 5e-5 for x in deltas),
                    t=sum(abs(x) <= 5e-5 for x in deltas),
                    l=sum(x < -5e-5 for x in deltas),
                )
            )
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = _manifest_rows(args)
    if args.max_cases > 0:
        rows = rows[: args.max_cases]
    best_lookup = baseline_best_lookup(args.baseline)
    history_eval = _history_eval_lookup(args.history_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[candidate-family] cases={len(rows)} output_root={output_root}")

    for idx, row in enumerate(rows, start=1):
        cfg, meta, _checkpoints, state_dicts, base_merged, base_weights, reference, param_names = _load_case(row, args)
        history_path = _history_merge_path(args.history_root, meta)
        if not history_path.exists():
            print(f"[WARN] missing history: {history_path}")
            continue
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
        candidates = _build_candidate_states(meta, state_dicts, base_merged, base_weights, reference, param_names, history_payload)
        hist_key = _case_key(meta)
        best = best_lookup.get(hist_key)
        hist_acc = history_eval.get(hist_key)

        for candidate, state in candidates.items():
            method = _candidate_method(candidate)
            eval_cfg = {
                **cfg,
                "method": method,
                "output_root": str(output_root),
                "device": args.device,
                "batch_size": args.small_batch_size,
                "num_workers": args.num_workers,
                "split": "test",
                "amp": str(args.device).startswith("cuda"),
            }
            existing = _existing_eval(output_root, eval_cfg) if args.resume else None
            if existing is not None:
                print(f"[candidate-family] skip {idx}/{len(rows)} {meta['dataset']} c{meta['num_clients']} b{meta['beta']:g} {candidate} acc={existing['test_acc']:.4f}")
                continue
            out_dir = make_merge_output_dir(output_root, eval_cfg)
            ckpt_path = _write_checkpoint(out_dir, meta, method, state)
            payload, _ = run_evaluate(eval_cfg, merged_dir=out_dir)
            payload["candidate"] = candidate
            payload["best_baseline"] = "" if best is None else best
            payload["history_acc"] = "" if hist_acc is None else hist_acc
            payload["delta_vs_best"] = "" if best is None else float(payload["test_acc"]) - best
            payload["delta_vs_history"] = "" if hist_acc is None else float(payload["test_acc"]) - hist_acc
            if args.delete_merged and ckpt_path.exists():
                ckpt_path.unlink()
            print(
                f"[candidate-family] done {idx}/{len(rows)} {meta['dataset']} "
                f"c{meta['num_clients']} b{meta['beta']:g} {candidate} acc={payload['test_acc']:.4f}"
            )

    eval_rows = _collect_eval_rows(output_root)
    best_lookup = baseline_best_lookup(args.baseline)
    history_eval = _history_eval_lookup(args.history_root)
    enriched = []
    for row in eval_rows:
        candidate = _candidate_label(row["method"])
        key = key_for_row(row)
        best = best_lookup.get(key)
        hist = history_eval.get(key)
        row = dict(row)
        row["candidate"] = candidate
        row["best_baseline"] = "" if best is None else best
        row["history_acc"] = "" if hist is None else hist
        row["delta_vs_best"] = "" if best is None else float(row["test_acc"]) - best
        row["delta_vs_history"] = "" if hist is None else float(row["test_acc"]) - hist
        enriched.append(row)
    save_csv(output_root / "reports" / "candidate_family_summary.csv", enriched)
    _write_summary_md(args.output_md, enriched)
    print(f"[candidate-family] summary: {output_root / 'reports' / 'candidate_family_summary.csv'}")
    print(f"[candidate-family] md: {args.output_md}")


if __name__ == "__main__":
    main()
