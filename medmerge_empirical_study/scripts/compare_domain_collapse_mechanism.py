#!/usr/bin/env python3
"""Compare where medical and natural-image AVG collapse mechanisms diverge."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.io import load_checkpoint, load_json
from utils.runtime import build_runtime
from utils.state_dict import average_state_dicts


DEFAULT_MEDICAL = "model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42"
DEFAULT_NATURAL = (
    "medmerge_empirical_study/results/experiment10_natural_image_imbalance_probe/"
    "natural_probe_hub/small/cifar10_32/resnet/clients_5/partial_label/seed_42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medical-case-dir", default=DEFAULT_MEDICAL)
    parser.add_argument("--natural-case-dir", default=DEFAULT_NATURAL)
    parser.add_argument("--data-root", default="Med_data")
    parser.add_argument("--out-dir", default="medmerge_empirical_study/results/experiment11_domain_collapse_mechanism")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def get_classifier(model: torch.nn.Module) -> tuple[str, torch.nn.Linear]:
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, torch.nn.Linear):
            for name, module in model.named_modules():
                if module is classifier:
                    return name, classifier
            return "classifier", classifier
    for name in ("fc", "head", "classifier"):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Linear):
            return name, module
    raise RuntimeError("Could not locate final Linear classifier")


def stage_hook_name(model: torch.nn.Module, name: str) -> torch.nn.Module | None:
    modules = dict(model.named_modules())
    return modules.get(name)


def pooled_activation(output: torch.Tensor) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 4:
        return output.detach().float().mean(dim=(2, 3)).cpu()
    return output.detach().float().flatten(1).cpu()


def balanced_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    recalls = []
    for cls in range(num_classes):
        mask = labels == cls
        support = int(mask.sum().item())
        recalls.append(float(((preds == cls) & mask).sum().item() / support) if support else 0.0)
    return float(sum(recalls) / max(num_classes, 1))


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    vals = []
    for cls in range(num_classes):
        tp = int(((preds == cls) & (labels == cls)).sum().item())
        fp = int(((preds == cls) & (labels != cls)).sum().item())
        fn = int(((preds != cls) & (labels == cls)).sum().item())
        denom = 2 * tp + fp + fn
        vals.append(2 * tp / denom if denom else 0.0)
    return float(sum(vals) / max(num_classes, 1))


def pred_entropy(pred_counts: list[int]) -> float:
    total = float(sum(pred_counts))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in pred_counts:
        if count:
            p = float(count) / total
            entropy -= p * math.log(p)
    return float(entropy)


@torch.no_grad()
def run_state(
    *,
    meta: dict,
    state_dict: OrderedDict,
    data_root: str,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    collect_stages: bool,
) -> dict:
    runtime = build_runtime(meta, data_root, split, batch_size, num_workers, device)
    model = runtime["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    forward_fn = runtime["forward_fn"]
    classifier_name, classifier = get_classifier(model)

    stage_names = ["bn1", "layer1", "layer2", "layer3", "layer4", "global_pool"]
    stage_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in stage_names}
    stage_chunks["classifier_input"] = []
    handles = []
    if collect_stages:
        for name in stage_names:
            module = stage_hook_name(model, name)
            if module is None:
                continue

            def make_hook(stage_name: str):
                def hook(_module, _inputs, output):
                    stage_chunks[stage_name].append(pooled_activation(output))

                return hook

            handles.append(module.register_forward_hook(make_hook(name)))

        def classifier_hook(_module, inputs, _output):
            stage_chunks["classifier_input"].append(inputs[0].detach().float().cpu())

        handles.append(classifier.register_forward_hook(classifier_hook))

    logits_chunks = []
    label_chunks = []
    for x, y in runtime["loader"]:
        x = x.to(device, non_blocking=True)
        logits = forward_fn(model, x)
        logits_chunks.append(logits.detach().float().cpu())
        label_chunks.append(y.detach().cpu())

    for handle in handles:
        handle.remove()

    logits = torch.cat(logits_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    preds = logits.argmax(dim=1)
    num_classes = int(meta["num_classes"])
    pred_counts = torch.bincount(preds, minlength=num_classes).tolist()
    true_counts = torch.bincount(labels, minlength=num_classes).tolist()
    sorted_logits, sorted_idx = torch.sort(logits, dim=1, descending=True)
    probs = torch.softmax(logits, dim=1)
    top_class = int(torch.tensor(pred_counts).argmax().item())
    out = {
        "classifier_name": classifier_name,
        "model": model,
        "logits": logits,
        "labels": labels,
        "preds": preds,
        "pred_counts": pred_counts,
        "true_counts": true_counts,
        "top_class": top_class,
        "num_samples": int(labels.numel()),
        "accuracy": float((preds == labels).float().mean().item()),
        "balanced_accuracy": balanced_accuracy(preds, labels, num_classes),
        "macro_f1": macro_f1(preds, labels, num_classes),
        "collapse_ratio": max(pred_counts) / max(int(labels.numel()), 1),
        "effective_pred_classes": int(sum(1 for c in pred_counts if c > 0)),
        "prediction_entropy": pred_entropy(pred_counts),
        "prediction_entropy_norm": pred_entropy(pred_counts) / math.log(max(num_classes, 2)),
        "mean_confidence": float(probs.max(dim=1).values.mean().item()),
        "mean_top1_top2_margin": float((sorted_logits[:, 0] - sorted_logits[:, 1]).mean().item()),
        "min_top1_top2_margin": float((sorted_logits[:, 0] - sorted_logits[:, 1]).min().item()),
        "top1_class_hist": [int((sorted_idx[:, 0] == i).sum().item()) for i in range(num_classes)],
    }
    if collect_stages:
        out["stages"] = {
            name: torch.cat(chunks, dim=0)
            for name, chunks in stage_chunks.items()
            if chunks
        }
    else:
        out["stages"] = {}
    return out


def feature_geometry(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    x = features.float()
    y = labels
    n = int(x.shape[0])
    norms = x.norm(dim=1)
    centroids = []
    within_vals = []
    present = []
    for cls in range(num_classes):
        mask = y == cls
        if int(mask.sum().item()) == 0:
            continue
        cls_x = x[mask]
        centroid = cls_x.mean(dim=0)
        centroids.append(centroid)
        present.append(cls)
        within_vals.append(((cls_x - centroid) ** 2).sum(dim=1).mean())
    centroid_t = torch.stack(centroids, dim=0)
    within = torch.stack(within_vals).mean()
    dists = torch.cdist(centroid_t, centroid_t, p=2)
    upper = dists[torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)]
    global_center = x.mean(dim=0)
    between = ((centroid_t - global_center) ** 2).sum(dim=1).mean()

    own_nearest_correct = 0
    own_margin_vals = []
    for start in range(0, n, 2048):
        batch = x[start:start + 2048]
        batch_labels = y[start:start + 2048]
        batch_d = torch.cdist(batch, centroid_t, p=2)
        nearest_idx = batch_d.argmin(dim=1)
        nearest_cls = torch.tensor(present, dtype=batch_labels.dtype)[nearest_idx]
        own_nearest_correct += int((nearest_cls == batch_labels).sum().item())
        for row, label in zip(batch_d, batch_labels):
            cls_pos = present.index(int(label.item()))
            own = row[cls_pos]
            masked = row.clone()
            masked[cls_pos] = float("inf")
            own_margin_vals.append(float(masked.min().item() - own.item()))
    own_margin = torch.tensor(own_margin_vals)

    cov_x = x - global_center
    cov = cov_x.t().matmul(cov_x) / max(n - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    eig_sum = eig.sum().item()
    if eig_sum > 0:
        p = eig / eig_sum
        eff_rank = float(torch.exp(-(p[p > 0] * torch.log(p[p > 0])).sum()).item())
    else:
        eff_rank = 0.0
    return {
        "feature_dim": int(x.shape[1]),
        "feature_norm_mean": float(norms.mean().item()),
        "feature_norm_std": float(norms.std(unbiased=False).item()),
        "feature_norm_cv": float(norms.std(unbiased=False).item() / max(norms.mean().item(), 1e-12)),
        "within_class_mse": float(within.item()),
        "between_centroid_mse": float(between.item()),
        "fisher_ratio": float((between / within.clamp_min(1e-12)).item()),
        "centroid_dist_mean": float(upper.mean().item()) if upper.numel() else 0.0,
        "centroid_dist_min": float(upper.min().item()) if upper.numel() else 0.0,
        "nearest_centroid_acc": float(own_nearest_correct / max(n, 1)),
        "own_centroid_margin_mean": float(own_margin.mean().item()),
        "own_centroid_margin_p05": float(torch.quantile(own_margin, 0.05).item()),
        "effective_rank": eff_rank,
    }


def dominant_margin_rows(domain: str, result: dict, top_class: int) -> list[dict]:
    logits = result["logits"]
    rows = []
    for cls in range(logits.shape[1]):
        if cls == top_class:
            continue
        delta = logits[:, top_class] - logits[:, cls]
        rows.append({
            "domain": domain,
            "dominant_class": top_class,
            "compare": f"{top_class}-{cls}",
            "mean_delta": f"{float(delta.mean().item()):.6f}",
            "std_delta": f"{float(delta.std(unbiased=False).item()):.6f}",
            "mean_over_std": f"{float(delta.mean().item() / max(delta.std(unbiased=False).item(), 1e-12)):.6f}",
            "min_delta": f"{float(delta.min().item()):.6f}",
            "p05_delta": f"{float(torch.quantile(delta, 0.05).item()):.6f}",
            "pct_positive": f"{float((delta > 0).float().mean().item()):.6f}",
        })
    return rows


def classifier_tensors(state: OrderedDict, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight_key = None
    bias_key = None
    for key, value in state.items():
        if value.ndim == 2 and value.shape[0] == num_classes and torch.is_floating_point(value):
            weight_key = key
        if value.ndim == 1 and value.shape[0] == num_classes and torch.is_floating_point(value):
            bias_key = key
    if weight_key is None:
        raise RuntimeError("classifier weight not found")
    w = state[weight_key].detach().cpu().float()
    b = state[bias_key].detach().cpu().float() if bias_key else torch.zeros(num_classes)
    return w, b


def head_on_avg_feature_rows(
    domain: str,
    states: list[OrderedDict],
    avg_state: OrderedDict,
    avg_features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    dominant_class: int,
) -> list[dict]:
    rows = []
    all_states = [(f"client_{i}", state) for i, state in enumerate(states)] + [("avg_head", avg_state)]
    for name, state in all_states:
        w, b = classifier_tensors(state, num_classes)
        logits = avg_features @ w.t() + b
        preds = logits.argmax(dim=1)
        pred_counts = torch.bincount(preds, minlength=num_classes).tolist()
        top_class = int(torch.tensor(pred_counts).argmax().item())
        rows.append({
            "domain": domain,
            "head": name,
            "top_class": top_class,
            "collapse_ratio": f"{max(pred_counts) / max(int(labels.numel()), 1):.6f}",
            "effective_pred_classes": int(sum(1 for c in pred_counts if c > 0)),
            "balanced_accuracy": f"{balanced_accuracy(preds, labels, num_classes):.6f}",
            "dominant_logit_mean": f"{float(logits[:, dominant_class].mean().item()):.6f}",
            "dominant_pred_fraction": f"{float((preds == dominant_class).float().mean().item()):.6f}",
            "pred_counts": json.dumps(pred_counts, ensure_ascii=False),
        })
    return rows


def logit_decomposition_rows(
    domain: str,
    avg_state: OrderedDict,
    avg_features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    dominant_class: int,
) -> tuple[list[dict], list[dict]]:
    w, b = classifier_tensors(avg_state, num_classes)
    logits = avg_features @ w.t() + b
    preds = logits.argmax(dim=1)
    pred_counts = torch.bincount(preds, minlength=num_classes).tolist()
    true_counts = torch.bincount(labels, minlength=num_classes).tolist()

    class_rows = []
    for cls in range(num_classes):
        logit = logits[:, cls]
        projection = avg_features @ w[cls]
        class_rows.append({
            "domain": domain,
            "class": cls,
            "is_dominant": int(cls == dominant_class),
            "weight_norm": f"{float(w[cls].norm().item()):.6f}",
            "bias": f"{float(b[cls].item()):.6f}",
            "projection_mean": f"{float(projection.mean().item()):.6f}",
            "projection_std": f"{float(projection.std(unbiased=False).item()):.6f}",
            "logit_mean": f"{float(logit.mean().item()):.6f}",
            "logit_std": f"{float(logit.std(unbiased=False).item()):.6f}",
            "logit_p05": f"{float(torch.quantile(logit, 0.05).item()):.6f}",
            "logit_p95": f"{float(torch.quantile(logit, 0.95).item()):.6f}",
            "pred_count": int(pred_counts[cls]),
            "true_count": int(true_counts[cls]),
        })

    pair_rows = []
    for cls in range(num_classes):
        if cls == dominant_class:
            continue
        projection_delta = avg_features @ (w[dominant_class] - w[cls])
        bias_delta = b[dominant_class] - b[cls]
        total_delta = projection_delta + bias_delta
        pair_rows.append({
            "domain": domain,
            "dominant_class": dominant_class,
            "compare": f"{dominant_class}-{cls}",
            "projection_mean": f"{float(projection_delta.mean().item()):.6f}",
            "projection_std": f"{float(projection_delta.std(unbiased=False).item()):.6f}",
            "projection_p05": f"{float(torch.quantile(projection_delta, 0.05).item()):.6f}",
            "projection_min": f"{float(projection_delta.min().item()):.6f}",
            "bias_delta": f"{float(bias_delta.item()):.6f}",
            "total_mean": f"{float(total_delta.mean().item()):.6f}",
            "total_std": f"{float(total_delta.std(unbiased=False).item()):.6f}",
            "total_mean_over_std": f"{float(total_delta.mean().item() / max(total_delta.std(unbiased=False).item(), 1e-12)):.6f}",
            "total_p05": f"{float(torch.quantile(total_delta, 0.05).item()):.6f}",
            "total_min": f"{float(total_delta.min().item()):.6f}",
            "pct_total_positive": f"{float((total_delta > 0).float().mean().item()):.6f}",
        })
    return class_rows, pair_rows


def param_group(name: str) -> str:
    if name.startswith("fc.") or ".fc." in name or name.startswith("head.") or name.startswith("classifier."):
        return "classifier"
    if "running_mean" in name or "running_var" in name:
        return "bn_running_stats"
    if ".bn" in name or name.startswith("bn"):
        return "bn_affine"
    if name.startswith("layer1"):
        return "layer1"
    if name.startswith("layer2"):
        return "layer2"
    if name.startswith("layer3"):
        return "layer3"
    if name.startswith("layer4"):
        return "layer4"
    if name.startswith("conv1"):
        return "stem_conv"
    return "other"


def parameter_dispersion(domain: str, states: list[OrderedDict], avg_state: OrderedDict) -> list[dict]:
    accum: dict[str, dict[str, float]] = {}
    for key, avg_value in avg_state.items():
        if not torch.is_tensor(avg_value) or not torch.is_floating_point(avg_value):
            continue
        values = [state[key].detach().cpu().float().flatten() for state in states]
        avg_flat = avg_value.detach().cpu().float().flatten()
        client_norm = torch.stack([v.norm() for v in values]).mean().item()
        delta_norm = torch.stack([(v - avg_flat).norm() for v in values]).mean().item()
        rel = delta_norm / max(client_norm, 1e-12)
        group = param_group(key)
        item = accum.setdefault(group, {"weighted_rel_sum": 0.0, "param_count": 0.0, "tensor_count": 0.0, "delta_norm_sum": 0.0})
        count = float(avg_flat.numel())
        item["weighted_rel_sum"] += rel * count
        item["param_count"] += count
        item["tensor_count"] += 1.0
        item["delta_norm_sum"] += delta_norm
    rows = []
    for group, vals in sorted(accum.items()):
        rows.append({
            "domain": domain,
            "group": group,
            "weighted_relative_l2_to_avg": f"{vals['weighted_rel_sum'] / max(vals['param_count'], 1.0):.6f}",
            "mean_delta_norm_per_tensor": f"{vals['delta_norm_sum'] / max(vals['tensor_count'], 1.0):.6f}",
            "param_count": int(vals["param_count"]),
            "tensor_count": int(vals["tensor_count"]),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def compact_model_row(domain: str, result: dict) -> dict:
    return {
        "domain": domain,
        "accuracy": f"{result['accuracy']:.6f}",
        "balanced_accuracy": f"{result['balanced_accuracy']:.6f}",
        "collapse_ratio": f"{result['collapse_ratio']:.6f}",
        "effective_pred_classes": result["effective_pred_classes"],
        "top_class": result["top_class"],
        "prediction_entropy_norm": f"{result['prediction_entropy_norm']:.6f}",
        "mean_confidence": f"{result['mean_confidence']:.6f}",
        "mean_top1_top2_margin": f"{result['mean_top1_top2_margin']:.6f}",
        "min_top1_top2_margin": f"{result['min_top1_top2_margin']:.6f}",
        "true_counts": json.dumps(result["true_counts"], ensure_ascii=False),
        "pred_counts": json.dumps(result["pred_counts"], ensure_ascii=False),
    }


def load_case(case_dir: Path) -> tuple[dict, list[OrderedDict], OrderedDict]:
    meta = load_json(case_dir / "meta.json")
    states = []
    for idx in range(int(meta["num_clients"])):
        ckpt = load_checkpoint(case_dir / f"client_{idx}.pt", device="cpu")
        states.append(ckpt["state_dict"])
    avg_state, _weights = average_state_dicts(states, [1.0] * len(states))
    return meta, states, avg_state


def process_case(
    *,
    domain: str,
    case_dir: Path,
    data_root: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict:
    meta, states, avg_state = load_case(case_dir)
    avg_result = run_state(
        meta=meta,
        state_dict=avg_state,
        data_root=data_root,
        split="test",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        collect_stages=True,
    )
    client_rows = []
    for idx, state in enumerate(states):
        result = run_state(
            meta=meta,
            state_dict=state,
            data_root=data_root,
            split="test",
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            collect_stages=False,
        )
        row = compact_model_row(f"{domain}:client_{idx}", result)
        row["domain"] = domain
        row["model"] = f"client_{idx}"
        client_rows.append(row)

    stage_rows = []
    labels = avg_result["labels"]
    num_classes = int(meta["num_classes"])
    for stage_name, features in avg_result["stages"].items():
        geom = feature_geometry(features, labels, num_classes)
        stage_rows.append({
            "domain": domain,
            "stage": stage_name,
            **{k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in geom.items()},
        })
    final_features = avg_result["stages"]["classifier_input"]
    class_decomp_rows, pair_decomp_rows = logit_decomposition_rows(
        domain,
        avg_state,
        final_features,
        labels,
        num_classes,
        int(avg_result["top_class"]),
    )
    return {
        "meta": meta,
        "states": states,
        "avg_state": avg_state,
        "avg_result": avg_result,
        "client_rows": client_rows,
        "stage_rows": stage_rows,
        "margin_rows": dominant_margin_rows(domain, avg_result, int(avg_result["top_class"])),
        "head_rows": head_on_avg_feature_rows(
            domain,
            states,
            avg_state,
            final_features,
            labels,
            num_classes,
            int(avg_result["top_class"]),
        ),
        "class_decomp_rows": class_decomp_rows,
        "pair_decomp_rows": pair_decomp_rows,
        "param_rows": parameter_dispersion(domain, states, avg_state),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    medical = process_case(
        domain="medical_derma",
        case_dir=Path(args.medical_case_dir),
        data_root=args.data_root,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    natural = process_case(
        domain="natural_cifar_imbalanced",
        case_dir=Path(args.natural_case_dir),
        data_root=args.data_root,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_rows = [
        compact_model_row("medical_derma:avg", medical["avg_result"]),
        compact_model_row("natural_cifar_imbalanced:avg", natural["avg_result"]),
    ]
    model_rows[0]["domain"] = "medical_derma"
    model_rows[0]["model"] = "avg"
    model_rows[1]["domain"] = "natural_cifar_imbalanced"
    model_rows[1]["model"] = "avg"
    model_rows.extend(medical["client_rows"])
    model_rows.extend(natural["client_rows"])
    stage_rows = medical["stage_rows"] + natural["stage_rows"]
    margin_rows = medical["margin_rows"] + natural["margin_rows"]
    head_rows = medical["head_rows"] + natural["head_rows"]
    class_decomp_rows = medical["class_decomp_rows"] + natural["class_decomp_rows"]
    pair_decomp_rows = medical["pair_decomp_rows"] + natural["pair_decomp_rows"]
    param_rows = medical["param_rows"] + natural["param_rows"]

    write_csv(out_dir / "model_output_comparison.csv", model_rows)
    write_csv(out_dir / "avg_stage_feature_geometry.csv", stage_rows)
    write_csv(out_dir / "avg_dominant_margin_comparison.csv", margin_rows)
    write_csv(out_dir / "head_on_avg_feature_comparison.csv", head_rows)
    write_csv(out_dir / "avg_class_logit_decomposition.csv", class_decomp_rows)
    write_csv(out_dir / "avg_pairwise_logit_decomposition.csv", pair_decomp_rows)
    write_csv(out_dir / "parameter_dispersion_comparison.csv", param_rows)

    final_stage_rows = [row for row in stage_rows if row["stage"] == "classifier_input"]
    key_margin_rows = []
    for row in margin_rows:
        if row["domain"] == "medical_derma" and row["compare"] in {"5-2", "5-4", "5-6"}:
            key_margin_rows.append(row)
        if row["domain"] == "natural_cifar_imbalanced" and row["compare"] in {"8-1", "8-3", "8-9"}:
            key_margin_rows.append(row)
    key_pair_decomp_rows = []
    for row in pair_decomp_rows:
        if row["domain"] == "medical_derma" and row["compare"] in {"5-2", "5-4", "5-6"}:
            key_pair_decomp_rows.append(row)
        if row["domain"] == "natural_cifar_imbalanced" and row["compare"] in {"8-1", "8-3", "8-9"}:
            key_pair_decomp_rows.append(row)
    key_class_rows = [
        row for row in class_decomp_rows
        if (row["domain"] == "medical_derma" and row["class"] in {2, 4, 5, 6})
        or (row["domain"] == "natural_cifar_imbalanced" and row["class"] in {1, 3, 8, 9})
    ]
    key_param_rows = [row for row in param_rows if row["group"] in {"classifier", "bn_running_stats", "layer4"}]

    md = [
        "# Medical vs Natural Collapse Mechanism Comparison",
        "",
        f"- Medical case: `{args.medical_case_dir}`",
        f"- Natural-image case: `{args.natural_case_dir}`",
        f"- Data root: `{args.data_root}`",
        f"- Device: `{device}`",
        "",
        "## Model Outputs",
        "",
        markdown_table(
            [row for row in model_rows if row["model"] == "avg"],
            [
                "domain",
                "model",
                "accuracy",
                "balanced_accuracy",
                "collapse_ratio",
                "effective_pred_classes",
                "top_class",
                "prediction_entropy_norm",
                "mean_top1_top2_margin",
                "pred_counts",
            ],
        ),
        "",
        "## Final Feature Geometry",
        "",
        markdown_table(
            final_stage_rows,
            [
                "domain",
                "stage",
                "feature_norm_mean",
                "feature_norm_cv",
                "within_class_mse",
                "between_centroid_mse",
                "fisher_ratio",
                "centroid_dist_mean",
                "nearest_centroid_acc",
                "own_centroid_margin_p05",
                "effective_rank",
            ],
        ),
        "",
        "## Dominant-Class Margin",
        "",
        markdown_table(
            key_margin_rows,
            [
                "domain",
                "dominant_class",
                "compare",
                "mean_delta",
                "std_delta",
                "mean_over_std",
                "min_delta",
                "p05_delta",
                "pct_positive",
            ],
        ),
        "",
        "## Class Logit Decomposition",
        "",
        markdown_table(
            key_class_rows,
            [
                "domain",
                "class",
                "is_dominant",
                "weight_norm",
                "bias",
                "projection_mean",
                "projection_std",
                "logit_mean",
                "logit_std",
                "logit_p05",
                "logit_p95",
                "pred_count",
                "true_count",
            ],
        ),
        "",
        "## Pairwise Logit Decomposition",
        "",
        markdown_table(
            key_pair_decomp_rows,
            [
                "domain",
                "dominant_class",
                "compare",
                "projection_mean",
                "projection_std",
                "projection_p05",
                "projection_min",
                "bias_delta",
                "total_mean",
                "total_std",
                "total_p05",
                "total_min",
                "pct_total_positive",
            ],
        ),
        "",
        "## Heads on AVG Features",
        "",
        markdown_table(
            head_rows,
            [
                "domain",
                "head",
                "top_class",
                "collapse_ratio",
                "effective_pred_classes",
                "balanced_accuracy",
                "dominant_pred_fraction",
                "pred_counts",
            ],
        ),
        "",
        "## Parameter Dispersion",
        "",
        markdown_table(
            key_param_rows,
            [
                "domain",
                "group",
                "weighted_relative_l2_to_avg",
                "mean_delta_norm_per_tensor",
                "param_count",
                "tensor_count",
            ],
        ),
        "",
        "## Reading",
        "",
        "- The decisive diagnostic is the dominant-class margin distribution, not the existence of non-IID itself.",
        "- In medical Derma, the dominant class has positive margin against close competitors on every test sample; in imbalanced CIFAR, the dominant class has negative low-tail margins against several competitors, so other classes can still win.",
        "- The final-feature table checks whether the averaged representation preserves class-dependent variation. The margin table checks whether that variation is large enough to overcome the dominant-class offset.",
    ]
    (out_dir / "domain_collapse_mechanism_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'domain_collapse_mechanism_comparison.md'}")


if __name__ == "__main__":
    main()
