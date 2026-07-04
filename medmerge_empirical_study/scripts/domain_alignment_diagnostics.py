#!/usr/bin/env python3
"""Run alignment diagnostics for medical-vs-natural collapse mechanisms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.io import load_checkpoint, load_json
from utils.runtime import build_reference_state, build_runtime
from utils.state_dict import average_state_dicts


DEFAULT_MEDICAL = "model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42"
DEFAULT_NATURAL = (
    "medmerge_empirical_study/results/experiment12_natural_image_3client_imbalance_probe/"
    "natural_probe_hub/small/cifar10_32/resnet/clients_3/partial_label/seed_42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medical-case-dir", default=DEFAULT_MEDICAL)
    parser.add_argument("--natural-case-dir", default=DEFAULT_NATURAL)
    parser.add_argument("--data-root", default="Med_data")
    parser.add_argument("--out-dir", default="medmerge_empirical_study/results/experiment13_alignment_diagnostics")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cka-max-samples", type=int, default=2000)
    parser.add_argument("--linear-epochs", type=int, default=80)
    parser.add_argument("--linear-lr", type=float, default=0.05)
    parser.add_argument("--linear-weight-decay", type=float, default=1e-4)
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


def classifier_tensors(state: OrderedDict, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight_key = None
    bias_key = None
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == num_classes and torch.is_floating_point(value):
            weight_key = key
        if torch.is_tensor(value) and value.ndim == 1 and value.shape[0] == num_classes and torch.is_floating_point(value):
            bias_key = key
    if weight_key is None:
        raise RuntimeError("classifier weight not found")
    w = state[weight_key].detach().cpu().float()
    b = state[bias_key].detach().cpu().float() if bias_key else torch.zeros(num_classes)
    return w, b


def pooled_activation(output: torch.Tensor) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 4:
        return output.detach().float().mean(dim=(2, 3)).cpu()
    return output.detach().float().flatten(1).cpu()


def balanced_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    vals = []
    for cls in range(num_classes):
        mask = labels == cls
        support = int(mask.sum().item())
        vals.append(float(((preds == cls) & mask).sum().item() / support) if support else 0.0)
    return float(sum(vals) / max(num_classes, 1))


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    vals = []
    for cls in range(num_classes):
        tp = int(((preds == cls) & (labels == cls)).sum().item())
        fp = int(((preds == cls) & (labels != cls)).sum().item())
        fn = int(((preds != cls) & (labels == cls)).sum().item())
        denom = 2 * tp + fp + fn
        vals.append(float(2 * tp / denom) if denom else 0.0)
    return float(sum(vals) / max(num_classes, 1))


def output_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    preds = logits.argmax(dim=1)
    pred_counts = torch.bincount(preds, minlength=num_classes).tolist()
    sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
    margins = sorted_logits[:, 0] - sorted_logits[:, 1]
    total = max(int(labels.numel()), 1)
    return {
        "accuracy": float((preds == labels).float().mean().item()),
        "balanced_accuracy": balanced_accuracy(preds, labels, num_classes),
        "macro_f1": macro_f1(preds, labels, num_classes),
        "collapse_ratio": max(pred_counts) / total,
        "effective_pred_classes": int(sum(1 for x in pred_counts if x > 0)),
        "top_class": int(torch.tensor(pred_counts).argmax().item()),
        "mean_top1_top2_margin": float(margins.mean().item()),
        "pred_counts": pred_counts,
    }


def load_case(case_dir: Path) -> tuple[dict, list[OrderedDict], OrderedDict]:
    meta = load_json(case_dir / "meta.json")
    states = []
    for idx in range(int(meta["num_clients"])):
        ckpt = load_checkpoint(case_dir / f"client_{idx}.pt", device="cpu")
        states.append(ckpt["state_dict"])
    avg_state, _weights = average_state_dicts(states, [1.0] * len(states))
    return meta, states, avg_state


@torch.no_grad()
def collect_features(
    *,
    meta: dict,
    state_dict: OrderedDict,
    data_root: str,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    stage_names: list[str],
    max_samples: int = 0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    runtime = build_runtime(meta, data_root, split, batch_size, num_workers, device)
    model = runtime["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    forward_fn = runtime["forward_fn"]
    modules = dict(model.named_modules())
    _classifier_name, classifier = get_classifier(model)
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in stage_names}
    label_chunks = []
    handles = []

    for name in stage_names:
        if name == "classifier_input":
            continue
        module = modules.get(name)
        if module is None:
            continue

        def make_hook(stage_name: str):
            def hook(_module, _inputs, output):
                chunks[stage_name].append(pooled_activation(output))

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    if "classifier_input" in stage_names:
        def classifier_hook(_module, inputs, _output):
            chunks["classifier_input"].append(inputs[0].detach().float().cpu())

        handles.append(classifier.register_forward_hook(classifier_hook))

    total = 0
    for x, y in runtime["loader"]:
        x = x.to(device, non_blocking=True)
        _ = forward_fn(model, x)
        label_chunks.append(y.detach().cpu())
        total += int(y.numel())
        if max_samples > 0 and total >= max_samples:
            break

    for handle in handles:
        handle.remove()

    labels = torch.cat(label_chunks, dim=0)
    features = {name: torch.cat(parts, dim=0) for name, parts in chunks.items() if parts}
    if max_samples > 0:
        labels = labels[:max_samples]
        features = {name: value[:max_samples] for name, value in features.items()}
    return features, labels


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = x.t().matmul(y)
    xtx = x.t().matmul(x)
    yty = y.t().matmul(y)
    numerator = xty.pow(2).sum()
    denom = torch.sqrt(xtx.pow(2).sum() * yty.pow(2).sum()).clamp_min(1e-12)
    return float((numerator / denom).item())


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


def task_vector_rows(domain: str, meta: dict, states: list[OrderedDict]) -> list[dict]:
    reference = build_reference_state(meta, device="cpu")
    groups = sorted({param_group(k) for k, v in states[0].items() if torch.is_tensor(v) and torch.is_floating_point(v)})
    rows = []
    for group in groups:
        deltas = []
        keys = [
            k for k, v in states[0].items()
            if torch.is_tensor(v)
            and torch.is_floating_point(v)
            and k in reference
            and torch.is_tensor(reference[k])
            and reference[k].shape == v.shape
            and param_group(k) == group
        ]
        if not keys:
            continue
        for state in states:
            parts = [(state[k].detach().cpu().float() - reference[k].detach().cpu().float()).flatten() for k in keys]
            deltas.append(torch.cat(parts, dim=0))
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                cosine = F.cosine_similarity(deltas[i], deltas[j], dim=0).item()
                rows.append({
                    "domain": domain,
                    "group": group,
                    "pair": f"client_{i}-client_{j}",
                    "cosine": f"{float(cosine):.6f}",
                    "delta_norm_i": f"{float(deltas[i].norm().item()):.6f}",
                    "delta_norm_j": f"{float(deltas[j].norm().item()):.6f}",
                    "num_params": int(deltas[i].numel()),
                })
    return rows


def cka_rows(
    *,
    domain: str,
    meta: dict,
    states: list[OrderedDict],
    avg_state: OrderedDict,
    data_root: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int,
) -> list[dict]:
    stage_names = ["bn1", "layer1", "layer2", "layer3", "layer4", "classifier_input"]
    model_states = [(f"client_{i}", state) for i, state in enumerate(states)] + [("avg", avg_state)]
    all_features = {}
    n_samples = 0
    for name, state in model_states:
        feats, labels = collect_features(
            meta=meta,
            state_dict=state,
            data_root=data_root,
            split="test",
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            stage_names=stage_names,
            max_samples=max_samples,
        )
        all_features[name] = feats
        n_samples = int(labels.numel())
    rows = []
    for stage in stage_names:
        for i in range(len(model_states)):
            for j in range(i + 1, len(model_states)):
                left = model_states[i][0]
                right = model_states[j][0]
                if stage not in all_features[left] or stage not in all_features[right]:
                    continue
                relation = "client-client" if left != "avg" and right != "avg" else "client-avg"
                rows.append({
                    "domain": domain,
                    "stage": stage,
                    "pair": f"{left}-{right}",
                    "relation": relation,
                    "cka": f"{linear_cka(all_features[left][stage], all_features[right][stage]):.6f}",
                    "num_samples": n_samples,
                    "feature_dim": int(all_features[left][stage].shape[1]),
                })
    return rows


def head_feature_swap_rows(
    *,
    domain: str,
    meta: dict,
    states: list[OrderedDict],
    avg_state: OrderedDict,
    data_root: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> list[dict]:
    num_classes = int(meta["num_classes"])
    feature_sources = [(f"client_{i}", state) for i, state in enumerate(states)] + [("avg", avg_state)]
    head_sources = [(f"client_{i}", state) for i, state in enumerate(states)] + [("avg", avg_state)]
    feature_cache = {}
    labels = None
    for name, state in feature_sources:
        feats, got_labels = collect_features(
            meta=meta,
            state_dict=state,
            data_root=data_root,
            split="test",
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            stage_names=["classifier_input"],
            max_samples=0,
        )
        feature_cache[name] = feats["classifier_input"]
        labels = got_labels
    assert labels is not None
    rows = []
    for feat_name, features in feature_cache.items():
        for head_name, state in head_sources:
            w, b = classifier_tensors(state, num_classes)
            logits = features @ w.t() + b
            metrics = output_metrics(logits, labels, num_classes)
            rows.append({
                "domain": domain,
                "feature_source": feat_name,
                "head_source": head_name,
                "is_matched_client": int(feat_name == head_name and feat_name.startswith("client_")),
                "is_avg_avg": int(feat_name == "avg" and head_name == "avg"),
                "accuracy": f"{metrics['accuracy']:.6f}",
                "balanced_accuracy": f"{metrics['balanced_accuracy']:.6f}",
                "macro_f1": f"{metrics['macro_f1']:.6f}",
                "collapse_ratio": f"{metrics['collapse_ratio']:.6f}",
                "effective_pred_classes": metrics["effective_pred_classes"],
                "top_class": metrics["top_class"],
                "mean_top1_top2_margin": f"{metrics['mean_top1_top2_margin']:.6f}",
                "pred_counts": json.dumps(metrics["pred_counts"], ensure_ascii=False),
            })
    return rows


def train_linear_probe(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std
    model = torch.nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    best = None
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(yb.numel())
            total += int(yb.numel())
        with torch.no_grad():
            val_logits = model(val_x.to(device)).detach().cpu()
        val_metrics = output_metrics(val_logits, val_y, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_collapse_ratio": val_metrics["collapse_ratio"],
        }
        history.append(row)
        if best is None or val_metrics["balanced_accuracy"] > best["val_balanced_accuracy"]:
            best = row
            best_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
    assert best_state is not None and best is not None
    model.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        test_logits = model(test_x.to(device)).detach().cpu()
    test_metrics = output_metrics(test_logits, test_y, num_classes)
    return {
        "best_epoch": int(best["epoch"]),
        "best_val_accuracy": float(best["val_accuracy"]),
        "best_val_balanced_accuracy": float(best["val_balanced_accuracy"]),
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_collapse_ratio": test_metrics["collapse_ratio"],
        "test_effective_pred_classes": test_metrics["effective_pred_classes"],
        "test_top_class": test_metrics["top_class"],
        "test_pred_counts": test_metrics["pred_counts"],
    }, history


def linear_probe_rows(
    *,
    domain: str,
    meta: dict,
    avg_state: OrderedDict,
    data_root: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[list[dict], list[dict]]:
    train_feats, train_labels = collect_features(
        meta=meta,
        state_dict=avg_state,
        data_root=data_root,
        split="train",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        stage_names=["classifier_input"],
    )
    val_feats, val_labels = collect_features(
        meta=meta,
        state_dict=avg_state,
        data_root=data_root,
        split="val",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        stage_names=["classifier_input"],
    )
    test_feats, test_labels = collect_features(
        meta=meta,
        state_dict=avg_state,
        data_root=data_root,
        split="test",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        stage_names=["classifier_input"],
    )
    num_classes = int(meta["num_classes"])
    w, b = classifier_tensors(avg_state, num_classes)
    original_logits = test_feats["classifier_input"] @ w.t() + b
    original = output_metrics(original_logits, test_labels, num_classes)
    probe, history = train_linear_probe(
        train_x=train_feats["classifier_input"],
        train_y=train_labels,
        val_x=val_feats["classifier_input"],
        val_y=val_labels,
        test_x=test_feats["classifier_input"],
        test_y=test_labels,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=1024,
    )
    rows = [
        {
            "domain": domain,
            "model": "avg_original_head",
            "train_samples": int(train_labels.numel()),
            "val_samples": int(val_labels.numel()),
            "test_samples": int(test_labels.numel()),
            "best_epoch": "",
            "accuracy": f"{original['accuracy']:.6f}",
            "balanced_accuracy": f"{original['balanced_accuracy']:.6f}",
            "macro_f1": f"{original['macro_f1']:.6f}",
            "collapse_ratio": f"{original['collapse_ratio']:.6f}",
            "effective_pred_classes": original["effective_pred_classes"],
            "top_class": original["top_class"],
            "pred_counts": json.dumps(original["pred_counts"], ensure_ascii=False),
        },
        {
            "domain": domain,
            "model": "avg_feature_linear_probe",
            "train_samples": int(train_labels.numel()),
            "val_samples": int(val_labels.numel()),
            "test_samples": int(test_labels.numel()),
            "best_epoch": probe["best_epoch"],
            "accuracy": f"{probe['test_accuracy']:.6f}",
            "balanced_accuracy": f"{probe['test_balanced_accuracy']:.6f}",
            "macro_f1": f"{probe['test_macro_f1']:.6f}",
            "collapse_ratio": f"{probe['test_collapse_ratio']:.6f}",
            "effective_pred_classes": probe["test_effective_pred_classes"],
            "top_class": probe["test_top_class"],
            "pred_counts": json.dumps(probe["test_pred_counts"], ensure_ascii=False),
        },
    ]
    hist_rows = [{"domain": domain, **row} for row in history]
    return rows, hist_rows


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


def summarize_numeric(rows: list[dict], group_keys: list[str], value_key: str) -> list[dict]:
    acc: dict[tuple, list[float]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        acc.setdefault(key, []).append(float(row[value_key]))
    out = []
    for key, vals in sorted(acc.items()):
        item = {k: v for k, v in zip(group_keys, key)}
        item[f"mean_{value_key}"] = f"{sum(vals) / len(vals):.6f}"
        item["num_pairs"] = len(vals)
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    cases = [
        ("medical_derma", Path(args.medical_case_dir)),
        ("natural_cifar_3client", Path(args.natural_case_dir)),
    ]
    all_cka_rows = []
    all_task_rows = []
    all_swap_rows = []
    all_probe_rows = []
    all_probe_history = []
    for domain, case_dir in cases:
        print(f"[INFO] processing {domain}: {case_dir}", flush=True)
        meta, states, avg_state = load_case(case_dir)
        print(f"[INFO] {domain}: CKA", flush=True)
        all_cka_rows.extend(
            cka_rows(
                domain=domain,
                meta=meta,
                states=states,
                avg_state=avg_state,
                data_root=args.data_root,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.cka_max_samples,
            )
        )
        print(f"[INFO] {domain}: task-vector cosine", flush=True)
        all_task_rows.extend(task_vector_rows(domain, meta, states))
        print(f"[INFO] {domain}: head-feature swaps", flush=True)
        all_swap_rows.extend(
            head_feature_swap_rows(
                domain=domain,
                meta=meta,
                states=states,
                avg_state=avg_state,
                data_root=args.data_root,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )
        print(f"[INFO] {domain}: linear probe", flush=True)
        probe_rows, probe_history = linear_probe_rows(
            domain=domain,
            meta=meta,
            avg_state=avg_state,
            data_root=args.data_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.linear_epochs,
            lr=args.linear_lr,
            weight_decay=args.linear_weight_decay,
        )
        all_probe_rows.extend(probe_rows)
        all_probe_history.extend(probe_history)

    write_csv(out_dir / "feature_cka_rows.csv", all_cka_rows)
    write_csv(out_dir / "task_vector_cosine_rows.csv", all_task_rows)
    write_csv(out_dir / "head_feature_swap_rows.csv", all_swap_rows)
    write_csv(out_dir / "linear_probe_rows.csv", all_probe_rows)
    write_csv(out_dir / "linear_probe_history.csv", all_probe_history)

    cka_summary = summarize_numeric(
        [row for row in all_cka_rows if row["relation"] == "client-client"],
        ["domain", "stage"],
        "cka",
    )
    task_summary = summarize_numeric(all_task_rows, ["domain", "group"], "cosine")
    write_csv(out_dir / "feature_cka_summary.csv", cka_summary)
    write_csv(out_dir / "task_vector_cosine_summary.csv", task_summary)

    swap_key_rows = [
        row for row in all_swap_rows
        if row["is_matched_client"] == 1 or row["is_avg_avg"] == 1 or row["feature_source"] == "avg"
    ]
    md = [
        "# Domain Alignment Diagnostics",
        "",
        f"- Medical case: `{args.medical_case_dir}`",
        f"- Natural case: `{args.natural_case_dir}`",
        f"- Device: `{device}`",
        f"- CKA max samples: `{args.cka_max_samples}`",
        "",
        "## Feature CKA Summary",
        "",
        markdown_table(
            cka_summary,
            ["domain", "stage", "mean_cka", "num_pairs"],
        ),
        "",
        "## Task-Vector Cosine Summary",
        "",
        markdown_table(
            [row for row in task_summary if row["group"] in {"classifier", "bn_running_stats", "layer3", "layer4"}],
            ["domain", "group", "mean_cosine", "num_pairs"],
        ),
        "",
        "## Head-Feature Swap Key Rows",
        "",
        markdown_table(
            swap_key_rows,
            [
                "domain",
                "feature_source",
                "head_source",
                "is_matched_client",
                "is_avg_avg",
                "accuracy",
                "balanced_accuracy",
                "collapse_ratio",
                "effective_pred_classes",
                "top_class",
                "pred_counts",
            ],
        ),
        "",
        "## Linear Probe",
        "",
        markdown_table(
            all_probe_rows,
            [
                "domain",
                "model",
                "train_samples",
                "best_epoch",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "collapse_ratio",
                "effective_pred_classes",
                "top_class",
                "pred_counts",
            ],
        ),
        "",
        "## Reading",
        "",
        "- CKA tests whether clients use aligned hidden representations at the same layer.",
        "- Task-vector cosine tests whether clients move away from the common reference in compatible parameter directions.",
        "- Head-feature swaps test whether one client's classifier can interpret another feature extractor or the averaged feature extractor.",
        "- Linear probe freezes the averaged feature extractor and retrains only a linear classifier; recovery means the averaged feature still contains class information, while failure means the feature itself is badly damaged.",
    ]
    (out_dir / "domain_alignment_diagnostics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'domain_alignment_diagnostics.md'}", flush=True)


if __name__ == "__main__":
    main()
