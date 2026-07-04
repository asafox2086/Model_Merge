#!/usr/bin/env python3
"""Diagnose why a merged classifier collapses to a single class.

The script focuses on one checkpoint directory and compares equal-weight AVG
against the original clients on source and public test sets. It also decomposes
the final linear head as z = Wf + b for the merged model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.io import load_checkpoint, load_json
from utils.runtime import build_runtime
from utils.state_dict import average_state_dicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        default="model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42",
    )
    parser.add_argument("--source-root", default="Med_data")
    parser.add_argument("--public-root", default="PublicMedFingerprint_data")
    parser.add_argument("--out-dir", default="medmerge_empirical_study/results/experiment6_collapse_mechanism")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    values = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()
        denom = (2 * tp + fp + fn)
        values.append((2 * tp / denom) if denom else 0.0)
    return float(sum(values) / max(len(values), 1))


def balanced_acc(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    recalls = []
    for cls in range(num_classes):
        support = (labels == cls).sum().item()
        if support:
            recalls.append(((preds == cls) & (labels == cls)).sum().item() / support)
        else:
            recalls.append(0.0)
    return float(sum(recalls) / max(len(recalls), 1))


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
    raise RuntimeError("Could not locate a final Linear classifier")


def run_model(
    *,
    meta: dict,
    state_dict: OrderedDict,
    data_root: str,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    collect_features: bool,
) -> dict:
    runtime = build_runtime(meta, data_root, split, batch_size, num_workers, device)
    model = runtime["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    forward_fn = runtime["forward_fn"]
    classifier_name, classifier = get_classifier(model)

    feature_chunks = []
    handle = None
    if collect_features:
        def hook(_module, inputs, _output):
            feature_chunks.append(inputs[0].detach().cpu())

        handle = classifier.register_forward_hook(hook)

    logits_chunks = []
    label_chunks = []
    with torch.no_grad():
        for x, y in runtime["loader"]:
            x = x.to(device, non_blocking=True)
            logits = forward_fn(model, x)
            logits_chunks.append(logits.detach().cpu())
            label_chunks.append(y.detach().cpu())

    if handle is not None:
        handle.remove()

    logits = torch.cat(logits_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    features = torch.cat(feature_chunks, dim=0) if feature_chunks else None
    preds = logits.argmax(dim=1)
    num_classes = int(meta["num_classes"])
    pred_counts = [int((preds == i).sum().item()) for i in range(num_classes)]
    true_counts = [int((labels == i).sum().item()) for i in range(num_classes)]
    top_pred_class = int(max(range(num_classes), key=lambda i: pred_counts[i]))
    n = int(labels.numel())
    probs = torch.softmax(logits, dim=1)
    sorted_logits, sorted_idx = torch.sort(logits, dim=1, descending=True)
    out = {
        "classifier_name": classifier_name,
        "logits": logits,
        "labels": labels,
        "features": features,
        "preds": preds,
        "n": n,
        "accuracy": float((preds == labels).float().mean().item()),
        "balanced_accuracy": balanced_acc(preds, labels, num_classes),
        "macro_f1": macro_f1(preds, labels, num_classes),
        "collapse_ratio": max(pred_counts) / max(n, 1),
        "effective_pred_classes": int(sum(1 for c in pred_counts if c > 0)),
        "top_pred_class": top_pred_class,
        "top_pred_fraction": pred_counts[top_pred_class] / max(n, 1),
        "pred_counts": pred_counts,
        "true_counts": true_counts,
        "mean_confidence": float(probs.max(dim=1).values.mean().item()),
        "mean_top1_top2_margin": float((sorted_logits[:, 0] - sorted_logits[:, 1]).mean().item()),
        "min_top1_top2_margin": float((sorted_logits[:, 0] - sorted_logits[:, 1]).min().item()),
        "top1_class_hist": [int((sorted_idx[:, 0] == i).sum().item()) for i in range(num_classes)],
    }
    return out


def compact_model_row(model_name: str, dataset_name: str, result: dict) -> dict:
    return {
        "model": model_name,
        "dataset": dataset_name,
        "n": result["n"],
        "accuracy": f"{result['accuracy']:.6f}",
        "balanced_accuracy": f"{result['balanced_accuracy']:.6f}",
        "macro_f1": f"{result['macro_f1']:.6f}",
        "collapse_ratio": f"{result['collapse_ratio']:.6f}",
        "effective_pred_classes": result["effective_pred_classes"],
        "top_pred_class": result["top_pred_class"],
        "top_pred_fraction": f"{result['top_pred_fraction']:.6f}",
        "mean_confidence": f"{result['mean_confidence']:.6f}",
        "mean_top1_top2_margin": f"{result['mean_top1_top2_margin']:.6f}",
        "min_top1_top2_margin": f"{result['min_top1_top2_margin']:.6f}",
        "true_counts": json.dumps(result["true_counts"], ensure_ascii=False),
        "pred_counts": json.dumps(result["pred_counts"], ensure_ascii=False),
    }


def final_layer_tables(model: torch.nn.Module, result: dict, collapse_class: int) -> tuple[list[dict], list[dict]]:
    _name, classifier = get_classifier(model)
    w = classifier.weight.detach().cpu()
    b = classifier.bias.detach().cpu() if classifier.bias is not None else torch.zeros(w.shape[0])
    features = result["features"]
    if features is None:
        raise RuntimeError("Final-layer decomposition requires collected features")
    wf = features @ w.t()
    logits = result["logits"]
    num_classes = w.shape[0]
    per_class = []
    for cls in range(num_classes):
        per_class.append(
            {
                "class": cls,
                "mean_wf": f"{as_float(wf[:, cls].mean()):.6f}",
                "std_wf": f"{as_float(wf[:, cls].std(unbiased=False)):.6f}",
                "bias": f"{as_float(b[cls]):.6f}",
                "mean_logit": f"{as_float(logits[:, cls].mean()):.6f}",
                "std_logit": f"{as_float(logits[:, cls].std(unbiased=False)):.6f}",
                "min_logit": f"{as_float(logits[:, cls].min()):.6f}",
                "max_logit": f"{as_float(logits[:, cls].max()):.6f}",
            }
        )

    pairwise = []
    for cls in range(num_classes):
        if cls == collapse_class:
            continue
        dyn_delta = wf[:, collapse_class] - wf[:, cls]
        total_delta = logits[:, collapse_class] - logits[:, cls]
        bias_delta = b[collapse_class] - b[cls]
        pairwise.append(
            {
                "compare": f"{collapse_class}-{cls}",
                "mean_dynamic_delta": f"{as_float(dyn_delta.mean()):.6f}",
                "std_dynamic_delta": f"{as_float(dyn_delta.std(unbiased=False)):.6f}",
                "bias_delta": f"{as_float(bias_delta):.6f}",
                "mean_total_delta": f"{as_float(total_delta.mean()):.6f}",
                "min_total_delta": f"{as_float(total_delta.min()):.6f}",
                "p05_total_delta": f"{as_float(torch.quantile(total_delta, 0.05)):.6f}",
                "pct_total_delta_positive": f"{float((total_delta > 0).float().mean().item()):.6f}",
            }
        )
    return per_class, pairwise


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    meta = load_json(case_dir / "meta.json")
    client_states = []
    for idx in range(int(meta["num_clients"])):
        ckpt = load_checkpoint(case_dir / f"client_{idx}.pt", device="cpu")
        client_states.append(ckpt["state_dict"])
    avg_state, avg_weights = average_state_dicts(client_states, [1.0] * len(client_states))

    model_rows = []
    detailed = {}
    datasets = {
        "source_test": args.source_root,
        "public_test": args.public_root,
    }
    states = {f"client_{i}": state for i, state in enumerate(client_states)}
    states["avg"] = avg_state

    avg_results = {}
    for model_name, state in states.items():
        detailed[model_name] = {}
        for dataset_name, root in datasets.items():
            result = run_model(
                meta=meta,
                state_dict=state,
                data_root=root,
                split="test",
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                collect_features=(model_name == "avg"),
            )
            detailed[model_name][dataset_name] = {
                key: value for key, value in result.items()
                if key not in {"logits", "labels", "features", "preds"}
            }
            model_rows.append(compact_model_row(model_name, dataset_name, result))
            if model_name == "avg":
                avg_results[dataset_name] = result

    # Rebuild one avg model for final-layer decomposition.
    runtime = build_runtime(meta, args.source_root, "test", args.batch_size, args.num_workers, device)
    avg_model = runtime["model"]
    avg_model.load_state_dict(avg_state, strict=True)
    avg_model.eval()

    per_class_rows = []
    pairwise_rows = []
    for dataset_name, result in avg_results.items():
        collapse_class = int(result["top_pred_class"])
        per_class, pairwise = final_layer_tables(avg_model, result, collapse_class)
        for row in per_class:
            row = {"dataset": dataset_name, **row}
            per_class_rows.append(row)
        for row in pairwise:
            row = {"dataset": dataset_name, **row}
            pairwise_rows.append(row)

    write_csv(out_dir / "model_level_diagnostic.csv", model_rows)
    write_csv(out_dir / "avg_final_layer_per_class.csv", per_class_rows)
    write_csv(out_dir / "avg_class5_pairwise_margin.csv", pairwise_rows)
    with (out_dir / "diagnostic_raw.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "case_dir": str(case_dir),
                "device": str(device),
                "avg_weights": avg_weights,
                "meta": meta,
                "models": detailed,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    focused_model_rows = [
        row for row in model_rows
        if row["model"] in {"client_0", "client_1", "client_2", "avg"}
    ]
    avg_source_summary = next(row for row in model_rows if row["model"] == "avg" and row["dataset"] == "source_test")
    avg_public_summary = next(row for row in model_rows if row["model"] == "avg" and row["dataset"] == "public_test")
    source_pairwise = [row for row in pairwise_rows if row["dataset"] == "source_test"]
    source_per_class = [row for row in per_class_rows if row["dataset"] == "source_test"]
    source_per_class_sorted = sorted(source_per_class, key=lambda r: float(r["mean_logit"]), reverse=True)

    md = []
    md.append("# Collapse Mechanism Diagnostic\n")
    md.append(f"- Case: `{case_dir}`")
    md.append(f"- Merge: equal-weight AVG, weights={avg_weights}")
    md.append(f"- Device: `{device}`")
    md.append("- Question: why does the merged model predict one class for almost every image?\n")
    md.append("## Model-Level Evidence\n")
    md.append(markdown_table(
        focused_model_rows,
        [
            "model",
            "dataset",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "collapse_ratio",
            "effective_pred_classes",
            "top_pred_class",
            "top_pred_fraction",
            "mean_top1_top2_margin",
            "true_counts",
            "pred_counts",
        ],
    ))
    md.append("\n## AVG Final-Layer Mean Logits on Source Test\n")
    md.append(markdown_table(
        source_per_class_sorted,
        ["class", "mean_wf", "std_wf", "bias", "mean_logit", "std_logit", "min_logit", "max_logit"],
    ))
    md.append("\n## Class-5 Margin Decomposition\n")
    md.append(markdown_table(
        pairwise_rows,
        [
            "dataset",
            "compare",
            "mean_dynamic_delta",
            "std_dynamic_delta",
            "bias_delta",
            "mean_total_delta",
            "min_total_delta",
            "p05_total_delta",
            "pct_total_delta_positive",
        ],
    ))
    md.append("\n## Short Reading\n")
    md.append(
        f"- AVG source test predicts class {avg_source_summary['top_pred_class']} for "
        f"{avg_source_summary['top_pred_fraction']} of samples; public test predicts class "
        f"{avg_public_summary['top_pred_class']} for {avg_public_summary['top_pred_fraction']} of samples."
    )
    md.append(
        "- Balanced accuracy close to 1/7 means the high source accuracy comes from matching the source test prior, not class-wise discrimination."
    )
    md.append(
        "- In the pairwise table, `pct_total_delta_positive=1.0` means class 5 logit is larger than that class for every sample; `min_total_delta>0` means even the closest sample cannot cross the decision boundary."
    )
    md.append(
        "- The collapse is therefore a logit-geometry result: after averaging non-IID clients, the final representation/head has a stable class-5 offset, while image-dependent variation is not enough to change the argmax."
    )
    (out_dir / "collapse_mechanism_diagnostic.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {out_dir / 'collapse_mechanism_diagnostic.md'}")
    print(f"Wrote {out_dir / 'model_level_diagnostic.csv'}")
    print(f"Wrote {out_dir / 'avg_final_layer_per_class.csv'}")
    print(f"Wrote {out_dir / 'avg_class5_pairwise_margin.csv'}")


if __name__ == "__main__":
    main()
