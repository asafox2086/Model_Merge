#!/usr/bin/env python3
"""Validate complete Pscore runs and export auditable comparison tables."""

import argparse
import ast
import csv
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["bloodmnist_224", "dermamnist_224", "organcmnist_224", "organsmnist_224", "chaoshengmnist_224"]
MODELS = ["resnet", "convnext", "vit_t", "swin_tiny"]


def key(row):
    return row["dataset"], row["model"], int(row["num_clients"]), float(row["beta"]), int(row["seed"])


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records, provenance = [], []
    code_hashes = set()
    for path in sorted(args.input_root.rglob("result.json")):
        row = json.loads(path.read_text())
        require(row["status"] == "OK" and row["split"] == "test" and row["method"] == "pscore_mlp", f"Invalid result: {path}")
        with np.load(path.parent / "test_predictions.npz") as archive:
            probabilities, labels = archive["probabilities"], archive["labels"]
        require(np.isfinite(probabilities).all(), f"Non-finite probability: {path}")
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        classes = probabilities.shape[1]
        confusion = np.bincount(labels * classes + probabilities.argmax(1), minlength=classes ** 2).reshape(classes, classes)
        np.testing.assert_array_equal(confusion, np.asarray(json.loads(row["confusion_matrix_json"])))
        support = confusion.sum(1)
        require(np.all(support > 0), "Unexpected missing test class")
        accuracy = np.trace(confusion) / confusion.sum()
        macro_f1 = np.mean(2 * np.diag(confusion) / (support + confusion.sum(0)))
        require(abs(accuracy - row["accuracy"]) < 1e-12 and abs(macro_f1 - row["macro_f1"]) < 1e-12, "Metric mismatch")
        protocol = json.loads((path.parent / "protocol.json").read_text())
        require(protocol["fit_split"] == "train" and protocol["selection_split"] == "val" and protocol["evaluation_split"] == "test", "Invalid split protocol")
        require(protocol["settings"]["epochs"] == 50, "Smoke-test epochs in formal results")
        history = json.loads((path.parent / "training_history.json").read_text())
        require(len(history) == 3 and all(len(trial["history"]) == 50 for trial in history), "Incomplete architecture grid")
        best_trial, best_epoch = max(((trial, epoch) for trial in history for epoch in trial["history"]),
                                    key=lambda pair: (pair[1]["val_accuracy"], -pair[1]["val_loss"]))
        require(row["selected_epoch"] == best_epoch["epoch"], "Validation epoch selection mismatch")
        require(row["selection_val_accuracy"] == best_epoch["val_accuracy"], "Validation accuracy mismatch")
        require(tuple(ast.literal_eval(row["selected_hidden_sizes"])) == tuple(best_trial["hidden_sizes"]), "Selected architecture mismatch")
        code_hashes.add(protocol["code_sha256"])
        records.append(row)
        provenance.append({"path": str(path.relative_to(args.input_root)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "protocol_sha256": hashlib.sha256((path.parent / "protocol.json").read_bytes()).hexdigest()})
    expected = set(itertools.product(DATASETS, MODELS, (3, 5, 7), (0.0, 0.01, 0.1), (42,)))
    require(len(records) == 180 and {key(row) for row in records} == expected, f"Incomplete or duplicate coverage: {len(records)}/180")
    require(len(code_hashes) == 1, "Mixed source code versions")
    source = ROOT / "My_merge_ret/reports/head_only_baselines_20260724_v5.csv"
    with source.open() as handle:
        baseline_rows = [row for row in csv.DictReader(handle) if row["status"] == "OK" and row["method"] != "head_fisher"]
    baseline_groups = {}
    for row in baseline_rows:
        baseline_groups.setdefault(key(row), []).append(row)
    for row in records:
        peers = baseline_groups[key(row)]
        require(len(peers) == 12, "Incomplete original comparator coverage")
        require(all(str(row["true_counts"]) == peer["true_counts"] for peer in peers), "Test population differs from original tables")
    summary = []
    for model, dataset, clients in itertools.product(MODELS, DATASETS, (3, 5, 7, "Avg")):
        selected = [row for row in records if row["model"] == model and row["dataset"] == dataset
                    and (clients == "Avg" or row["num_clients"] == clients)]
        peers = [row for row in baseline_rows if row["model"] == model and row["dataset"] == dataset
                 and (clients == "Avg" or int(row["num_clients"]) == clients)]
        lamp = [row for row in peers if row["method"] == "lamp_merge"]
        summary.append({"model": model, "dataset": dataset, "K": clients, "cases": len(selected),
                        "accuracy_percent": 100 * mean(row["accuracy"] for row in selected),
                        "macro_f1_percent": 100 * mean(row["macro_f1"] for row in selected),
                        "lamp_accuracy_percent": 100 * mean(float(row["accuracy"]) for row in lamp),
                        "lamp_macro_f1_percent": 100 * mean(float(row["macro_f1"]) for row in lamp)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pscore_raw.csv", records)
    write_csv(args.output_dir / "pscore_summary.csv", summary)
    audit = {"cases": len(records), "prediction_arrays_verified": len(records), "selection_histories_verified": len(records),
             "source_code_sha256": next(iter(code_hashes)), "comparator_source": str(source.relative_to(ROOT)),
             "comparator_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "case_files": provenance}
    (args.output_dir / "pscore_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = ["# Pscore-MLP (adapted)：完整实验结果", "",
             "监督式预测融合；train 拟合融合器，val 选择架构/epoch，test 最终评估。所有数值为 ACC / Macro-F1（%）。", "",
             "| Backbone | Dataset | K=3 | K=5 | K=7 | Avg | LAMP Avg |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for model, dataset in itertools.product(MODELS, DATASETS):
        cells = [row for row in summary if row["model"] == model and row["dataset"] == dataset]
        values = [f"{row['accuracy_percent']:.2f} / {row['macro_f1_percent']:.2f}" for row in cells]
        average = cells[-1]
        lines.append("| " + " | ".join([model, dataset, *values, f"{average['lamp_accuracy_percent']:.2f} / {average['lamp_macro_f1_percent']:.2f}"]) + " |")
    lines.extend(["", "以上是非联邦的 Pscore-MLP 分类适配结果，不能称为原始乳腺摄影系统的直接复现。融合器访问训练概率及标签，与无训练参数合并的方法信息预算不同。", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))
    print(f"Verified {len(records)} cases, predictions, validation selection and original test populations.")


if __name__ == "__main__":
    main()
