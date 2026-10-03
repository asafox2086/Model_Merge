#!/usr/bin/env python3
"""Verify TMI metrics and regenerate the unaveraged Swin table from saved runs."""

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "My_merge_ret/reports/head_only_baselines_20260724_v5.csv"
PAPER = ROOT / "paper/LAMP_Merge_TMI.tex"
DATASETS = (
    "bloodmnist_224", "dermamnist_224", "organcmnist_224",
    "organsmnist_224", "chaoshengmnist_224",
)
MODELS = {
    "swin_tiny": "swin-tiny", "convnext": "convnext",
    "resnet": "resnet", "vit_t": "vit-tiny",
}
METHODS = {
    "Weight Avg.": "head_avg", "TIES": "head_ties",
    "DARE-Linear": "head_dare_linear", "DARE-TIES": "head_dare_ties",
    "RegMean": "head_regmean", "Breadcrumbs": "head_breadcrumbs",
    "Model Stock": "head_model_stock", "Iso-C": "head_iso_c",
    "FREE-Merging": "head_free_merge", "RobustMerge": "head_robustmerge",
    "FROM": "head_from", r"\textbf{LAMP-Merge}": "lamp_merge",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_metrics(rows):
    text = PAPER.read_text()
    checked = 0
    for model, label in MODELS.items():
        table = text.split(r"\label{tab:client-avg-" + label + "}", 1)[1]
        table = table.split(r"\end{table*}", 1)[0]
        for display, method in METHODS.items():
            lines = [line for line in table.splitlines() if line.startswith(display + " &") or line.startswith(display + r"\pub")]
            require(len(lines) == 1, f"Ambiguous table row: {model}/{method}")
            cells = lines[0].split(" &")[1:]
            require(len(cells) == 16, f"Expected 16 cells: {model}/{method}")
            expected = []
            for dataset in (DATASETS[0], DATASETS[1], DATASETS[2], DATASETS[4]):
                for clients in (3, 5, 7, None):
                    selected = [row for row in rows if row["model"] == model and row["method"] == method
                                and row["dataset"] == dataset
                                and (clients is None or int(row["num_clients"]) == clients)]
                    require(len(selected) == (9 if clients is None else 3), f"Incomplete group: {model}/{method}/{dataset}/{clients}")
                    expected.append(tuple(mean(float(row[key]) for row in selected) * 100 for key in ("accuracy", "macro_f1")))
            for cell, pair in zip(cells, expected):
                plain = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", cell)
                match = re.search(r"([0-9]+\.[0-9]+) / ([0-9]+\.[0-9]+)", plain)
                require(match is not None, f"Missing metric pair: {cell}")
                require(tuple(match.groups()) == tuple(f"{value:.2f}" for value in pair), f"Metric mismatch: {model}/{method}/{cell}/{pair}")
                checked += 1
    for row in rows:
        matrix = json.loads(row["confusion_matrix_json"])
        classes = len(matrix)
        support = [sum(values) for values in matrix]
        require(all(value > 0 for value in support), f"Absent test class: {row['dataset']}")
        predicted = [sum(values[column] for values in matrix) for column in range(classes)]
        accuracy = sum(matrix[index][index] for index in range(classes)) / sum(support)
        macro_f1 = mean(2 * matrix[index][index] / (support[index] + predicted[index]) for index in range(classes))
        require(abs(accuracy - float(row["accuracy"])) < 1e-12, "ACC/confusion mismatch")
        require(abs(macro_f1 - float(row["macro_f1"])) < 1e-12, "Macro-F1/confusion mismatch")
    return {"main_table_metric_pairs": checked, "confusion_matrices_verified": len(rows), "all_test_classes_present": True}


def grid_artifacts(rows):
    lamp = [row for row in rows if row["method"] == "lamp_merge"]
    means = {model: mean(float(row["accuracy"]) for row in lamp if row["model"] == model) for model in MODELS}
    require(max(means, key=means.get) == "swin_tiny", "Selected backbone is no longer the highest-ACC backbone")
    selected = sorted((row for row in lamp if row["model"] == "swin_tiny"),
                      key=lambda row: (int(row["num_clients"]), float(row["beta"]), DATASETS.index(row["dataset"])))
    require(len(selected) == 45, "Expected 45 unaveraged Swin cases")
    fields = ["dataset", "model", "num_clients", "beta", "seed", "split", "accuracy", "macro_f1"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(selected)
    lines = [
        r"\begin{table*}[!t]", r"\centering", r"\small",
        r"\caption{Unaveraged LAMP-Merge test ACC / Macro-F1 (\%) on Swin-Tiny. Each cell is one run (seed 42); no averaging over $K$ or $\beta$. US denotes ChaoShengMNIST.}",
        r"\label{tab:swin-full-grid}", r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{cc|ccccc}", r"\hline",
        r"$K$ & $\beta$ & Blood & Derma & Organ-C & Organ-S & US \\", r"\hline",
    ]
    for clients in (3, 5, 7):
        for beta in (0.0, 0.01, 0.1):
            cells = []
            for dataset in DATASETS:
                values = [row for row in selected if int(row["num_clients"]) == clients and float(row["beta"]) == beta and row["dataset"] == dataset]
                require(len(values) == 1, "Duplicate or missing grid case")
                row = values[0]
                cells.append(f"{100 * float(row['accuracy']):.2f} / {100 * float(row['macro_f1']):.2f}")
            lines.append(f"{clients} & {beta:g} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    return means, buffer.getvalue(), "\n".join(lines)


def audit_pscore(rows):
    text = PAPER.read_text()
    if "Pscore-MLP (adapted)\\pub" not in text:
        return {}
    source = ROOT / "paper/data/pscore/pscore_raw.csv"
    with source.open() as handle:
        additional = list(csv.DictReader(handle))
    require(len(additional) == 180, "Incomplete Pscore source")
    checked = 0
    for model, label in MODELS.items():
        table = text.split(r"\label{tab:client-avg-" + label + "}", 1)[1].split(r"\end{table*}", 1)[0]
        pscore_line = next(line for line in table.splitlines() if line.startswith("Pscore-MLP (adapted)"))
        lamp_line = next(line for line in table.splitlines() if line.startswith(r"\textbf{LAMP-Merge} &"))
        pscore_cells = pscore_line.split(" &")[1:]
        gain_cells = re.findall(r"\}\{([+-][0-9]+\.[0-9]+) / ([+-][0-9]+\.[0-9]+)\}", lamp_line)
        require(len(pscore_cells) == 16 and len(gain_cells) == 16, "Malformed extended main table")
        index = 0
        for dataset in (DATASETS[0], DATASETS[1], DATASETS[2], DATASETS[4]):
            for clients in (3, 5, 7, None):
                group = [row for row in rows + additional if row["model"] == model and row["dataset"] == dataset
                         and (clients is None or int(row["num_clients"]) == clients)]
                means = {method: tuple(100 * mean(float(row[metric]) for row in group if row["method"] == method)
                                       for metric in ("accuracy", "macro_f1"))
                         for method in [*METHODS.values(), "pscore_mlp"]}
                plain = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", pscore_cells[index])
                match = re.search(r"([0-9]+\.[0-9]+) / ([0-9]+\.[0-9]+)", plain)
                require(match and match.groups() == tuple(f"{value:.2f}" for value in means["pscore_mlp"]), "Pscore table mismatch")
                gains = tuple(means["lamp_merge"][metric] - max(pair[metric] for method, pair in means.items() if method != "lamp_merge") for metric in range(2))
                require(gain_cells[index] == tuple(f"{value:+.2f}" for value in gains), "Extended baseline gain mismatch")
                index += 1
                checked += 1
    return {"pscore_table_metric_pairs": checked, "updated_gain_pairs": checked,
            "pscore_source_sha256": digest(source)}


def hyperparameters():
    summaries = {}
    for name, filename, keys, chosen in (
        ("DPR", "超参数分析_诊断原型重建.csv", ("Support exponent gamma", "Prototype head scale s"), (0.55, 18.75)),
        ("LPC", "超参数分析_长尾患病率校准.csv", ("Activation threshold tau", "Calibration strength lambda"), (2.5, 4.25)),
    ):
        path = ROOT / "paper/data" / filename
        with path.open() as handle:
            next(handle)
            rows = list(csv.DictReader(handle))
        require(len(rows) == 50 and all(int(row["Raw cells"]) == 180 for row in rows), "Unexpected hyperparameter grid")
        best = max(rows, key=lambda row: float(row["Mean ACC (%)"]))
        selected = [row for row in rows if tuple(float(row[key]) for key in keys) == chosen]
        require(len(selected) == 1, "Missing selected hyperparameters")
        chosen_acc = float(selected[0]["Mean ACC (%)"])
        summaries[name] = {
            "source": str(path.relative_to(ROOT)), "sha256": digest(path), "points": len(rows),
            "min_acc_percent": min(float(row["Mean ACC (%)"]) for row in rows),
            "max_acc_percent": float(best["Mean ACC (%)"]), "best_parameters": {key: float(best[key]) for key in keys},
            "selected_acc_percent": chosen_acc, "gap_pp": round(float(best["Mean ACC (%)"]) - chosen_acc, 4),
            "interpretation": "Retrospective test-set sensitivity; not evidence of independent validation selection",
        }
    return summaries


def audit_partitions(rows):
    import numpy as np
    import torch

    results = []
    labels_by_dataset = {}
    for row in rows:
        if row["method"] != "lamp_merge":
            continue
        dataset = row["dataset"]
        if dataset not in labels_by_dataset:
            with np.load(ROOT / "Med_data" / f"{dataset}.npz") as archive:
                labels_by_dataset[dataset] = archive["train_labels"].reshape(-1).astype(int)
        labels = labels_by_dataset[dataset]
        beta_name = f"{float(row['beta']):g}".replace(".", "p")
        suffix = Path("small") / dataset / row["model"] / f"clients_{row['num_clients']}" / f"beta_{beta_name}" / f"seed_{row['seed']}"
        partition = ROOT / "outputs/lamp_merge_client_partitions" / suffix / "client_indices.npz"
        stats = ROOT / "outputs/lamp_merge_client_local_proto_stats" / suffix / "prototype_stats.pt"
        payload = torch.load(stats, map_location="cpu", weights_only=False)
        require(payload["feature_space"] == "reference_model" and payload["source_split"] == "train", "Unexpected prototype provenance")
        pooled = np.zeros(int(payload["num_classes"]), dtype=np.int64)
        feature_error = 0.0
        with np.load(partition) as archive:
            indices = np.concatenate([archive[f"client_{index}"] for index in range(int(row["num_clients"]))])
            require(np.array_equal(np.sort(indices), np.arange(labels.size)), "Overlapping, omitted, or invalid training indices")
            for client in payload["clients"]:
                client_indices = archive[f"client_{int(client['client_id'])}"]
                counts = np.asarray(client["class_prevalence_counts"], dtype=np.int64)
                features = np.asarray(client["class_feature_counts"], dtype=float)
                require(np.array_equal(counts, np.bincount(labels[client_indices], minlength=pooled.size)), "Client prevalence differs from labels")
                feature_error = max(feature_error, float(np.max(np.abs(features - counts))))
                pooled += counts
        require(np.array_equal(pooled, np.bincount(labels, minlength=pooled.size)), "Pooled prior differs from training labels")
        require(feature_error == 0, "Feature support and prevalence differ")
        results.append({
            **{key: row[key] for key in ("dataset", "model", "num_clients", "beta", "seed")},
            "samples": int(labels.size), "assigned": int(indices.size), "unique": int(np.unique(indices).size),
            "prior_count_max_error": 0, "feature_prevalence_max_error": feature_error,
            "partition_path": str(partition.relative_to(ROOT)), "partition_sha256": digest(partition),
            "stats_path": str(stats.relative_to(ROOT)), "stats_sha256": digest(stats),
        })
    require(len(results) == 180, "Incomplete partition audit")
    return json.dumps(results, ensure_ascii=False, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write generated evidence instead of checking it")
    parser.add_argument("--partitions", action="store_true", help="Also inspect local partitions and trusted prototype payloads (NumPy/PyTorch required)")
    args = parser.parse_args()
    with SOURCE.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] in METHODS.values()]
    require(len(rows) == 2160 and all(row["status"] == "OK" and row["split"] == "test" and row["seed"] == "42" for row in rows), "Unexpected run coverage")
    audit = audit_metrics(rows)
    audit.update(audit_pscore(rows))
    means, csv_text, tex_text = grid_artifacts(rows)
    evidence = {
        "source": str(SOURCE.relative_to(ROOT)), "sha256": digest(SOURCE), **audit,
        "lamp_mean_accuracy_by_backbone": means, "selected_backbone": "swin_tiny",
        "selection_rule": "Highest LAMP mean ACC across all five datasets and all nine K/beta settings",
        "hyperparameter_grids": hyperparameters(),
    }
    artifacts = {
        "swin_full_grid.csv": csv_text, "swin_full_grid.tex": tex_text,
        "tmi_revision_audit.json": json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
    }
    if args.partitions:
        artifacts["tmi_partition_audit.json"] = audit_partitions(rows)
    for filename, content in artifacts.items():
        path = ROOT / "paper/data" / filename
        if args.write:
            path.write_text(content)
        else:
            require(path.read_text() == content, f"Stale generated evidence: {path}")
    print(json.dumps({**audit, "generated_files_checked": len(artifacts), "partitions_audited": 180 if args.partitions else None}))


if __name__ == "__main__":
    main()
