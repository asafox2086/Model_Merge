#!/usr/bin/env python3
"""Export numeric data used by the final LAMP-Merge manuscript."""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FINAL_DIR = ROOT / "final_LAMP"
REPORT_DIR = ROOT / "My_merge_ret" / "reports"
OUT_DIR = FINAL_DIR / "paper_data"
CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
SOURCE_DIR = OUT_DIR / "source_csv"

DATASETS = ["Blood", "Derma", "Organ-C", "Organ-S", "Ultrasound"]
DATASET_KEYS = {
    "bloodmnist_224": "Blood",
    "dermamnist_224": "Derma",
    "organcmnist_224": "Organ-C",
    "organsmnist_224": "Organ-S",
    "chaoshengmnist_224": "Ultrasound",
}
DATASET_ORDER = {v: i for i, v in enumerate(DATASETS)}
BACKBONE_LABELS = {
    "tab:client-avg-resnet": ("resnet", "ResNet"),
    "tab:client-avg-convnext": ("convnext", "ConvNeXt"),
    "tab:client-avg-vit-t": ("vit_t", "ViT-Tiny"),
    "tab:client-avg-swin-tiny": ("swin_tiny", "Swin-Tiny"),
}
MAIN_COLUMNS = [
    f"{dataset}_{col}"
    for dataset in DATASETS
    for col in ["K=3", "K=5", "K=7", "Avg"]
]


def ensure_dirs() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for path in [CSV_DIR, FIG_DIR, SOURCE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def clean_tex_cell(cell: str) -> str:
    cell = cell.strip()
    cell = re.sub(r"\\\\$", "", cell).strip()
    cell = re.sub(r"\\cite\{[^{}]*\}", "", cell)
    for _ in range(8):
        old = cell
        cell = re.sub(r"\\(?:texttt|textbf|underline|avgcell|metriclampcell|dropdown|gainup|gaindown|neutraldelta|tblhead)\{([^{}]*)\}", r"\1", cell)
        if cell == old:
            break
    cell = cell.replace(r"\_", "_")
    cell = cell.replace(r"\uparrow", "↑").replace(r"\downarrow", "↓")
    cell = cell.replace("$", "")
    cell = re.sub(r"\s+", " ", cell)
    return cell.strip()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def format_float(value: str | float, digits: int = 4) -> str:
    if value in ("", None, "--"):
        return str(value or "")
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def markdown_table(rows: list[dict], fieldnames: list[str], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    out = ["| " + " | ".join(fieldnames) + " |"]
    out.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in shown:
        out.append("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |")
    if max_rows is not None and len(rows) > max_rows:
        out.append(f"\n仅展示前 {max_rows} 行；完整数据见 CSV。")
    return "\n".join(out)


def extract_main_tables(tex: str) -> tuple[list[dict], dict[str, list[dict]]]:
    long_rows: list[dict] = []
    wide_by_backbone: dict[str, list[dict]] = {}
    for label, (backbone, backbone_label) in BACKBONE_LABELS.items():
        label_pos = tex.index(rf"\label{{{label}}}")
        block = tex[label_pos: tex.index(r"\end{tabular}", label_pos)]
        rows: list[dict] = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or "&" not in line or not line.endswith(r"\\"):
                continue
            if line.startswith("&") or line.startswith(r"\tblhead") or line.startswith(r"\cline"):
                continue
            if "texttt" not in line and "LAMP-Merge" not in line:
                continue
            parts = [clean_tex_cell(x) for x in line.split("&")]
            method = parts[0]
            values = parts[1:]
            if len(values) != len(MAIN_COLUMNS):
                raise RuntimeError(f"Unexpected column count for {label}: {len(values)}")
            wide = {"backbone": backbone_label, "method": method}
            wide.update(dict(zip(MAIN_COLUMNS, values)))
            rows.append(wide)
            for dataset in DATASETS:
                for metric in ["K=3", "K=5", "K=7", "Avg"]:
                    col = f"{dataset}_{metric}"
                    long_rows.append(
                        {
                            "backbone": backbone_label,
                            "method": method,
                            "dataset": dataset,
                            "client_setting": metric,
                            "client_average_acc": values[MAIN_COLUMNS.index(col)],
                        }
                    )
        wide_by_backbone[backbone] = rows
        write_csv(CSV_DIR / f"main_client_average_{backbone}.csv", rows, ["backbone", "method"] + MAIN_COLUMNS)
    write_csv(CSV_DIR / "main_client_average_long.csv", long_rows, ["backbone", "method", "dataset", "client_setting", "client_average_acc"])
    return long_rows, wide_by_backbone


def extract_simple_table(tex: str, label: str, name: str, fields: list[str]) -> list[dict]:
    label_pos = tex.index(rf"\label{{{label}}}")
    block = tex[label_pos: tex.index(r"\end{tabular}", label_pos)]
    rows: list[dict] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line.endswith(r"\\") or "&" not in line:
            continue
        if line.startswith(r"\tblhead") or line.startswith(r"\hline"):
            continue
        parts = [clean_tex_cell(x) for x in line.split("&")]
        if len(parts) != len(fields):
            continue
        rows.append(dict(zip(fields, parts)))
    write_csv(CSV_DIR / f"{name}.csv", rows, fields)
    return rows


def copy_sources() -> None:
    source_files = [
        REPORT_DIR / "lamp_merge_hparam_interaction_full.csv",
        REPORT_DIR / "lamp_merge_hparam_interaction_full_client_average.csv",
        REPORT_DIR / "lamp_merge_internal_ablation_full.csv",
        REPORT_DIR / "lamp_merge_internal_ablation_full_client_average.csv",
        REPORT_DIR / "lamp_merge_prototype_geometry_overall.csv",
        REPORT_DIR / "lamp_merge_prototype_geometry_full.csv",
        REPORT_DIR / "prediction_diagnostics_full.csv",
        REPORT_DIR / "lamp_merge_client_avg_ablation_cells.csv",
        REPORT_DIR / "lamp_merge_client_avg_ablation_by_dataset.csv",
        ROOT / "outputs" / "final_lamp_paper_tsne_rebuild_20260715" / "reports" / "eval_summary.csv",
        ROOT / "outputs" / "final_lamp_paper_tsne_rebuild_20260715" / "reports" / "merge_summary.csv",
        ROOT / "outputs" / "final_lamp_paper_tsne_rebuild_20260715" / "reports" / "batch_status.csv",
    ]
    for src in source_files:
        if src.exists():
            shutil.copy2(src, SOURCE_DIR / src.name)
    for fig in (FINAL_DIR / "figures").glob("*.png"):
        shutil.copy2(fig, FIG_DIR / fig.name)


def build_hparam_matrices() -> tuple[list[dict], list[dict]]:
    rows = read_csv(REPORT_DIR / "lamp_merge_hparam_interaction_full.csv")
    matrices = []
    for module, out_name in [
        ("diagnostic prototype reconstruction", "figure_hparam_m1_gamma_by_s_matrix.csv"),
        ("long-tail prevalence calibration", "figure_hparam_m2_tau_by_lambda_matrix.csv"),
    ]:
        sub = [r for r in rows if r["module"] == module]
        x_values = sorted({float(r["x_value"]) for r in sub})
        curve_values = sorted({float(r["curve_value"]) for r in sub})
        x_symbol = sub[0]["x_symbol"]
        curve_symbol = sub[0]["curve_symbol"]
        lookup = {(float(r["curve_value"]), float(r["x_value"])): r["client_average_mean_acc"] for r in sub}
        matrix_rows = []
        for curve in curve_values:
            row = {curve_symbol: format_float(curve, 2)}
            for x in x_values:
                row[f"{x_symbol}={format_float(x, 2)}"] = format_float(lookup.get((curve, x), ""), 6)
            matrix_rows.append(row)
        fields = [curve_symbol] + [f"{x_symbol}={format_float(x, 2)}" for x in x_values]
        write_csv(CSV_DIR / out_name, matrix_rows, fields)
        matrices.append({"name": out_name, "rows": matrix_rows, "fields": fields, "module": module})
    return matrices[0], matrices[1]


def find_manifest_meta(dataset: str, model: str, num_clients: str, beta: str, seed: str) -> Path:
    with (ROOT / "model_hub" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row.get("dataset") == dataset
                and row.get("model") == model
                and row.get("num_clients") == num_clients
                and row.get("beta") == beta
                and row.get("seed") == seed
            ):
                return ROOT / "model_hub" / row["meta_path"]
    raise FileNotFoundError(f"meta not found for {dataset}/{model}/K={num_clients}/beta={beta}/seed={seed}")


def build_tsne_probability_coordinates() -> list[dict]:
    """Reconstruct the Blood/ResNet joint output-probability t-SNE data.

    The final figure visualizes output-probability representations for
    BloodMNIST-224, ResNet, K=3, beta=0.01. The original plotting step saved the
    bitmap only, so this function regenerates the numeric coordinates from the
    merged checkpoints recorded in prediction_diagnostics_full.csv.
    """

    try:
        import numpy as np
        import torch
        from sklearn.manifold import TSNE
        from utils import load_checkpoint, load_json
        from utils.runtime import build_runtime
        from utils.state_dict import extract_state_dict
    except Exception as exc:
        path = CSV_DIR / "figure_tsne_blood_resnet_probabilities_README.txt"
        path.write_text(f"t-SNE coordinate export skipped because required runtime packages are unavailable: {exc}\n", encoding="utf-8")
        return []

    dataset = "bloodmnist_224"
    model_name = "resnet"
    num_clients = "3"
    beta = "0.01"
    seed = "42"
    methods = [
        ("lamp_merge:full", "LAMP-Merge"),
        ("avg", "avg"),
        ("ties", "TIES-Merging"),
        ("dare_linear", "DARE-Linear"),
    ]
    method_lookup = dict(methods)
    rebuild_root = ROOT / "outputs" / "final_lamp_paper_tsne_rebuild_20260715" / "merged" / "small" / dataset / model_name / "clients_3" / "beta_0p01" / "seed_42"
    rebuild_method_dir = {
        "lamp_merge:full": "lamp_merge",
        "avg": "avg",
        "ties": "ties",
        "dare_linear": "dare_linear",
    }
    diag_rows = read_csv(REPORT_DIR / "prediction_diagnostics_full.csv")
    ckpt_by_method = {}
    for row in diag_rows:
        if (
            row.get("dataset") == dataset
            and row.get("model") == model_name
            and row.get("num_clients") == num_clients
            and row.get("beta") == beta
            and row.get("seed") == seed
            and row.get("method") in method_lookup
        ):
            ckpt_by_method[row["method"]] = row["checkpoint_path"]
    for method_key, method_dir in rebuild_method_dir.items():
        rebuilt = rebuild_root / method_dir / "merged.pt"
        if rebuilt.exists():
            ckpt_by_method[method_key] = str(rebuilt)
    missing = [m for m, _ in methods if m not in ckpt_by_method]
    if missing:
        raise FileNotFoundError(f"missing checkpoints for t-SNE methods: {missing}")

    meta = load_json(find_manifest_meta(dataset, model_name, num_clients, beta, seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runtime = build_runtime(
        meta=meta,
        data_root=str(ROOT / "Med_data"),
        split="test",
        batch_size=256,
        num_workers=0,
        device=device,
    )
    model = runtime["model"]
    loader = runtime["loader"]
    forward_fn = runtime["forward_fn"]
    num_classes = int(meta.get("num_classes") or len(meta.get("class_names", [])))

    all_probs = []
    row_meta = []
    labels_reference = None
    for method_key, method_label in methods:
        checkpoint = load_checkpoint(Path(ckpt_by_method[method_key]), device="cpu")
        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
        model.to(device)
        model.eval()
        probs_for_method = []
        labels_for_method = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                logits = forward_fn(model, x)
                probs = torch.softmax(logits, dim=1).detach().cpu().float().numpy()
                probs_for_method.append(probs)
                labels_for_method.append(y.detach().cpu().numpy().reshape(-1))
        probs_np = np.concatenate(probs_for_method, axis=0)
        labels_np = np.concatenate(labels_for_method, axis=0).astype(int)
        if labels_reference is None:
            labels_reference = labels_np
        all_probs.append(probs_np)
        for sample_index, true_label in enumerate(labels_np.tolist()):
            row_meta.append((method_label, sample_index, true_label))

    points = np.vstack(all_probs)
    perplexity = max(5, min(40, (points.shape[0] - 1) // 4))
    coords = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=1701,
    ).fit_transform(points)

    out = []
    offset = 0
    for probs_np, (method_key, method_label) in zip(all_probs, methods):
        labels = labels_reference if labels_reference is not None else np.zeros(probs_np.shape[0], dtype=int)
        for sample_index in range(probs_np.shape[0]):
            row = {
                "dataset": dataset,
                "backbone": "ResNet",
                "num_clients": num_clients,
                "beta": beta,
                "method": method_label,
                "sample_index": str(sample_index),
                "true_label": str(int(labels[sample_index])),
                "tsne_x": format_float(coords[offset + sample_index, 0], 6),
                "tsne_y": format_float(coords[offset + sample_index, 1], 6),
            }
            for c in range(num_classes):
                row[f"prob_{c}"] = format_float(probs_np[sample_index, c], 8)
            out.append(row)
        offset += probs_np.shape[0]
    fields = ["dataset", "backbone", "num_clients", "beta", "method", "sample_index", "true_label", "tsne_x", "tsne_y"] + [
        f"prob_{c}" for c in range(num_classes)
    ]
    write_csv(CSV_DIR / "figure_tsne_blood_resnet_output_probability_coordinates.csv", out, fields)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def build_module_ablation_dataset_table() -> list[dict]:
    source = REPORT_DIR / "lamp_merge_client_avg_ablation_cells.csv"
    rows = read_csv(source)
    rows = [r for r in rows if r["model"] != "openai/clip-vit-base-patch32"]
    methods = ["LAMP-Merge", "M1 only", "avg+M2", "avg"]
    by_dataset = defaultdict(lambda: defaultdict(list))
    for row in rows:
        dataset = DATASET_KEYS.get(row["dataset"], row["dataset"])
        for method in methods:
            by_dataset[dataset][method].append(float(row[method]))
    out = []
    for dataset in sorted(by_dataset, key=lambda x: DATASET_ORDER.get(x, 99)):
        item = {"dataset": dataset, "client_average_cells": str(len(next(iter(by_dataset[dataset].values()))))}
        for method in methods:
            vals = by_dataset[dataset][method]
            item[f"{method}_mean_acc"] = format_float(sum(vals) / len(vals), 4)
        out.append(item)
    fields = ["dataset", "client_average_cells"] + [f"{m}_mean_acc" for m in methods]
    write_csv(CSV_DIR / "figure_module_ablation_dataset_accuracy_4backbone.csv", out, fields)
    return out


def parse_counts(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        return []
    if text.startswith("["):
        return [float(x) for x in json.loads(text)]
    return [float(x) for x in text.split()]


def build_prediction_distribution_table() -> list[dict]:
    rows = read_csv(REPORT_DIR / "prediction_diagnostics_full.csv")
    rows = [r for r in rows if r["dataset"] == "chaoshengmnist_224" and r["source"] == "merge"]
    methods = ["lamp_merge:full", "avg", "ties", "dare_linear"]
    accum = defaultdict(lambda: defaultdict(float))
    true_accum = defaultdict(float)
    method_totals = defaultdict(float)
    true_total = 0.0
    for row in rows:
        pred_counts = parse_counts(row["predicted_class_counts"])
        true_counts = parse_counts(row["true_counts"])
        if row["method"] in methods:
            total = sum(pred_counts)
            if total:
                method_totals[row["method"]] += 1.0
                for i, c in enumerate(pred_counts):
                    accum[row["method"]][i] += c / total
        if row["method"] == "lamp_merge:full":
            total = sum(true_counts)
            if total:
                true_total += 1.0
                for i, c in enumerate(true_counts):
                    true_accum[i] += c / total
    out = []
    labels = {"lamp_merge:full": "LAMP-Merge", "avg": "avg", "ties": "ties", "dare_linear": "dare_linear"}
    for method in methods:
        n = method_totals[method]
        item = {"series": labels[method]}
        for i in range(8):
            item[f"class_{i}"] = format_float(accum[method][i] / n if n else "", 6)
        out.append(item)
    true_item = {"series": "true_distribution"}
    for i in range(8):
        true_item[f"class_{i}"] = format_float(true_accum[i] / true_total if true_total else "", 6)
    out.append(true_item)
    fields = ["series"] + [f"class_{i}" for i in range(8)]
    write_csv(CSV_DIR / "figure_predicted_distribution_ultrasound.csv", out, fields)
    return out


def build_prototype_geometry_table() -> list[dict]:
    rows = read_csv(REPORT_DIR / "lamp_merge_prototype_geometry_overall.csv")
    keep = [
        "label",
        "cases",
        "mean_mean_pairwise_distance",
        "mean_mean_nearest_class_distance",
        "mean_prototype_consistency",
        "mean_prototype_to_client_alignment",
    ]
    out = []
    for row in rows:
        out.append(
            {
                "setting": row["label"],
                "cases": row["cases"],
                "D_pair": format_float(row["mean_mean_pairwise_distance"], 6),
                "D_near": format_float(row["mean_mean_nearest_class_distance"], 6),
                "A_client": format_float(row["mean_prototype_consistency"], 6),
                "A_proto": format_float(row["mean_prototype_to_client_alignment"], 6),
            }
        )
    fields = ["setting", "cases", "D_pair", "D_near", "A_client", "A_proto"]
    write_csv(CSV_DIR / "figure_prototype_geometry_internal_ablation.csv", out, fields)
    return out


def build_markdown(
    wide_by_backbone: dict[str, list[dict]],
    diag_rows: list[dict],
    prev_rows: list[dict],
    collapse_rows: list[dict],
    hparam_m1: dict,
    hparam_m2: dict,
    module_dataset: list[dict],
    pred_dist: list[dict],
    proto_geom: list[dict],
    tsne_rows: list[dict],
) -> None:
    lines = []
    lines.append("# LAMP-Merge 最终论文数据归档")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("本目录汇总最新 `final_LAMP/final.tex` 论文中使用的数值数据。正式论文表格直接从 TeX 正文抽取；图像对应的数据在可追溯的情况下由实验报告 CSV 重新聚合得到。")
    lines.append("")
    lines.append("## 文件说明")
    lines.append("")
    lines.append("- `csv/main_client_average_*.csv`：论文表 1--4 的 client-average ACC 数值。")
    lines.append("- `csv/table_diagnostic_internal_ablation.csv`：诊断原型重建模块的内部消融表。")
    lines.append("- `csv/table_prevalence_internal_ablation.csv`：长尾患病率校准模块的内部消融表。")
    lines.append("- `csv/table_prediction_collapse_diagnostics.csv`：预测坍缩诊断的新指标表。")
    lines.append("- `csv/figure_*`：论文图像对应的绘图数据。")
    lines.append("- `source_csv/`：用于生成归档数据的原始报告 CSV 备份。")
    lines.append("- `figures/`：最终论文中使用的图片备份。")
    lines.append("")

    lines.append("## 主实验 Client-Average ACC 表")
    for backbone, rows in wide_by_backbone.items():
        label = rows[0]["backbone"] if rows else backbone
        fields = ["method"] + MAIN_COLUMNS
        lines.append(f"\n### {label}")
        lines.append(markdown_table(rows, fields))

    lines.append("\n## 消融实验表")
    lines.append("\n### 诊断原型重建模块")
    lines.append(markdown_table(diag_rows, ["Setting", "Mean Acc", "Delta vs. LAMP"]))
    lines.append("\n### 长尾患病率校准模块")
    lines.append(markdown_table(prev_rows, ["Setting", "Mean Acc", "Delta vs. LAMP"]))

    lines.append("\n## 新指标诊断表")
    lines.append(markdown_table(collapse_rows, ["Metric", "LAMP-Merge", "Best reference", "Delta vs. LAMP"]))

    lines.append("\n## 图像绘图数据")
    lines.append("\n### 模块消融的数据集级准确率")
    lines.append("每个数据集行对四个视觉骨干网络和三个客户端数量取平均；旧版 CLIP-ViT 行已排除，因此该表与最新论文的四骨干设置一致。")
    lines.append(markdown_table(module_dataset, ["dataset", "client_average_cells", "LAMP-Merge_mean_acc", "M1 only_mean_acc", "avg+M2_mean_acc", "avg_mean_acc"]))

    lines.append("\n### 超参数敏感性：诊断原型重建")
    lines.append("行表示证据指数 `gamma`，列表示原型分类头缩放因子 `s`，表中数值为完整实验范围下的 client-average ACC。该表使用之前正式的 5×5 超参数结果，不包含当前仍在运行的扩展实验。")
    lines.append(markdown_table(hparam_m1["rows"], hparam_m1["fields"]))

    lines.append("\n### 超参数敏感性：长尾患病率校准")
    lines.append("行表示长尾校准触发阈值 `tau`，列表示校准强度 `lambda`，表中数值为完整实验范围下的 client-average ACC。该表使用之前正式的 5×5 超参数结果，不包含当前仍在运行的扩展实验。")
    lines.append(markdown_table(hparam_m2["rows"], hparam_m2["fields"]))

    lines.append("\n### Ultrasound 预测类别分布")
    lines.append("表中数值为 Ultrasound 数据集上各类别的平均预测比例，聚合范围包括四个视觉骨干网络、三个客户端数量以及三个 Dirichlet 偏斜水平。")
    lines.append(markdown_table(pred_dist, ["series"] + [f"class_{i}" for i in range(8)]))

    lines.append("\n### 原型几何指标")
    lines.append("`D_pair` 表示不同诊断类别全局原型之间的平均距离；`D_near` 表示每个类别到最近错误类别的距离；`A_client` 表示同一类别在不同客户端原型之间的一致性；`A_proto` 表示全局原型与参与聚合的客户端原型之间的对齐程度。")
    lines.append(markdown_table(proto_geom, ["setting", "cases", "D_pair", "D_near", "A_client", "A_proto"]))

    lines.append("\n### t-SNE 图")
    lines.append("Blood/ResNet 联合输出概率 t-SNE 图的坐标数据已导出到 `csv/figure_tsne_blood_resnet_output_probability_coordinates.csv`。每一行对应某个融合方法在一张测试图像上的输出概率向量，并给出其在共享 t-SNE 平面中的二维坐标。")
    if tsne_rows:
        fields = ["dataset", "backbone", "num_clients", "beta", "method", "sample_index", "true_label", "tsne_x", "tsne_y"] + [
            key for key in tsne_rows[0] if key.startswith("prob_")
        ]
        lines.append(markdown_table(tsne_rows, fields, max_rows=20))

    (OUT_DIR / "论文数据总表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    copy_sources()
    tex = read_text(FINAL_DIR / "final.tex")
    _, wide_by_backbone = extract_main_tables(tex)
    diag_rows = extract_simple_table(tex, "tab:diagnostic-internal-ablation", "table_diagnostic_internal_ablation", ["Setting", "Mean Acc", "Delta vs. LAMP"])
    prev_rows = extract_simple_table(tex, "tab:prevalence-internal-ablation", "table_prevalence_internal_ablation", ["Setting", "Mean Acc", "Delta vs. LAMP"])
    collapse_rows = extract_simple_table(tex, "tab:collapse-key", "table_prediction_collapse_diagnostics", ["Metric", "LAMP-Merge", "Best reference", "Delta vs. LAMP"])
    hparam_m1, hparam_m2 = build_hparam_matrices()
    module_dataset = build_module_ablation_dataset_table()
    pred_dist = build_prediction_distribution_table()
    proto_geom = build_prototype_geometry_table()
    tsne_rows = build_tsne_probability_coordinates()
    build_markdown(wide_by_backbone, diag_rows, prev_rows, collapse_rows, hparam_m1, hparam_m2, module_dataset, pred_dist, proto_geom, tsne_rows)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
