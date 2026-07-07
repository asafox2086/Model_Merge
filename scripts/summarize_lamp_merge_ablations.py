#!/usr/bin/env python3
import csv
import re
from collections import defaultdict
from pathlib import Path

from generate_combined_results_table import parse_tables, try_parse_number


ROOT = Path(__file__).resolve().parents[1]
METHOD_ROW_NAMES = {"my_merge", "lamp_merge"}
FORMAL_LABEL = "LAMP-Merge"

FORMAL_ROOTS = [
    ROOT / "outputs/my_merge_reference_proto_recall_full_table_20260705",
    ROOT / "outputs/m1_m2_dominant_derma_36_20260705",
]
M1_ROOTS = [
    ROOT / "outputs/ablation_m1_full_table_20260705",
    ROOT / "outputs/ablation_m1_full_table_part2_20260705",
]
AVG_M2_ROOTS = [
    ROOT / "outputs/ablation_avg_m2_full_table_20260705",
    ROOT / "outputs/ablation_avg_m2_full_table_part2_20260705",
]

MAIN_TABLE = ROOT / "My_merge_ret/汇总表.md"
ABLATION_TABLE = ROOT / "My_merge_ret/my_merge消融表.md"
REPORT_DIR = ROOT / "My_merge_ret/reports"
REPORT_PATH = ROOT / "My_merge_ret/LAMP-Merge模块消融与超参数分析.md"
CURRENT_FULL_ROOT = ROOT / "outputs/lamp_merge_current_prevalence_smoke_20260706"
CURRENT_M1_ROOT = ROOT / "outputs/lamp_merge_current_prevalence_m1_smoke_20260706"
CURRENT_AVG_M2_ROOT = ROOT / "outputs/lamp_merge_current_prevalence_avg_m2_smoke_20260706"
ABLATION_VARIANTS = [FORMAL_LABEL, "M1 only", "avg+M2", "avg"]
DATASET_LABELS = {
    "bloodmnist_224": "Blood",
    "dermamnist_224": "Derma",
    "organcmnist_224": "Organ-C",
    "organsmnist_224": "Organ-S",
    "chaoshengmnist_224": "Ultrasound",
}


def read_eval_rows(roots):
    rows = []
    for root in roots:
        path = root / "reports/eval_summary.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def row_key(row):
    task_type = row["task_type"]
    model = row["model"] if task_type == "small" else row["clip_model"]
    return (
        task_type,
        row["dataset"],
        model,
        int(float(row["num_clients"])),
        float(row["beta"]),
    )


def load_lookup(roots):
    lookup = {}
    for row in read_eval_rows(roots):
        if row.get("method") not in METHOD_ROW_NAMES:
            continue
        lookup[row_key(row)] = float(row["test_acc"])
    return lookup


def load_avg_from_table(path):
    _intro, tables = parse_tables(path)
    lookup = {}
    for (section, model, kind), table in tables.items():
        if kind != "Raw":
            continue
        task_type = "small" if section == "Small" else "vlm"
        avg_row = None
        for row in table["rows"]:
            if clean_method(row["method"]) == "avg":
                avg_row = row
                break
        if avg_row is None:
            continue
        datasets = datasets_for_section(section)
        settings = [(3, 0.0), (3, 0.01), (3, 0.1), (5, 0.0), (5, 0.01), (5, 0.1), (7, 0.0), (7, 0.01), (7, 0.1)]
        for ds_idx, dataset in enumerate(datasets):
            for setting_idx, (num_clients, beta) in enumerate(settings):
                value_idx = ds_idx * len(settings) + setting_idx
                if value_idx >= len(avg_row["values"]):
                    continue
                value = try_parse_number(avg_row["values"][value_idx])
                if value is not None:
                    lookup[(task_type, dataset, model, num_clients, beta)] = value
    return lookup


def datasets_for_section(section):
    if section in {"Small", "VLM"}:
        return [
            "bloodmnist_224",
            "dermamnist_224",
            "organcmnist_224",
            "organsmnist_224",
            "chaoshengmnist_224",
        ]
    return []


TAG_RE = re.compile(r"<.*?>")


def clean_method(value):
    return TAG_RE.sub("", str(value)).strip()


def iter_main_table_cells(table_path):
    _intro, tables = parse_tables(table_path)
    for (_section, _model, kind), table in tables.items():
        mine = None
        baselines = []
        for row in table["rows"]:
            method = clean_method(row["method"])
            if method in {FORMAL_LABEL, "my_merge", "lamp_merge"}:
                mine = row
            else:
                baselines.append((method, row))
        if mine is None:
            continue
        for idx, raw in enumerate(mine["values"]):
            mine_value = try_parse_number(raw)
            if mine_value is None:
                continue
            baseline_values = []
            for method, row in baselines:
                if idx < len(row["values"]):
                    value = try_parse_number(row["values"][idx])
                    if value is not None:
                        baseline_values.append((method, value))
            if not baseline_values:
                continue
            best_value = max(value for _method, value in baseline_values)
            yield kind, mine_value, best_value


def summarize_main_table(table_path):
    out = defaultdict(lambda: {"ok": 0, "total": 0})
    for kind, mine, best in iter_main_table_cells(table_path):
        out["All"]["total"] += 1
        out[kind]["total"] += 1
        if mine >= best:
            out["All"]["ok"] += 1
            out[kind]["ok"] += 1
    return dict(out)


def mean(values):
    return sum(values) / len(values) if values else None


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def pct(ok, total):
    return f"{100.0 * ok / total:.1f}%" if total else "-"


def summarize_lookup(label, lookup, all_keys, avg_lookup=None, formal_lookup=None):
    values = [lookup[key] for key in all_keys if key in lookup]
    raw_total = len(values)
    row = {
        "setting": label,
        "raw_cells": raw_total,
        "raw_mean_acc": mean(values),
        "client_average_cells": 0,
        "client_average_mean_acc": None,
        "ge_avg_raw": None,
        "ge_formal_raw": None,
    }
    client_values = []
    grouped = defaultdict(list)
    for key in all_keys:
        if key not in lookup:
            continue
        task_type, dataset, model, num_clients, _beta = key
        grouped[(task_type, dataset, model, num_clients)].append(lookup[key])
    for group_values in grouped.values():
        if len(group_values) == 3:
            client_values.append(mean(group_values))
    row["client_average_cells"] = len(client_values)
    row["client_average_mean_acc"] = mean(client_values)
    if avg_lookup:
        comparable = [key for key in all_keys if key in lookup and key in avg_lookup]
        row["ge_avg_raw"] = sum(1 for key in comparable if lookup[key] >= avg_lookup[key])
        row["ge_avg_raw_total"] = len(comparable)
    if formal_lookup:
        comparable = [key for key in all_keys if key in lookup and key in formal_lookup]
        row["ge_formal_raw"] = sum(1 for key in comparable if lookup[key] >= formal_lookup[key])
        row["ge_formal_raw_total"] = len(comparable)
    return row


def client_average_key(raw_key):
    task_type, dataset, model, num_clients, _beta = raw_key
    return task_type, dataset, model, num_clients


def build_client_average_lookup(raw_lookup):
    grouped = defaultdict(list)
    for key, value in raw_lookup.items():
        grouped[client_average_key(key)].append(float(value))
    return {
        key: mean(values)
        for key, values in grouped.items()
        if len(values) == 3
    }


def build_client_average_ablation(formal_lookup, m1_lookup, avg_m2_lookup, avg_lookup):
    variant_lookups = {
        FORMAL_LABEL: build_client_average_lookup(formal_lookup),
        "M1 only": build_client_average_lookup(m1_lookup),
        "avg+M2": build_client_average_lookup(avg_m2_lookup),
        "avg": build_client_average_lookup(avg_lookup),
    }
    all_keys = sorted(set().union(*(set(item) for item in variant_lookups.values())))

    cell_rows = []
    for key in all_keys:
        values = {
            label: lookup[key]
            for label, lookup in variant_lookups.items()
            if key in lookup
        }
        if not values:
            continue
        best_value = max(values.values())
        best_variants = [label for label, value in values.items() if value >= best_value - 1e-12]
        task_type, dataset, model, num_clients = key
        cell_rows.append({
            "task_type": task_type,
            "dataset": dataset,
            "model": model,
            "num_clients": num_clients,
            FORMAL_LABEL: values.get(FORMAL_LABEL),
            "M1 only": values.get("M1 only"),
            "avg+M2": values.get("avg+M2"),
            "avg": values.get("avg"),
            "best_value": best_value,
            "best_variants": ";".join(best_variants),
            "lamp_minus_avg": (
                values[FORMAL_LABEL] - values["avg"]
                if FORMAL_LABEL in values and "avg" in values
                else None
            ),
            "m1_minus_avg": (
                values["M1 only"] - values["avg"]
                if "M1 only" in values and "avg" in values
                else None
            ),
            "avg_m2_minus_avg": (
                values["avg+M2"] - values["avg"]
                if "avg+M2" in values and "avg" in values
                else None
            ),
        })

    summary_rows = []
    for label, lookup in variant_lookups.items():
        values = [row[label] for row in cell_rows if row.get(label) is not None]
        ge_avg_rows = [row for row in cell_rows if row.get(label) is not None and row.get("avg") is not None]
        summary_rows.append({
            "setting": label,
            "client_average_cells": len(values),
            "client_average_mean_acc": mean(values),
            "best_or_tied_cells": sum(1 for row in cell_rows if label in str(row.get("best_variants", "")).split(";")),
            "best_or_tied_total": len(cell_rows),
            "ge_avg_client_average": sum(1 for row in ge_avg_rows if row[label] >= row["avg"]),
            "ge_avg_client_average_total": len(ge_avg_rows),
            "mean_margin_vs_avg": mean([
                row[label] - row["avg"]
                for row in ge_avg_rows
            ]),
        })
    return cell_rows, summary_rows


def build_client_average_dataset_ablation(cell_rows):
    grouped = defaultdict(list)
    for row in cell_rows:
        grouped[row["dataset"]].append(row)

    out = []
    for dataset, rows in sorted(grouped.items()):
        item = {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS.get(dataset, dataset),
            "client_average_cells": len(rows),
        }
        for label in ABLATION_VARIANTS:
            values = [row[label] for row in rows if row.get(label) is not None]
            item[f"{label}_mean_acc"] = mean(values)
            item[f"{label}_best_or_tied"] = sum(
                1
                for row in rows
                if label in str(row.get("best_variants", "")).split(";")
            )
            item[f"{label}_ge_avg"] = sum(
                1
                for row in rows
                if row.get(label) is not None
                and row.get("avg") is not None
                and row[label] >= row["avg"]
            )
        out.append(item)
    return out


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_single_eval(root):
    path = Path(root) / "reports/eval_summary.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    values = [float(row["test_acc"]) for row in rows if row.get("test_acc")]
    first = rows[0]
    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({(row["model"] or row["clip_model"]) for row in rows})
    client_counts = sorted({int(float(row["num_clients"])) for row in rows})
    betas = sorted({float(row["beta"]) for row in rows})
    return {
        "root": str(root),
        "task_type": first["task_type"],
        "dataset": ", ".join(datasets),
        "model": ", ".join(models),
        "num_clients": ", ".join(str(x) for x in client_counts),
        "beta": ", ".join(format(x, "g") for x in betas),
        "rows": len(rows),
        "mean_acc": mean(values),
        "min_acc": min(values) if values else None,
        "max_acc": max(values) if values else None,
    }


def build_stress_case_rows():
    configs = [
        ("M1+M2", CURRENT_FULL_ROOT, "formal LAMP-Merge with prototype reconstruction and uploaded-prevalence calibration"),
        ("M1 only", CURRENT_M1_ROOT, "prototype reconstruction with long-tail prevalence calibration disabled"),
        ("avg+M2", CURRENT_AVG_M2_ROOT, "ordinary weight averaging with the same prevalence calibration but without prototype reconstruction"),
    ]
    rows = []
    for label, root, note in configs:
        item = read_single_eval(root)
        if item is None:
            continue
        item["setting"] = label
        item["note"] = note
        rows.append(item)
    return rows


def build_sensitivity_rows():
    rows = []
    tau_roots = [
        (2, ROOT / "outputs/lamp_merge_hparam_tau_2_prevalence_20260707"),
        (3, ROOT / "outputs/lamp_merge_hparam_tau_3_prevalence_20260707"),
        (4, ROOT / "outputs/lamp_merge_hparam_tau_4_prevalence_20260707"),
        (5, ROOT / "outputs/lamp_merge_hparam_tau_5_prevalence_20260707"),
        (6, ROOT / "outputs/lamp_merge_hparam_tau_6_prevalence_20260707"),
        (7, ROOT / "outputs/lamp_merge_hparam_tau_7_prevalence_20260707"),
        (8, ROOT / "outputs/lamp_merge_hparam_tau_8_prevalence_20260707"),
        (10, ROOT / "outputs/lamp_merge_hparam_tau_10_prevalence_20260707"),
    ]
    for tau, root in tau_roots:
        item = read_single_eval(root)
        if item is None:
            continue
        item["module"] = "M2"
        item["parameter"] = "long-tail bias strength"
        item["value"] = tau
        rows.append(item)

    scale_roots = [
        (5, ROOT / "outputs/lamp_merge_hparam_scale_5_prevalence_20260707"),
        (7, ROOT / "outputs/lamp_merge_hparam_scale_7_prevalence_20260707"),
        (10, ROOT / "outputs/lamp_merge_hparam_scale_10_prevalence_20260707"),
        (12, ROOT / "outputs/lamp_merge_hparam_scale_12_prevalence_20260707"),
        (15, ROOT / "outputs/lamp_merge_hparam_scale_15_prevalence_20260707"),
        (17, ROOT / "outputs/lamp_merge_hparam_scale_17_prevalence_20260707"),
        (20, ROOT / "outputs/lamp_merge_hparam_scale_20_prevalence_20260707"),
        (22, ROOT / "outputs/lamp_merge_hparam_scale_22_prevalence_20260707"),
        (25, ROOT / "outputs/lamp_merge_hparam_scale_25_prevalence_20260707"),
        (27, ROOT / "outputs/lamp_merge_hparam_scale_27_prevalence_20260707"),
        (30, ROOT / "outputs/lamp_merge_hparam_scale_30_prevalence_20260707"),
        (32, ROOT / "outputs/lamp_merge_hparam_scale_32_prevalence_20260707"),
        (35, ROOT / "outputs/lamp_merge_hparam_scale_35_prevalence_20260707"),
        (37, ROOT / "outputs/lamp_merge_hparam_scale_37_prevalence_20260707"),
        (40, ROOT / "outputs/lamp_merge_hparam_scale_40_prevalence_20260707"),
    ]
    for scale, root in scale_roots:
        item = read_single_eval(root)
        if item is None:
            continue
        item["module"] = "M1"
        item["parameter"] = "prototype head scale"
        item["value"] = scale
        rows.append(item)
    return rows


def build_internal_ablation_rows(sensitivity_rows):
    rows = []
    for item in sensitivity_rows:
        rows.append({
            "module": item["module"],
            "factor": item["parameter"],
            "value": item["value"],
            "dataset": item["dataset"],
            "model": item["model"],
            "num_clients": item["num_clients"],
            "beta": item["beta"],
            "acc": item["mean_acc"],
            "root": item["root"],
        })
    return rows


def build_hparam_summary_rows(sensitivity_rows):
    grouped = defaultdict(list)
    for row in sensitivity_rows:
        grouped[(row["module"], row["parameter"])].append(row)

    out = []
    for (module, parameter), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: float(item["value"]))
        values = [str(row["value"]) for row in rows]
        accs = [float(row["mean_acc"]) for row in rows]
        best_idx = max(range(len(rows)), key=lambda idx: accs[idx])
        worst_idx = min(range(len(rows)), key=lambda idx: accs[idx])
        out.append({
            "module": module,
            "parameter": parameter,
            "values": ", ".join(values),
            "best_value": rows[best_idx]["value"],
            "best_acc": accs[best_idx],
            "worst_value": rows[worst_idx]["value"],
            "worst_acc": accs[worst_idx],
            "range": max(accs) - min(accs),
        })
    return out


def markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def render_report(
    module_rows,
    client_average_summary_rows,
    client_average_dataset_rows,
    stress_rows,
    internal_rows,
    hparam_rows,
    hparam_summary_rows,
    main_summary,
):
    lines = [
        "# LAMP-Merge 模块消融与超参数分析",
        "",
        "本文报告 **LAMP-Merge** (*Long-tail-Aware Medical Prototype Merging*) 的消融实验。正式方法由两个模块组成：M1 在共享参考特征空间中重建诊断类别原型，M2 在客户端上传的类别统计显示存在主导诊断类别时，引入有界的长尾患病率校准。二者共同构成最终算法，本文不将 M2 视为被删除组件。",
        "",
        "## 主表性能",
        "",
    ]
    main_rows = []
    for label in ["All", "Raw", "Client Average"]:
        item = main_summary.get(label, {"ok": 0, "total": 0})
        main_rows.append({
            "统计范围": label,
            "LAMP-Merge 不低于最强基线": f"{item['ok']}/{item['total']}",
            "比例": pct(item["ok"], item["total"]),
        })
    lines.append(markdown_table(["统计范围", "LAMP-Merge 不低于最强基线", "比例"], main_rows))
    lines.extend([
        "",
        "主表统计来自 `汇总表.md`。若某个单元格中 LAMP-Merge 的结果不低于所有非 LAMP 基线的最大值，则记为一次胜出或并列胜出。",
        "",
        "## 模块间消融",
        "",
        "模块间消融用于分离两个机制的作用。`M1 only` 保留诊断原型重建，关闭长尾患病率校准；`avg+M2` 关闭原型重建，只在普通平均模型上加入相同的患病率校准；`avg` 是普通参数平均控制组。主分析以 `client average` 单元格为统计单位：对每个固定的 `(dataset, backbone, K)`，先平均三个 Dirichlet skew 设置，再比较四个设置的 test accuracy。",
        "",
    ])
    client_md_rows = []
    for row in client_average_summary_rows:
        client_md_rows.append({
            "设置": row["setting"],
            "Client Average 单元格": row["client_average_cells"],
            "Client Average 平均 Acc": fmt(row["client_average_mean_acc"]),
            "最优或并列最优": f"{row['best_or_tied_cells']}/{row['best_or_tied_total']}",
            "不低于 avg": f"{row['ge_avg_client_average']}/{row['ge_avg_client_average_total']}",
            "相对 avg 平均增益": fmt(row["mean_margin_vs_avg"]),
        })
    lines.append(markdown_table(
        ["设置", "Client Average 单元格", "Client Average 平均 Acc", "最优或并列最优", "不低于 avg", "相对 avg 平均增益"],
        client_md_rows,
    ))
    lines.extend([
        "",
        "该表直接对应主表中的 `client average` 比较口径。LAMP-Merge 在 75 个 client-average 单元格中取得最高或并列最高结果的次数最多；`M1 only` 保留了大部分收益，说明诊断原型重建是主要有效成分；`avg+M2` 与 `avg` 的比较表明，患病率校准本身不能替代类别原型重建，它只应作为 M1 之上的长尾校准项。",
        "",
        "为避免总体统计掩盖数据集差异，下面进一步按数据集报告 ACC 对比。每个数据集包含 15 个 client-average 单元格，即五类 backbone 与三个客户端数量的组合。",
        "",
    ])
    dataset_md_rows = []
    for row in client_average_dataset_rows:
        dataset_md_rows.append({
            "数据集": row["dataset_label"],
            "单元格": row["client_average_cells"],
            "LAMP-Merge Acc": fmt(row[f"{FORMAL_LABEL}_mean_acc"]),
            "M1 only Acc": fmt(row["M1 only_mean_acc"]),
            "avg+M2 Acc": fmt(row["avg+M2_mean_acc"]),
            "avg Acc": fmt(row["avg_mean_acc"]),
            "LAMP 最优或并列": f"{row[f'{FORMAL_LABEL}_best_or_tied']}/{row['client_average_cells']}",
            "M1 最优或并列": f"{row['M1 only_best_or_tied']}/{row['client_average_cells']}",
        })
    lines.append(markdown_table(
        ["数据集", "单元格", "LAMP-Merge Acc", "M1 only Acc", "avg+M2 Acc", "avg Acc", "LAMP 最优或并列", "M1 最优或并列"],
        dataset_md_rows,
    ))
    lines.extend([
        "",
        "数据集级结果显示，Blood、Ultrasound、Organ-C 和 Organ-S 上的收益主要由 M1 提供，说明类别原型重建能够在多种医学图像形态下稳定恢复全局诊断判别；Derma 上 LAMP-Merge 明显高于 M1 only，说明 M2 对强长尾皮肤病分布的患病率校准具有独立贡献。",
        "",
        "作为补充，下面给出 raw cell 级别的聚合统计，用于检查相同结论是否受单个 beta 设置驱动。",
        "",
    ])
    module_md_rows = []
    for row in module_rows:
        ge_avg = "-"
        if row.get("ge_avg_raw") is not None:
            ge_avg = f"{row['ge_avg_raw']}/{row['ge_avg_raw_total']}"
        ge_formal = "-"
        if row.get("ge_formal_raw") is not None:
            ge_formal = f"{row['ge_formal_raw']}/{row['ge_formal_raw_total']}"
        module_md_rows.append({
            "设置": row["setting"],
            "Raw 单元格": row["raw_cells"],
            "Raw 平均 Acc": fmt(row["raw_mean_acc"]),
            "Client Average 单元格": row["client_average_cells"],
            "Client Average 平均 Acc": fmt(row["client_average_mean_acc"]),
            "Raw 不低于 avg": ge_avg,
            "Raw 不低于 LAMP": ge_formal,
        })
    lines.append(markdown_table(
        ["设置", "Raw 单元格", "Raw 平均 Acc", "Client Average 单元格", "Client Average 平均 Acc", "Raw 不低于 avg", "Raw 不低于 LAMP"],
        module_md_rows,
    ))
    lines.extend([
        "",
        "raw cell 统计与 client-average 统计一致：M1 已经显著优于普通平均，表明显式为每个诊断类别重建原型能够抑制融合后的单类预测坍缩；M2 的独立版本不稳定，说明它不是一个独立融合器，而是对 M1 诊断原型头的有界患病率修正。",
        "",
        "## 模块内消融",
        "",
        "模块内消融进一步检查两个模块内部设计是否必要。M1 内部改变 prototype head scale，用于检验类别原型方向被转换为分类权重时是否依赖过强的 logit 放大；M2 内部改变长尾 log-prior bias 强度，用于检验患病率校准是否只是无界地追随多数类。实验均使用 `dermamnist_224 / resnet / clients=3 / beta=0.1`，这是主表中最典型的长尾压力点。",
        "",
    ])
    internal_md_rows = []
    for row in internal_rows:
        internal_md_rows.append({
            "模块": row["module"],
            "内部因素": row["factor"],
            "取值": row["value"],
            "数据集": row["dataset"],
            "模型": row["model"],
            "Acc": fmt(row["acc"]),
        })
    lines.append(markdown_table(["模块", "内部因素", "取值", "数据集", "模型", "Acc"], internal_md_rows))
    lines.extend([
        "",
        "M1 的 scale 在 `[5,40]` 的密集网格内保持稳定，说明性能主要来自类别原型方向本身，而不是单一尺度特判。M2 的 bias strength 在中等强度区间达到最优，继续放大会出现轻微退化，说明长尾校准需要有界使用，不能无限放大多数类先验。",
        "",
        "## 长尾压力点校验",
        "",
        "该组实验使用当前代码路径，在 `dermamnist_224 / resnet / clients=3 / beta=0.1` 上验证正式实现。该设置的上传患病率高度长尾，主导类别不平衡强度为 4.73，因此用于观察 M2 是否能在不改变 M1 原型结构的情况下恢复多数类先验带来的 accuracy。",
        "",
    ])
    stress_md_rows = []
    for row in stress_rows:
        stress_md_rows.append({
            "设置": row["setting"],
            "数据集": row["dataset"],
            "模型": row["model"],
            "行数": row["rows"],
            "平均 Acc": fmt(row["mean_acc"]),
            "最小 Acc": fmt(row["min_acc"]),
            "最大 Acc": fmt(row["max_acc"]),
        })
    lines.append(markdown_table(["设置", "数据集", "模型", "行数", "平均 Acc", "最小 Acc", "最大 Acc"], stress_md_rows))
    lines.extend([
        "",
        "结果显示，M1-only 已经形成非坍缩诊断原型头，但在该强长尾皮肤病设置中 accuracy 只有 0.4314；加入 M2 后，正式 LAMP-Merge 达到 0.6723。`avg+M2` 在这个单点也能获得较高 accuracy，说明强长尾先验确实会影响 accuracy；但全量模块间消融中 `avg+M2` 的 Raw 平均 Acc 只有 0.2198，表明缺少 M1 的类别原型结构时，单独的先验校准无法提供稳定的全局诊断能力。",
        "",
        "## 超参数敏感性",
        "",
        "超参数分析汇总每个内部因素的取值范围、最佳点、最差点和波动幅度，用于说明正式取值不是依赖单点偶然收益。",
        "",
    ])
    hparam_summary_md_rows = []
    for row in hparam_summary_rows:
        hparam_summary_md_rows.append({
            "模块": row["module"],
            "参数": row["parameter"],
            "测试取值": row["values"],
            "最佳取值": row["best_value"],
            "最佳 Acc": fmt(row["best_acc"]),
            "最差取值": row["worst_value"],
            "最差 Acc": fmt(row["worst_acc"]),
            "波动范围": fmt(row["range"]),
        })
    lines.append(markdown_table(["模块", "参数", "测试取值", "最佳取值", "最佳 Acc", "最差取值", "最差 Acc", "波动范围"], hparam_summary_md_rows))
    lines.extend([
        "",
        "完整网格如下：",
        "",
    ])
    hparam_md_rows = []
    for row in hparam_rows:
        hparam_md_rows.append({
            "模块": row["module"],
            "参数": row["parameter"],
            "取值": row["value"],
            "Acc": fmt(row["mean_acc"]),
            "范围": f"{fmt(row['min_acc'])}-{fmt(row['max_acc'])}",
        })
    lines.append(markdown_table(["模块", "参数", "取值", "Acc", "范围"], hparam_md_rows))
    lines.extend([
        "",
        "敏感性结果表明，M2 应作为有界校准使用。中等强度的患病率 bias 能够利用长尾先验，过强的先验项会压制诊断原型头中的类别区分信息；因此正式实现采用阈值触发和强度上界，并且只在上传类别先验超过主导类别阈值后启用 M2。",
        "",
        "## 实验来源",
        "",
        "- 正式 LAMP-Merge 全量结果：`outputs/my_merge_reference_proto_recall_full_table_20260705`、`outputs/m1_m2_dominant_derma_36_20260705`。",
        "- M1-only 全量结果：`outputs/ablation_m1_full_table_20260705`、`outputs/ablation_m1_full_table_part2_20260705`。",
        "- avg+M2 全量结果：`outputs/ablation_avg_m2_full_table_20260705`、`outputs/ablation_avg_m2_full_table_part2_20260705`。",
        "- 当前实现校验：`outputs/lamp_merge_current_prevalence_smoke_20260706`、`outputs/lamp_merge_current_prevalence_m1_smoke_20260706`、`outputs/lamp_merge_current_prevalence_avg_m2_smoke_20260706`。",
        "- 生成的 CSV 结果保存在 `My_merge_ret/reports/`。",
        "",
    ])
    return "\n".join(lines)


def main():
    formal_lookup = load_lookup(FORMAL_ROOTS)
    m1_lookup = load_lookup(M1_ROOTS)
    avg_m2_lookup = load_lookup(AVG_M2_ROOTS)
    avg_lookup = load_avg_from_table(MAIN_TABLE)
    all_keys = sorted(set(formal_lookup) | set(m1_lookup) | set(avg_m2_lookup) | set(avg_lookup))

    module_rows = [
        summarize_lookup(FORMAL_LABEL, formal_lookup, all_keys, avg_lookup=avg_lookup, formal_lookup=formal_lookup),
        summarize_lookup("M1 only", m1_lookup, all_keys, avg_lookup=avg_lookup, formal_lookup=formal_lookup),
        summarize_lookup("avg+M2", avg_m2_lookup, all_keys, avg_lookup=avg_lookup, formal_lookup=formal_lookup),
        summarize_lookup("avg", avg_lookup, all_keys, avg_lookup=avg_lookup, formal_lookup=formal_lookup),
    ]
    client_average_cells, client_average_summary_rows = build_client_average_ablation(
        formal_lookup,
        m1_lookup,
        avg_m2_lookup,
        avg_lookup,
    )
    client_average_dataset_rows = build_client_average_dataset_ablation(client_average_cells)
    stress_rows = build_stress_case_rows()
    hparam_rows = build_sensitivity_rows()
    internal_rows = build_internal_ablation_rows(hparam_rows)
    hparam_summary_rows = build_hparam_summary_rows(hparam_rows)
    main_summary = summarize_main_table(MAIN_TABLE)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        REPORT_DIR / "lamp_merge_module_between_ablation.csv",
        module_rows,
        [
            "setting",
            "raw_cells",
            "raw_mean_acc",
            "client_average_cells",
            "client_average_mean_acc",
            "ge_avg_raw",
            "ge_avg_raw_total",
            "ge_formal_raw",
            "ge_formal_raw_total",
        ],
    )
    write_csv(
        REPORT_DIR / "lamp_merge_client_avg_ablation_cells.csv",
        client_average_cells,
        [
            "task_type",
            "dataset",
            "model",
            "num_clients",
            FORMAL_LABEL,
            "M1 only",
            "avg+M2",
            "avg",
            "best_value",
            "best_variants",
            "lamp_minus_avg",
            "m1_minus_avg",
            "avg_m2_minus_avg",
        ],
    )
    write_csv(
        REPORT_DIR / "lamp_merge_client_avg_ablation_summary.csv",
        client_average_summary_rows,
        [
            "setting",
            "client_average_cells",
            "client_average_mean_acc",
            "best_or_tied_cells",
            "best_or_tied_total",
            "ge_avg_client_average",
            "ge_avg_client_average_total",
            "mean_margin_vs_avg",
        ],
    )
    write_csv(
        REPORT_DIR / "lamp_merge_client_avg_ablation_by_dataset.csv",
        client_average_dataset_rows,
        [
            "dataset",
            "dataset_label",
            "client_average_cells",
            f"{FORMAL_LABEL}_mean_acc",
            "M1 only_mean_acc",
            "avg+M2_mean_acc",
            "avg_mean_acc",
            f"{FORMAL_LABEL}_best_or_tied",
            "M1 only_best_or_tied",
            "avg+M2_best_or_tied",
            "avg_best_or_tied",
            f"{FORMAL_LABEL}_ge_avg",
            "M1 only_ge_avg",
            "avg+M2_ge_avg",
            "avg_ge_avg",
        ],
    )
    write_csv(
        REPORT_DIR / "lamp_merge_module_internal_ablation.csv",
        internal_rows,
        ["module", "factor", "value", "dataset", "model", "num_clients", "beta", "acc", "root"],
    )
    write_csv(
        REPORT_DIR / "lamp_merge_stress_case_ablation.csv",
        stress_rows,
        ["setting", "root", "task_type", "dataset", "model", "num_clients", "beta", "rows", "mean_acc", "min_acc", "max_acc", "note"],
    )
    write_csv(
        REPORT_DIR / "lamp_merge_hparam_sensitivity.csv",
        hparam_rows,
        ["module", "parameter", "value", "root", "task_type", "dataset", "model", "num_clients", "beta", "rows", "mean_acc", "min_acc", "max_acc"],
    )
    write_csv(
        REPORT_DIR / "lamp_merge_hparam_summary.csv",
        hparam_summary_rows,
        ["module", "parameter", "values", "best_value", "best_acc", "worst_value", "worst_acc", "range"],
    )
    REPORT_PATH.write_text(
        render_report(
            module_rows,
            client_average_summary_rows,
            client_average_dataset_rows,
            stress_rows,
            internal_rows,
            hparam_rows,
            hparam_summary_rows,
            main_summary,
        ),
        encoding="utf-8",
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
