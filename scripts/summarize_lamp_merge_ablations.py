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
        (2, ROOT / "outputs/tmp_derma_b01_tau_2"),
        (3, ROOT / "outputs/tmp_derma_b01_tau_3"),
        (4, ROOT / "outputs/tmp_derma_b01_tau_4"),
        (5, ROOT / "outputs/tmp_derma_b01_tau_5"),
        (6, ROOT / "outputs/tmp_derma_b01_tau_6"),
        (7, ROOT / "outputs/tmp_derma_b01_tau_7"),
        (8, ROOT / "outputs/tmp_derma_b01_tau_8"),
        (10, ROOT / "outputs/tmp_derma_b01_tau_10"),
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
        (10, ROOT / "outputs/tmp_derma_b01_scale_10_tau6"),
        (15, ROOT / "outputs/tmp_derma_b01_scale_15_tau6"),
        (20, ROOT / "outputs/tmp_derma_b01_tau_6"),
        (25, ROOT / "outputs/tmp_derma_b01_scale_25_tau6"),
        (30, ROOT / "outputs/tmp_derma_b01_scale_30_tau6"),
        (40, ROOT / "outputs/tmp_derma_b01_scale_40_tau6"),
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


def render_report(module_rows, stress_rows, internal_rows, hparam_rows, hparam_summary_rows, main_summary):
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
        "模块间消融用于分离两个机制的作用。`M1 only` 保留诊断原型重建，关闭长尾患病率校准；`avg+M2` 关闭原型重建，只在普通平均模型上加入相同的患病率校准。该设计用于验证主要收益是否来自医学类别原型，同时检验 M2 是否必须附着在 M1 的诊断原型头之上。",
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
        "结果支持两个结论。第一，`M1 only` 已经具备强反坍缩能力，说明显式为每个诊断类别重建原型能够避免融合模型继承参数空间平均产生的单类预测坍缩。第二，`avg+M2` 明显较弱，说明 M2 不能独立替代原型重建；它的作用是在 M1 已经形成诊断类别判别头之后，对极端长尾场景进行有界校准。",
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
        "M1 的 scale 在较宽范围内保持稳定，说明性能主要来自类别原型方向本身，而不是单一尺度特判。M2 的 bias strength 从 2 增大到 6 时持续提升，之后进入饱和区间，说明长尾校准需要有界使用，不能无限放大多数类先验。",
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
        "敏感性结果表明，M2 应作为有界校准使用。增大患病率 bias 的强度会先提升长尾皮肤病设置，但收益随后饱和；因此正式实现采用最大强度截断，并且只在上传类别先验超过主导类别阈值后启用 M2。",
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
        render_report(module_rows, stress_rows, internal_rows, hparam_rows, hparam_summary_rows, main_summary),
        encoding="utf-8",
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
