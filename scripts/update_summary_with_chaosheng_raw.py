#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


START = "<!-- CHAOSHENG_RAW_M2_START -->"
END = "<!-- CHAOSHENG_RAW_M2_END -->"
DATASET = "chaoshengmnist_224"
SETTINGS = [(3, 0.0), (3, 0.01), (3, 0.1), (5, 0.0), (5, 0.01), (5, 0.1), (7, 0.0), (7, 0.01), (7, 0.1)]
ABLATIONS = ["full", "no_client_information", "no_fusion_selection", "avg_only"]
ABLATION_LABELS = {
    "full": "full",
    "no_client_information": "-M1",
    "no_fusion_selection": "-M2",
    "avg_only": "avg",
}
MODEL_ORDER = ["resnet", "convnext", "vit_t", "swin_tiny", "clip-vit-base-patch32"]
MODEL_LABELS = {
    "resnet": "resnet",
    "convnext": "convnext",
    "vit_t": "vit_t",
    "swin_tiny": "swin_tiny",
    "clip-vit-base-patch32": "clip-vit-base-patch32",
}


def fmt(value):
    return "-" if value is None else f"{value:.4f}"


def display_beta(beta):
    return f"{beta:g}"


def model_name(row):
    if row["task_type"] == "small":
        return row["model"]
    return row["clip_model"].split("/")[-1]


def discover_eval_csvs(grid_root):
    return sorted(Path(grid_root).glob("*/*/reports/eval_summary.csv"))


def load_lookup(grid_root):
    lookup = {}
    for csv_path in discover_eval_csvs(grid_root):
        ablation = csv_path.relative_to(grid_root).parts[0]
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("method") != "my_merge" or row.get("dataset") != DATASET:
                    continue
                key = (ablation, model_name(row), int(float(row["num_clients"])), float(row["beta"]))
                lookup[key] = float(row["test_acc"])
    return lookup


def load_fusion_rules(grid_root):
    counts = defaultdict(lambda: defaultdict(int))
    for json_path in Path(grid_root).glob("*/*/merged/**/merge_result.json"):
        ablation = json_path.relative_to(grid_root).parts[0]
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("dataset") != DATASET:
            continue
        info = payload.get("method_info", {})
        rule = info.get("fusion_rule") or info.get("selected_candidate") or "-"
        counts[ablation][rule] += 1
    return counts


def render_table(lines, lookup, model):
    headers = [f"c{c}_b{display_beta(b)}" for c, b in SETTINGS]
    lines.append(f"### {MODEL_LABELS.get(model, model)}")
    lines.append("")
    lines.append("| variant | " + " | ".join(headers) + " | mean |")
    lines.append("| --- | " + " | ".join("---:" for _ in headers) + " | ---: |")
    for ablation in ABLATIONS:
        values = [lookup.get((ablation, model, c, b)) for c, b in SETTINGS]
        mean = sum(v for v in values if v is not None) / max(1, sum(v is not None for v in values))
        rendered = [fmt(v) for v in values]
        lines.append(f"| `{ABLATION_LABELS[ablation]}` | " + " | ".join(rendered) + f" | {fmt(mean)} |")
    lines.append("")


def render_section(grid_root):
    lookup = load_lookup(grid_root)
    fusion_counts = load_fusion_rules(grid_root)
    lines = [
        START,
        "## Chaosheng Raw Check With Rewritten M2",
        "",
        "- Dataset: `chaoshengmnist_224`.",
        "- This section is generated from the new M2 subset run, not manually filled.",
        f"- Source: `{grid_root}`.",
        "- `full` now uses `direct_evidence_routed_fusion`; `-M2` and `avg` are the average-fusion controls.",
        "",
        "### Fusion Rule Counts",
        "",
        "| variant | fusion rule counts |",
        "| --- | --- |",
    ]
    for ablation in ABLATIONS:
        items = ", ".join(f"`{name}`={count}" for name, count in sorted(fusion_counts[ablation].items())) or "-"
        lines.append(f"| `{ABLATION_LABELS[ablation]}` | {items} |")
    lines.append("")

    lines.extend(["### Raw Accuracy By Model", ""])
    for model in MODEL_ORDER:
        render_table(lines, lookup, model)
    lines.extend([END, ""])
    return "\n".join(lines)


def replace_section(text, section):
    if START in text and END in text:
        before = text.split(START, 1)[0].rstrip()
        after = text.split(END, 1)[1].lstrip()
        return before + "\n\n" + section + after
    marker = "<!-- M2_ROUTED_SMOKE_START -->"
    if marker in text:
        return text.replace(marker, section + marker, 1)
    title_end = text.find("\n## ")
    if title_end == -1:
        return text.rstrip() + "\n\n" + section
    return text[:title_end].rstrip() + "\n\n" + section + text[title_end:]


def main():
    grid_root = Path("outputs/my_merge_m2_routed_chaosheng_raw_20260528_163125/my_merge_ablation_grid")
    summary_path = Path("My_merge_ret/汇总表.md")
    report_path = Path("My_merge_ret/reports/chaosheng_raw_m2_check.md")
    section = render_section(grid_root)
    summary_path.write_text(replace_section(summary_path.read_text(encoding="utf-8"), section), encoding="utf-8")
    report_path.write_text(section.replace(START + "\n", "").replace("\n" + END, ""), encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
