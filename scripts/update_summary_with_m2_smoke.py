#!/usr/bin/env python3
import csv
import json
from pathlib import Path


CASE = ("small", "bloodmnist_224", "resnet", 3, 0.0)
START = "<!-- M2_ROUTED_SMOKE_START -->"
END = "<!-- M2_ROUTED_SMOKE_END -->"


def fmt(value):
    return f"{value:.4f}"


def read_eval_rows(csv_path):
    rows = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") != "my_merge":
                continue
            model = row["model"] if row["task_type"] == "small" else row["clip_model"]
            key = (
                row["task_type"],
                row["dataset"],
                model,
                int(float(row["num_clients"])),
                float(row["beta"]),
            )
            if key == CASE:
                rows.append(row)
    return rows


def old_ablation_rows(grid_root):
    out = []
    for ablation in ["full", "no_client_information", "no_fusion_selection", "avg_only"]:
        for csv_path in sorted((grid_root / ablation).glob("**/reports/eval_summary.csv")):
            for row in read_eval_rows(csv_path):
                out.append(
                    {
                        "source": "old full-run",
                        "variant": ablation,
                        "test_acc": float(row["test_acc"]),
                        "selected_candidate": "-",
                        "fusion_rule": "-",
                    }
                )
    return out


def smoke_row(smoke_root):
    eval_rows = read_eval_rows(smoke_root / "reports" / "eval_summary.csv")
    if not eval_rows:
        return []
    selected_candidate = "-"
    fusion_rule = "-"
    merge_jsons = sorted(smoke_root.glob("**/merge_result.json"))
    if merge_jsons:
        payload = json.loads(merge_jsons[0].read_text(encoding="utf-8"))
        method_info = payload.get("method_info", {})
        selected_candidate = method_info.get("selected_candidate", "-")
        fusion_rule = method_info.get("fusion_rule", "-")
    row = eval_rows[0]
    return [
        {
            "source": "new M2 smoke",
            "variant": "full_routed_m2",
            "test_acc": float(row["test_acc"]),
            "selected_candidate": selected_candidate,
            "fusion_rule": fusion_rule,
        }
    ]


def render(rows):
    lines = [
        START,
        "## New M2 Smoke Check",
        "",
        "- This is a single-case smoke result for the rewritten M2, not a full rerun.",
        "- Case: `small / bloodmnist_224 / resnet / c=3 / beta=0 / seed=42`.",
        "- Old rows are automatically read from the previous full-run output; the new row is read from `outputs/m2_routed_smoke2`.",
        "",
        "| source | variant | test_acc | selected_candidate | fusion_rule |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | `{row['variant']}` | {fmt(row['test_acc'])} | `{row['selected_candidate']}` | `{row['fusion_rule']}` |"
        )
    lines.extend(["", END, ""])
    return "\n".join(lines)


def replace_section(text, section):
    if START in text and END in text:
        before = text.split(START, 1)[0].rstrip()
        after = text.split(END, 1)[1].lstrip()
        return before + "\n\n" + section + after
    title_end = text.find("\n## ")
    if title_end == -1:
        return text.rstrip() + "\n\n" + section
    return text[:title_end].rstrip() + "\n\n" + section + text[title_end:]


def main():
    grid_root = Path("outputs/my_merge_full_weights_20260523_171624/my_merge_ablation_grid")
    smoke_root = Path("outputs/m2_routed_smoke2")
    summary_path = Path("My_merge_ret/汇总表.md")
    report_path = Path("My_merge_ret/reports/m2_routed_smoke_check.md")

    rows = old_ablation_rows(grid_root) + smoke_row(smoke_root)
    section = render(rows)
    report_path.write_text(section.replace(START + "\n", "").replace("\n" + END, ""), encoding="utf-8")
    summary_path.write_text(replace_section(summary_path.read_text(encoding="utf-8"), section), encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
