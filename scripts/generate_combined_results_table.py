#!/usr/bin/env python3
import argparse
import re
from copy import deepcopy
from pathlib import Path


HEADING_RE = re.compile(r"^(#{2,4})\s+(.*)$")
TD_RE = re.compile(r"<td>(.*?)</td>")
TABLE_START = "<table>"
TABLE_END = "</table>"
TBODY_START = "  <tbody>"
TBODY_END = "  </tbody>"


def parse_args():
    p = argparse.ArgumentParser("Merge baseline and my_merge master tables with ranking highlights")
    p.add_argument("--base", required=True, help="Baseline markdown, e.g. result/all_results.md")
    p.add_argument("--mine", required=True, help="my_merge markdown, e.g. My_merge_ret/all_results_my_merge.md")
    p.add_argument("--dest", required=True, help="Destination markdown path")
    return p.parse_args()


def parse_tables(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    intro_lines = []
    tables = {}
    current_section = None
    current_model = None
    current_table_kind = None
    in_table = False
    table_lines = []

    for line in lines:
        heading = HEADING_RE.match(line)
        if heading and not in_table:
            level, title = heading.groups()
            if level == "##":
                current_section = title
            elif level == "###":
                current_model = title
            elif level == "####":
                current_table_kind = title
            if not tables and not current_section:
                intro_lines.append(line)
            continue

        if not tables and current_section is None:
            intro_lines.append(line)

        if line.strip() == TABLE_START:
            in_table = True
            table_lines = [line]
            continue

        if in_table:
            table_lines.append(line)
            if line.strip() == TABLE_END:
                key = (current_section, current_model, current_table_kind)
                tables[key] = parse_single_table(table_lines)
                in_table = False
                table_lines = []
            continue

    return intro_lines, tables


def parse_single_table(lines):
    tbody_idx = lines.index(TBODY_START)
    tbody_end_idx = lines.index(TBODY_END)
    header_lines = lines[:tbody_idx + 1]
    footer_lines = lines[tbody_end_idx:]
    body_lines = lines[tbody_idx + 1:tbody_end_idx]
    rows = []
    row_buf = []
    for line in body_lines:
        row_buf.append(line)
        if line.strip() == "</tr>":
            cells = TD_RE.findall("\n".join(row_buf))
            if cells:
                rows.append({
                    "method": cells[0],
                    "values": cells[1:],
                })
            row_buf = []
    return {
        "header_lines": header_lines,
        "footer_lines": footer_lines,
        "rows": rows,
    }


def try_parse_number(value):
    text = re.sub(r"<.*?>", "", value).strip()
    if text in {"", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def highlight_rows(rows):
    if not rows:
        return rows
    row_count = len(rows)
    col_count = len(rows[0]["values"])
    highlighted = deepcopy(rows)

    for col_idx in range(col_count):
        values = []
        for row_idx in range(row_count):
            num = try_parse_number(rows[row_idx]["values"][col_idx])
            if num is not None:
                values.append(num)
        if not values:
            continue
        unique_desc = sorted(set(values), reverse=True)
        top = unique_desc[0]
        second = unique_desc[1] if len(unique_desc) > 1 else None

        for row_idx in range(row_count):
            raw = rows[row_idx]["values"][col_idx]
            num = try_parse_number(raw)
            if num is None:
                continue
            if num == top:
                highlighted[row_idx]["values"][col_idx] = f"<strong>{raw}</strong>"
            elif second is not None and num == second:
                highlighted[row_idx]["values"][col_idx] = f"<ins>{raw}</ins>"
    return highlighted


def merge_tables(base_tables, my_tables):
    merged = {}
    for key, base_table in base_tables.items():
        rows = deepcopy(base_table["rows"])
        mine = my_tables.get(key)
        if mine and mine["rows"]:
            existing_methods = {row["method"] for row in rows}
            for my_row in mine["rows"]:
                if my_row["method"] not in existing_methods:
                    rows.append(my_row)
                    existing_methods.add(my_row["method"])
        merged[key] = {
            "header_lines": base_table["header_lines"],
            "footer_lines": base_table["footer_lines"],
            "rows": highlight_rows(rows),
        }
    return merged


def render_table(table):
    lines = []
    lines.extend(table["header_lines"])
    for row in table["rows"]:
        lines.append("    <tr>")
        lines.append(f"      <td>{row['method']}</td>")
        for value in row["values"]:
            lines.append(f"      <td>{value}</td>")
        lines.append("    </tr>")
    lines.extend(table["footer_lines"])
    return lines


def build_output(base_intro, merged_tables):
    order = []
    for section in ["Small", "VLM"]:
        if section == "Small":
            models = ["resnet", "convnext", "vit_t", "swin_tiny"]
        else:
            models = ["openai/clip-vit-base-patch32"]
        for model in models:
            for kind in ["Raw", "Client Average"]:
                key = (section, model, kind)
                if key in merged_tables:
                    order.append(key)

    lines = [
        "# Experiment Master Tables",
        "",
        "- Combined from `result/all_results.md` and the generated my_merge result table.",
        "- Original baseline values are preserved; this file adds the formal `my_merge` row.",
        "- Highlight rule: highest value in each column is `<strong>bold</strong>`, second-highest distinct value is `<ins>underlined</ins>`.",
        "",
    ]

    current_section = None
    current_model = None
    for section, model, kind in order:
        if section != current_section:
            lines.append(f"## {section}")
            lines.append("")
            current_section = section
            current_model = None
        if model != current_model:
            lines.append(f"### {model}")
            lines.append("")
            current_model = model
        lines.append(f"#### {kind}")
        lines.append("")
        lines.extend(render_table(merged_tables[(section, model, kind)]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    base_intro, base_tables = parse_tables(Path(args.base))
    _my_intro, my_tables = parse_tables(Path(args.mine))
    merged = merge_tables(base_tables, my_tables)
    content = build_output(base_intro, merged)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
