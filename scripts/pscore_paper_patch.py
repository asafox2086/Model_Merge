#!/usr/bin/env python3
"""Print an apply_patch patch adding audited Pscore rows and updating main-table gains."""

import argparse
import csv
import difflib
import re
from pathlib import Path
from statistics import mean

from audit_tmi_revision import DATASETS, METHODS, MODELS, PAPER, ROOT, SOURCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    with SOURCE.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] in METHODS.values()]
    with args.results.open() as handle:
        added = list(csv.DictReader(handle))
    if len(added) != 180 or any(row["status"] != "OK" for row in added):
        raise ValueError("Only complete audited results can enter the paper")
    rows.extend(added)
    original = PAPER.read_text()
    updated = original
    for model, label in MODELS.items():
        start = updated.index(r"\label{tab:client-avg-" + label + "}")
        end = updated.index(r"\end{table*}", start)
        table = updated[start:end]
        means = {}
        for method in [*METHODS.values(), "pscore_mlp"]:
            values = []
            for dataset in (DATASETS[0], DATASETS[1], DATASETS[2], DATASETS[4]):
                for clients in (3, 5, 7, None):
                    selected = [row for row in rows if row["model"] == model and row["method"] == method
                                and row["dataset"] == dataset
                                and (clients is None or int(row["num_clients"]) == clients)]
                    values.append(tuple(100 * mean(float(row[metric]) for row in selected) for metric in ("accuracy", "macro_f1")))
            means[method] = values
        maxima = [tuple(max(means[method][index][metric] for method in means) for metric in range(2)) for index in range(16)]
        baseline_maxima = [tuple(max(means[method][index][metric] for method in means if method != "lamp_merge") for metric in range(2)) for index in range(16)]

        def cells(method):
            output = []
            for index, pair in enumerate(means[method]):
                values = [r"\textbf{" + f"{value:.2f}" + "}" if abs(value - maxima[index][metric]) < 1e-10 else f"{value:.2f}"
                          for metric, value in enumerate(pair)]
                display = " / ".join(values)
                if method == "lamp_merge":
                    gain = " / ".join(f"{value - baseline_maxima[index][metric]:+.2f}" for metric, value in enumerate(pair))
                    display = r"\tworowbestreshl{" + display + "}{" + gain + "}"
                output.append(display)
            return " & ".join(output)

        for display, method in METHODS.items():
            for line in table.splitlines():
                if line.startswith(display + " &") or line.startswith(display + r"\pub"):
                    replacement = line.split(" &", 1)[0] + " & " + cells(method) + r" \\"
                    table = table.replace(line, replacement, 1)
                    break
        new_row = r"Pscore-MLP (adapted)\pub{TMI 2022} & " + cells("pscore_mlp") + r" \\" + "\n"
        if "Pscore-MLP (adapted)" in table:
            table = re.sub(r"^Pscore-MLP \(adapted\).*\n", lambda match: new_row, table, flags=re.MULTILINE)
        else:
            marker = "\\hline \\hline\n\\rowcolor[HTML]{FFF9C4}"
            if marker not in table:
                raise ValueError("Cannot locate LAMP row insertion point")
            table = table.replace(marker, "\\hdashline\n" + new_row + "\n" + marker, 1)
        updated = updated[:start] + table + updated[end:]
    difference = list(difflib.unified_diff(original.splitlines(), updated.splitlines(), n=3, lineterm=""))
    if difference:
        print("*** Begin Patch")
        print("*** Update File: paper/LAMP_Merge_TMI.tex")
        for line in difference[2:]:
            print("@@" if line.startswith("@@") else line)
        print("*** End Patch")


if __name__ == "__main__":
    main()
