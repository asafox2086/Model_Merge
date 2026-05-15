#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("Plot my_merge hyperparameter sensitivity from eval summaries")
    p.add_argument("--root", required=True, help="Root containing one subdirectory per hyperparameter setting")
    p.add_argument("--dest-dir", required=True)
    return p.parse_args()


def setting_name(root, csv_path):
    rel = csv_path.parent.parent.relative_to(root)
    return rel.parts[0] if rel.parts else "unknown"


def discover(root):
    root = Path(root)
    paths = set(root.glob("*/reports/eval_summary.csv"))
    paths.update(root.glob("*/*/reports/eval_summary.csv"))
    return sorted(path for path in paths if path.is_file())


def load_rows(root):
    rows = []
    for csv_path in discover(root):
        setting = setting_name(Path(root), csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("method") != "my_merge":
                    continue
                row["setting"] = setting
                row["test_acc"] = float(row["test_acc"])
                rows.append(row)
    return rows


def parse_setting(setting):
    out = {}
    for key in ["stats", "eval", "bn"]:
        match = re.search(rf"{key}([0-9]+|all)", setting)
        if match:
            raw = match.group(1)
            out[key] = 0 if raw == "all" else int(raw)
    return out


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(rows):
    by_setting = defaultdict(list)
    by_dataset_setting = defaultdict(list)
    for row in rows:
        by_setting[row["setting"]].append(row["test_acc"])
        by_dataset_setting[(row["dataset"], row["setting"])].append(row["test_acc"])
    summary = []
    for setting, values in sorted(by_setting.items()):
        params = parse_setting(setting)
        summary.append(
            {
                "setting": setting,
                "stats": params.get("stats", ""),
                "eval": params.get("eval", ""),
                "bn": params.get("bn", ""),
                "mean_acc": mean(values),
                "num_rows": len(values),
            }
        )
    dataset_summary = []
    for (dataset, setting), values in sorted(by_dataset_setting.items()):
        dataset_summary.append(
            {
                "dataset": dataset,
                "setting": setting,
                "mean_acc": mean(values),
                "num_rows": len(values),
            }
        )
    return summary, dataset_summary


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_overall(summary, dest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["setting"] for row in summary]
    values = [row["mean_acc"] for row in summary]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4.8), dpi=160)
    ax.plot(range(len(labels)), values, marker="o", linewidth=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("mean accuracy")
    ax.set_title("my_merge hyperparameter sensitivity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)


def plot_by_dataset(dataset_summary, dest_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in dataset_summary:
        grouped[row["dataset"]].append(row)
    for dataset, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["setting"])
        labels = [row["setting"] for row in rows]
        values = [row["mean_acc"] for row in rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4.8), dpi=160)
        ax.plot(range(len(labels)), values, marker="o", linewidth=2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("mean accuracy")
        ax.set_title(f"hyperparameter sensitivity: {dataset}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(dest_dir) / f"{dataset}_hparam_sensitivity.png")
        plt.close(fig)


def write_markdown(path, summary, dataset_summary):
    lines = [
        "# my_merge Hyperparameter Sensitivity",
        "",
        "This report is intentionally separate from the main result table. It is used to inspect whether diagnostic client weights and candidate selection are sensitive to validation batch counts and BN recalibration.",
        "",
        "## Overall",
        "",
        "| setting | stats | eval | bn | mean_acc | rows |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['setting']} | {row['stats']} | {row['eval']} | {row['bn']} | {row['mean_acc']:.4f} | {row['num_rows']} |"
        )
    lines.extend(["", "![overall](overall_hparam_sensitivity.png)", "", "## Dataset Plots", ""])
    seen = sorted({row["dataset"] for row in dataset_summary})
    for dataset in seen:
        lines.append(f"- `{dataset}`: ![]({dataset}_hparam_sensitivity.png)")
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = load_rows(Path(args.root))
    summary, dataset_summary = summarize(rows)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    save_csv(dest_dir / "hparam_summary.csv", summary)
    save_csv(dest_dir / "hparam_dataset_summary.csv", dataset_summary)
    plot_overall(summary, dest_dir / "overall_hparam_sensitivity.png")
    plot_by_dataset(dataset_summary, dest_dir)
    write_markdown(dest_dir / "hparam_sensitivity.md", summary, dataset_summary)
    print(f"wrote {dest_dir / 'hparam_sensitivity.md'}")


if __name__ == "__main__":
    main()
