#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser("Collect completed my_merge experiment reports into My_merge_ret")
    p.add_argument("--ablation-root", required=True)
    p.add_argument("--hparam-root", default="")
    p.add_argument("--dest-root", default=str(ROOT / "My_merge_ret"))
    p.add_argument("--run-name", default="")
    p.add_argument("--commit", action="store_true")
    p.add_argument("--push", action="store_true")
    p.add_argument("--remote", default="MM")
    p.add_argument("--branch", default="main")
    return p.parse_args()


def copy_file(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree_files(src_dir, dst_dir, patterns):
    src_dir = Path(src_dir)
    copied = []
    if not src_dir.exists():
        return copied
    for pattern in patterns:
        for src in sorted(src_dir.glob(pattern)):
            if src.is_file():
                rel = src.relative_to(src_dir)
                dst = Path(dst_dir) / rel
                copy_file(src, dst)
                copied.append(dst)
    return copied


def copy_ablation_reports(ablation_root, dest):
    ablation_root = Path(ablation_root)
    dest = Path(dest)
    copied = []
    copied.extend(
        copy_tree_files(
            ablation_root / "reports",
            dest / "ablation_grid_reports",
            ["*.md", "*.csv", "*.txt", "*.json"],
        )
    )
    for subdir in sorted(path for path in ablation_root.iterdir() if path.is_dir() and path.name != "reports"):
        report_dir = subdir / "reports"
        copied.extend(
            copy_tree_files(
                report_dir,
                dest / "ablation_runs" / subdir.name / "reports",
                ["*.md", "*.csv", "*.txt", "*.json", "diagnostic_figures/*.png"],
            )
        )
    return copied


def copy_hparam_reports(hparam_root, dest):
    if not hparam_root:
        return []
    hparam_root = Path(hparam_root)
    dest = Path(dest)
    copied = []
    if not hparam_root.exists():
        return copied
    copied.extend(
        copy_tree_files(
            hparam_root / "reports",
            dest / "hparam_reports",
            ["*.md", "*.csv", "*.txt", "*.json", "hparam_plots/*"],
        )
    )
    for subdir in sorted(path for path in hparam_root.iterdir() if path.is_dir() and path.name != "reports"):
        copied.extend(
            copy_tree_files(
                subdir / "reports",
                dest / "hparam_runs" / subdir.name / "reports",
                ["*.md", "*.csv", "*.txt", "*.json"],
            )
        )
    return copied


def write_location_doc(dest_root, run_dir, ablation_root, hparam_root):
    dest_root = Path(dest_root)
    run_dir = Path(run_dir)
    rel_run = run_dir.relative_to(ROOT)
    doc = dest_root / "docs" / "my_merge_results_location.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    has_hparam = bool(hparam_root) and Path(hparam_root).exists()
    lines = [
        "# my_merge Result Locations",
        "",
        f"Updated at: `{datetime.now().isoformat(timespec='seconds')}`.",
        "",
        "## Source Output Roots",
        "",
        f"- Ablation source: `{Path(ablation_root)}`",
        f"- Hyperparameter source: `{Path(hparam_root)}`" if has_hparam else "- Hyperparameter source: not collected in this run",
        "",
        "## Preserved Result Folder",
        "",
        f"- Collected results: `{rel_run}`",
        "",
        "## Main Files",
        "",
        f"- Method paper-style description: `My_merge_ret/docs/my_merge_paper_method.md`",
        f"- Combined baseline + ablation table: `{rel_run}/ablation_grid_reports/all_results_ablation_combined.md`",
        f"- Ablation summary: `{rel_run}/ablation_grid_reports/ablation_summary.md`",
        f"- Ablation grid config: `{rel_run}/ablation_grid_reports/ablation_grid_config.txt`",
        "",
    ]
    if has_hparam:
        lines.extend(
            [
                f"- Hyperparameter sensitivity report: `{rel_run}/hparam_reports/hparam_plots/hparam_sensitivity.md`",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-Ablation Diagnostics",
            "",
            f"- Per-ablation eval/merge summaries: `{rel_run}/ablation_runs/<ablation>/reports/`",
            f"- Client diagnostic weight tables: `{rel_run}/ablation_runs/<ablation>/reports/my_merge_client_diagnostic_weights.md`",
            f"- Weight + PCA visualization tables: `{rel_run}/ablation_runs/<ablation>/reports/my_merge_weight_visualization_table.md`",
            f"- PCA figure folder: `{rel_run}/ablation_runs/<ablation>/reports/diagnostic_figures/`",
            "",
            "## Notes",
            "",
            "- `outputs/` remains ignored and is not pushed; this folder only preserves the report artifacts needed for reading and comparison.",
            "- When PCA images are enabled, they are generated before merged checkpoints are deleted, so the repository can keep visual evidence without storing large model checkpoints.",
        ]
    )
    if has_hparam:
        lines.append("- Hyperparameter plots are intentionally separate from the main table; they are for sensitivity analysis rather than leaderboard comparison.")
    lines.append("")
    doc.write_text("\n".join(lines), encoding="utf-8")
    copy_file(doc, run_dir / "RESULTS_LOCATION.md")
    return doc


def publish_root_summary(dest_root, run_dir):
    src = Path(run_dir) / "ablation_grid_reports" / "all_results_ablation_combined.md"
    dst = Path(dest_root) / "汇总表.md"
    if copy_file(src, dst):
        return dst
    return None


def run(cmd, cwd=ROOT, allow_fail=False):
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout.strip()


def maybe_commit_and_push(paths, remote, branch, do_commit, do_push):
    if not do_commit and not do_push:
        return []
    outputs = []
    rel_paths = [str(Path(path).relative_to(ROOT)) for path in paths]
    outputs.append(run(["git", "add", *rel_paths]))
    if do_commit:
        status = run(["git", "status", "--short"])
        if status:
            outputs.append(run(["git", "commit", "-m", "Add completed my_merge experiment reports"], allow_fail=True))
        else:
            outputs.append("nothing to commit")
    if do_push:
        outputs.append(run(["git", "push", remote, branch]))
    return outputs


def main():
    args = parse_args()
    ablation_root = Path(args.ablation_root)
    hparam_root = Path(args.hparam_root) if args.hparam_root else None
    run_name = args.run_name or f"my_merge_completed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dest_root = Path(args.dest_root)
    run_dir = dest_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    copied.extend(copy_ablation_reports(ablation_root, run_dir))
    copied.extend(copy_hparam_reports(hparam_root, run_dir))
    location_doc = write_location_doc(dest_root, run_dir, ablation_root, hparam_root)
    root_summary = publish_root_summary(dest_root, run_dir)
    copied.append(location_doc)
    copied.append(run_dir / "RESULTS_LOCATION.md")
    if root_summary is not None:
        copied.append(root_summary)

    print(f"collected {len(copied)} files into {run_dir}")
    commit_paths = [run_dir, location_doc]
    if root_summary is not None:
        commit_paths.append(root_summary)
    for output in maybe_commit_and_push(commit_paths, args.remote, args.branch, args.commit, args.push):
        if output:
            print(output)


if __name__ == "__main__":
    main()
