#!/usr/bin/env python3
"""Check source syntax, links, paper assets and frozen snapshot integrity."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("methods", "model", "dataset", "utils", "evaluators", "scripts", "exp_analyze", "experiments")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true", help="Also check relocated file hashes")
    args = parser.parse_args()
    errors = []
    sources = list(ROOT.glob("*.py"))
    for directory in SOURCE_DIRS:
        sources.extend((ROOT / directory).rglob("*.py"))
    for path in sources:
        try:
            ast.parse(path.read_text(), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(str(exc))
    for directory in SOURCE_DIRS:
        for path in (ROOT / directory).rglob("*"):
            if path.is_symlink() and not path.exists():
                errors.append(f"Broken link: {path.relative_to(ROOT)}")
    shells = list(ROOT.glob("*.sh"))
    for directory in ("scripts", "exp_analyze"):
        shells.extend((ROOT / directory).glob("*.sh"))
    for path in shells:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode:
            errors.append(result.stderr)
    tree = ast.parse((ROOT / "methods/__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            if not (ROOT / "methods" / (node.module + ".py")).is_file():
                errors.append(f"Missing method implementation: {node.module}")
    for path in (ROOT / "configs").rglob("*.json"):
        cfg = json.loads(path.read_text())
        for field in ("model_hub_root", "data_root", "output_root"):
            if field in cfg and Path(cfg[field]).is_absolute():
                errors.append(f"Nonportable config path: {path.relative_to(ROOT)}:{field}")
    for path in (ROOT / "paper").glob("*.tex"):
        source = "\n".join(line.split("%", 1)[0] for line in path.read_text().splitlines())
        for asset in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", source):
            if "#" not in asset and not (ROOT / "paper" / asset).is_file():
                errors.append(f"Missing paper graphic: {asset}")
    for record in json.loads((ROOT / "paper/snapshot.json").read_text())["files"]:
        path = ROOT / record["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            errors.append(f"Changed frozen paper asset: {record['path']}")
    if args.archive:
        archive_manifest = json.loads((ROOT / "remove/manifest.json").read_text())
        removed_roots = archive_manifest.get("removed_manuscript_roots", [])
        for record in archive_manifest["files"]:
            if not record["destination"].startswith("remove/"):
                continue
            path = ROOT / record["destination"]
            if not path.exists():
                if any(path == ROOT / removed or (ROOT / removed) in path.parents for removed in removed_roots):
                    continue
                if record["tracked_before"]:
                    errors.append(f"Missing tracked archive: {record['destination']}")
            elif "sha256" in record and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                errors.append(f"Changed archive: {record['destination']}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(sources)} Python entries, {len(shells)} shell entries, links, method imports, configs and paper snapshot.")


if __name__ == "__main__":
    main()
