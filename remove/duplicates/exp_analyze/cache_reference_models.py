#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import load_json
from utils.reference_cache import get_reference_bundle_path
from utils.runtime import build_reference_bundle


def parse_args():
    p = argparse.ArgumentParser('Materialize local reference model bundles for reproducible merges')
    p.add_argument('--model-hub-root', type=str, default=str(ROOT / 'model_hub'))
    p.add_argument('--task-type', choices=['all', 'small', 'vlm'], default='all')
    p.add_argument('--datasets', nargs='*', default=None)
    p.add_argument('--small-models', nargs='*', default=None)
    p.add_argument('--clip-models', nargs='*', default=None)
    return p.parse_args()


def load_manifest_rows(model_hub_root):
    manifest_path = Path(model_hub_root) / 'manifest.csv'
    with manifest_path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def selected_rows(rows, args):
    filtered = rows
    if args.task_type != 'all':
        filtered = [row for row in filtered if row['task_type'] == args.task_type]
    if args.datasets:
        allowed = set(args.datasets)
        filtered = [row for row in filtered if row['dataset'] in allowed]
    if args.small_models:
        allowed = set(args.small_models)
        filtered = [row for row in filtered if row['task_type'] != 'small' or row['model'] in allowed]
    if args.clip_models:
        allowed = {item.split('/')[-1] for item in args.clip_models}
        filtered = [row for row in filtered if row['task_type'] != 'vlm' or row['clip_model'] in allowed]
    return filtered


def cache_key(meta):
    path = get_reference_bundle_path(meta)
    return str(path.resolve())


def main():
    args = parse_args()
    rows = selected_rows(load_manifest_rows(args.model_hub_root), args)
    seen = set()
    built = 0
    reused = 0

    for row in rows:
        meta_path = Path(args.model_hub_root) / row['meta_path']
        meta = load_json(meta_path)
        key = cache_key(meta)
        if key in seen:
            continue
        seen.add(key)
        existed = Path(key).exists()
        build_reference_bundle(meta, device='cpu')
        if existed:
            reused += 1
        else:
            built += 1
        print(f"[cache] {'reused' if existed else 'built'} {key}")

    print(f'[cache] done | unique={len(seen)} built={built} reused={reused}')


if __name__ == '__main__':
    main()
