#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from utils.io import load_json


def parse_args():
    p = argparse.ArgumentParser('Inspect model_hub')
    p.add_argument('--model-hub-root', type=str, default='model_hub')
    p.add_argument('--limit', type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.model_hub_root)
    manifest = root / 'manifest.csv'
    if not manifest.exists():
        raise FileNotFoundError(f'manifest not found: {manifest}')
    print(f'model_hub: {root}')
    print(f'manifest: {manifest}')
    shown = 0
    for meta_path in sorted(root.glob('small/*/*/clients_*/beta_*/seed_*/meta.json')):
        meta = load_json(meta_path)
        print(f"[small] {meta['dataset']} | {meta['model']} | c={meta['num_clients']} | beta={meta['beta']} | seed={meta['seed']}")
        shown += 1
        if shown >= args.limit:
            return
    for meta_path in sorted(root.glob('vlm/*/*/clients_*/beta_*/seed_*/meta.json')):
        meta = load_json(meta_path)
        print(f"[vlm] {meta['dataset']} | {meta['model']} | c={meta['num_clients']} | beta={meta['beta']} | seed={meta['seed']}")
        shown += 1
        if shown >= args.limit:
            return


if __name__ == '__main__':
    main()
