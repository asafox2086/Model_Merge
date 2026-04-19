#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def parse_args():
    p = argparse.ArgumentParser('Export model_hub manifest as task list')
    p.add_argument('--model-hub-root', type=str, default='model_hub')
    p.add_argument('--output', type=str, default='outputs/reports/tasks.csv')
    return p.parse_args()


def main():
    args = parse_args()
    manifest = Path(args.model_hub_root) / 'manifest.csv'
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with manifest.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'tasks saved to: {out}')


if __name__ == '__main__':
    main()
