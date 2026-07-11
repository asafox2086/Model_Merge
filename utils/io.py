import csv
import json
from pathlib import Path

import torch


def load_json(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path):
    cfg = load_json(path)
    if not isinstance(cfg, dict):
        raise TypeError(f'Config must be a JSON object: {path}')
    return cfg


def load_checkpoint(path, device='cpu'):
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
