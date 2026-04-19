#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from evaluators import evaluate_small_checkpoint, evaluate_vlm_checkpoint
from utils import append_summary_row, load_checkpoint, load_config, load_json, make_eval_output_dir, save_json


def validate_cfg_against_meta(cfg, meta):
    if 'task_type' in cfg and cfg['task_type'] != meta['task_type']:
        raise ValueError(f"task_type mismatch: cfg={cfg['task_type']} meta={meta['task_type']}")
    if 'dataset' in cfg and cfg['dataset'] != meta['dataset']:
        raise ValueError(f"dataset mismatch: cfg={cfg['dataset']} meta={meta['dataset']}")
    if meta['task_type'] == 'small':
        if 'model' in cfg and cfg['model'] != meta['model']:
            raise ValueError(f"model mismatch: cfg={cfg['model']} meta={meta['model']}")
    elif meta['task_type'] == 'vlm':
        clip_cfg = cfg.get('clip_model', '').split('/')[-1] if cfg.get('clip_model') else ''
        if clip_cfg and clip_cfg != meta['model']:
            raise ValueError(f"clip_model mismatch: cfg={clip_cfg} meta={meta['model']}")


def resolve_eval_inputs(args):
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = {}

    if args.merged_dir:
        merged_dir = Path(args.merged_dir)
        ckpt_path = merged_dir / 'merged.pt'
        meta_path = merged_dir / 'meta.json'
    else:
        ckpt_path = Path(args.checkpoint_path)
        meta_path = Path(args.meta_path)
        merged_dir = ckpt_path.parent

    if not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
    if not meta_path.exists():
        raise FileNotFoundError(f'meta not found: {meta_path}')

    meta = load_json(meta_path)
    validate_cfg_against_meta(cfg, meta)
    cfg.setdefault('task_type', meta['task_type'])
    cfg.setdefault('dataset', meta['dataset'])
    cfg.setdefault('model', meta['model'])
    if 'clip_model' in meta:
        cfg.setdefault('clip_model', meta.get('clip_model', ''))
    cfg.setdefault('num_clients', meta['num_clients'])
    cfg.setdefault('beta', meta['beta'])
    cfg.setdefault('seed', meta['seed'])
    cfg.setdefault('method', 'avg')
    cfg.setdefault('split', 'test')
    cfg.setdefault('batch_size', int(meta.get('batch_size', 64)))
    cfg.setdefault('num_workers', 4)
    cfg.setdefault('device', 'cpu')
    cfg.setdefault('amp', False)
    return cfg, meta, ckpt_path, merged_dir


def run_evaluate(cfg, merged_dir=None, checkpoint_path=None, meta_path=None):
    if merged_dir is not None:
        class Args: pass
        args = Args()
        args.config = ''
        args.merged_dir = str(merged_dir)
        args.checkpoint_path = ''
        args.meta_path = ''
        runtime_cfg = cfg.copy()
        temp_cfg_path = None
        resolved_cfg, meta, ckpt_path, merged_dir = resolve_eval_inputs(args)
        resolved_cfg.update(runtime_cfg)
        cfg = resolved_cfg
    else:
        class Args: pass
        args = Args()
        args.config = ''
        args.merged_dir = ''
        args.checkpoint_path = str(checkpoint_path)
        args.meta_path = str(meta_path)
        resolved_cfg, meta, ckpt_path, merged_dir = resolve_eval_inputs(args)
        resolved_cfg.update(cfg)
        cfg = resolved_cfg

    checkpoint = load_checkpoint(ckpt_path, device='cpu')
    device = torch.device(cfg.get('device', 'cpu'))

    if meta['task_type'] == 'small':
        result = evaluate_small_checkpoint(
            meta=meta,
            checkpoint=checkpoint,
            data_root=cfg['data_root'],
            split=cfg.get('split', 'test'),
            device=device,
            batch_size=int(cfg.get('batch_size', 64)),
            num_workers=int(cfg.get('num_workers', 4)),
            amp=bool(cfg.get('amp', False)),
        )
    elif meta['task_type'] == 'vlm':
        result = evaluate_vlm_checkpoint(
            meta=meta,
            checkpoint=checkpoint,
            data_root=cfg['data_root'],
            split=cfg.get('split', 'test'),
            device=device,
            batch_size=int(cfg.get('batch_size', 64)),
            num_workers=int(cfg.get('num_workers', 4)),
            amp=bool(cfg.get('amp', False)),
        )
    else:
        raise ValueError(f"Unsupported task_type: {meta['task_type']}")

    out_dir = make_eval_output_dir(cfg['output_root'], cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / 'eval.json'
    payload = {
        'task_type': meta['task_type'],
        'dataset': meta['dataset'],
        'model': meta['model'],
        'clip_model': meta.get('clip_model', ''),
        'num_clients': meta['num_clients'],
        'beta': meta['beta'],
        'seed': meta['seed'],
        'method': cfg.get('method', 'avg'),
        'split': cfg.get('split', 'test'),
        'checkpoint_path': str(ckpt_path),
        'merged_dir': str(merged_dir),
        'test_acc': float(result['acc']),
        'test_loss': float(result['loss']),
        'num_samples': int(result['num_samples']),
    }
    save_json(eval_path, payload)
    append_summary_row(Path(cfg['output_root']) / 'reports' / 'eval_summary.csv', payload)
    return payload, eval_path


def parse_args():
    p = argparse.ArgumentParser('Evaluate merged checkpoints')
    p.add_argument('--config', type=str, default='')
    p.add_argument('--merged-dir', type=str, default='')
    p.add_argument('--checkpoint-path', type=str, default='')
    p.add_argument('--meta-path', type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        payload, eval_path = run_evaluate(cfg, merged_dir=args.merged_dir or None, checkpoint_path=args.checkpoint_path or None, meta_path=args.meta_path or None)
    else:
        cfg = {}
        payload, eval_path = run_evaluate(cfg, merged_dir=args.merged_dir or None, checkpoint_path=args.checkpoint_path or None, meta_path=args.meta_path or None)
    print(f'eval result saved to: {eval_path}')
    print(f"test_acc={payload['test_acc']:.4f} test_loss={payload['test_loss']:.4f}")


if __name__ == '__main__':
    main()
