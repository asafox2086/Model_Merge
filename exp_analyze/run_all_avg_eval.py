#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path
import sys

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import run_evaluate
from merge import METHOD_DEFAULTS, run_merge
from methods.lamp_merge_analysis import LAMP_MERGE_ABLATION_MODES
from methods import normalize_method_name
from utils import load_json, make_eval_output_dir, save_csv, save_json
from utils.lamp_merge_stats import validate_lamp_merge_stats_root


CASE_KEY_FIELDS = (
    'task_type',
    'dataset',
    'model',
    'clip_model',
    'num_clients',
    'beta',
    'seed',
    'method',
    'merge_weight_mode',
    'lamp_merge_ablation_mode',
)

STATUS_FIELDS = [
    'status',
    'task_type',
    'dataset',
    'model',
    'clip_model',
    'num_clients',
    'beta',
    'seed',
    'method',
    'merge_weight_mode',
    'lamp_merge_ablation_mode',
    'eval_json',
    'merged_deleted',
    'seconds',
    'test_acc',
    'test_loss',
    'updated_at',
    'error',
]


def ts():
    return datetime.now().strftime('%H:%M:%S')


def parse_args():
    p = argparse.ArgumentParser('Run merge + eval for all model_hub entries')
    default_model_hub = ROOT / 'model_hub'
    default_data_root = ROOT / 'Med_data'
    model_hub_default = (
        str(default_model_hub)
        if (default_model_hub / 'manifest.csv').exists()
        else '/data1/users/weiyipan/ML/MedMNSITMerge/model_hub'
    )
    data_root_default = (
        str(default_data_root)
        if default_data_root.exists()
        else '/data1/users/weiyipan/FL/data/Med_data'
    )
    p.add_argument(
        '--model-hub-root',
        type=str,
        default=model_hub_default,
    )
    p.add_argument(
        '--data-root',
        type=str,
        default=data_root_default,
    )
    p.add_argument('--output-root', type=str, default='')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--small-batch-size', type=int, default=128)
    p.add_argument('--vlm-batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--task-type', choices=['all', 'small', 'vlm'], default='all')
    p.add_argument('--datasets', nargs='*', default=None)
    p.add_argument('--small-models', nargs='*', default=None)
    p.add_argument('--clip-models', nargs='*', default=None)
    p.add_argument('--num-clients', nargs='*', type=int, default=None)
    p.add_argument('--betas', nargs='*', type=float, default=None)
    p.add_argument('--skip', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--delete-merged', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--method', type=str, default='avg')
    p.add_argument('--merge-weight-mode', choices=['auto', 'sample', 'equal'], default='auto')
    p.add_argument('--density', type=float, default=METHOD_DEFAULTS['density'])
    p.add_argument('--dare-seed', type=int, default=METHOD_DEFAULTS['dare_seed'])
    p.add_argument('--fisher-eps', type=float, default=METHOD_DEFAULTS['fisher_eps'])
    p.add_argument('--fisher-normalize-weight', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fisher-minimal-weight', type=float, default=METHOD_DEFAULTS['fisher_minimal_weight'])
    p.add_argument('--regmean-eps', type=float, default=METHOD_DEFAULTS['regmean_eps'])
    p.add_argument('--regmean-reduce-non-diagonal-ratio', type=float, default=METHOD_DEFAULTS['regmean_reduce_non_diagonal_ratio'])
    p.add_argument('--breadcrumbs-top-k-keep', type=float, default=METHOD_DEFAULTS['breadcrumbs_top_k_keep'])
    p.add_argument('--breadcrumbs-top-k-remove', type=float, default=METHOD_DEFAULTS['breadcrumbs_top_k_remove'])
    p.add_argument('--breadcrumbs-alpha', type=float, default=METHOD_DEFAULTS['breadcrumbs_alpha'])
    p.add_argument('--model-stock-k', type=float, default=METHOD_DEFAULTS['model_stock_k'])
    p.add_argument('--adamerging-epochs', type=int, default=METHOD_DEFAULTS['adamerging_epochs'])
    p.add_argument('--adamerging-lr', type=float, default=METHOD_DEFAULTS['adamerging_lr'])
    p.add_argument('--adamerging-prior', type=float, default=METHOD_DEFAULTS['adamerging_prior'])
    p.add_argument('--adamerging-max-batches', type=int, default=METHOD_DEFAULTS['adamerging_max_batches'])
    p.add_argument('--from-k', type=float, default=METHOD_DEFAULTS['from_k'])
    p.add_argument('--iso-common-space-fraction', type=float, default=METHOD_DEFAULTS['iso_common_space_fraction'])
    p.add_argument('--free-filter-ratio', type=float, default=METHOD_DEFAULTS['free_filter_ratio'])
    p.add_argument('--free-scaling', type=float, default=METHOD_DEFAULTS['free_scaling'])
    p.add_argument('--free-include-all-2d', action=argparse.BooleanOptionalAction, default=METHOD_DEFAULTS['free_include_all_2d'])
    p.add_argument('--robustmerge-mask-ratio', type=float, default=METHOD_DEFAULTS['robustmerge_mask_ratio'])
    p.add_argument('--robustmerge-att-ratio', type=float, default=METHOD_DEFAULTS['robustmerge_att_ratio'])
    p.add_argument('--robustmerge-fuse-weight', type=float, default=METHOD_DEFAULTS['robustmerge_fuse_weight'])
    p.add_argument('--robustmerge-include-all-2d', action=argparse.BooleanOptionalAction, default=METHOD_DEFAULTS['robustmerge_include_all_2d'])
    p.add_argument('--stats-split', type=str, default=METHOD_DEFAULTS['stats_split'])
    p.add_argument('--stats-batch-size', type=int, default=METHOD_DEFAULTS['stats_batch_size'])
    p.add_argument('--stats-num-workers', type=int, default=METHOD_DEFAULTS['stats_num_workers'])
    p.add_argument('--fisher-max-batches', type=int, default=METHOD_DEFAULTS['fisher_max_batches'])
    p.add_argument('--regmean-max-batches', type=int, default=METHOD_DEFAULTS['regmean_max_batches'])
    p.add_argument('--regmean-max-dim', type=int, default=METHOD_DEFAULTS['regmean_max_dim'])
    p.add_argument('--lamp-merge-prototype-root', dest='lamp_merge_prototype_root', type=str, default='')
    p.add_argument('--lamp-merge-proto-count-power', dest='lamp_merge_proto_count_power', type=float, default=METHOD_DEFAULTS['lamp_merge_proto_count_power'])
    p.add_argument('--lamp-merge-reference-head-scale', dest='lamp_merge_reference_head_scale', type=float, default=METHOD_DEFAULTS['lamp_merge_reference_head_scale'])
    p.add_argument('--lamp-merge-prevalence-threshold', dest='lamp_merge_prevalence_threshold', type=float, default=0.5)
    p.add_argument(
        '--lamp-merge-reference-prior-threshold',
        dest='lamp_merge_reference_prior_threshold',
        type=float,
        default=METHOD_DEFAULTS['lamp_merge_reference_prior_threshold'],
        help='Dominant-class imbalance ratio threshold for M2 long-tail calibration.',
    )
    p.add_argument(
        '--lamp-merge-reference-prior-max-tau',
        dest='lamp_merge_reference_prior_max_tau',
        type=float,
        default=METHOD_DEFAULTS['lamp_merge_reference_prior_max_tau'],
        help='Maximum centered log-prior strength used by M2.',
    )
    p.add_argument(
        '--lamp-merge-reference-prior-tau',
        dest='lamp_merge_reference_prior_tau',
        type=float,
        default=None,
        help='Explicit centered log-prior strength for controlled M2 sensitivity experiments.',
    )
    p.add_argument(
        '--lamp-merge-ablation-mode',
        dest='lamp_merge_ablation_mode',
        choices=sorted(LAMP_MERGE_ABLATION_MODES),
        default='full',
        help='LAMP-Merge ablation mode for module-level and module-internal studies.',
    )
    p.add_argument(
        '--lamp-merge-ablation-seed',
        dest='lamp_merge_ablation_seed',
        type=int,
        default=1701,
        help='Deterministic seed for LAMP-Merge synthetic and shuffled-label ablations.',
    )
    p.add_argument(
        '--lamp-merge-prevalence-smoothing',
        dest='lamp_merge_prevalence_smoothing',
        type=float,
        default=1.0,
        help='Additive smoothing used by the smoothed-prevalence internal ablation.',
    )
    return p.parse_args()


def load_manifest(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def filter_manifest(rows, args):
    selected = rows
    if args.task_type != 'all':
        selected = [row for row in selected if row['task_type'] == args.task_type]
    if args.datasets:
        ds = set(args.datasets)
        selected = [row for row in selected if row['dataset'] in ds]
    if args.small_models:
        sm = set(args.small_models)
        selected = [row for row in selected if row['task_type'] != 'small' or row['model'] in sm]
    if args.clip_models:
        cm = set(args.clip_models)
        selected = [row for row in selected if row['task_type'] != 'vlm' or row['clip_model'] in cm]
    if args.num_clients:
        client_counts = {int(x) for x in args.num_clients}
        selected = [row for row in selected if int(row['num_clients']) in client_counts]
    if args.betas:
        betas = {format(float(x), 'g') for x in args.betas}
        selected = [row for row in selected if format(float(row['beta']), 'g') in betas]
    if args.skip > 0:
        selected = selected[args.skip:]
    if args.limit > 0:
        selected = selected[:args.limit]
    return selected


def normalize_row(row):
    return {
        'task_type': row['task_type'],
        'dataset': row['dataset'],
        'model': row['model'],
        'clip_model': row['clip_model'],
        'num_clients': int(row['num_clients']),
        'beta': float(row['beta']),
        'seed': int(row['seed']),
    }


def resolve_merge_weight_mode(args):
    if args.merge_weight_mode != 'auto':
        return args.merge_weight_mode
    return 'equal'


def build_cfg(row, args):
    item = normalize_row(row)
    cfg = {
        'task_type': item['task_type'],
        'dataset': item['dataset'],
        'num_clients': item['num_clients'],
        'beta': item['beta'],
        'seed': item['seed'],
        'method': normalize_method_name(args.method),
        'merge_weight_mode': resolve_merge_weight_mode(args),
        'model_hub_root': args.model_hub_root,
        'data_root': args.data_root,
        'device': args.device,
        'num_workers': args.num_workers,
        'output_root': args.output_root,
        'split': 'test',
        'amp': str(args.device).startswith('cuda'),
        'density': args.density,
        'dare_seed': args.dare_seed,
        'fisher_eps': args.fisher_eps,
        'fisher_normalize_weight': args.fisher_normalize_weight,
        'fisher_minimal_weight': args.fisher_minimal_weight,
        'regmean_eps': args.regmean_eps,
        'regmean_reduce_non_diagonal_ratio': args.regmean_reduce_non_diagonal_ratio,
        'breadcrumbs_top_k_keep': args.breadcrumbs_top_k_keep,
        'breadcrumbs_top_k_remove': args.breadcrumbs_top_k_remove,
        'breadcrumbs_alpha': args.breadcrumbs_alpha,
        'model_stock_k': args.model_stock_k,
        'adamerging_epochs': args.adamerging_epochs,
        'adamerging_lr': args.adamerging_lr,
        'adamerging_prior': args.adamerging_prior,
        'adamerging_max_batches': args.adamerging_max_batches,
        'from_k': args.from_k,
        'iso_common_space_fraction': args.iso_common_space_fraction,
        'free_filter_ratio': args.free_filter_ratio,
        'free_scaling': args.free_scaling,
        'free_include_all_2d': args.free_include_all_2d,
        'robustmerge_mask_ratio': args.robustmerge_mask_ratio,
        'robustmerge_att_ratio': args.robustmerge_att_ratio,
        'robustmerge_fuse_weight': args.robustmerge_fuse_weight,
        'robustmerge_include_all_2d': args.robustmerge_include_all_2d,
        'stats_split': args.stats_split,
        'stats_batch_size': args.stats_batch_size,
        'stats_num_workers': args.stats_num_workers,
        'fisher_max_batches': args.fisher_max_batches,
        'regmean_max_batches': args.regmean_max_batches,
        'regmean_max_dim': args.regmean_max_dim,
    }
    if args.lamp_merge_prototype_root:
        cfg['lamp_merge_prototype_root'] = args.lamp_merge_prototype_root
    cfg['lamp_merge_proto_count_power'] = args.lamp_merge_proto_count_power
    cfg['lamp_merge_reference_head_scale'] = args.lamp_merge_reference_head_scale
    cfg['lamp_merge_prevalence_threshold'] = args.lamp_merge_prevalence_threshold
    cfg['lamp_merge_reference_prior_threshold'] = args.lamp_merge_reference_prior_threshold
    cfg['lamp_merge_reference_prior_max_tau'] = args.lamp_merge_reference_prior_max_tau
    if args.lamp_merge_reference_prior_tau is not None:
        cfg['lamp_merge_reference_prior_tau'] = args.lamp_merge_reference_prior_tau
    cfg['lamp_merge_ablation_mode'] = args.lamp_merge_ablation_mode
    cfg['lamp_merge_ablation_seed'] = args.lamp_merge_ablation_seed
    cfg['lamp_merge_prevalence_smoothing'] = args.lamp_merge_prevalence_smoothing
    if item['task_type'] == 'small':
        cfg['model'] = item['model']
        cfg['batch_size'] = args.small_batch_size
    else:
        cfg['clip_model'] = row['clip_model']
        cfg['batch_size'] = args.vlm_batch_size
    return cfg


def build_eval_path(cfg):
    return make_eval_output_dir(cfg['output_root'], cfg) / 'eval.json'


def normalize_case_value(field, value):
    if field == 'beta':
        return format(float(value), 'g')
    if field in {'num_clients', 'seed'}:
        return str(int(value))
    return str(value or '')


def case_key(row):
    return tuple(normalize_case_value(field, row.get(field, '')) for field in CASE_KEY_FIELDS)


def load_status_rows(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    status_rows = {}
    for row in rows:
        normalized = {field: row.get(field, '') for field in STATUS_FIELDS}
        status_rows[case_key(normalized)] = normalized
    return status_rows


def save_status_rows(path, status_rows):
    rows = [status_rows[key] for key in sorted(status_rows.keys())]
    save_csv(path, [{field: row.get(field, '') for field in STATUS_FIELDS} for row in rows])


def update_status_row(path, status_rows, row):
    normalized = {field: row.get(field, '') for field in STATUS_FIELDS}
    status_rows[case_key(normalized)] = normalized
    save_status_rows(path, status_rows)


def append_csv_row(path, row, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, '') for field in fields})


def build_status_row(cfg, *, status, eval_json='', merged_deleted='', seconds='', test_acc='', test_loss='', error=''):
    return {
        'status': status,
        'task_type': cfg['task_type'],
        'dataset': cfg['dataset'],
        'model': cfg.get('model', ''),
        'clip_model': cfg.get('clip_model', ''),
        'num_clients': cfg['num_clients'],
        'beta': cfg['beta'],
        'seed': cfg['seed'],
        'method': cfg['method'],
        'merge_weight_mode': cfg['merge_weight_mode'],
        'lamp_merge_ablation_mode': cfg.get('lamp_merge_ablation_mode', ''),
        'eval_json': eval_json,
        'merged_deleted': merged_deleted,
        'seconds': seconds,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'error': error,
    }


def load_valid_eval_payload(cfg):
    eval_json = build_eval_path(cfg)
    if not eval_json.exists():
        return None, eval_json
    try:
        payload = load_json(eval_json)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, eval_json

    expected = {
        'task_type': cfg['task_type'],
        'dataset': cfg['dataset'],
        'num_clients': int(cfg['num_clients']),
        'seed': int(cfg['seed']),
        'method': cfg['method'],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return None, eval_json
    if cfg['task_type'] == 'small':
        if payload.get('model') != cfg['model']:
            return None, eval_json
    else:
        if payload.get('clip_model', '').split('/')[-1] != cfg['clip_model'].split('/')[-1]:
            return None, eval_json
    try:
        payload_beta = format(float(payload.get('beta')), 'g')
        cfg_beta = format(float(cfg['beta']), 'g')
    except (TypeError, ValueError):
        return None, eval_json
    if payload_beta != cfg_beta:
        return None, eval_json
    try:
        float(payload['test_acc'])
        float(payload['test_loss'])
    except (KeyError, TypeError, ValueError):
        return None, eval_json
    return payload, eval_json


def release_case_resources():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    args = parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    method = normalize_method_name(args.method)
    if method in {'lamp_merge', 'lamp_merge_analysis'}:
        if not args.lamp_merge_prototype_root:
            raise SystemExit(
                'LAMP-Merge requires --lamp-merge-prototype-root pointing to client-local aggregate statistics.'
            )
        validate_lamp_merge_stats_root(Path(args.lamp_merge_prototype_root), min_files=1)
    if not args.output_root:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_root = f'/data1/users/weiyipan/ML/MedMNSITMerge/outputs/{args.method}_{stamp}'
    manifest = load_manifest(Path(args.model_hub_root) / 'manifest.csv')
    manifest = filter_manifest(manifest, args)

    status_csv = Path(args.output_root) / 'reports' / 'batch_status.csv'
    status_rows = load_status_rows(status_csv)
    save_json(Path(args.output_root) / 'reports' / 'batch_config.json', vars(args))

    print(f'[{ts()}] batch start | tasks={len(manifest)} | output_root={args.output_root} | method={args.method}')
    for idx, row in enumerate(manifest, start=1):
        cfg = build_cfg(row, args)
        label = f"{cfg['task_type']}:{cfg['dataset']}:{cfg.get('model') or cfg.get('clip_model')}|c={cfg['num_clients']}|b={cfg['beta']}|s={cfg['seed']}|m={cfg['method']}|w={cfg['merge_weight_mode']}"
        existing_payload, eval_json = load_valid_eval_payload(cfg)
        if args.resume and existing_payload is not None:
            update_status_row(
                status_csv,
                status_rows,
                build_status_row(
                    cfg,
                    status='SKIP',
                    eval_json=str(eval_json),
                    seconds=0.0,
                    test_acc=float(existing_payload['test_acc']),
                    test_loss=float(existing_payload['test_loss']),
                ),
            )
            print(f'[{ts()}] skip ({idx}/{len(manifest)}) {label} | valid eval exists')
            continue
        if args.resume and eval_json.exists():
            print(f'[{ts()}] stale eval ignored ({idx}/{len(manifest)}) {label} | invalid eval.json')
        t0 = time.time()
        try:
            print(f'[{ts()}] start ({idx}/{len(manifest)}) {label}')
            merge_info = run_merge(cfg)
            print(f'[{ts()}] merged ({idx}/{len(manifest)}) {label}')
            payload, eval_path = run_evaluate(cfg, merged_dir=merge_info['merged_dir'])
            merged_ckpt = Path(merge_info['merged_checkpoint'])
            removed = False
            if args.delete_merged and merged_ckpt.exists():
                merged_ckpt.unlink()
                removed = True
            update_status_row(
                status_csv,
                status_rows,
                build_status_row(
                    cfg,
                    status='OK',
                    eval_json=str(eval_path),
                    merged_deleted=removed,
                    seconds=round(time.time() - t0, 2),
                    test_acc=payload['test_acc'],
                    test_loss=payload['test_loss'],
                ),
            )
            print(f"[{ts()}] done ({idx}/{len(manifest)}) {label} | acc={payload['test_acc']:.4f} | loss={payload['test_loss']:.4f} | removed={removed}")
        except Exception as exc:
            update_status_row(
                status_csv,
                status_rows,
                build_status_row(
                    cfg,
                    status='FAIL',
                    seconds=round(time.time() - t0, 2),
                    merged_deleted=False,
                    error=str(exc),
                ),
            )
            print(f'[{ts()}] fail ({idx}/{len(manifest)}) {label} | {exc}')
        finally:
            release_case_resources()
    print(f'[{ts()}] batch done | output_root={args.output_root}')


if __name__ == '__main__':
    main()
