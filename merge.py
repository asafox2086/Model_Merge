#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from methods import (
    merge_avg,
    merge_adamerging,
    merge_breadcrumbs,
    merge_dare_linear,
    merge_dare_ties,
    merge_fisher,
    merge_from,
    merge_free,
    merge_iso_c,
    merge_iso_cts,
    merge_lamp_merge,
    merge_model_stock,
    merge_my_merge,
    merge_regmean,
    merge_robustmerge,
    merge_ties,
    normalize_method_name,
)
from utils import (
    append_summary_row,
    ensure_checkpoint_files,
    ensure_state_dicts_compatible,
    ensure_task_matches_config,
    extract_state_dict,
    find_hub_experiment_dir,
    load_checkpoint,
    load_config,
    load_hub_meta,
    make_merge_output_dir,
    save_json,
    set_seed,
)
from utils.runtime import build_reference_bundle
from utils.statistics import collect_linear_covariances, compute_fisher_diagonal


METHOD_DEFAULTS = {
    'density': 0.5,
    'dare_seed': 42,
    'fisher_eps': 1e-8,
    'fisher_normalize_weight': True,
    'fisher_minimal_weight': 1e-6,
    'regmean_eps': 1e-6,
    'regmean_reduce_non_diagonal_ratio': 1.0,
    'breadcrumbs_top_k_keep': 0.2,
    'breadcrumbs_top_k_remove': 0.1,
    'breadcrumbs_alpha': 1.0,
    'model_stock_k': 2.0,
    'adamerging_epochs': 50,
    'adamerging_lr': 1e-2,
    'adamerging_prior': 0.3,
    'adamerging_max_batches': 1,
    'from_k': 1.0,
    'iso_common_space_fraction': 0.8,
    'free_filter_ratio': 0.7,
    'free_scaling': 1.0,
    'free_include_all_2d': False,
    'robustmerge_mask_ratio': 0.2,
    'robustmerge_att_ratio': 0.2,
    'robustmerge_fuse_weight': 2.0,
    'robustmerge_include_all_2d': True,
    'stats_split': 'val',
    'stats_batch_size': 0,
    'stats_num_workers': 0,
    'fisher_max_batches': 0,
    'regmean_max_batches': 0,
    'regmean_max_dim': 1024,
}


def resolve_client_weights(meta, cfg, method):
    mode = cfg.get('merge_weight_mode', 'sample')
    if mode == 'sample':
        weights = [int(item['num_samples']) for item in meta['clients']]
    elif mode == 'equal':
        weights = [1 for _ in meta['clients']]
    else:
        raise ValueError(f'Unsupported merge_weight_mode: {mode}')
    return weights


def merge_with_method(method, state_dicts, weights, meta, checkpoints, cfg):
    if method == 'avg':
        merged_state_dict, method_info = merge_avg(state_dicts, weights)
    elif method == 'ties':
        base_state, param_names = build_reference_bundle(meta, device='cpu')
        merged_state_dict, method_info = merge_ties(
            state_dicts,
            base_state,
            weights,
            density=float(cfg['density']),
            param_keys=param_names,
        )
    elif method == 'dare_linear':
        base_state, param_names = build_reference_bundle(meta, device='cpu')
        merged_state_dict, method_info = merge_dare_linear(
            state_dicts,
            base_state,
            weights,
            density=float(cfg['density']),
            seed=int(cfg['dare_seed']),
            param_keys=param_names,
        )
    elif method == 'dare_ties':
        base_state, param_names = build_reference_bundle(meta, device='cpu')
        merged_state_dict, method_info = merge_dare_ties(
            state_dicts,
            base_state,
            weights,
            density=float(cfg['density']),
            seed=int(cfg['dare_seed']),
            param_keys=param_names,
        )
    elif method == 'fisher':
        fisher_stats = [compute_fisher_diagonal(meta, ckpt, cfg) for ckpt in checkpoints]
        merged_state_dict, method_info = merge_fisher(
            state_dicts,
            fisher_stats,
            weights,
            eps=float(cfg['fisher_eps']),
            normalize_fisher_weight=bool(cfg['fisher_normalize_weight']),
            minimal_fisher_weight=float(cfg['fisher_minimal_weight']),
        )
    elif method == 'regmean':
        cov_stats = [collect_linear_covariances(meta, ckpt, cfg) for ckpt in checkpoints]
        merged_state_dict, method_info = merge_regmean(
            state_dicts,
            cov_stats,
            weights,
            eps=float(cfg['regmean_eps']),
            reduce_non_diagonal_ratio=float(cfg['regmean_reduce_non_diagonal_ratio']),
        )
    elif method == 'breadcrumbs':
        base_state, param_names = build_reference_bundle(meta, device='cpu')
        merged_state_dict, method_info = merge_breadcrumbs(
            state_dicts,
            base_state,
            weights,
            top_k_keep=float(cfg['breadcrumbs_top_k_keep']),
            top_k_remove=float(cfg['breadcrumbs_top_k_remove']),
            alpha=float(cfg['breadcrumbs_alpha']),
            param_keys=param_names,
        )
    elif method == 'model_stock':
        base_state, _ = build_reference_bundle(meta, device='cpu')
        merged_state_dict, method_info = merge_model_stock(
            state_dicts,
            base_state,
            weights,
            k=float(cfg['model_stock_k']),
        )
    elif method in {'lamp_merge', 'my_merge'}:
        merged_state_dict, method_info = merge_lamp_merge(
            state_dicts,
            weights,
            meta=meta,
            checkpoints=checkpoints,
            cfg=cfg,
        )
    elif method == 'adamerging':
        base_state, _ = build_reference_bundle(meta, device='cpu')
        merged_state_dict, method_info = merge_adamerging(meta, state_dicts, base_state, weights, cfg)
    elif method == 'from':
        base_state, _ = build_reference_bundle(meta, device='cpu')
        merged_state_dict, method_info = merge_from(
            state_dicts,
            base_state,
            weights,
            k=float(cfg['from_k']),
        )
    elif method == 'iso_c':
        base_state, _ = build_reference_bundle(meta, device='cpu')
        merged_state_dict, method_info = merge_iso_c(
            state_dicts,
            base_state,
            weights,
            cfg,
        )
    elif method == 'iso_cts':
        base_state, _ = build_reference_bundle(meta, device='cpu')
        merged_state_dict, method_info = merge_iso_cts(
            state_dicts,
            base_state,
            weights,
            cfg,
        )
    elif method == 'free_merge':
        base_state, _ = build_reference_bundle(meta, device='cpu')
        merged_state_dict, method_info = merge_free(
            state_dicts,
            base_state,
            weights,
            cfg,
        )
    elif method == 'robustmerge':
        base_state, _ = build_reference_bundle(meta, device='cpu')
        merged_state_dict, method_info = merge_robustmerge(
            state_dicts,
            base_state,
            weights,
            cfg,
        )
    else:
        raise ValueError(f'Unsupported merge method: {method}')
    return merged_state_dict, method_info


def run_merge(cfg):
    set_seed(int(cfg.get('seed', 42)))
    method = normalize_method_name(cfg.get('method', 'avg'))
    cfg = {**METHOD_DEFAULTS, **cfg, 'method': method}

    exp_dir = find_hub_experiment_dir(cfg['model_hub_root'], cfg)
    meta = load_hub_meta(exp_dir)
    ensure_task_matches_config(meta, cfg)
    ckpt_paths = ensure_checkpoint_files(exp_dir, meta)

    checkpoints = [load_checkpoint(path, device='cpu') for path in ckpt_paths]
    state_dicts = [extract_state_dict(obj) for obj in checkpoints]
    ensure_state_dicts_compatible(state_dicts)

    weights = resolve_client_weights(meta, cfg, method)
    merged_state_dict, method_info = merge_with_method(method, state_dicts, weights, meta, checkpoints, cfg)

    out_dir = make_merge_output_dir(cfg['output_root'], cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_ckpt_path = out_dir / 'merged.pt'
    merged_meta_path = out_dir / 'meta.json'
    merge_result_path = out_dir / 'merge_result.json'

    merged_checkpoint = {
        'state_dict': merged_state_dict,
        'meta': {
            'task_type': meta['task_type'],
            'dataset': meta['dataset'],
            'model': meta['model'],
            'clip_model': meta.get('clip_model'),
            'num_clients': meta['num_clients'],
            'beta': meta['beta'],
            'seed': meta['seed'],
            'method': cfg['method'],
            'merge_weight_mode': cfg['merge_weight_mode'],
        },
    }
    torch.save(merged_checkpoint, merged_ckpt_path)
    save_json(merged_meta_path, meta)

    merge_result = {
        'task_type': meta['task_type'],
        'dataset': meta['dataset'],
        'model': meta['model'],
        'clip_model': meta.get('clip_model', ''),
        'num_clients': meta['num_clients'],
        'beta': meta['beta'],
        'seed': meta['seed'],
        'method': cfg['method'],
        'merge_weight_mode': cfg['merge_weight_mode'],
        'method_info': method_info,
        'source_clients': [item['checkpoint'] for item in meta['clients']],
        'merged_checkpoint': str(merged_ckpt_path),
        'meta_path': str(merged_meta_path),
    }
    save_json(merge_result_path, merge_result)
    append_summary_row(
        Path(cfg['output_root']) / 'reports' / 'merge_summary.csv',
        {
            'task_type': merge_result['task_type'],
            'dataset': merge_result['dataset'],
            'model': merge_result['model'],
            'clip_model': merge_result['clip_model'],
            'num_clients': merge_result['num_clients'],
            'beta': merge_result['beta'],
            'seed': merge_result['seed'],
            'method': merge_result['method'],
            'merge_weight_mode': merge_result['merge_weight_mode'],
            'merged_checkpoint': merge_result['merged_checkpoint'],
            'meta_path': merge_result['meta_path'],
        },
    )
    return {
        'merged_dir': str(out_dir),
        'merged_checkpoint': str(merged_ckpt_path),
        'meta_path': str(merged_meta_path),
        'merge_result_path': str(merge_result_path),
        'meta': meta,
    }


def parse_args():
    p = argparse.ArgumentParser('Merge client checkpoints from model_hub')
    p.add_argument('--config', type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    result = run_merge(cfg)
    print(f"merged checkpoint saved to: {result['merged_checkpoint']}")
    print(f"merge result saved to: {result['merge_result_path']}")


if __name__ == '__main__':
    main()
