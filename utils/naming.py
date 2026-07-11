from pathlib import Path

from .hub import beta_to_dirname


def make_merge_output_dir(output_root, config):
    root = Path(output_root) / 'merged' / config['task_type'] / config['dataset']
    model_name = config['model'] if config['task_type'] == 'small' else config['clip_model'].split('/')[-1]
    return root / model_name / f"clients_{int(config['num_clients'])}" / beta_to_dirname(config['beta']) / f"seed_{int(config['seed'])}" / config['method']


def make_eval_output_dir(output_root, config):
    root = Path(output_root) / 'eval' / config['task_type'] / config['dataset']
    model_name = config['model'] if config['task_type'] == 'small' else config['clip_model'].split('/')[-1]
    return root / model_name / f"clients_{int(config['num_clients'])}" / beta_to_dirname(config['beta']) / f"seed_{int(config['seed'])}" / config['method']
