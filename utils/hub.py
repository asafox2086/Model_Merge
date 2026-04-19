from pathlib import Path

from .io import load_json


def beta_to_dirname(beta):
    return f"beta_{format(float(beta), 'g').replace('.', 'p')}"


def client_dirname(num_clients):
    return f'clients_{int(num_clients)}'


def seed_dirname(seed):
    return f'seed_{int(seed)}'


def find_hub_experiment_dir(model_hub_root, config):
    root = Path(model_hub_root)
    task_type = config['task_type']
    dataset = config['dataset']
    num_clients = client_dirname(config['num_clients'])
    beta_dir = beta_to_dirname(config['beta'])
    seed_dir = seed_dirname(config['seed'])

    if task_type == 'small':
        model = config['model']
        exp_dir = root / 'small' / dataset / model / num_clients / beta_dir / seed_dir
    elif task_type == 'vlm':
        clip_model = config['clip_model'].split('/')[-1]
        exp_dir = root / 'vlm' / dataset / clip_model / num_clients / beta_dir / seed_dir
    else:
        raise ValueError(f'Unsupported task_type: {task_type}')

    if not exp_dir.exists():
        raise FileNotFoundError(f'Experiment dir not found: {exp_dir}')
    return exp_dir


def load_hub_meta(exp_dir):
    meta_path = Path(exp_dir) / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'meta.json not found: {meta_path}')
    return load_json(meta_path)
