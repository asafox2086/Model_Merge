from pathlib import Path


def ensure_task_matches_config(meta, config):
    expected_task = config['task_type']
    if meta.get('task_type') != expected_task:
        raise ValueError(f"task_type mismatch: meta={meta.get('task_type')} config={expected_task}")
    if meta.get('dataset') != config['dataset']:
        raise ValueError(f"dataset mismatch: meta={meta.get('dataset')} config={config['dataset']}")
    if expected_task == 'small':
        if meta.get('model') != config['model']:
            raise ValueError(f"model mismatch: meta={meta.get('model')} config={config['model']}")
    else:
        clip_dir = config['clip_model'].split('/')[-1]
        if meta.get('model') != clip_dir:
            raise ValueError(f"clip model mismatch: meta={meta.get('model')} config={clip_dir}")


def ensure_checkpoint_files(exp_dir, meta):
    exp_dir = Path(exp_dir)
    clients = meta.get('clients', [])
    if not clients:
        raise ValueError('No clients found in meta.json')
    paths = []
    for item in clients:
        ckpt = exp_dir / item['checkpoint']
        if not ckpt.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
        paths.append(ckpt)
    return paths


def ensure_state_dicts_compatible(state_dicts):
    if not state_dicts:
        raise ValueError('No state_dicts provided for merge')
    ref = state_dicts[0]
    ref_keys = list(ref.keys())
    for idx, sd in enumerate(state_dicts[1:], start=1):
        keys = list(sd.keys())
        if keys != ref_keys:
            raise ValueError(f'state_dict keys mismatch at index={idx}')
        for key in ref_keys:
            if tuple(sd[key].shape) != tuple(ref[key].shape):
                raise ValueError(f'shape mismatch at index={idx}, key={key}: {tuple(sd[key].shape)} vs {tuple(ref[key].shape)}')
