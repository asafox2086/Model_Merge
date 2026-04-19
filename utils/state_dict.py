from collections import OrderedDict

import torch


def extract_state_dict(checkpoint_obj):
    if not isinstance(checkpoint_obj, dict):
        raise TypeError(f'Checkpoint must be a dict, got: {type(checkpoint_obj).__name__}')
    if 'state_dict' not in checkpoint_obj:
        raise KeyError('Checkpoint missing state_dict')
    return checkpoint_obj['state_dict']


def average_state_dicts(state_dicts, weights):
    if len(state_dicts) != len(weights):
        raise ValueError('state_dicts and weights length mismatch')
    total = float(sum(weights))
    if total <= 0:
        raise ValueError('weights sum must be positive')
    norm = [float(w) / total for w in weights]
    merged = OrderedDict()
    keys = list(state_dicts[0].keys())
    for key in keys:
        values = [sd[key] for sd in state_dicts]
        first = values[0]
        if torch.is_floating_point(first):
            out = first.detach().clone() * norm[0]
            for value, weight in zip(values[1:], norm[1:]):
                out.add_(value.detach(), alpha=weight)
            merged[key] = out
        else:
            merged[key] = first.detach().clone()
    return merged, norm
