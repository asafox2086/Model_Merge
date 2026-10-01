from collections import OrderedDict

import torch

from utils.state_dict import average_state_dicts


def _fisher_l2_norm(fisher_stats):
    values = [value.detach().cpu().float().reshape(-1) for value in fisher_stats.values() if torch.is_floating_point(value)]
    if not values:
        return torch.tensor(0.0)
    return torch.norm(torch.cat(values), p=2)


def merge_fisher(state_dicts, fisher_stats, weights, eps=1e-8, normalize_fisher_weight=True, minimal_fisher_weight=1e-6):
    avg_state, norm = average_state_dicts(state_dicts, weights)
    merged = OrderedDict((key, value.detach().cpu().clone()) for key, value in avg_state.items())

    scaling = torch.tensor(norm, dtype=torch.float32)
    if normalize_fisher_weight:
        fisher_norms = torch.stack([_fisher_l2_norm(item) for item in fisher_stats])
        inv = 1.0 / (fisher_norms + float(minimal_fisher_weight))
        inv = inv / inv.sum().clamp_min(float(minimal_fisher_weight))
        scaling = scaling * inv

    for key, first in avg_state.items():
        if not torch.is_floating_point(first):
            continue
        numerator = torch.zeros_like(first, dtype=torch.float32)
        denominator = torch.zeros_like(first, dtype=torch.float32)
        for state_dict, fisher_dict, scale in zip(state_dicts, fisher_stats, scaling.tolist()):
            fisher_weight = fisher_dict.get(key)
            if fisher_weight is None:
                continue
            value = state_dict[key].detach().cpu().float()
            coeff = fisher_weight.detach().cpu().float() + float(minimal_fisher_weight)
            coeff = coeff * float(scale)
            numerator.add_(coeff * value)
            denominator.add_(coeff)
        valid = denominator > float(eps)
        if valid.any():
            out = avg_state[key].detach().cpu().float()
            out[valid] = numerator[valid] / denominator[valid]
            merged[key] = out.to(dtype=first.dtype)
    return merged, {
        'implementation': 'official_fisher_core',
        'eps': float(eps),
        'normalize_fisher_weight': bool(normalize_fisher_weight),
        'minimal_fisher_weight': float(minimal_fisher_weight),
        'normalized_weights': norm,
        'effective_scaling': scaling.tolist(),
    }
