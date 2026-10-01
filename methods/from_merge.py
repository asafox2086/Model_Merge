from collections import OrderedDict

import torch

from utils.state_dict import average_state_dicts


def _task_vector_norm_power(state_dict, base_state, power: float) -> float:
    values = []
    for key, base_value in base_state.items():
        if key not in state_dict:
            continue
        value = state_dict[key]
        if not torch.is_floating_point(value):
            continue
        delta = value.detach().cpu().float() - base_value.detach().cpu().float()
        values.append(torch.sum(delta * delta))
    if not values:
        return 1.0
    norm = torch.sqrt(torch.stack(values).sum()).item()
    return float(norm ** power)


def merge_from(state_dicts, base_state, weights, k=1.0):
    avg_state, norm_weights = average_state_dicts(state_dicts, weights)
    coeffs = [
        float(norm_weight) * _task_vector_norm_power(sd, base_state, power=float(k))
        for sd, norm_weight in zip(state_dicts, norm_weights)
    ]
    total = sum(coeffs)
    if total <= 0:
        coeffs = [1.0 for _ in state_dicts]
        total = float(len(coeffs))
    norm = [c / total for c in coeffs]
    merged = OrderedDict((key, value.detach().cpu().clone()) for key, value in avg_state.items())
    for key, base_value in base_state.items():
        if key not in state_dicts[0]:
            continue
        first = state_dicts[0][key]
        if not torch.is_floating_point(first):
            merged[key] = first.detach().cpu().clone()
            continue
        out = base_value.detach().cpu().float().clone()
        delta_sum = torch.zeros_like(out)
        for state_dict, coeff in zip(state_dicts, norm):
            delta = state_dict[key].detach().cpu().float() - base_value.detach().cpu().float()
            delta_sum.add_(delta, alpha=float(coeff))
        merged[key] = (out + delta_sum).to(dtype=first.dtype)
    return merged, {
        'implementation': 'paper_from_closed_form_task_vector_scaling',
        'k': float(k),
        'effective_scaling': norm,
        'normalized_weights': norm_weights,
    }
