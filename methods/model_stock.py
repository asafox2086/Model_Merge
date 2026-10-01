from collections import OrderedDict
from math import sqrt

import torch

from utils.state_dict import average_state_dicts

EPS = 1e-8


def _cosine(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    num = torch.sum(vec1 * vec2)
    den = sqrt(float(torch.sum(vec1 ** 2) * torch.sum(vec2 ** 2)) + EPS)
    if den <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    out = num / den
    return torch.clamp(out, min=-1.0, max=1.0)


def _compute_ratio(cosine_val: torch.Tensor, k: float = 2.0) -> float:
    cos = float(cosine_val.detach().cpu())
    return float(k * cos / (((k - 1.0) * cos) + 1.0 + EPS))


def merge_model_stock(state_dicts, base_state, weights, k=2.0):
    avg_state, norm = average_state_dicts(state_dicts, weights)
    merged = OrderedDict((key, value.detach().cpu().clone()) for key, value in avg_state.items())
    ratios = {}
    num_models = len(state_dicts)
    for key, base_value in base_state.items():
        if key not in state_dicts[0]:
            continue
        first = state_dicts[0][key]
        if not torch.is_floating_point(first):
            merged[key] = first.detach().cpu().clone()
            continue
        deltas = [sd[key].detach().cpu().float() - base_value.detach().cpu().float() for sd in state_dicts]
        mean_delta = torch.zeros_like(deltas[0])
        for delta, coef in zip(deltas, norm):
            mean_delta.add_(delta, alpha=float(coef))
        if num_models == 2:
            cosine_val = _cosine(deltas[0], deltas[1])
        else:
            cosine_val = torch.tensor(0.0, dtype=torch.float32)
            for delta, coef in zip(deltas, norm):
                cosine_val = cosine_val + float(coef) * _cosine(delta, mean_delta)
        ratio = _compute_ratio(cosine_val, k=float(k))
        ratios[key] = ratio
        merged[key] = (base_value.detach().cpu().float() + ratio * mean_delta).to(dtype=first.dtype)
    return merged, {
        'implementation': 'paper_model_stock_core_with_weighted_multiclient_extension',
        'k': float(k),
        'normalized_weights': norm,
        'num_models': num_models,
    }
