from collections import OrderedDict

import torch

from utils.state_dict import average_state_dicts


def _reduce_non_diagonal_elements(regmean_weights, reduce_non_diagonal_ratio):
    ratio = float(reduce_non_diagonal_ratio)
    if ratio >= 1.0:
        return regmean_weights
    eye = torch.eye(regmean_weights.shape[0], dtype=regmean_weights.dtype, device=regmean_weights.device)
    return regmean_weights * eye + regmean_weights * (1.0 - eye) * ratio


def merge_regmean(state_dicts, cov_stats, weights, eps=1e-6, reduce_non_diagonal_ratio=1.0):
    avg_state, norm = average_state_dicts(state_dicts, weights)
    merged = OrderedDict((key, value.detach().cpu().clone()) for key, value in avg_state.items())

    for key, first in avg_state.items():
        if not (key.endswith('.weight') and first.ndim == 2 and torch.is_floating_point(first)):
            continue
        module_name = key[:-len('.weight')]
        if not all(module_name in stats for stats in cov_stats):
            continue

        sum_cov = None
        sum_cov_weight = None
        for state_dict, stats, weight in zip(state_dicts, cov_stats, norm):
            cov = stats[module_name].detach().cpu().float()
            cov = _reduce_non_diagonal_elements(cov, reduce_non_diagonal_ratio=reduce_non_diagonal_ratio)
            w = state_dict[key].detach().cpu().float()
            cov_weight = cov @ w.transpose(0, 1)
            if sum_cov is None:
                sum_cov = cov * float(weight)
                sum_cov_weight = cov_weight * float(weight)
            else:
                sum_cov.add_(cov, alpha=float(weight))
                sum_cov_weight.add_(cov_weight, alpha=float(weight))

        eye = torch.eye(sum_cov.shape[0], dtype=torch.float32)
        merged_param = torch.linalg.pinv(sum_cov + float(eps) * eye) @ sum_cov_weight
        merged[key] = merged_param.transpose(0, 1).to(dtype=first.dtype)

    return merged, {
        'implementation': 'official_regmean_core',
        'eps': float(eps),
        'reduce_non_diagonal_ratio': float(reduce_non_diagonal_ratio),
        'normalized_weights': norm,
    }
