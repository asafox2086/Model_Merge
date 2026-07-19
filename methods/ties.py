import torch

from utils.state_dict import average_state_dicts

from .common import build_task_matrix, disjoint_merge, elect_sign, mask_smallest_magnitude, overlay_param_dict, vector_to_param_dict


def merge_ties(state_dicts, base_state, weights, density=0.5, param_keys=None, scaling_coefficient=1.0):
    if param_keys is None:
        param_keys = [key for key, value in base_state.items() if torch.is_floating_point(value)]
    avg_state, norm = average_state_dicts(state_dicts, weights)
    task_matrix, base_vector = build_task_matrix(state_dicts, base_state, param_keys)
    trimmed = mask_smallest_magnitude(task_matrix, preserve_density=density)
    elected_sign = elect_sign(trimmed)
    merged_delta = disjoint_merge(trimmed, elected_sign, weights=weights)
    merged_vector = base_vector + float(scaling_coefficient) * merged_delta
    merged_params = vector_to_param_dict(merged_vector, base_state, param_keys)
    merged_state = overlay_param_dict(avg_state, merged_params)
    return merged_state, {
        'implementation': 'official_ties_core',
        'preserve_density': float(density),
        'mask_rate': float(1.0 - density),
        'scaling_coefficient': float(scaling_coefficient),
        'normalized_weights': norm,
    }
