from utils.state_dict import average_state_dicts

from .common import build_task_matrix, disjoint_merge, elect_sign, mask_smallest_magnitude, overlay_param_dict, random_prune, vector_to_param_dict


def merge_dare_linear(state_dicts, base_state, weights, density=0.5, seed=42, param_keys=None, scaling_coefficient=1.0):
    if param_keys is None:
        param_keys = list(base_state.keys())
    avg_state, norm = average_state_dicts(state_dicts, weights)
    task_matrix, base_vector = build_task_matrix(state_dicts, base_state, param_keys)
    dared = random_prune(task_matrix, preserve_density=density, seed=seed)
    coeff = dared.new_tensor(norm).reshape(-1, 1)
    merged_delta = (dared * coeff).sum(dim=0)
    merged_vector = base_vector + float(scaling_coefficient) * merged_delta
    merged_params = vector_to_param_dict(merged_vector, base_state, param_keys)
    merged_state = overlay_param_dict(avg_state, merged_params)
    return merged_state, {
        'implementation': 'official_dare_linear_core',
        'preserve_density': float(density),
        'mask_rate': float(1.0 - density),
        'seed': int(seed),
        'scaling_coefficient': float(scaling_coefficient),
        'normalized_weights': norm,
    }


def merge_dare_ties(state_dicts, base_state, weights, density=0.5, seed=42, param_keys=None, scaling_coefficient=1.0):
    if param_keys is None:
        param_keys = list(base_state.keys())
    avg_state, norm = average_state_dicts(state_dicts, weights)
    task_matrix, base_vector = build_task_matrix(state_dicts, base_state, param_keys)
    dared = random_prune(task_matrix, preserve_density=density, seed=seed)
    trimmed = mask_smallest_magnitude(dared, preserve_density=density)
    elected_sign = elect_sign(trimmed)
    merged_delta = disjoint_merge(trimmed, elected_sign, weights=weights)
    merged_vector = base_vector + float(scaling_coefficient) * merged_delta
    merged_params = vector_to_param_dict(merged_vector, base_state, param_keys)
    merged_state = overlay_param_dict(avg_state, merged_params)
    return merged_state, {
        'implementation': 'official_dare_ties_core',
        'preserve_density': float(density),
        'mask_rate': float(1.0 - density),
        'seed': int(seed),
        'scaling_coefficient': float(scaling_coefficient),
        'normalized_weights': norm,
    }
