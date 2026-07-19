from collections import OrderedDict

import torch
import torch.nn as nn


def normalize_weights(weights):
    total = float(sum(weights))
    if total <= 0:
        raise ValueError('weights sum must be positive')
    return [float(w) / total for w in weights]


def flatten_param_vector(state_dict, param_keys):
    if not param_keys:
        return torch.zeros(0, dtype=torch.float32)
    tensors = [state_dict[key].detach().cpu().float().reshape(-1) for key in param_keys]
    return nn.utils.parameters_to_vector(tensors)


def vector_to_param_dict(vector, template_state, param_keys):
    params = [template_state[key].detach().cpu().float().clone() for key in param_keys]
    if params:
        nn.utils.vector_to_parameters(vector, params)
    return OrderedDict((key, value.to(dtype=template_state[key].dtype)) for key, value in zip(param_keys, params))


def build_task_matrix(state_dicts, base_state, param_keys):
    base_vector = flatten_param_vector(base_state, param_keys)
    deltas = [flatten_param_vector(sd, param_keys) - base_vector for sd in state_dicts]
    if not deltas:
        raise ValueError('No state_dicts provided to build task matrix')
    return torch.vstack(deltas), base_vector


def mask_smallest_magnitude(task_matrix, preserve_density):
    preserve_density = float(preserve_density)
    preserve_density = max(0.0, min(1.0, preserve_density))
    if preserve_density >= 1.0:
        return task_matrix
    if preserve_density <= 0.0:
        return torch.zeros_like(task_matrix)
    num_params = int(task_matrix.shape[1])
    num_mask = int(num_params * (1.0 - preserve_density))
    if num_mask <= 0:
        return task_matrix
    if num_mask >= num_params:
        return torch.zeros_like(task_matrix)
    kth_values = task_matrix.abs().kthvalue(k=num_mask, dim=1, keepdim=True).values
    mask = task_matrix.abs() >= kth_values
    return task_matrix * mask.to(task_matrix.dtype)


def elect_sign(task_matrix):
    return torch.sign(task_matrix.sum(dim=0))


def disjoint_merge(task_matrix, param_signs, weights=None):
    if weights is None:
        coeff = torch.full((task_matrix.shape[0], 1), 1.0 / task_matrix.shape[0], dtype=task_matrix.dtype)
    else:
        coeff = torch.tensor(normalize_weights(weights), dtype=task_matrix.dtype).reshape(-1, 1)
    preserve_mask = (((param_signs.unsqueeze(0) > 0) & (task_matrix > 0)) |
                     ((param_signs.unsqueeze(0) < 0) & (task_matrix < 0)))
    preserve = task_matrix * preserve_mask.to(task_matrix.dtype)
    numerator = (preserve * coeff).sum(dim=0)
    denominator = (preserve_mask.to(task_matrix.dtype) * coeff).sum(dim=0)
    out = torch.zeros_like(numerator)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def random_prune(task_matrix, preserve_density, seed):
    preserve_density = float(preserve_density)
    preserve_density = max(0.0, min(1.0, preserve_density))
    if preserve_density >= 1.0:
        return task_matrix
    if preserve_density <= 0.0:
        return torch.zeros_like(task_matrix)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    mask = torch.rand(task_matrix.shape, generator=generator, device=task_matrix.device) < preserve_density
    return task_matrix * mask.to(task_matrix.dtype) / preserve_density


def overlay_param_dict(base_state, param_dict):
    merged = OrderedDict()
    for key, value in base_state.items():
        if key in param_dict:
            merged[key] = param_dict[key].detach().cpu().clone()
        else:
            merged[key] = value.detach().cpu().clone()
    return merged
