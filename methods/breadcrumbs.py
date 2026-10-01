import torch

from utils.state_dict import average_state_dicts

from .common import build_task_matrix, overlay_param_dict, vector_to_param_dict


def _topk_mask(values: torch.Tensor, keep_count: int) -> torch.Tensor:
    if keep_count <= 0:
        return torch.zeros_like(values, dtype=torch.bool)
    if keep_count >= values.numel():
        return torch.ones_like(values, dtype=torch.bool)
    _, indices = torch.topk(values, keep_count)
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask.scatter_(0, indices, True)
    return mask


def _middle_keep_vector(vector: torch.Tensor, top_k_keep: float, top_k_remove: float, remove_first: bool = True) -> torch.Tensor:
    flat = vector.reshape(-1)
    magnitudes = flat.abs()
    total = flat.numel()
    keep_count = int(total * float(top_k_keep))
    remove_count = int(total * float(top_k_remove))
    if total == 0 or keep_count <= 0:
        return torch.zeros_like(vector)

    available_mask = torch.ones_like(flat, dtype=torch.bool)
    if remove_first and remove_count > 0:
        remove_mask = _topk_mask(magnitudes, remove_count)
        available_mask &= ~remove_mask
    candidate_scores = magnitudes.masked_fill(~available_mask, -1.0)
    keep_mask = _topk_mask(candidate_scores, min(keep_count, int(available_mask.sum().item())))

    if not remove_first and remove_count > 0:
        remove_mask = _topk_mask((magnitudes * keep_mask.to(magnitudes.dtype)), remove_count)
        keep_mask &= ~remove_mask

    out = torch.zeros_like(flat)
    out[keep_mask] = flat[keep_mask]
    return out.reshape_as(vector)


def merge_breadcrumbs(state_dicts, base_state, weights, top_k_keep=0.2, top_k_remove=0.1, alpha=1.0, remove_first=True, param_keys=None):
    if param_keys is None:
        param_keys = [key for key, value in base_state.items() if torch.is_floating_point(value)]
    avg_state, norm = average_state_dicts(state_dicts, weights)
    task_matrix, base_vector = build_task_matrix(state_dicts, base_state, param_keys)
    filtered = torch.vstack([
        _middle_keep_vector(row, top_k_keep=top_k_keep, top_k_remove=top_k_remove, remove_first=remove_first)
        for row in task_matrix
    ])
    coeff = filtered.new_tensor(norm).reshape(-1, 1)
    merged_delta = (filtered * coeff).sum(dim=0)
    merged_vector = base_vector + float(alpha) * merged_delta
    merged_params = vector_to_param_dict(merged_vector, base_state, param_keys)
    merged_state = overlay_param_dict(avg_state, merged_params)
    return merged_state, {
        'implementation': 'paper_breadcrumbs_global_task_vector_middle_keep',
        'top_k_keep': float(top_k_keep),
        'top_k_remove': float(top_k_remove),
        'alpha': float(alpha),
        'remove_first': bool(remove_first),
        'normalized_weights': norm,
    }
