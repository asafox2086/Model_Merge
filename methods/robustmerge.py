from collections import OrderedDict

import torch


def _official_clamp(x: torch.Tensor, min_ratio=0.0, max_ratio=0.0):
    if x.ndim == 1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min_v = sorted_x[int(d * min_ratio)]
        max_v = sorted_x[int(d * (1 - max_ratio) - 1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min_v = sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max_v = sorted_x[:, int(d * (1 - max_ratio) - 1)].unsqueeze(1)
    return torch.clamp(x, min_v, max_v)


def merge_robustmerge(state_dicts, base_state, weights, cfg):
    mask_ratio = float(cfg.get('robustmerge_mask_ratio', 0.2))
    att_ratio = float(cfg.get('robustmerge_att_ratio', 0.2))
    fuse_weight = float(cfg.get('robustmerge_fuse_weight', 2.0))
    include_all_2d = bool(cfg.get('robustmerge_include_all_2d', True))
    merged = OrderedDict()
    total = float(sum(weights)) if sum(weights) > 0 else float(len(weights))
    norm = [float(w) / total for w in (weights if sum(weights) > 0 else [1] * len(weights))]
    for key, base_value in base_state.items():
        if key not in state_dicts[0]:
            continue
        first = state_dicts[0][key]
        if not torch.is_floating_point(first):
            merged[key] = first.detach().cpu().clone()
            continue
        if first.ndim != 2 and not include_all_2d:
            out = base_value.detach().cpu().float().clone()
            for sd, coef in zip(state_dicts, norm):
                delta = sd[key].detach().cpu().float() - base_value.detach().cpu().float()
                out.add_(delta, alpha=float(coef))
            merged[key] = out.to(dtype=first.dtype)
            continue
        if first.ndim == 2:
            stack = torch.stack([sd[key].detach().cpu().float() - base_value.detach().cpu().float() for sd in state_dicts], dim=0)
            t, d1, d2 = stack.shape
            flat = stack.reshape(t, -1)
            k = max(1, min(flat.shape[1], int(flat.shape[1] * (1 - mask_ratio))))
            kth_values, _ = flat.abs().kthvalue(k, dim=1, keepdim=True)
            masks = (flat.abs() >= kth_values).reshape(t, d1, d2)
            trimmed = masks * stack
            denom = torch.sum(torch.abs(trimmed), dim=-1).clamp_min(1e-8)
            s_vector = torch.sum(torch.abs(stack), dim=-1) / denom
            scale = _official_clamp(s_vector, 1 - att_ratio, 0.0)
            weighted = torch.tensor(norm, dtype=stack.dtype).reshape(t, 1, 1) * float(fuse_weight)
            numerator = torch.sum(weighted * scale.unsqueeze(2) * trimmed, dim=0)
            denominator = torch.sum(weighted * scale.unsqueeze(2), dim=0).clamp_min(1e-8)
            merged_delta = numerator / denominator
            merged[key] = (base_value.detach().cpu().float() + merged_delta).to(dtype=first.dtype)
        else:
            out = base_value.detach().cpu().float().clone()
            for sd, coef in zip(state_dicts, norm):
                delta = sd[key].detach().cpu().float() - base_value.detach().cpu().float()
                out.add_(delta, alpha=float(coef))
            merged[key] = out.to(dtype=first.dtype)
    return merged, {
        'implementation': 'paper_robustmerge_core_adapted_to_full_parameter_matrices',
        'venue': 'NeurIPS',
        'year': 2025,
        'mask_ratio': mask_ratio,
        'att_ratio': att_ratio,
        'fuse_weight': fuse_weight,
        'include_all_2d': include_all_2d,
        'normalized_weights': norm,
    }
