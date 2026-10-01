from collections import OrderedDict

import torch


def _fft_filter_delta(delta: torch.Tensor, filter_ratio: float) -> torch.Tensor:
    freq_domain = torch.fft.fftshift(torch.fft.fft2(delta))
    h, w = freq_domain.shape
    low_radius = min(h, w) // 10
    high_radius = min(h, w) * float(filter_ratio)
    center = (h // 2, w // 2)
    y, x = torch.meshgrid(torch.arange(h, device=delta.device), torch.arange(w, device=delta.device), indexing='ij')
    dist = torch.sqrt((x - center[1]).float() ** 2 + (y - center[0]).float() ** 2)
    low_freq_mask = (dist <= low_radius).float()
    band_pass_mask = ((dist >= low_radius) & (dist <= high_radius)).float()
    low_freq_component = freq_domain * low_freq_mask
    band_pass_component = freq_domain * band_pass_mask
    kept = torch.fft.ifftshift(low_freq_component + band_pass_component)
    return torch.fft.ifft2(kept).real


def merge_free(state_dicts, base_state, weights, cfg):
    filter_ratio = float(cfg.get('free_filter_ratio', 0.7))
    scaling = float(cfg.get('free_scaling', 1.0))
    include_all_2d = bool(cfg.get('free_include_all_2d', False))
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
        out = base_value.detach().cpu().float().clone()
        delta_sum = torch.zeros_like(out)
        for state_dict, coef in zip(state_dicts, norm):
            delta = state_dict[key].detach().cpu().float() - base_value.detach().cpu().float()
            use_fft = delta.ndim == 2 and (
                include_all_2d or ('weight' in key and ('mlp' in key or 'attn' in key))
            )
            if use_fft:
                delta = _fft_filter_delta(delta, filter_ratio=filter_ratio)
            delta_sum.add_(delta, alpha=float(coef))
        merged[key] = (out + scaling * delta_sum).to(dtype=first.dtype)
    return merged, {
        'implementation': 'paper_free_frequency_filter_core_adapted_without_router',
        'paper_family': 'FREE-Merging / FR-Merging',
        'venue': 'ICCV',
        'year': 2025,
        'filter_ratio': filter_ratio,
        'scaling': scaling,
        'include_all_2d': include_all_2d,
        'normalized_weights': norm,
    }
