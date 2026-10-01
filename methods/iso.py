from collections import OrderedDict

import torch

from .common import normalize_weights


def _device_from_cfg(cfg):
    dev = str(cfg.get('device', 'cpu'))
    return torch.device(dev if dev else 'cpu')


def _task_deltas(state_dicts, base_state, key, device):
    return [state_dict[key].detach().to(device).float() - base_state[key].detach().to(device).float() for state_dict in state_dicts]


def _safe_svd(matrix):
    out_dtype = matrix.dtype
    attempts = [matrix]
    if matrix.dtype != torch.float64:
        attempts.append(matrix.double())
    for candidate in attempts:
        try:
            u, s, v = torch.linalg.svd(candidate, full_matrices=False)
            return u.to(out_dtype), s.to(out_dtype), v.to(out_dtype)
        except RuntimeError:
            pass
    jitter = 1e-6 * torch.randn_like(attempts[-1])
    u, s, v = torch.linalg.svd(attempts[-1] + jitter, full_matrices=False)
    return u.to(out_dtype), s.to(out_dtype), v.to(out_dtype)


def merge_iso_c(state_dicts, base_state, weights, cfg):
    device = _device_from_cfg(cfg)
    merged = OrderedDict()
    num_models = len(state_dicts)
    norm = normalize_weights(weights)
    for key, base_value in base_state.items():
        deltas = _task_deltas(state_dicts, base_state, key, device)
        merged_delta = torch.zeros_like(deltas[0])
        for delta, coef in zip(deltas, norm):
            merged_delta.add_(delta, alpha=float(coef))
        is_2d = len(merged_delta.shape) == 2 and 'text_projection' not in key
        if is_2d:
            u, s, v = _safe_svd(merged_delta)
            s_mean = torch.ones_like(s) * s.mean()
            merged_delta = torch.linalg.multi_dot((u, torch.diag(s_mean), v))
        out = base_value.detach().to(device).float() + merged_delta
        merged[key] = out.detach().cpu().to(dtype=base_value.dtype)
    return merged, {
        'implementation': 'paper_iso_merging_iso_c_weighted_2d_adapted',
        'paper': 'No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces',
        'venue': 'ICML',
        'year': 2025,
        'num_models': num_models,
        'normalized_weights': norm,
    }


def merge_iso_cts(state_dicts, base_state, weights, cfg):
    device = _device_from_cfg(cfg)
    common_space_fraction = float(cfg.get('iso_common_space_fraction', 0.8))
    num_models = len(state_dicts)
    norm = normalize_weights(weights)
    merged = OrderedDict()
    for key, base_value in base_state.items():
        deltas = _task_deltas(state_dicts, base_state, key, device)
        shape_ = deltas[0].shape
        is_2d = len(shape_) == 2 and 'text_projection' not in key
        if not is_2d:
            merged_delta = torch.zeros_like(deltas[0])
            for delta, coef in zip(deltas, norm):
                merged_delta.add_(delta, alpha=float(coef))
            out = base_value.detach().to(device).float() + merged_delta
            merged[key] = out.detach().cpu().to(dtype=base_value.dtype)
            continue

        combined_w = torch.zeros_like(deltas[0])
        for delta, coef in zip(deltas, norm):
            combined_w.add_(delta, alpha=float(coef))
        min_dim = min(shape_)
        common_space_index_s = int(min_dim * common_space_fraction)
        task_specific_total = round((min_dim - common_space_index_s) / num_models) * num_models
        common_space_index_s = min_dim - task_specific_total

        u, s, v = _safe_svd(combined_w)
        common_space_u = u[:, :common_space_index_s]
        common_space_s = s[:common_space_index_s]
        common_space_v = v[:common_space_index_s, :]

        n_dims_per_task = int((min_dim - common_space_index_s) / num_models)
        combined_space_u = torch.zeros_like(u, device=device)
        combined_space_s = torch.zeros_like(s, device=device)
        combined_space_v = torch.zeros_like(v, device=device)

        for idx, delta in enumerate(deltas):
            w_ts = delta - common_space_u @ common_space_u.T @ delta
            u_ts, s_ts, v_ts = _safe_svd(w_ts)
            start = idx * n_dims_per_task
            end = (idx + 1) * n_dims_per_task
            combined_space_u[:, start:end] = u_ts[:, :n_dims_per_task]
            combined_space_s[start:end] = s_ts[:n_dims_per_task]
            combined_space_v[start:end, :] = v_ts[:n_dims_per_task, :]

        start = num_models * n_dims_per_task
        end = start + common_space_index_s
        combined_space_u[:, start:end] = common_space_u
        combined_space_s[start:end] = common_space_s
        combined_space_v[start:end, :] = common_space_v

        u_u, _, v_u = _safe_svd(combined_space_u)
        u_v, _, v_v = _safe_svd(combined_space_v)
        combined_space_u = u_u @ v_u
        combined_space_v = u_v @ v_v
        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()

        merged_delta = torch.linalg.multi_dot((combined_space_u, torch.diag(combined_space_s), combined_space_v))
        out = base_value.detach().to(device).float() + merged_delta
        merged[key] = out.detach().cpu().to(dtype=base_value.dtype)

    return merged, {
        'implementation': 'paper_iso_merging_iso_cts_weighted_2d_adapted',
        'paper': 'No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces',
        'venue': 'ICML',
        'year': 2025,
        'num_models': num_models,
        'common_space_fraction': common_space_fraction,
        'normalized_weights': norm,
    }
