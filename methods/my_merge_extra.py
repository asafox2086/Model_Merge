from collections import OrderedDict

import torch


EPS = 1e-8


def _normalize(values, fallback=None):
    scores = torch.as_tensor(values, dtype=torch.float32)
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = float(scores.sum().item())
    if total <= 0.0:
        if fallback is not None:
            return torch.as_tensor(fallback, dtype=torch.float32)
        return torch.ones_like(scores) / max(1, scores.numel())
    return scores / total


def _weighted_average(values, weights):
    out = values[0].detach().clone() * float(weights[0].item())
    for value, weight in zip(values[1:], weights[1:]):
        out.add_(value.detach(), alpha=float(weight.item()))
    return out


def _is_classifier_tensor(key, tensor, num_classes):
    name = key.lower()
    if num_classes <= 0 or not torch.is_floating_point(tensor):
        return False
    if tensor.ndim == 1 and tensor.shape[0] == num_classes:
        return any(token in name for token in ("head", "classifier", "fc", "linear"))
    if tensor.ndim == 2 and tensor.shape[0] == num_classes:
        return any(token in name for token in ("head", "classifier", "fc", "linear"))
    return False


def _param_group(key):
    name = key.lower()
    if any(token in name for token in ("conv1", "stem", "patch_embed", "features.0", "layer1", "blocks.0", "blocks.1")):
        return "early"
    if any(token in name for token in ("head", "classifier", "fc", "layer4", "blocks.10", "blocks.11")):
        return "late"
    return "mid"


def _client_classes(meta, client_idx):
    if not meta:
        return set()
    clients = meta.get("clients", [])
    if client_idx >= len(clients):
        return set()
    return {int(cls) for cls in clients[client_idx].get("classes", [])}


def _model_consistency_weights(state_dicts, priors):
    keys = [
        key
        for key, value in state_dicts[0].items()
        if torch.is_tensor(value) and torch.is_floating_point(value) and value.ndim >= 2
    ]
    if not keys:
        return priors

    distances = torch.zeros(len(state_dicts), dtype=torch.float32)
    used = 0
    for key in keys:
        values = [state[key].detach().float() for state in state_dicts]
        mean_value = sum(value * float(weight.item()) for value, weight in zip(values, priors))
        denom = mean_value.pow(2).mean().sqrt().item() + EPS
        for idx, value in enumerate(values):
            distances[idx] += (value - mean_value).pow(2).mean().sqrt().item() / denom
        used += 1
    distances = distances / max(1, used)
    consistency = _normalize(priors / (distances + distances.mean() + EPS), fallback=priors)
    return _normalize(0.45 * priors + 0.55 * consistency, fallback=priors)


def _classifier_row_weights(base_weights, cls_idx, meta):
    if not meta:
        return base_weights
    scores = base_weights.clone()
    for client_idx in range(scores.numel()):
        seen = _client_classes(meta, client_idx)
        if seen:
            scores[client_idx] *= 1.35 if cls_idx in seen else 0.75
    return _normalize(scores, fallback=base_weights)


def merge_my_merge_extra(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    """A compact post-hoc checkpoint fusion method.

    This method is deliberately small: it does not run diagnostics, validation
    candidate selection, plotting, or ablation bookkeeping.  It computes a
    consistency-calibrated client weight from the checkpoints themselves, uses
    slightly different weights for early/middle/late parameters, and applies
    class-aware row weights to classifier heads when client class metadata is
    available.
    """

    if not state_dicts:
        raise ValueError("state_dicts must not be empty")

    priors = _normalize(weights)
    consistency_weights = _model_consistency_weights(state_dicts, priors)
    num_classes = int((meta or {}).get("num_classes", 0))
    layer_blend = {"early": 0.35, "mid": 0.55, "late": 0.75}

    merged = OrderedDict()
    for key in state_dicts[0].keys():
        values = [state[key] for state in state_dicts]
        first = values[0]
        if not torch.is_floating_point(first):
            merged[key] = values[int(torch.argmax(consistency_weights).item())].detach().clone()
            continue

        if _is_classifier_tensor(key, first, num_classes):
            out = first.detach().clone().float().zero_()
            rows = first.shape[0] if first.ndim >= 1 else 0
            for cls_idx in range(rows):
                row_weights = _classifier_row_weights(consistency_weights, cls_idx, meta)
                row_values = [value[cls_idx] for value in values]
                out[cls_idx] = _weighted_average(row_values, row_weights)
            merged[key] = out.to(dtype=first.dtype)
            continue

        group = _param_group(key)
        blend = layer_blend[group]
        layer_weights = _normalize((1.0 - blend) * priors + blend * consistency_weights, fallback=priors)
        merged[key] = _weighted_average(values, layer_weights)

    return merged, {
        "implementation": "compact_medical_checkpoint_fusion",
        "medical_only": False,
        "base_weights": [float(v) for v in priors.tolist()],
        "consistency_weights": [float(v) for v in consistency_weights.tolist()],
        "fusion_rule": "checkpoint_consistency_layerwise_classaware",
    }


__all__ = ["merge_my_merge_extra"]
