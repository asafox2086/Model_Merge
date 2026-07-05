from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.runtime import build_reference_bundle
from utils.hub import beta_to_dirname
from utils.state_dict import average_state_dicts


EPS = 1e-8

MEDICAL_IMAGE_DATASETS = {
    "bloodmnist_224",
    "chaoshengmnist_224",
    "dermamnist_224",
    "organamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "pathmnist_224",
}


def _is_medical_image_task(meta):
    return meta.get("task_type") in {"small", "vlm"} and meta.get("dataset") in MEDICAL_IMAGE_DATASETS


def _normalize(values, dim=0):
    values = torch.as_tensor(values, dtype=torch.float32).clamp_min(0.0)
    return values / values.sum(dim=dim, keepdim=True).clamp_min(EPS)


def _uploaded_client_value(client_stats, idx, field):
    if not client_stats:
        return None
    clients = client_stats.get("clients", [])
    if idx >= len(clients):
        return None
    return clients[idx].get(field)


def _client_class_matrix(meta, client_stats=None):
    num_classes = int(meta["num_classes"])
    rows = []
    reliability_rows = []
    supports = []
    clients = meta.get("clients", [])
    for idx, client in enumerate(clients):
        row = torch.zeros(num_classes, dtype=torch.float32)
        seen = [int(c) for c in client.get("classes", []) if 0 <= int(c) < num_classes]
        num_samples = float(client.get("num_samples", 1.0))
        uploaded_counts = (
            _uploaded_client_value(client_stats, idx, "class_counts")
            or client.get("class_counts")
            or client.get("class_support")
            or client.get("label_counts")
        )
        if isinstance(uploaded_counts, dict):
            for cls_key, count in uploaded_counts.items():
                cls_idx = int(cls_key)
                if 0 <= cls_idx < num_classes:
                    row[cls_idx] = float(count)
        elif isinstance(uploaded_counts, (list, tuple)) and len(uploaded_counts) == num_classes:
            row = torch.tensor([float(x) for x in uploaded_counts], dtype=torch.float32)
        elif seen:
            per_class = num_samples / float(len(seen))
            for cls_idx in seen:
                row[cls_idx] = per_class
        else:
            row.fill_(num_samples / float(max(1, num_classes)))
        rows.append(row)
        supports.append(num_samples)
        val_acc = client.get("best_val_acc", None)
        if val_acc is None:
            reliability = 1.0
        else:
            random_prior = 1.0 / float(max(1, len(seen) or num_classes))
            score = (float(val_acc) - random_prior) / max(1.0 - random_prior, EPS)
            reliability = max(0.05, score + 0.20)
        reliability_row = torch.full((num_classes,), float(reliability), dtype=torch.float32)
        reliability_rows.append(reliability_row)

    if not rows:
        raise ValueError("my_merge requires at least one client in meta['clients'].")
    class_counts = torch.stack(rows, dim=0).clamp_min(0.0)
    reliability = torch.stack(reliability_rows, dim=0).clamp(0.05, 2.0)
    support = torch.tensor(supports, dtype=torch.float32).clamp_min(1.0)
    return class_counts, reliability, support


def _class_client_weights(meta, cfg, client_stats=None):
    class_counts, reliability, support = _client_class_matrix(meta, client_stats=client_stats)
    rho = float(cfg.get("my_merge_class_count_power", 0.70))
    eta = float(cfg.get("my_merge_reliability_power", 0.35))
    floor = float(cfg.get("my_merge_class_weight_floor", 0.015))

    evidence = (class_counts + 1.0).pow(rho) * reliability.pow(eta)
    class_weights = _normalize(evidence, dim=0)
    sample_weights = _normalize(support, dim=0).view(-1)
    if floor > 0:
        class_weights = _normalize((1.0 - floor) * class_weights + floor * sample_weights.view(-1, 1), dim=0)

    class_prior = _normalize(class_counts.sum(dim=0), dim=0).view(-1)
    balanced_client = class_weights.mean(dim=1)
    balanced_client = _normalize(balanced_client, dim=0).view(-1)
    return {
        "class_counts": class_counts,
        "class_weights": class_weights,
        "client_weights": balanced_client,
        "sample_weights": sample_weights,
        "class_prior": class_prior,
        "reliability": reliability.mean(dim=1),
        "class_reliability": reliability,
    }


def _name_tokens(key):
    return {part.lower() for part in key.replace("/", ".").split(".")}


def _is_classifier_tensor(key, tensor, num_classes):
    if not torch.is_floating_point(tensor) or tensor.ndim < 1 or int(tensor.shape[0]) != int(num_classes):
        return False
    tokens = _name_tokens(key)
    head_tokens = {"head", "fc", "classifier", "classif", "last_linear", "logits"}
    if tokens & head_tokens:
        return True
    tail = key.lower().split(".")[-1]
    return tail in {"weight", "bias"} and any(token in key.lower() for token in head_tokens)


def _weighted_tensor(values, weights):
    out = torch.zeros_like(values[0].detach().cpu().float())
    for value, weight in zip(values, weights.tolist()):
        out.add_(value.detach().cpu().float(), alpha=float(weight))
    return out


def _merge_classifier_tensor(key, state_dicts, class_weights, class_prior, cfg):
    values = [sd[key] for sd in state_dicts]
    first = values[0]
    out = torch.zeros_like(first.detach().cpu().float())
    for cls_idx in range(int(class_weights.shape[1])):
        row = torch.zeros_like(out[cls_idx])
        for value, weight in zip(values, class_weights[:, cls_idx].tolist()):
            row.add_(value[cls_idx].detach().cpu().float(), alpha=float(weight))
        out[cls_idx] = row

    if first.ndim == 1:
        tau = float(cfg.get("my_merge_prior_bias_tau", 0.15))
        if tau != 0:
            centered_log_prior = torch.log(class_prior.clamp_min(EPS))
            centered_log_prior = centered_log_prior - centered_log_prior.mean()
            out = out - tau * centered_log_prior.to(out.device)
    return out.to(dtype=first.dtype)


def _mask_delta(delta, density):
    density = max(0.0, min(1.0, float(density)))
    if density >= 1.0 or delta.numel() == 0:
        return delta
    if density <= 0.0:
        return torch.zeros_like(delta)
    num_mask = int(delta.numel() * (1.0 - density))
    if num_mask <= 0:
        return delta
    if num_mask >= delta.numel():
        return torch.zeros_like(delta)
    threshold = delta.abs().reshape(-1).kthvalue(k=num_mask).values
    return delta * (delta.abs() >= threshold).to(delta.dtype)


def _merge_task_tensor(key, state_dicts, base_state, weights, cfg):
    first = state_dicts[0][key]
    base = base_state.get(key)
    if base is None or tuple(base.shape) != tuple(first.shape):
        values = [sd[key] for sd in state_dicts]
        return _weighted_tensor(values, weights).to(dtype=first.dtype)

    density = float(cfg.get("my_merge_task_density", 0.55))
    alpha = float(cfg.get("my_merge_task_alpha", 1.0))
    base_f = base.detach().cpu().float()
    deltas = []
    for state_dict in state_dicts:
        delta = state_dict[key].detach().cpu().float() - base_f
        deltas.append(_mask_delta(delta, density))
    stacked = torch.stack(deltas, dim=0)
    coeff = weights.to(dtype=stacked.dtype).view(-1, *([1] * (stacked.ndim - 1)))
    sign_vote = (stacked * coeff).sum(dim=0)
    signs = torch.sign(sign_vote)
    preserve = (((signs > 0).unsqueeze(0) & (stacked > 0)) |
                ((signs < 0).unsqueeze(0) & (stacked < 0)))
    numerator = (stacked * preserve.to(stacked.dtype) * coeff).sum(dim=0)
    denominator = (preserve.to(stacked.dtype) * coeff).sum(dim=0)
    merged_delta = torch.zeros_like(base_f)
    valid = denominator > 0
    merged_delta[valid] = numerator[valid] / denominator[valid]
    return (base_f + alpha * merged_delta).to(dtype=first.dtype)


def _merge_classifier_task_tensor(key, state_dicts, base_state, class_weights, class_prior, cfg):
    first = state_dicts[0][key]
    base = base_state.get(key)
    if base is None or tuple(base.shape) != tuple(first.shape):
        return _merge_classifier_tensor(key, state_dicts, class_weights, class_prior, cfg)

    base_f = base.detach().cpu().float()
    out = base_f.clone()
    for cls_idx in range(int(class_weights.shape[1])):
        row_delta = torch.zeros_like(out[cls_idx])
        base_row = base_f[cls_idx]
        for value, weight in zip([sd[key] for sd in state_dicts], class_weights[:, cls_idx].tolist()):
            row_delta.add_(value[cls_idx].detach().cpu().float() - base_row, alpha=float(weight))
        out[cls_idx] = base_row + row_delta

    if first.ndim == 1:
        tau = float(cfg.get("my_merge_prior_bias_tau", 0.0))
        if tau != 0:
            centered_log_prior = torch.log(class_prior.clamp_min(EPS))
            centered_log_prior = centered_log_prior - centered_log_prior.mean()
            out = out - tau * centered_log_prior.to(out.device)
    return out.to(dtype=first.dtype)


def _merge_running_var(key, state_dicts, weights, merged):
    mean_key = key[:-len("running_var")] + "running_mean"
    values = [sd[key] for sd in state_dicts]
    if mean_key not in merged or any(mean_key not in sd for sd in state_dicts):
        return _weighted_tensor(values, weights).to(dtype=values[0].dtype)
    merged_mean = merged[mean_key].detach().cpu().float()
    out = torch.zeros_like(values[0].detach().cpu().float())
    for state_dict, weight in zip(state_dicts, weights.tolist()):
        local_mean = state_dict[mean_key].detach().cpu().float()
        local_var = state_dict[key].detach().cpu().float()
        out.add_(local_var + (local_mean - merged_mean).square(), alpha=float(weight))
    return out.clamp_min(0.0).to(dtype=values[0].dtype)


def _prototype_stats_path(meta, cfg):
    direct = (
        cfg.get("my_merge_prototype_stats_path")
        or cfg.get("my_merge_client_prototype_path")
        or cfg.get("my_merge_proto_stats_path")
    )
    if direct:
        path = Path(direct)
        return path if path.exists() else None
    root = cfg.get("my_merge_prototype_root") or cfg.get("my_merge_client_prototype_root")
    if not root:
        return None
    model_name = meta.get("model") or meta.get("clip_model", "").replace("/", "__")
    path = (
        Path(root)
        / str(meta.get("task_type"))
        / str(meta.get("dataset"))
        / str(model_name)
        / f"clients_{int(meta.get('num_clients'))}"
        / beta_to_dirname(meta.get("beta"))
        / f"seed_{int(meta.get('seed'))}"
        / "prototype_stats.pt"
    )
    return path if path.exists() else None


def _load_prototype_stats(meta, cfg):
    path = _prototype_stats_path(meta, cfg)
    if path is None:
        return None, ""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid my_merge prototype stats payload: {path}")
    return payload, str(path)


def _stack_client_stat(proto_stats, field, num_clients, num_classes):
    clients = proto_stats.get("clients", [])
    rows = []
    for idx in range(num_clients):
        item = clients[idx] if idx < len(clients) else {}
        value = item.get(field)
        if value is None:
            rows.append(torch.zeros(num_classes, dtype=torch.float32))
        else:
            rows.append(torch.as_tensor(value, dtype=torch.float32).view(num_classes))
    return torch.stack(rows, dim=0)


def _uploaded_prevalence_counts(proto_stats, stats, num_clients, num_classes):
    if not isinstance(proto_stats, dict):
        return stats["class_counts"].sum(dim=0).detach().cpu().float().clamp_min(0.0)
    counts = _stack_client_stat(proto_stats, "class_prevalence_counts", num_clients, num_classes)
    if float(counts.sum().item()) <= 0:
        counts = _stack_client_stat(proto_stats, "label_counts", num_clients, num_classes)
    if float(counts.sum().item()) <= 0:
        counts = stats["class_counts"]
    return counts.sum(dim=0).detach().cpu().float().clamp_min(0.0)


def _global_prototypes(proto_stats, meta, stats, cfg):
    num_clients = len(meta.get("clients", []))
    num_classes = int(meta["num_classes"])
    clients = proto_stats.get("clients", [])
    if not clients:
        return None
    means = []
    for idx in range(num_clients):
        if idx >= len(clients) or clients[idx].get("class_feature_mean") is None:
            return None
        means.append(torch.as_tensor(clients[idx]["class_feature_mean"], dtype=torch.float32))
    means = torch.stack(means, dim=0)
    if means.ndim != 3 or means.shape[1] != num_classes:
        return None

    counts = _stack_client_stat(proto_stats, "class_counts", num_clients, num_classes)
    if float(counts.sum().item()) <= 0:
        counts = stats["class_counts"]
    rho = float(cfg.get("my_merge_proto_count_power", 0.45))
    evidence = (counts + 1.0).pow(rho)
    evidence = evidence * (counts > 0).to(evidence.dtype)
    weights = evidence / evidence.sum(dim=0, keepdim=True).clamp_min(EPS)
    prototypes = (means * weights.unsqueeze(-1)).sum(dim=0)
    class_counts = _uploaded_prevalence_counts(proto_stats, stats, num_clients, num_classes)
    valid = counts.sum(dim=0) > 0
    return {
        "prototypes": prototypes,
        "class_counts": class_counts,
        "prototype_class_counts": counts.sum(dim=0).clamp_min(0.0),
        "valid": valid,
        "prototype_weights": weights,
    }


def _classifier_pair(state, num_classes):
    weight_key = None
    for key, tensor in state.items():
        if _is_classifier_tensor(key, tensor, num_classes) and tensor.ndim == 2:
            weight_key = key
            break
    if weight_key is None:
        return None, None
    prefix = weight_key.rsplit(".", 1)[0]
    bias_key = f"{prefix}.bias"
    if bias_key not in state or state[bias_key].ndim != 1 or int(state[bias_key].shape[0]) != int(num_classes):
        for key, tensor in state.items():
            if _is_classifier_tensor(key, tensor, num_classes) and tensor.ndim == 1:
                bias_key = key
                break
    return weight_key, bias_key if bias_key in state else None


def _is_reference_prototype_stats(client_stats):
    return isinstance(client_stats, dict) and client_stats.get("feature_space") == "reference_model"


def _reference_prior_tau(class_prior, num_classes, cfg):
    imbalance_ratio = float((class_prior.max() * float(num_classes)).item())
    threshold = float(cfg.get("my_merge_reference_prior_threshold", 2.5))
    max_tau = float(cfg.get("my_merge_reference_prior_max_tau", 6.0))
    saturation = float(cfg.get("my_merge_reference_prior_saturation", 3.0))
    if imbalance_ratio <= threshold or max_tau <= 0:
        return 0.0, imbalance_ratio

    denom = torch.log(torch.tensor(max(saturation / threshold, 1.0001))).item()
    ratio = torch.log(torch.tensor(max(imbalance_ratio / threshold, 1.0))).item()
    prior_tau = max_tau * max(0.0, min(1.0, ratio / max(denom, EPS)))
    return prior_tau, imbalance_ratio


def _prevalence_calibration_strength(class_prior, num_classes):
    dominant_prior = float(class_prior.max().item())
    if dominant_prior <= 0.5:
        return 0.0
    return float(num_classes) * dominant_prior


def _apply_prevalence_bias(state, meta, class_counts, cfg, *, enabled=True):
    num_classes = int(meta["num_classes"])
    weight_key, bias_key = _classifier_pair(state, num_classes)
    class_counts = class_counts.detach().cpu().float().clamp_min(0.0)
    class_prior = class_counts / class_counts.sum().clamp_min(EPS)
    prior_tau, imbalance_ratio = _reference_prior_tau(class_prior, num_classes, cfg)
    if not enabled:
        prior_tau = 0.0

    trace = {
        "used": bool(prior_tau != 0.0 and bias_key is not None),
        "weight_key": weight_key,
        "bias_key": bias_key,
        "prior_tau": float(prior_tau),
        "imbalance_ratio": float(imbalance_ratio),
        "class_counts": [float(x) for x in class_counts.tolist()],
        "class_prior": [float(x) for x in class_prior.tolist()],
    }
    if weight_key is None:
        trace.update({"used": False, "reason": "classifier_weight_not_found"})
        return trace
    if bias_key is None:
        trace.update({"used": False, "reason": "classifier_bias_not_found"})
        return trace
    if prior_tau == 0.0:
        trace.update({"used": False, "reason": "balanced_or_disabled"})
        return trace

    centered_log_prior = torch.log(class_prior.clamp_min(EPS))
    centered_log_prior = centered_log_prior - centered_log_prior.mean()
    bias = state[bias_key].detach().cpu().float()
    state[bias_key] = (bias + prior_tau * centered_log_prior).to(dtype=state[bias_key].dtype)
    return trace


def _merge_reference_prototype_model(base_state, proto_stats, meta, stats, cfg):
    proto = _global_prototypes(proto_stats, meta, stats, cfg)
    if proto is None:
        return None, {"used": False, "reason": "missing_or_invalid_reference_prototypes"}
    num_classes = int(meta["num_classes"])
    state = OrderedDict((key, value.detach().cpu().clone()) for key, value in base_state.items())
    weight_key, bias_key = _classifier_pair(state, num_classes)
    if weight_key is None:
        return None, {"used": False, "reason": "classifier_weight_not_found"}

    weight0 = state[weight_key].detach().cpu().float()
    prototypes = proto["prototypes"].detach().cpu().float()
    valid = proto["valid"].detach().cpu().bool()
    if prototypes.ndim != 2 or prototypes.shape[0] != num_classes or prototypes.shape[1] != weight0.shape[1]:
        return None, {
            "used": False,
            "reason": "prototype_classifier_shape_mismatch",
            "prototype_shape": list(prototypes.shape),
            "classifier_shape": list(weight0.shape),
        }

    mode = str(cfg.get("my_merge_reference_head_mode", "cosine")).lower()
    scale = float(cfg.get("my_merge_reference_head_scale", 20.0))
    class_prior = proto["class_counts"] / proto["class_counts"].sum().clamp_min(EPS)
    ablation_mode = str(cfg.get("my_merge_ablation_mode", "full")).lower()
    prior_enabled = ablation_mode in {"full", "m1_m2"}
    prevalence_strength = _prevalence_calibration_strength(class_prior, num_classes) if prior_enabled else 0.0
    imbalance_ratio = float((class_prior.max() * float(num_classes)).item())

    if mode == "euclidean":
        head_weight = 2.0 * prototypes
        head_bias = -prototypes.square().sum(dim=1)
    else:
        head_weight = scale * F.normalize(prototypes, dim=1)
        head_bias = torch.zeros(num_classes, dtype=torch.float32)

    if prevalence_strength != 0.0:
        centered_log_prior = torch.log(class_prior.clamp_min(EPS))
        centered_log_prior = centered_log_prior - centered_log_prior.mean()
        head_bias = head_bias + prevalence_strength * centered_log_prior

    missing = ~valid
    if missing.any():
        head_weight[missing] = weight0[missing]
        if bias_key is not None:
            head_bias[missing] = state[bias_key].detach().cpu().float()[missing]

    state[weight_key] = head_weight.to(dtype=state[weight_key].dtype)
    if bias_key is not None:
        state[bias_key] = head_bias.to(dtype=state[bias_key].dtype)

    return state, {
        "used": True,
        "feature_space": "reference_model",
        "weight_key": weight_key,
        "bias_key": bias_key,
        "valid_classes": int(valid.sum().item()),
        "head_mode": mode,
        "head_scale": scale,
        "prevalence_bias_scale": prevalence_strength,
        "prior_enabled": prior_enabled,
        "prior_disabled_reason": (
            "" if prevalence_strength != 0.0
            else "no_absolute_dominant_diagnosis_or_m1_only_ablation"
        ),
        "imbalance_ratio": imbalance_ratio,
        "class_counts": [float(x) for x in proto["class_counts"].tolist()],
        "prototype_class_counts": [float(x) for x in proto["prototype_class_counts"].tolist()],
        "prototype_weights": [[float(v) for v in row] for row in proto["prototype_weights"].tolist()],
    }


def _merge_avg_with_prevalence_bias(state_dicts, weights, meta, stats, cfg):
    merged, normalized_weights = average_state_dicts(state_dicts, weights)
    merged = OrderedDict((key, value.detach().cpu().clone()) for key, value in merged.items())
    prevalence_trace = _apply_prevalence_bias(merged, meta, stats["class_counts"].sum(dim=0), cfg, enabled=True)
    return merged, {
        "avg_weights": [float(x) for x in normalized_weights],
        "prevalence_bias": prevalence_trace,
        "class_prior": [float(x) for x in stats["class_prior"].tolist()],
        "class_counts": [[float(v) for v in row] for row in stats["class_counts"].tolist()],
    }


def _apply_anti_collapse_head_correction(state, meta, stats, cfg):
    num_classes = int(meta["num_classes"])
    weight_key, bias_key = _classifier_pair(state, num_classes)
    if weight_key is None:
        return {"used": False, "reason": "classifier_weight_not_found"}

    class_prior = stats["class_prior"].detach().cpu().float().clamp_min(EPS)
    class_prior = class_prior / class_prior.sum().clamp_min(EPS)
    centered_log_prior = torch.log(class_prior) - torch.log(class_prior).mean()

    prior_tau = float(cfg.get("my_merge_prior_bias_tau", 0.85))
    confidence_tau = float(cfg.get("my_merge_confidence_bias_tau", 0.60))
    norm_gamma = float(cfg.get("my_merge_head_norm_gamma", 0.45))
    norm_min = float(cfg.get("my_merge_head_norm_min", 0.70))
    norm_max = float(cfg.get("my_merge_head_norm_max", 1.45))

    weight = state[weight_key].detach().cpu().float()
    if weight.ndim != 2:
        return {"used": False, "reason": "classifier_weight_is_not_matrix", "weight_key": weight_key}

    with torch.no_grad():
        norms = weight.norm(dim=1).clamp_min(EPS)
        target_norm = norms.median().clamp_min(EPS)
        equalizer = (target_norm / norms).pow(norm_gamma)
        prior_equalizer = (class_prior / class_prior.log().mean().exp().clamp_min(EPS)).pow(-0.20)
        scale = (equalizer * prior_equalizer).clamp(norm_min, norm_max)
        corrected_weight = weight * scale.view(-1, 1)
        state[weight_key] = corrected_weight.to(dtype=state[weight_key].dtype)

        bias_used = False
        if bias_key is not None:
            bias = state[bias_key].detach().cpu().float()
            corrected_bias = bias - prior_tau * centered_log_prior
            class_counts = stats["class_counts"]
            class_reliability = stats["class_reliability"]
            confidence = (class_counts * class_reliability).sum(dim=0) / class_counts.sum(dim=0).clamp_min(EPS)
            centered_log_conf = torch.log(confidence.clamp_min(0.05)) - torch.log(confidence.clamp_min(0.05)).mean()
            corrected_bias = corrected_bias + confidence_tau * centered_log_conf
            state[bias_key] = corrected_bias.to(dtype=state[bias_key].dtype)
            bias_used = True

    return {
        "used": True,
        "weight_key": weight_key,
        "bias_key": bias_key,
        "bias_corrected": bias_used,
        "prior_bias_tau": prior_tau,
        "confidence_bias_tau": confidence_tau,
        "head_norm_gamma": norm_gamma,
        "head_norm_clip": [norm_min, norm_max],
        "class_prior": [float(x) for x in class_prior.tolist()],
        "row_scale": [float(x) for x in scale.tolist()],
    }


def _merge_state_dicts(state_dicts, base_state, stats, meta, cfg):
    num_classes = int(meta["num_classes"])
    client_weights = stats["client_weights"]
    class_weights = stats["class_weights"]
    sample_weights = stats["sample_weights"]
    class_prior = stats["class_prior"]
    state = OrderedDict()
    classifier_keys = []
    bn_mean_keys = []
    bn_var_keys = []

    for key, first in state_dicts[0].items():
        if not torch.is_floating_point(first):
            if key.endswith("num_batches_tracked"):
                state[key] = torch.stack([sd[key].detach().cpu() for sd in state_dicts]).max(dim=0).values
            else:
                state[key] = first.detach().cpu().clone()
            continue
        if _is_classifier_tensor(key, first, num_classes):
            state[key] = _merge_classifier_task_tensor(key, state_dicts, base_state, class_weights, class_prior, cfg)
            classifier_keys.append(key)
        elif key.endswith("running_mean"):
            values = [sd[key] for sd in state_dicts]
            state[key] = _weighted_tensor(values, client_weights).to(dtype=first.dtype)
            bn_mean_keys.append(key)
        elif key.endswith("running_var"):
            bn_var_keys.append(key)
        else:
            state[key] = _merge_task_tensor(key, state_dicts, base_state, client_weights, cfg)

    for key in bn_var_keys:
        state[key] = _merge_running_var(key, state_dicts, client_weights, state)

    return state, {
        "classifier_keys": classifier_keys,
        "bn_running_mean_keys": bn_mean_keys,
        "bn_running_var_keys": bn_var_keys,
        "client_weights": [float(x) for x in client_weights.tolist()],
        "sample_weights": [float(x) for x in sample_weights.tolist()],
        "class_prior": [float(x) for x in class_prior.tolist()],
        "class_weights": [[float(v) for v in row] for row in class_weights.tolist()],
        "client_reliability": [float(x) for x in stats["reliability"].tolist()],
    }


def _client_rows(meta, stats):
    rows = []
    class_counts = stats["class_counts"]
    class_weights = stats["class_weights"]
    for idx, client in enumerate(meta.get("clients", [])):
        rows.append({
            "client_index": int(idx),
            "client_name": client.get("checkpoint", f"client_{idx}.pt"),
            "seen_classes": [int(x) for x in client.get("classes", [])],
            "num_samples": int(client.get("num_samples", 0)),
            "best_val_acc": float(client.get("best_val_acc", 0.0)),
            "balanced_merge_weight": float(stats["client_weights"][idx].item()),
            "sample_weight": float(stats["sample_weights"][idx].item()),
            "class_counts": [float(x) for x in class_counts[idx].tolist()],
            "class_merge_weights": [float(x) for x in class_weights[idx].tolist()],
        })
    return rows


def merge_my_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    if meta is None or cfg is None:
        raise ValueError("my_merge requires task metadata and runtime config.")
    if not _is_medical_image_task(meta):
        raise ValueError(f"my_merge is defined only for medical image tasks, got dataset={meta.get('dataset')}.")
    if not state_dicts:
        raise ValueError("my_merge requires at least one state_dict.")

    cfg = cfg or {}
    client_stats, client_stats_path = _load_prototype_stats(meta, cfg)
    stats = _class_client_weights(meta, cfg, client_stats=client_stats)
    ablation_mode = str(cfg.get("my_merge_ablation_mode", "full")).lower()
    if ablation_mode not in {"full", "m1_only", "m1_m2", "avg_m2"}:
        raise ValueError(f"Unsupported my_merge_ablation_mode: {ablation_mode}")

    if ablation_mode == "avg_m2":
        merged, trace = _merge_avg_with_prevalence_bias(state_dicts, weights, meta, stats, cfg)
        fusion_rule = "avg_plus_prevalence_bias_ablation"
        trace["reference_prototype_head"] = {"used": False, "reason": "avg_m2_ablation"}
        trace["anti_collapse_head"] = {"used": False, "reason": "not_used_by_two_module_ablation"}
        trace["client_stats_path"] = client_stats_path
        return merged, {
            "implementation": "diagnostic_prototype_merge_v1",
            "ablation_mode": ablation_mode,
            "medical_only": True,
            "fusion_rule": fusion_rule,
            "privacy": {
                "server_reads_private_images": False,
                "uses_public_probe": False,
                "uses_candidate_selection": False,
                "client_uploads": [
                    "checkpoint",
                    "effective_class_support_counts",
                ],
            },
            "modules": {
                "M1": "disabled for ablation",
                "M2": "server calibrates the averaged classifier with the uploaded class prevalence",
            },
            "modality": meta.get("dataset"),
            "num_clients": int(len(state_dicts)),
            "base_input_weights": [float(x) for x in weights],
            "client_medical_evidence": _client_rows(meta, stats),
            **trace,
        }

    base_state, _ = build_reference_bundle(meta, device="cpu")
    reference_trace = None
    if _is_reference_prototype_stats(client_stats):
        reference_merged, reference_trace = _merge_reference_prototype_model(base_state, client_stats, meta, stats, cfg)
        if reference_merged is not None:
            merged = reference_merged
            trace = {
                "reference_prototype_head": reference_trace,
                "client_weights": [float(x) for x in stats["client_weights"].tolist()],
                "sample_weights": [float(x) for x in stats["sample_weights"].tolist()],
                "class_prior": [float(x) for x in stats["class_prior"].tolist()],
                "class_weights": [[float(v) for v in row] for row in stats["class_weights"].tolist()],
                "client_reliability": [float(x) for x in stats["reliability"].tolist()],
            }
            fusion_rule = "single_checkpoint_reference_prototype_medical_merge"
        else:
            merged, trace = _merge_state_dicts(state_dicts, base_state, stats, meta, cfg)
            trace["reference_prototype_head"] = reference_trace
            fusion_rule = "single_checkpoint_source_confidence_signed_task_merge"
    else:
        merged, trace = _merge_state_dicts(state_dicts, base_state, stats, meta, cfg)
        fusion_rule = "single_checkpoint_source_confidence_signed_task_merge"
    if fusion_rule == "single_checkpoint_reference_prototype_medical_merge":
        trace["anti_collapse_head"] = {"used": False, "reason": "reference_prototype_head_already_balanced"}
    else:
        trace["anti_collapse_head"] = _apply_anti_collapse_head_correction(merged, meta, stats, cfg)
    trace["client_stats_path"] = client_stats_path
    return merged, {
        "implementation": "diagnostic_prototype_merge_v1",
        "ablation_mode": ablation_mode,
        "medical_only": True,
        "fusion_rule": fusion_rule,
        "privacy": {
            "server_reads_private_images": False,
            "uses_public_probe": False,
            "uses_candidate_selection": False,
            "client_uploads": [
                "checkpoint",
                "effective_class_support_counts",
                "reference_backbone_class_prototypes",
            ],
        },
        "modules": {
            "M1_diagnostic_prototype_upload": (
                "clients summarize each diagnostic class with support counts and "
                "reference-backbone feature means"
            ),
            "M2_collapse_free_head_synthesis": (
                "server aggregates class-wise prototypes and synthesizes a cosine "
                "prototype classifier without prevalence-bias correction"
            ),
            "discarded_prevalence_prior": "not part of the formal two-module method",
        },
        "task_density": float(cfg.get("my_merge_task_density", 0.55)),
        "task_alpha": float(cfg.get("my_merge_task_alpha", 1.0)),
        "modality": meta.get("dataset"),
        "num_clients": int(len(state_dicts)),
        "base_input_weights": [float(x) for x in weights],
        "client_medical_evidence": _client_rows(meta, stats),
        **trace,
    }
