from collections import OrderedDict
import json
from pathlib import Path

import torch

from utils.state_dict import average_state_dicts


EPS = 1e-8

BN_MOMENT_NAMES = [
    "bn_running_mean",
    "bn_running_var",
    "bn_num_batches_tracked",
]

# Kept for older scripts that import FEATURE_NAMES from this module.
FEATURE_NAMES = BN_MOMENT_NAMES

def _is_image_checkpoint_task(meta):
    return (meta or {}).get("task_type") in {"small", "vlm"}


def _domain_control_enabled(cfg):
    value = (cfg or {}).get("my_merge_domain_control", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _bool_cfg(cfg, name, default=False):
    value = (cfg or {}).get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _tokens(value):
    if value in (None, "", False):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def _ablation_labels(cfg):
    labels = _tokens((cfg or {}).get("my_merge_ablation", "full"))
    labels.extend(_tokens((cfg or {}).get("my_merge_disable", "")))
    return labels or ["full"]


def _has_label(cfg, name):
    target = str(name).lower().replace("-", "_")
    return target in {raw.lower().replace("-", "_") for raw in _ablation_labels(cfg)}


def _disable_bml(cfg):
    return any(
        _has_label(cfg, name)
        for name in (
            "no_bml",
            "no_cel",
            "no_mel",
            "no_m1",
            "no_client_information",
            "no_medical_evidence_ledger",
            "no_clinical_evidence_ledger",
        )
    )


def _disable_bcm(cfg):
    return any(
        _has_label(cfg, name)
        for name in (
            "no_bcm",
            "no_pcm",
            "no_mccm",
            "no_m2",
            "no_fusion_selection",
            "no_morphology_calibrated_conflict_merging",
            "no_protocol_calibrated_merge",
        )
    )


def _normalize(values, fallback=None):
    values = torch.as_tensor(values, dtype=torch.float32)
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = values.sum()
    if float(total.item()) > 0:
        return values / total
    if fallback is not None:
        fallback = torch.as_tensor(fallback, dtype=torch.float32)
        return fallback / fallback.sum().clamp_min(EPS)
    return torch.ones_like(values) / float(max(1, values.numel()))


def _smooth01(value):
    x = max(0.0, min(1.0, float(value)))
    return x * x * (3.0 - 2.0 * x)


def _weighted_average(values, weights):
    weights = torch.as_tensor(weights, dtype=torch.float32)
    out = values[0].detach().float() * float(weights[0])
    for value, weight in zip(values[1:], weights[1:]):
        out.add_(value.detach().float(), alpha=float(weight))
    return out


def _bn_var_key(mean_key):
    return mean_key[: -len("running_mean")] + "running_var"


def _bn_num_batches_key(mean_key):
    return mean_key[: -len("running_mean")] + "num_batches_tracked"


def _is_bn_running_key(key):
    return key.endswith("running_mean") or key.endswith("running_var")


def _common_bn_pairs(state_dicts):
    if not state_dicts:
        return []
    common = set(state_dicts[0].keys())
    for state in state_dicts[1:]:
        common &= set(state.keys())

    pairs = []
    for key in state_dicts[0].keys():
        if not key.endswith("running_mean"):
            continue
        var_key = _bn_var_key(key)
        if key not in common or var_key not in common:
            continue
        mean_shape = tuple(state_dicts[0][key].shape)
        var_shape = tuple(state_dicts[0][var_key].shape)
        if mean_shape != var_shape:
            continue
        ok = True
        for state in state_dicts[1:]:
            ok = ok and tuple(state[key].shape) == mean_shape and tuple(state[var_key].shape) == var_shape
        if ok:
            pairs.append((key, var_key, _bn_num_batches_key(key)))
    return pairs


def _client_sample_prior(meta, base_weights):
    base = _normalize(base_weights)
    return _sample_evidence_profile(meta, base)["sample_prior"]


def _sample_evidence_profile(meta, base_weights):
    base = _normalize(base_weights)
    samples = []
    clients = (meta or {}).get("clients", [])
    for idx in range(base.numel()):
        client = clients[idx] if idx < len(clients) else {}
        samples.append(float(client.get("num_samples", 1.0)))
    sample_counts = torch.tensor(samples, dtype=torch.float32).clamp_min(1.0)
    sample = _normalize(sample_counts, base)
    if sample.numel() <= 1:
        return {
            "sample_prior": sample,
            "sample_mass": sample,
            "mild_sample_prior": sample,
            "max_case_share": float(sample.max().item()) if sample.numel() else 0.0,
            "evidence_concentration_gate": 0.0,
        }
    mild_sample = _normalize(torch.sqrt(sample_counts), base)
    mild_prior = _normalize(0.50 * base + 0.50 * mild_sample, base)
    max_share = float(sample.max().item())
    evidence_gate = _smooth01((max_share - 0.45) / 0.25)
    return {
        "sample_prior": _normalize((1.0 - evidence_gate) * mild_prior + evidence_gate * sample, base),
        "sample_mass": sample,
        "mild_sample_prior": mild_prior,
        "max_case_share": max_share,
        "evidence_concentration_gate": float(evidence_gate),
    }


def _bn_profile_distances(state_dicts, pairs, centroid_weights):
    num_clients = len(state_dicts)
    distances = torch.zeros(num_clients, dtype=torch.float32)
    mean_terms = torch.zeros(num_clients, dtype=torch.float32)
    var_terms = torch.zeros(num_clients, dtype=torch.float32)
    used = 0

    for mean_key, var_key, _num_key in pairs:
        means = torch.stack([state[mean_key].detach().float() for state in state_dicts], dim=0)
        variances = torch.stack([state[var_key].detach().float().clamp_min(EPS) for state in state_dicts], dim=0)
        weight_shape = (len(state_dicts),) + (1,) * (means.ndim - 1)
        w = torch.as_tensor(centroid_weights, dtype=torch.float32).view(weight_shape)
        mean_bar = (w * means).sum(dim=0)
        log_var = torch.log(variances)
        log_var_bar = (w * log_var).sum(dim=0)
        var_bar = torch.exp(log_var_bar).clamp_min(EPS)

        layer_mean = ((means - mean_bar.view((1,) + tuple(mean_bar.shape))).square() / var_bar.view((1,) + tuple(var_bar.shape))).mean(
            dim=tuple(range(1, means.ndim))
        )
        layer_var = (log_var - log_var_bar.view((1,) + tuple(log_var_bar.shape))).square().mean(
            dim=tuple(range(1, log_var.ndim))
        )
        mean_terms += layer_mean.cpu()
        var_terms += layer_var.cpu()
        distances += (layer_mean + 0.50 * layer_var).cpu()
        used += 1

    if used > 0:
        distances /= float(used)
        mean_terms /= float(used)
        var_terms /= float(used)
    return distances, mean_terms, var_terms


def _build_bn_moment_ledger(state_dicts, base_weights, meta, cfg):
    base = _normalize(base_weights)
    evidence_profile = _sample_evidence_profile(meta, base)
    sample_prior = evidence_profile["sample_prior"]
    pairs = _common_bn_pairs(state_dicts)
    num_clients = len(state_dicts)

    if not pairs:
        if _bool_cfg(cfg, "my_merge_require_bn", False):
            raise ValueError(
                "my_merge BN-only mode requires client checkpoints with BatchNorm running_mean/running_var buffers."
            )
        return {
            "has_bn": False,
            "bn_pairs": [],
            "base_weights": base,
            "sample_prior": sample_prior,
            "bn_weights": base,
            "bn_consistency": torch.ones(num_clients, dtype=torch.float32),
            "bn_distance": torch.zeros(num_clients, dtype=torch.float32),
            "bn_mean_shift": torch.zeros(num_clients, dtype=torch.float32),
            "bn_var_shift": torch.zeros(num_clients, dtype=torch.float32),
            "heterogeneity": 0.0,
            "merge_gate": 0.0,
            "temperature": 1.0,
            "evidence_profile": evidence_profile,
            "trace": {
                "enabled": False,
                "reason": "no_batchnorm_running_statistics",
                "max_case_share": float(evidence_profile["max_case_share"]),
                "evidence_concentration_gate": float(evidence_profile["evidence_concentration_gate"]),
                "privacy": "no BN moments were present; no raw samples were requested",
            },
        }

    if _disable_bml(cfg):
        return {
            "has_bn": True,
            "bn_pairs": pairs,
            "base_weights": base,
            "sample_prior": sample_prior,
            "bn_weights": base,
            "bn_consistency": torch.ones(num_clients, dtype=torch.float32),
            "bn_distance": torch.zeros(num_clients, dtype=torch.float32),
            "bn_mean_shift": torch.zeros(num_clients, dtype=torch.float32),
            "bn_var_shift": torch.zeros(num_clients, dtype=torch.float32),
            "heterogeneity": 0.0,
            "merge_gate": 0.0,
            "temperature": 1.0,
            "client_rows": [],
            "evidence_profile": evidence_profile,
            "trace": {
                "enabled": False,
                "reason": "disabled_by_ablation",
                "rule": "clinical_evidence_ledger",
                "bn_layer_count": len(pairs),
                "base_weights": [float(x) for x in base.tolist()],
                "bn_weights": [float(x) for x in base.tolist()],
                "max_case_share": float(evidence_profile["max_case_share"]),
                "evidence_concentration_gate": float(evidence_profile["evidence_concentration_gate"]),
                "privacy": "BN moments were available but CEL was disabled for ablation",
            },
        }

    distances, mean_shift, var_shift = _bn_profile_distances(state_dicts, pairs, sample_prior)
    positive = distances[distances > 0]
    if positive.numel() > 0:
        auto_temp = float(torch.median(positive).item()) * 1.50 + 0.05
    else:
        auto_temp = 1.0
    temperature = float((cfg or {}).get("my_merge_bn_temperature", auto_temp))
    temperature = max(temperature, 0.05)

    consistency = torch.exp(-distances / temperature).clamp(0.05, 1.0)
    bn_weights = _normalize(sample_prior * (0.35 + 0.65 * consistency), sample_prior)
    heterogeneity = float((sample_prior * distances).sum().item() / temperature)
    merge_gate = _smooth01(
        (heterogeneity - float((cfg or {}).get("my_merge_bn_gate_floor", 0.15)))
        / float((cfg or {}).get("my_merge_bn_gate_width", 0.85))
    )

    client_rows = []
    clients = (meta or {}).get("clients", [])
    for idx in range(num_clients):
        client = clients[idx] if idx < len(clients) else {}
        client_rows.append(
            {
                "client_index": idx,
                "client_name": client.get("checkpoint", f"client_{idx}.pt"),
                "num_samples": int(client.get("num_samples", 0)),
                "seen_classes": [int(cls) for cls in client.get("classes", [])],
                "base_weight": float(base[idx].item()),
                "sample_prior": float(sample_prior[idx].item()),
                "bn_weight": float(bn_weights[idx].item()),
                "bn_consistency": float(consistency[idx].item()),
                "bn_distance": float(distances[idx].item()),
                "bn_mean_shift": float(mean_shift[idx].item()),
                "bn_var_shift": float(var_shift[idx].item()),
            }
        )

    return {
        "has_bn": True,
        "bn_pairs": pairs,
        "base_weights": base,
        "sample_prior": sample_prior,
        "bn_weights": bn_weights,
        "bn_consistency": consistency,
        "bn_distance": distances,
        "bn_mean_shift": mean_shift,
        "bn_var_shift": var_shift,
        "heterogeneity": heterogeneity,
        "merge_gate": float(merge_gate),
        "temperature": float(temperature),
        "client_rows": client_rows,
        "evidence_profile": evidence_profile,
        "trace": {
            "enabled": True,
            "rule": "clinical_evidence_ledger",
            "bn_layer_count": len(pairs),
            "bn_layers": [mean_key[: -len(".running_mean")] if mean_key.endswith(".running_mean") else mean_key for mean_key, _var, _num in pairs],
            "temperature": float(temperature),
            "heterogeneity": float(heterogeneity),
            "merge_gate": float(merge_gate),
            "max_case_share": float(evidence_profile["max_case_share"]),
            "evidence_concentration_gate": float(evidence_profile["evidence_concentration_gate"]),
            "sample_mass": [float(x) for x in evidence_profile["sample_mass"].tolist()],
            "mild_sample_prior": [float(x) for x in evidence_profile["mild_sample_prior"].tolist()],
            "base_weights": [float(x) for x in base.tolist()],
            "sample_prior": [float(x) for x in sample_prior.tolist()],
            "bn_weights": [float(x) for x in bn_weights.tolist()],
            "bn_consistency": [float(x) for x in consistency.tolist()],
            "bn_distance": [float(x) for x in distances.tolist()],
            "client_rows": client_rows,
            "privacy": "clients upload checkpoints with aggregate BN running moments only; no raw images, per-sample logits, activations, or candidate feedback are used",
        },
    }


def _merge_running_stat(values, weights, key, dtype):
    weights = torch.as_tensor(weights, dtype=torch.float32)
    if key.endswith("running_var"):
        out = torch.zeros_like(values[0].detach().float())
        for value, weight in zip(values, weights):
            out.add_(torch.log(value.detach().float().clamp_min(EPS)), alpha=float(weight))
        return torch.exp(out).to(dtype)
    return _weighted_average(values, weights).to(dtype)


def _is_classifier_tensor(key, tensor, num_classes):
    lowered = key.lower()
    return (
        torch.is_floating_point(tensor)
        and tensor.ndim in (1, 2)
        and tensor.shape[0] == int(num_classes)
        and any(token in lowered for token in ("fc", "classifier", "head", "logit", "proj"))
    )


def _client_class_sets(meta, num_classes):
    sets = []
    for client in (meta or {}).get("clients", []):
        seen = sorted({int(x) for x in client.get("classes", []) if 0 <= int(x) < int(num_classes)})
        sets.append(seen)
    return sets


def _class_support_profile(meta, num_classes):
    class_sets = _client_class_sets(meta, num_classes)
    if not class_sets or int(num_classes) <= 0:
        return {
            "support_counts": [],
            "mean_support": 0.0,
            "singleton_fraction": 0.0,
            "non_singleton_fraction": 0.0,
            "client_class_fractions": [],
            "mean_client_class_fraction": 0.0,
            "max_client_class_fraction": 0.0,
            "partition_gate": 0.0,
            "broad_client_gate": 0.0,
            "clear_partition_gate": 0.0,
            "false_specialty_gate": 0.0,
            "specialization_gate": 0.0,
        }
    counts = []
    for cls_idx in range(int(num_classes)):
        counts.append(sum(1 for seen in class_sets if cls_idx in seen))
    singleton_fraction = sum(1 for value in counts if value <= 1) / float(max(1, len(counts)))
    client_class_fractions = [
        len(seen) / float(max(1, int(num_classes)))
        for seen in class_sets
    ]
    max_client_class_fraction = max(client_class_fractions) if client_class_fractions else 0.0
    mean_client_class_fraction = (
        sum(client_class_fractions) / float(max(1, len(client_class_fractions)))
        if client_class_fractions
        else 0.0
    )

    # Clinical specialists should form a near-partition of labels.  A clear
    # singleton partition is strong evidence, but a moderate singleton rate is
    # still useful when no client broadly covers the label space.
    partition_gate = _smooth01((singleton_fraction - 0.45) / 0.20)
    clear_partition_gate = _smooth01((singleton_fraction - 0.78) / 0.10)
    broad_client_gate = _smooth01((max_client_class_fraction - 0.40) / 0.20)
    compact_specialty_gate = _smooth01((singleton_fraction - 0.35) / 0.25) * (1.0 - broad_client_gate)
    false_specialty_gate = broad_client_gate * (1.0 - clear_partition_gate)
    specialization_gate = max(
        0.0,
        min(1.0, max(partition_gate * (1.0 - false_specialty_gate), compact_specialty_gate)),
    )
    return {
        "support_counts": counts,
        "mean_support": float(sum(counts) / float(max(1, len(counts)))),
        "singleton_fraction": float(singleton_fraction),
        "non_singleton_fraction": float(1.0 - singleton_fraction),
        "client_class_fractions": [float(x) for x in client_class_fractions],
        "mean_client_class_fraction": float(mean_client_class_fraction),
        "max_client_class_fraction": float(max_client_class_fraction),
        "partition_gate": float(partition_gate),
        "broad_client_gate": float(broad_client_gate),
        "compact_specialty_gate": float(compact_specialty_gate),
        "clear_partition_gate": float(clear_partition_gate),
        "false_specialty_gate": float(false_specialty_gate),
        "specialization_gate": float(specialization_gate),
    }


def _class_head_weights(meta, base_weights, bn_weights, num_classes, cfg=None, prior_entropy_deficit=0.0):
    base = torch.as_tensor(base_weights, dtype=torch.float32)
    bn = torch.as_tensor(bn_weights, dtype=torch.float32)
    class_sets = _client_class_sets(meta, num_classes)
    if not class_sets:
        return (
            torch.stack([bn.clone() for _ in range(int(num_classes))], dim=0),
            torch.zeros(int(num_classes)),
            {"specialization_gate": 0.0},
        )

    support_profile = _class_support_profile(meta, num_classes)
    sparse_overlap_gate = (
        _smooth01((2.20 - float(support_profile["mean_support"])) / 0.45)
        * (1.0 - _smooth01((float(support_profile["singleton_fraction"]) - 0.32) / 0.18))
        * (1.0 - float(support_profile["broad_client_gate"]))
        * _smooth01((float(prior_entropy_deficit) - 0.07) / 0.05)
    )
    specialization_gate = max(float(support_profile["specialization_gate"]), float(sparse_overlap_gate))
    support_profile["sparse_overlap_specialty_gate"] = float(sparse_overlap_gate)
    support_profile["specialization_gate"] = float(specialization_gate)
    fragmentation_gate = _smooth01((float(support_profile["singleton_fraction"]) - 0.50) / 0.20)
    imbalance_gate = fragmentation_gate * _smooth01(
        (float(prior_entropy_deficit) - float((cfg or {}).get("my_merge_support_mass_prior_floor", 0.06)))
        / float((cfg or {}).get("my_merge_support_mass_prior_width", 0.03))
    )
    support_mass_floor = float((cfg or {}).get("my_merge_support_mass_floor", 0.15))
    support_mass_width = float((cfg or {}).get("my_merge_support_mass_width", 0.12))

    rows = []
    strengths = []
    support_masses = []
    support_reliability = []
    for cls_idx in range(int(num_classes)):
        support = torch.tensor([1.0 if cls_idx in seen else 0.0 for seen in class_sets[: bn.numel()]], dtype=torch.float32)
        support_count = int(support.sum().item())
        if support_count <= 0:
            rows.append(bn.clone())
            strengths.append(0.0)
            support_masses.append(0.0)
            support_reliability.append(0.0)
            continue

        support_floor = 0.05
        support_mass = float((bn * support).sum().item())
        reliability = _smooth01((support_mass - support_mass_floor) / support_mass_width)
        routed = _normalize(bn * (support_floor + support), bn)
        strength = specialization_gate * (0.30 + 0.25 * max(0, 3 - support_count))
        strength *= 1.0 - imbalance_gate * (1.0 - reliability)
        strength = max(0.0, min(0.98, strength))
        rows.append(_normalize((1.0 - strength) * bn + strength * routed, base))
        strengths.append(strength)
        support_masses.append(support_mass)
        support_reliability.append(float(reliability))
    support_profile.update(
        {
            "support_mass_prior_gate": float(imbalance_gate),
            "support_mass_fragmentation_gate": float(fragmentation_gate),
            "support_mass_floor": float(support_mass_floor),
            "support_mass_width": float(support_mass_width),
            "class_support_mass": [float(x) for x in support_masses],
            "class_support_reliability": [float(x) for x in support_reliability],
        }
    )
    return torch.stack(rows, dim=0), torch.tensor(strengths, dtype=torch.float32), support_profile


def _reduce_non_diagonal(covariance, ratio):
    ratio = float(ratio)
    if ratio >= 1.0:
        return covariance
    eye = torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device)
    return covariance * eye + covariance * (1.0 - eye) * ratio


def _linear_covariance_merge(key, values, cov_stats, weights, cfg):
    if not key.endswith(".weight") or values[0].ndim != 2 or cov_stats is None:
        return None
    module_name = key[: -len(".weight")]
    if not all(isinstance(stats, dict) and module_name in stats for stats in cov_stats):
        return None
    in_features = int(values[0].shape[1])
    covariances = []
    for stats in cov_stats:
        cov = stats[module_name].detach().cpu().float()
        if cov.ndim != 2 or cov.shape[0] != in_features or cov.shape[1] != in_features:
            return None
        covariances.append(
            _reduce_non_diagonal(cov, float((cfg or {}).get("my_merge_cov_reduce_non_diagonal_ratio", 1.0)))
        )

    weights = torch.as_tensor(weights, dtype=torch.float32)
    sum_cov = None
    sum_cov_weight = None
    for value, covariance, weight in zip(values, covariances, weights):
        w = value.detach().cpu().float()
        cov_weight = covariance @ w.transpose(0, 1)
        if sum_cov is None:
            sum_cov = covariance * float(weight)
            sum_cov_weight = cov_weight * float(weight)
        else:
            sum_cov.add_(covariance, alpha=float(weight))
            sum_cov_weight.add_(cov_weight, alpha=float(weight))
    eps = float((cfg or {}).get("my_merge_cov_eps", (cfg or {}).get("regmean_eps", 1e-6)))
    eye = torch.eye(sum_cov.shape[0], dtype=torch.float32)
    merged = torch.linalg.pinv(sum_cov + eps * eye) @ sum_cov_weight
    return merged.transpose(0, 1)


def _linear_covariance_gate(weights, meta, num_classes, cfg):
    weights = torch.as_tensor(weights, dtype=torch.float32)
    effective_clients = float(1.0 / weights.square().sum().clamp_min(EPS).item())
    support = _class_support_profile(meta, num_classes)
    silo_gate = _smooth01(
        (effective_clients - float((cfg or {}).get("my_merge_cov_effective_client_floor", 5.0)))
        / float((cfg or {}).get("my_merge_cov_effective_client_width", 1.0))
    )
    overlap_gate = _smooth01(
        (support["non_singleton_fraction"] - float((cfg or {}).get("my_merge_cov_overlap_floor", 0.20)))
        / float((cfg or {}).get("my_merge_cov_overlap_width", 0.45))
    )
    return max(0.0, min(1.0, silo_gate * overlap_gate)), {
        "effective_clients": effective_clients,
        "class_support_counts": support["support_counts"],
        "mean_class_support": support["mean_support"],
        "singleton_class_fraction": support["singleton_fraction"],
        "non_singleton_class_fraction": support["non_singleton_fraction"],
        "silo_gate": float(silo_gate),
        "overlap_gate": float(overlap_gate),
    }


def _collect_linear_covariances(meta, checkpoints, cfg, *, gate_trace=None):
    if not _bool_cfg(cfg, "my_merge_use_linear_covariance", False):
        return None, {"enabled": False, "reason": "disabled_by_config"}
    if gate_trace is not None and float(gate_trace.get("linear_covariance_gate", 0.0)) <= 0.0:
        return None, {
            "enabled": False,
            "reason": "gate_zero",
            **gate_trace,
            "privacy": "no linear covariance was requested because the case is handled by BN/class routing",
        }
    uploaded = (cfg or {}).get("my_merge_linear_covariances")
    if uploaded is not None:
        return uploaded, {
            "enabled": True,
            "source": "uploaded_client_linear_covariances",
            "privacy": "aggregate linear input second moments only",
        }
    if not checkpoints:
        return None, {"enabled": False, "reason": "missing_checkpoints"}
    try:
        from utils.statistics import collect_linear_covariances

        stats = [collect_linear_covariances(meta, checkpoint, cfg) for checkpoint in checkpoints]
    except Exception as exc:
        return None, {"enabled": False, "reason": f"collection_failed: {exc}"}
    module_names = sorted(set.intersection(*(set(item.keys()) for item in stats))) if stats else []
    return stats, {
        "enabled": True,
        "source": "client_side_linear_covariance_simulation",
        "module_names": module_names,
        "privacy": "in deployment each client uploads aggregate G=E[z z^T] for linear heads; no raw images or per-sample features are sent",
    }


def _class_prior_logit(meta, num_classes):
    counts = torch.zeros(int(num_classes), dtype=torch.float32)
    for client in (meta or {}).get("clients", []):
        seen = [int(x) for x in client.get("classes", []) if 0 <= int(x) < int(num_classes)]
        if not seen:
            continue
        per_class = float(client.get("num_samples", 0.0)) / float(len(seen))
        for cls_idx in seen:
            counts[cls_idx] += per_class
    if float(counts.sum().item()) <= 0:
        return torch.zeros(int(num_classes), dtype=torch.float32), 0.0
    prior = (counts + 1.0) / (counts.sum() + float(num_classes))
    centered = torch.log(prior.clamp_min(EPS))
    centered = centered - centered.mean()
    entropy = -(prior * torch.log(prior.clamp_min(EPS))).sum() / torch.log(torch.tensor(float(num_classes)))
    return centered, max(0.0, min(1.0, 1.0 - float(entropy.item())))


def _is_classifier_bias_tensor(key, tensor, num_classes):
    lowered = key.lower()
    return (
        torch.is_floating_point(tensor)
        and tensor.ndim == 1
        and tensor.shape[0] == int(num_classes)
        and "bias" in lowered
        and any(token in lowered for token in ("fc", "classifier", "head", "logit", "proj"))
    )


def _apply_bn_calibrated_merge(state_dicts, base_merged, ledger, meta, cfg, cov_stats=None):
    if not ledger["has_bn"]:
        return OrderedDict((k, v.detach().clone()) for k, v in base_merged.items()), {
            "enabled": False,
            "reason": "no_batchnorm_running_statistics",
            "fusion_rule": "client_average_fallback",
        }

    weights = ledger["bn_weights"]
    num_classes = int((meta or {}).get("num_classes", 0))
    prior_logit, prior_entropy_deficit = _class_prior_logit(meta, num_classes)
    class_weights, class_strength, specialization_profile = _class_head_weights(
        meta,
        ledger["base_weights"],
        weights,
        num_classes,
        cfg=cfg,
        prior_entropy_deficit=prior_entropy_deficit,
    )
    cov_gate, cov_gate_trace = _linear_covariance_gate(weights, meta, num_classes, cfg)
    cov_weights = ledger["base_weights"]
    prior_strength = (
        0.0
        if _disable_bcm(cfg)
        else float((cfg or {}).get("my_merge_class_prior_strength", 0.20))
        * _smooth01((prior_entropy_deficit - 0.08) / 0.30)
    )

    merged = OrderedDict()
    updated_tensors = 0
    running_stat_tensors = 0
    covariance_tensors = 0

    for key, base_value in base_merged.items():
        values = [state[key] for state in state_dicts if key in state and tuple(state[key].shape) == tuple(base_value.shape)]
        if len(values) != len(state_dicts):
            merged[key] = base_value.detach().clone()
            continue

        if _is_bn_running_key(key):
            merged[key] = _merge_running_stat(values, weights, key, base_value.dtype)
            running_stat_tensors += 1
            continue

        if key.endswith("num_batches_tracked"):
            try:
                merged[key] = torch.stack([value.detach().cpu() for value in values], dim=0).max(dim=0).values.to(base_value.dtype)
            except RuntimeError:
                merged[key] = base_value.detach().clone()
            continue

        if not torch.is_floating_point(base_value):
            merged[key] = base_value.detach().clone()
            continue

        if num_classes > 0 and _is_classifier_tensor(key, base_value, num_classes):
            cov_out = _linear_covariance_merge(key, values, cov_stats, cov_weights, cfg)
            out = _weighted_average(values, weights)
            for cls_idx in range(int(num_classes)):
                routed = _weighted_average([value[cls_idx] for value in values], class_weights[cls_idx])
                strength = 0.0 if _disable_bcm(cfg) else float(class_strength[cls_idx].item())
                out[cls_idx] = (1.0 - strength) * out[cls_idx] + strength * routed
            if cov_out is not None and not _disable_bcm(cfg) and cov_gate > 0:
                out = (1.0 - cov_gate) * out + cov_gate * cov_out.to(out.device)
                covariance_tensors += 1
            merged[key] = out.to(base_value.dtype)
            updated_tensors += 1
            continue

        cov_out = _linear_covariance_merge(key, values, cov_stats, cov_weights, cfg)
        if cov_out is not None and not _disable_bcm(cfg) and cov_gate > 0:
            avg_out = _weighted_average(values, weights)
            merged[key] = ((1.0 - cov_gate) * avg_out + cov_gate * cov_out.to(avg_out.device)).to(base_value.dtype)
            covariance_tensors += 1
        else:
            merged[key] = _weighted_average(values, weights).to(base_value.dtype)
        updated_tensors += 1

    prior_bias_tensors = 0
    if prior_strength > 0.0 and prior_logit.numel() == int(num_classes):
        for key, value in list(merged.items()):
            if _is_classifier_bias_tensor(key, value, num_classes):
                merged[key] = (value.detach().float() + prior_strength * prior_logit.to(value.device)).to(value.dtype)
                prior_bias_tensors += 1

    return merged, {
        "enabled": not _disable_bcm(cfg),
        "rule": "protocol_calibrated_merge",
        "reason": "disabled_by_ablation" if _disable_bcm(cfg) else "enabled",
        "updated_tensors": int(updated_tensors),
        "running_stat_tensors": int(running_stat_tensors),
        "linear_covariance_gate": float(cov_gate),
        "linear_covariance_gate_trace": cov_gate_trace,
        "linear_covariance_weight_rule": "base_client_weights",
        "linear_covariance_tensors": int(covariance_tensors),
        "class_specialization_gate": float(specialization_profile.get("specialization_gate", 0.0)),
        "class_specialization_profile": specialization_profile,
        "class_head_strength": [float(x) for x in class_strength.tolist()],
        "class_prior_entropy_deficit": float(prior_entropy_deficit),
        "class_prior_strength": float(prior_strength),
        "prior_bias_tensors": int(prior_bias_tensors),
        "privacy": "server uses checkpoints, aggregate BN moments, and class/sample counts only in the main method; no raw images, per-sample logits, activations, or candidate validation feedback",
    }


def merge_my_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    if meta is None or cfg is None:
        raise ValueError("my_merge requires task metadata and runtime config.")
    domain_control = _domain_control_enabled(cfg)
    if not _is_image_checkpoint_task(meta):
        raise ValueError(
            f"my_merge requires image checkpoint metadata with BN-compatible spatial features, got task_type={meta.get('task_type')}."
        )

    base_merged, base_weights = average_state_dicts(state_dicts, weights)
    base = torch.as_tensor(base_weights, dtype=torch.float32)
    ablation_labels = {x.lower().replace("-", "_") for x in _ablation_labels(cfg)}
    if "avg_only" in ablation_labels:
        return base_merged, {
            "implementation": "clinical_evidence_ledger_pcm_avg_only",
            "clinical_site_silo_method": True,
            "domain_control": bool(domain_control),
            "fusion_rule": "client_average",
            "base_weights": [float(x) for x in base.tolist()],
        }

    ledger = _build_bn_moment_ledger(state_dicts, base, meta, cfg)
    pre_cov_gate, pre_cov_trace = _linear_covariance_gate(
        ledger["bn_weights"],
        meta,
        int((meta or {}).get("num_classes", 0)),
        cfg,
    )
    pre_cov_trace = {"linear_covariance_gate": float(pre_cov_gate), **pre_cov_trace}
    cov_stats, cov_trace = _collect_linear_covariances(meta, checkpoints, cfg, gate_trace=pre_cov_trace)
    merged, merge_trace = _apply_bn_calibrated_merge(state_dicts, base_merged, ledger, meta, cfg, cov_stats=cov_stats)

    fusion_path = "CEL + PCM" if ledger["has_bn"] else "client_average_fallback(no_bn)"
    return merged, {
        "implementation": "clinical_evidence_ledger_pcm_v1",
        "clinical_site_silo_method": True,
        "domain_control": bool(domain_control),
        "fusion_solution": "single_shot_private_bn_moment_and_specialty_partition_merge",
        "fusion_rule": "clinical_evidence_ledger_protocol_calibrated_merge",
        "fusion_path": fusion_path,
        "observed_medical_property": (
            "medical image clients often differ by scanner, staining, acquisition protocol, organ window, and lesion scale; "
            "these site shifts appear as channel-wise activation mean/variance changes in BN buffers, while NLP token models "
            "usually use LayerNorm and do not expose image-channel spatial moment statistics"
        ),
        "module_1": "Clinical Evidence Ledger (CEL)",
        "module_2": "Protocol-Calibrated Merge (PCM)",
        "base_weights": [float(x) for x in ledger["base_weights"].tolist()],
        "sample_prior": [float(x) for x in ledger["sample_prior"].tolist()],
        "fusion_weights": [float(x) for x in ledger["bn_weights"].tolist()],
        "bn_consistency": [float(x) for x in ledger["bn_consistency"].tolist()],
        "bn_distance": [float(x) for x in ledger["bn_distance"].tolist()],
        "bn_layer_count": int(len(ledger["bn_pairs"])),
        "bn_heterogeneity": float(ledger["heterogeneity"]),
        "client_diagnostic_information": ledger.get("client_rows", []),
        "routing_summary": {
            "module_1": ledger["trace"],
            "module_2": merge_trace,
            "fusion_equation": (
                "For BN layer l and client i, CEL forms d_i = mean_l[ ||mu_i^l-mu_bar^l||^2/(sigma_bar^l)^2 "
                "+ 0.5||log v_i^l-log v_bar^l||^2 ]. It sets omega_i = Norm(p_i(0.35+0.65 exp(-d_i/tau))), "
                "where p_i is the sample evidence prior. PCM merges trainable tensors with omega, merges BN means "
                "arithmetically, merges BN variances geometrically, and routes classifier rows only when the client "
                "class layout forms a clinical-specialty partition."
            ),
            "privacy_audit": {
                "server_raw_data_access": False,
                "per_sample_logits": False,
                "per_sample_activations": False,
                "client_validation_feedback": False,
                "uploaded_statistics": "checkpoint parameters plus aggregate BN running mean/variance and sample/class metadata; optional linear-head covariances are disabled in the main method",
            },
        },
        "method_trace": {
            "clinical_evidence_ledger": ledger["trace"],
            "linear_covariance": cov_trace,
            "protocol_calibrated_merge": merge_trace,
        },
    }


def _feature_summary_path(meta, cfg):
    root = (cfg or {}).get("my_merge_feature_summary_root")
    if not root:
        return None
    model_name = meta.get("model") if meta.get("task_type") == "small" else str(meta.get("clip_model", meta.get("model", ""))).split("/")[-1]
    beta = str(meta.get("beta", "0")).replace(".", "p")
    name = f"{meta.get('task_type')}__{meta.get('dataset')}__{model_name}__c{meta.get('num_clients')}__b{beta}__s{meta.get('seed')}.json"
    return Path(root) / name


def _summary_from_payload(payload, num_classes=None):
    out = dict(payload or {})
    out.setdefault("source", "uploaded_bn_moment_summary")
    out.setdefault("feature_names", BN_MOMENT_NAMES)
    return out


def _load_feature_summary_file(meta, cfg):
    path = _feature_summary_path(meta, cfg)
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("source", "bn_moment_summary_file")
    return payload


def _collect_feature_summary(meta, cfg):
    uploaded = (cfg or {}).get("my_merge_feature_summary") or _load_feature_summary_file(meta, cfg)
    if isinstance(uploaded, dict):
        return _summary_from_payload(uploaded, int(meta.get("num_classes", 0)))
    return {
        "source": "bn_only_no_morphology_summary",
        "feature_names": BN_MOMENT_NAMES,
        "privacy_note": "BN-only my_merge does not collect morphology features or per-sample diagnostics.",
    }


def _param_group(key, value=None, num_classes=None):
    if _is_bn_running_key(key) or key.endswith("num_batches_tracked"):
        return "normalization_stat"
    return "representation"
