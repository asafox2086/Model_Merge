"""Incremental, asynchronous variant of LAMP-Merge.

``new_lamp_merge`` retains only sufficient statistics for the prototype
aggregation.  A server can therefore update and deploy a checkpoint whenever a
new client upload arrives, without revisiting earlier client payloads.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from methods.lamp_merge import (
    EPS,
    _cfg_value,
    _classifier_pair,
    _dominant_prior_threshold,
    _is_medical_image_task,
    _is_reference_prototype_stats,
    _load_prototype_stats,
    _prevalence_calibration_strength,
    _validate_prevalence_provenance,
)
from utils.runtime import build_reference_bundle


def _vector(value, num_classes, field_name):
    tensor = torch.as_tensor(value, dtype=torch.float32).view(-1)
    if int(tensor.numel()) != int(num_classes):
        raise ValueError(f"{field_name} must contain {num_classes} values, got {int(tensor.numel())}.")
    return tensor


def _client_counts(client, num_classes, fields, required=True):
    for field in fields:
        if client.get(field) is not None:
            return _vector(client[field], num_classes, field)
    if required:
        raise ValueError(f"Client statistic is missing one of: {', '.join(fields)}.")
    return torch.zeros(num_classes, dtype=torch.float32)


def initialize_async_lamp_state(meta, cfg):
    """Create a serializable server state before receiving any client upload."""
    if not _is_medical_image_task(meta):
        raise ValueError(f"new_lamp_merge is defined only for medical image tasks, got {meta.get('dataset')}.")

    base_state, _ = build_reference_bundle(meta, device="cpu")
    base_state = OrderedDict((key, value.detach().cpu().clone()) for key, value in base_state.items())
    num_classes = int(meta["num_classes"])
    weight_key, bias_key = _classifier_pair(base_state, num_classes)
    if weight_key is None:
        raise ValueError("Could not locate classifier weight tensor for new_lamp_merge.")
    feature_dim = int(base_state[weight_key].shape[1])
    return {
        "version": 1,
        "base_state": base_state,
        "weight_key": weight_key,
        "bias_key": bias_key,
        "prototype_numerator": torch.zeros(num_classes, feature_dim, dtype=torch.float32),
        "evidence_sum": torch.zeros(num_classes, dtype=torch.float32),
        "prototype_class_counts": torch.zeros(num_classes, dtype=torch.float32),
        "prevalence_counts": torch.zeros(num_classes, dtype=torch.float32),
        "received_client_ids": [],
    }


def add_async_client(state, client, client_id, meta, cfg):
    """Apply one client upload in-place and return the updated server state."""
    client_id = int(client_id)
    if client_id in state["received_client_ids"]:
        raise ValueError(f"Client {client_id} has already been incorporated into the async state.")

    num_classes = int(meta["num_classes"])
    means = torch.as_tensor(client.get("class_feature_mean"), dtype=torch.float32)
    expected_dim = int(state["prototype_numerator"].shape[1])
    if means.ndim != 2 or tuple(means.shape) != (num_classes, expected_dim):
        raise ValueError(
            f"Invalid class_feature_mean for client {client_id}: got {list(means.shape)}, "
            f"expected [{num_classes}, {expected_dim}]."
        )

    feature_counts = _client_counts(
        client,
        num_classes,
        ("class_feature_counts", "class_counts"),
    ).clamp_min(0.0)
    prevalence_counts = _client_counts(client, num_classes, ("class_prevalence_counts",)).clamp_min(0.0)
    gamma = float(_cfg_value(cfg, "lamp_merge_proto_count_power", default=0.55))
    evidence = (feature_counts + 1.0).pow(gamma) * (feature_counts > 0).to(torch.float32)

    state["prototype_numerator"] += evidence.unsqueeze(1) * means
    state["evidence_sum"] += evidence
    state["prototype_class_counts"] += feature_counts
    state["prevalence_counts"] += prevalence_counts
    state["received_client_ids"].append(client_id)
    return state


def synthesize_async_lamp_state(state, meta, cfg):
    """Materialize the immediately deployable checkpoint represented by ``state``."""
    num_classes = int(meta["num_classes"])
    merged = OrderedDict((key, value.detach().cpu().clone()) for key, value in state["base_state"].items())
    evidence_sum = state["evidence_sum"].detach().cpu().float()
    observed = evidence_sum > 0
    prototypes = state["prototype_numerator"].detach().cpu().float() / evidence_sum[:, None].clamp_min(EPS)

    scale = float(_cfg_value(cfg, "lamp_merge_reference_head_scale", default=18.75))
    head_weight = merged[state["weight_key"]].detach().cpu().float().clone()
    if bool(observed.any()):
        head_weight[observed] = scale * F.normalize(prototypes[observed], dim=1)
    merged[state["weight_key"]] = head_weight.to(dtype=merged[state["weight_key"]].dtype)

    prevalence_counts = state["prevalence_counts"].detach().cpu().float()
    all_classes_observed = bool((prevalence_counts > 0).all())
    prevalence_strength = 0.0
    class_prior = torch.zeros(num_classes, dtype=torch.float32)
    if all_classes_observed:
        class_prior = prevalence_counts / prevalence_counts.sum().clamp_min(EPS)
        prevalence_strength = _prevalence_calibration_strength(class_prior, num_classes, cfg)

    if state["bias_key"] is not None:
        head_bias = torch.zeros(num_classes, dtype=torch.float32)
        if prevalence_strength != 0.0:
            centered_log_prior = torch.log(class_prior.clamp_min(EPS))
            head_bias = prevalence_strength * (centered_log_prior - centered_log_prior.mean())
        merged[state["bias_key"]] = head_bias.to(dtype=merged[state["bias_key"]].dtype)
    elif prevalence_strength != 0.0:
        raise ValueError("Long-tail prevalence calibration requires a classifier bias tensor.")

    trace = {
        "used": True,
        "head_mode": "incremental_cosine_prototype",
        "evidence_mode": "running_support_power",
        "evidence_gamma": float(_cfg_value(cfg, "lamp_merge_proto_count_power", default=0.55)),
        "head_scale": scale,
        "num_clients": len(state["received_client_ids"]),
        "received_client_ids": list(state["received_client_ids"]),
        "unobserved_classes": torch.nonzero(~observed, as_tuple=False).view(-1).tolist(),
        "prototype_class_counts": [float(value) for value in state["prototype_class_counts"].tolist()],
        "class_counts": [float(value) for value in prevalence_counts.tolist()],
        "class_prior": [float(value) for value in class_prior.tolist()],
        "prevalence_bias_scale": float(prevalence_strength),
        "prevalence_threshold": _dominant_prior_threshold(cfg, num_classes),
    }
    return merged, trace


def merge_new_lamp_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    """One-shot compatibility wrapper built from the asynchronous state API."""
    if meta is None or cfg is None:
        raise ValueError("new_lamp_merge requires task metadata and runtime config.")
    prototype_stats, client_stats_path = _load_prototype_stats(meta, cfg)
    if not _is_reference_prototype_stats(prototype_stats):
        raise ValueError("new_lamp_merge requires client-side reference prototype statistics.")
    _validate_prevalence_provenance(prototype_stats)

    state = initialize_async_lamp_state(meta, cfg)
    clients = prototype_stats.get("clients", [])
    expected_clients = int(meta.get("num_clients") or len(clients))
    if len(clients) < expected_clients:
        raise ValueError(f"Expected {expected_clients} client statistics, got {len(clients)}.")
    for index in range(expected_clients):
        add_async_client(state, clients[index], index, meta, cfg)
    merged, trace = synthesize_async_lamp_state(state, meta, cfg)
    trace["client_stats_path"] = client_stats_path
    return merged, {
        "method_name": "New LAMP-Merge",
        "implementation": "new_lamp_merge_async_v1",
        "fusion_rule": "incremental_long_tail_aware_medical_prototype_merging",
        "num_clients": len(state["received_client_ids"]),
        "reference_prototype_head": trace,
    }
