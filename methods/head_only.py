from collections import OrderedDict

import torch

from utils.runtime import build_reference_bundle

from .avg import merge_avg
from .breadcrumbs import merge_breadcrumbs
from .dare import merge_dare_linear, merge_dare_ties
from .fisher import merge_fisher
from .from_merge import merge_from
from .free_merge import merge_free
from .iso import merge_iso_c
from .model_stock import merge_model_stock
from .regmean import merge_regmean
from .robustmerge import merge_robustmerge
from .ties import merge_ties


def _name_tokens(key):
    return {part.lower() for part in key.replace("/", ".").split(".")}


def is_classifier_tensor(key, tensor, num_classes):
    if not torch.is_floating_point(tensor) or tensor.ndim < 1 or int(tensor.shape[0]) != int(num_classes):
        return False
    tokens = _name_tokens(key)
    head_tokens = {"head", "fc", "classifier", "classif", "last_linear", "logits"}
    if tokens & head_tokens:
        return True
    tail = key.lower().split(".")[-1]
    return tail in {"weight", "bias"} and any(token in key.lower() for token in head_tokens)


def classifier_param_keys(state, num_classes):
    keys = [
        key
        for key, tensor in state.items()
        if is_classifier_tensor(key, tensor, num_classes)
    ]
    weight_keys = [key for key in keys if state[key].ndim == 2]
    if not weight_keys:
        raise ValueError("No classifier-head weight tensor was found.")
    prefix = weight_keys[0].rsplit(".", 1)[0]
    ordered = [weight_keys[0]]
    bias_key = f"{prefix}.bias"
    if bias_key in state and bias_key in keys:
        ordered.append(bias_key)
    else:
        ordered.extend(key for key in keys if key != weight_keys[0] and state[key].ndim == 1)
    return ordered


def _filter_states(state_dicts, keys):
    out = []
    for state in state_dicts:
        out.append(OrderedDict((key, state[key]) for key in keys))
    return out


def _filter_base(base_state, keys):
    return OrderedDict((key, base_state[key]) for key in keys)


def _overlay_head_on_reference(head_state, reference_state, head_keys):
    merged = OrderedDict((key, value.detach().cpu().clone()) for key, value in reference_state.items())
    for key in head_keys:
        if key not in head_state:
            raise ValueError(f"Head-only merge did not produce classifier parameter: {key}")
        if merged[key].shape != head_state[key].shape:
            raise ValueError(
                f"Incompatible classifier parameter shape for {key}: "
                f"reference={list(merged[key].shape)}, merged={list(head_state[key].shape)}"
            )
        merged[key] = head_state[key].detach().cpu().to(dtype=merged[key].dtype)
    return merged


def _filter_stats(stats_list, head_keys):
    wanted_modules = {key[:-len(".weight")] for key in head_keys if key.endswith(".weight")}
    filtered = []
    for stats in stats_list:
        item = {}
        for key, value in stats.items():
            if key in head_keys or key in wanted_modules:
                item[key] = value
        filtered.append(item)
    return filtered


def merge_head_only(base_method, state_dicts, weights, meta, checkpoints=None, cfg=None, fisher_stats=None, cov_stats=None):
    """Run a baseline merge only in classifier-head parameter space."""
    if meta is None:
        raise ValueError("head-only baseline requires task metadata.")
    cfg = cfg or {}
    base_method = str(base_method).strip().lower()
    num_classes = int(meta["num_classes"])
    reference_state, _ = build_reference_bundle(meta, device="cpu")
    head_keys = classifier_param_keys(reference_state, num_classes)
    head_states = _filter_states(state_dicts, head_keys)
    head_base = _filter_base(reference_state, head_keys)

    if base_method == "avg":
        merged_head, info = merge_avg(head_states, weights)
    elif base_method == "ties":
        merged_head, info = merge_ties(
            head_states,
            head_base,
            weights,
            density=float(cfg["density"]),
            param_keys=head_keys,
        )
    elif base_method == "dare_linear":
        merged_head, info = merge_dare_linear(
            head_states,
            head_base,
            weights,
            density=float(cfg["density"]),
            seed=int(cfg["dare_seed"]),
            param_keys=head_keys,
        )
    elif base_method == "dare_ties":
        merged_head, info = merge_dare_ties(
            head_states,
            head_base,
            weights,
            density=float(cfg["density"]),
            seed=int(cfg["dare_seed"]),
            param_keys=head_keys,
        )
    elif base_method == "fisher":
        if fisher_stats is None:
            raise ValueError("head_fisher requires fisher statistics.")
        merged_head, info = merge_fisher(
            head_states,
            _filter_stats(fisher_stats, head_keys),
            weights,
            eps=float(cfg["fisher_eps"]),
            normalize_fisher_weight=bool(cfg["fisher_normalize_weight"]),
            minimal_fisher_weight=float(cfg["fisher_minimal_weight"]),
        )
    elif base_method == "regmean":
        if cov_stats is None:
            raise ValueError("head_regmean requires covariance statistics.")
        merged_head, info = merge_regmean(
            head_states,
            _filter_stats(cov_stats, head_keys),
            weights,
            eps=float(cfg["regmean_eps"]),
            reduce_non_diagonal_ratio=float(cfg["regmean_reduce_non_diagonal_ratio"]),
        )
    elif base_method == "breadcrumbs":
        merged_head, info = merge_breadcrumbs(
            head_states,
            head_base,
            weights,
            top_k_keep=float(cfg["breadcrumbs_top_k_keep"]),
            top_k_remove=float(cfg["breadcrumbs_top_k_remove"]),
            alpha=float(cfg["breadcrumbs_alpha"]),
            param_keys=head_keys,
        )
    elif base_method == "model_stock":
        merged_head, info = merge_model_stock(
            head_states,
            head_base,
            weights,
            k=float(cfg["model_stock_k"]),
        )
    elif base_method == "from":
        merged_head, info = merge_from(
            head_states,
            head_base,
            weights,
            k=float(cfg["from_k"]),
        )
    elif base_method == "iso_c":
        merged_head, info = merge_iso_c(head_states, head_base, weights, cfg)
    elif base_method == "free_merge":
        merged_head, info = merge_free(head_states, head_base, weights, cfg)
    elif base_method == "robustmerge":
        merged_head, info = merge_robustmerge(head_states, head_base, weights, cfg)
    else:
        raise ValueError(f"Unsupported head-only baseline: {base_method}")

    merged = _overlay_head_on_reference(merged_head, reference_state, head_keys)
    info = dict(info) if isinstance(info, dict) else {"baseline_info": info}
    info.update(
        {
            "method_name": f"Head-{base_method}",
            "implementation": f"head_only_{info.get('implementation', base_method)}",
            "head_only": True,
            "base_method": base_method,
            "backbone_source": "reference_initialization",
            "head_keys": list(head_keys),
        }
    )
    return merged, info
