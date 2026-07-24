from collections import OrderedDict

import torch

from utils.runtime import build_reference_bundle
from utils.state_dict import average_state_dicts


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
        bias_key = None
        for key, tensor in state.items():
            if _is_classifier_tensor(key, tensor, num_classes) and tensor.ndim == 1:
                bias_key = key
                break
    return weight_key, bias_key


def merge_avg_head(state_dicts, weights, meta=None):
    if meta is None:
        raise ValueError("avg_head requires task metadata.")

    num_classes = int(meta["num_classes"])
    base_state, _ = build_reference_bundle(meta, device="cpu")
    avg_state, normalized_weights = average_state_dicts(state_dicts, weights)
    merged = OrderedDict((key, value.detach().cpu().clone()) for key, value in base_state.items())

    base_weight_key, base_bias_key = _classifier_pair(merged, num_classes)
    avg_weight_key, avg_bias_key = _classifier_pair(avg_state, num_classes)
    if base_weight_key is None or avg_weight_key is None:
        raise ValueError("avg_head requires classifier weights in both reference and client models.")
    if merged[base_weight_key].shape != avg_state[avg_weight_key].shape:
        raise ValueError(
            "Reference and averaged classifier weights have incompatible shapes: "
            f"reference={list(merged[base_weight_key].shape)}, "
            f"averaged={list(avg_state[avg_weight_key].shape)}"
        )

    merged[base_weight_key] = avg_state[avg_weight_key].detach().cpu().to(dtype=merged[base_weight_key].dtype)
    if base_bias_key is not None:
        if avg_bias_key is None:
            bias = torch.zeros_like(merged[base_bias_key], dtype=torch.float32)
        else:
            bias = avg_state[avg_bias_key].detach().cpu().float()
        merged[base_bias_key] = bias.to(dtype=merged[base_bias_key].dtype)

    return merged, {
        "method_name": "Avg-Head",
        "implementation": "reference_backbone_with_averaged_client_classifier_head",
        "backbone_source": "reference_initialization",
        "head_source": "weighted_average_of_client_classifier_heads",
        "weight_key": base_weight_key,
        "bias_key": base_bias_key,
        "normalized_weights": [float(x) for x in normalized_weights],
    }
