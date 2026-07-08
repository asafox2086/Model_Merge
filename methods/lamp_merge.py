from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.hub import beta_to_dirname
from utils.runtime import build_reference_bundle


EPS = 1e-8

LAMP_MERGE_ABLATION_MODES = {"full"}

MEDICAL_IMAGE_DATASETS = {
    "bloodmnist_224",
    "chaoshengmnist_224",
    "dermamnist_224",
    "organamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "pathmnist_224",
}


def _cfg_value(cfg, *names, default=None):
    for name in names:
        value = cfg.get(name)
        if value is not None:
            return value
    return default


def _is_medical_image_task(meta):
    return meta.get("task_type") in {"small", "vlm"} and meta.get("dataset") in MEDICAL_IMAGE_DATASETS


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


def _prototype_stats_path(meta, cfg):
    direct = (
        cfg.get("lamp_merge_prototype_stats_path")
        or cfg.get("lamp_merge_client_prototype_path")
        or cfg.get("lamp_merge_proto_stats_path")
        or cfg.get("my_merge_prototype_stats_path")
        or cfg.get("my_merge_client_prototype_path")
        or cfg.get("my_merge_proto_stats_path")
    )
    if direct:
        path = Path(direct)
        return path if path.exists() else None

    root = (
        cfg.get("lamp_merge_prototype_root")
        or cfg.get("lamp_merge_client_prototype_root")
        or cfg.get("my_merge_prototype_root")
        or cfg.get("my_merge_client_prototype_root")
    )
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
        raise ValueError(f"Invalid LAMP-Merge prototype statistics payload: {path}")
    return payload, str(path)


def _is_reference_prototype_stats(proto_stats):
    return isinstance(proto_stats, dict) and proto_stats.get("feature_space") == "reference_model"


def _num_clients(meta, proto_stats):
    meta_clients = len(meta.get("clients", []))
    payload_clients = len(proto_stats.get("clients", [])) if isinstance(proto_stats, dict) else 0
    configured = int(meta.get("num_clients") or 0)
    return max(meta_clients, payload_clients, configured)


def _client_stat_matrix(proto_stats, fields, num_clients, num_classes):
    clients = proto_stats.get("clients", [])
    rows = []
    for idx in range(num_clients):
        item = clients[idx] if idx < len(clients) else {}
        value = None
        for field in fields:
            value = item.get(field)
            if value is not None:
                break

        row = torch.zeros(num_classes, dtype=torch.float32)
        if isinstance(value, dict):
            for cls_key, count in value.items():
                cls_idx = int(cls_key)
                if 0 <= cls_idx < num_classes:
                    row[cls_idx] = float(count)
        elif value is not None:
            tensor = torch.as_tensor(value, dtype=torch.float32).view(-1)
            if int(tensor.numel()) != int(num_classes):
                raise ValueError(
                    f"Expected {num_classes} class statistics for fields {fields}, got {int(tensor.numel())}."
                )
            row = tensor
        rows.append(row)
    return torch.stack(rows, dim=0)


def _client_feature_means(proto_stats, num_clients, num_classes):
    clients = proto_stats.get("clients", [])
    means = []
    for idx in range(num_clients):
        if idx >= len(clients) or clients[idx].get("class_feature_mean") is None:
            raise ValueError(f"Missing reference feature means for client {idx}.")
        tensor = torch.as_tensor(clients[idx]["class_feature_mean"], dtype=torch.float32)
        if tensor.ndim != 2 or int(tensor.shape[0]) != int(num_classes):
            raise ValueError(
                f"Invalid class_feature_mean shape for client {idx}: {list(tensor.shape)}; "
                f"expected [{num_classes}, feature_dim]."
            )
        means.append(tensor)
    return torch.stack(means, dim=0)


def _evidence_matrix(feature_counts, cfg):
    gamma = float(_cfg_value(cfg, "lamp_merge_proto_count_power", "my_merge_proto_count_power", default=0.45))
    evidence = (feature_counts + 1.0).pow(gamma)
    return evidence * (feature_counts > 0).to(evidence.dtype), gamma


def _prevalence_counts(proto_stats, feature_counts, meta):
    num_classes = int(meta["num_classes"])
    num_clients = _num_clients(meta, proto_stats)
    counts = _client_stat_matrix(
        proto_stats,
        ("class_prevalence_counts", "class_prior_counts", "label_counts"),
        num_clients,
        num_classes,
    )
    if float(counts.sum().item()) <= 0:
        return torch.zeros_like(feature_counts), "missing"
    return counts, "uploaded_prevalence_counts"


def _global_prototypes(proto_stats, meta, cfg):
    num_classes = int(meta["num_classes"])
    num_clients = _num_clients(meta, proto_stats)
    if num_clients <= 0:
        raise ValueError("LAMP-Merge requires at least one uploaded client prototype payload.")

    means = _client_feature_means(proto_stats, num_clients, num_classes)
    feature_counts = _client_stat_matrix(
        proto_stats,
        ("class_feature_counts", "class_counts"),
        num_clients,
        num_classes,
    )
    if float(feature_counts.sum().item()) <= 0:
        raise ValueError("Uploaded prototype statistics contain no class support counts.")

    evidence, gamma = _evidence_matrix(feature_counts, cfg)
    evidence_per_class = evidence.sum(dim=0)
    missing_classes = torch.nonzero(evidence_per_class <= 0, as_tuple=False).view(-1).tolist()
    if missing_classes:
        raise ValueError(f"Uploaded prototypes do not cover classes: {missing_classes}")

    weights = evidence / evidence_per_class.view(1, -1).clamp_min(EPS)
    prototypes = (means * weights.unsqueeze(-1)).sum(dim=0)
    prevalence_counts, prevalence_source = _prevalence_counts(proto_stats, feature_counts, meta)

    return {
        "prototypes": prototypes,
        "class_counts": prevalence_counts.sum(dim=0).clamp_min(0.0),
        "class_prevalence_source": prevalence_source,
        "prototype_class_counts": feature_counts.sum(dim=0).clamp_min(0.0),
        "prototype_weights": weights,
        "evidence_gamma": gamma,
        "num_clients": num_clients,
    }


def _dominant_prior_threshold(cfg, num_classes):
    imbalance_threshold = float(_cfg_value(
        cfg,
        "lamp_merge_reference_prior_threshold",
        "my_merge_reference_prior_threshold",
        default=2.5,
    ))
    if imbalance_threshold > 0:
        return imbalance_threshold / float(num_classes)
    prevalence_threshold = _cfg_value(cfg, "lamp_merge_prevalence_threshold", "my_merge_prevalence_threshold")
    if prevalence_threshold is not None:
        return float(prevalence_threshold)
    return 0.5


def _prevalence_calibration_strength(class_prior, num_classes, cfg):
    if bool(_cfg_value(
        cfg,
        "lamp_merge_disable_prevalence_calibration",
        "my_merge_disable_prevalence_calibration",
        default=False,
    )):
        return 0.0
    if float(class_prior.sum().item()) <= 0:
        return 0.0
    threshold = _dominant_prior_threshold(cfg, num_classes)
    dominant_prior = float(class_prior.max().item())
    if dominant_prior <= threshold:
        return 0.0

    explicit_tau = _cfg_value(cfg, "lamp_merge_reference_prior_tau", "my_merge_reference_prior_tau")
    if explicit_tau is not None and float(explicit_tau) >= 0.0:
        return float(explicit_tau)
    return float(_cfg_value(
        cfg,
        "lamp_merge_reference_prior_max_tau",
        "my_merge_reference_prior_max_tau",
        default=6.0,
    ))


def _centered_log_prior_bias(class_prior, strength):
    centered_log_prior = torch.log(class_prior.clamp_min(EPS))
    centered_log_prior = centered_log_prior - centered_log_prior.mean()
    return float(strength) * centered_log_prior


def _synthesize_reference_prototype_model(base_state, proto_stats, meta, cfg):
    proto = _global_prototypes(proto_stats, meta, cfg)
    num_classes = int(meta["num_classes"])
    state = OrderedDict((key, value.detach().cpu().clone()) for key, value in base_state.items())
    weight_key, bias_key = _classifier_pair(state, num_classes)
    if weight_key is None:
        raise ValueError("Could not locate classifier weight tensor for LAMP-Merge prototype head synthesis.")

    classifier_weight = state[weight_key].detach().cpu().float()
    prototypes = proto["prototypes"].detach().cpu().float()
    if prototypes.ndim != 2 or prototypes.shape[0] != num_classes or prototypes.shape[1] != classifier_weight.shape[1]:
        raise ValueError(
            "Prototype feature dimension does not match classifier weight: "
            f"prototype_shape={list(prototypes.shape)}, classifier_shape={list(classifier_weight.shape)}"
        )

    scale = float(_cfg_value(
        cfg,
        "lamp_merge_reference_head_scale",
        "my_merge_reference_head_scale",
        default=20.0,
    ))
    class_counts = proto["class_counts"]
    class_prior = class_counts / class_counts.sum().clamp_min(EPS)
    prevalence_strength = _prevalence_calibration_strength(class_prior, num_classes, cfg)
    imbalance_ratio = float((class_prior.max() * float(num_classes)).item())

    head_weight = scale * F.normalize(prototypes, dim=1)
    head_bias = torch.zeros(num_classes, dtype=torch.float32)
    if prevalence_strength != 0.0:
        head_bias = head_bias + _centered_log_prior_bias(class_prior, prevalence_strength)

    state[weight_key] = head_weight.to(dtype=state[weight_key].dtype)
    if bias_key is not None:
        state[bias_key] = head_bias.to(dtype=state[bias_key].dtype)
    elif prevalence_strength != 0.0:
        raise ValueError("Long-tail prevalence calibration requires a classifier bias tensor.")

    return state, {
        "used": True,
        "feature_space": "reference_model",
        "weight_key": weight_key,
        "bias_key": bias_key,
        "head_mode": "cosine_prototype",
        "prototype_mode": "reference_prototype",
        "evidence_mode": "support_power",
        "prevalence_mode": "uploaded",
        "evidence_gamma": proto["evidence_gamma"],
        "head_scale": scale,
        "prevalence_bias_scale": prevalence_strength,
        "prevalence_threshold": _dominant_prior_threshold(cfg, num_classes),
        "reference_prior_threshold": float(_cfg_value(
            cfg,
            "lamp_merge_reference_prior_threshold",
            "my_merge_reference_prior_threshold",
            default=2.5,
        )),
        "reference_prior_max_tau": float(_cfg_value(
            cfg,
            "lamp_merge_reference_prior_max_tau",
            "my_merge_reference_prior_max_tau",
            default=6.0,
        )),
        "imbalance_ratio": imbalance_ratio,
        "num_clients": int(proto["num_clients"]),
        "class_counts": [float(x) for x in class_counts.tolist()],
        "class_prevalence_source": proto["class_prevalence_source"],
        "class_prior": [float(x) for x in class_prior.tolist()],
        "prototype_class_counts": [float(x) for x in proto["prototype_class_counts"].tolist()],
        "prototype_weights": [[float(v) for v in row] for row in proto["prototype_weights"].tolist()],
    }


def merge_lamp_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    if meta is None or cfg is None:
        raise ValueError("LAMP-Merge requires task metadata and runtime config.")
    if not _is_medical_image_task(meta):
        raise ValueError(f"LAMP-Merge is defined only for medical image tasks, got dataset={meta.get('dataset')}.")

    cfg = cfg or {}
    requested_mode = str(cfg.get("lamp_merge_ablation_mode") or "full").strip().lower()
    if requested_mode != "full":
        raise ValueError(
            "The release LAMP-Merge implementation is fixed to the formal method. "
            "Use method=lamp_merge_analysis for ablation experiments."
        )

    client_stats, client_stats_path = _load_prototype_stats(meta, cfg)
    if not _is_reference_prototype_stats(client_stats):
        raise ValueError(
            "LAMP-Merge requires client-side reference prototype statistics. "
            "Please export prototype_stats.pt before running this method."
        )

    base_state, _ = build_reference_bundle(meta, device="cpu")
    merged, trace = _synthesize_reference_prototype_model(base_state, client_stats, meta, cfg)
    trace["client_stats_path"] = client_stats_path

    prevalence_trace = {
        "used": bool(float(trace.get("prevalence_bias_scale", 0.0)) != 0.0),
        "prevalence_bias_scale": float(trace.get("prevalence_bias_scale", 0.0)),
        "imbalance_ratio": float(trace.get("imbalance_ratio", 0.0)),
        "class_prior": trace.get("class_prior", []),
    }

    base_input_weights = []
    if weights is not None:
        base_input_weights = [float(x) for x in torch.as_tensor(weights, dtype=torch.float32).view(-1).tolist()]

    return merged, {
        "method_name": "LAMP-Merge",
        "implementation": "lamp_merge_release_v1",
        "medical_only": True,
        "fusion_rule": "long_tail_aware_medical_prototype_merging",
        "privacy": {
            "server_reads_private_images": False,
            "uses_public_probe": False,
            "client_uploads": [
                "checkpoint",
                "reference_backbone_class_prototypes",
                "class_support_counts",
                "class_prevalence_counts",
            ],
        },
        "modules": {
            "M1_diagnostic_prototype_reconstruction": (
                "clients upload per-diagnostic-class support counts and reference-backbone "
                "feature means; the server aggregates them into a cosine prototype head"
            ),
            "M2_long_tail_prevalence_calibration": (
                "the server adds a centered log-prevalence bias when uploaded class "
                "prevalence indicates a dominant diagnosis"
            ),
        },
        "modality": meta.get("dataset"),
        "num_clients": int(trace.get("num_clients", _num_clients(meta, client_stats))),
        "base_input_weights": base_input_weights,
        "reference_prototype_head": trace,
        "long_tail_prevalence_calibration": prevalence_trace,
    }
