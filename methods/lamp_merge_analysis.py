from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.hub import beta_to_dirname
from utils.runtime import build_reference_bundle
from utils.state_dict import average_state_dicts


EPS = 1e-8

LAMP_MERGE_ABLATION_MODES = {
    "full",
    "m1_only",
    "avg_m2",
    "no_prevalence",
    "prototype_head_agg",
    "head_agg",
    "global_feature_mean",
    "support_only",
    "prototype_shuffle",
    "uniform_client_weight",
    "binary_support",
    "global_client_size_weight",
    "uniform_prevalence",
    "smoothed_prevalence",
}

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
        if name in cfg and cfg.get(name) is not None:
            return cfg.get(name)
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


def _normalize_ablation_mode(mode):
    mode = str(mode or "full").strip().lower()
    aliases = {
        "prototype_head": "prototype_head_agg",
        "classifier_head": "prototype_head_agg",
        "classifier_head_agg": "prototype_head_agg",
        "head_agg": "prototype_head_agg",
        "head_aggregation": "prototype_head_agg",
        "shuffled_prototype": "prototype_shuffle",
        "shuffle": "prototype_shuffle",
        "global_mean": "global_feature_mean",
        "synthetic_support": "support_only",
        "support_synthetic": "support_only",
        "uniform_weight": "uniform_client_weight",
        "client_size_weight": "global_client_size_weight",
        "disable_prevalence": "no_prevalence",
    }
    return aliases.get(mode, mode)


def _ablation_components(mode):
    mode = _normalize_ablation_mode(mode)
    if mode in {"m1_only", "no_prevalence"}:
        return {
            "prototype_mode": "reference_prototype",
            "evidence_mode": "support_power",
            "prevalence_mode": "disabled",
            "shuffle_prototypes": False,
        }
    if mode == "avg_m2":
        return {
            "prototype_mode": "average_checkpoint",
            "evidence_mode": "support_power",
            "prevalence_mode": "uploaded",
            "shuffle_prototypes": False,
        }
    if mode == "prototype_head_agg":
        return {
            "prototype_mode": "classifier_head_aggregation",
            "evidence_mode": "support_power",
            "prevalence_mode": "uploaded",
            "shuffle_prototypes": False,
        }
    if mode == "global_feature_mean":
        return {
            "prototype_mode": "global_feature_mean",
            "evidence_mode": "support_power",
            "prevalence_mode": "uploaded",
            "shuffle_prototypes": False,
        }
    if mode == "support_only":
        return {
            "prototype_mode": "support_only_synthetic",
            "evidence_mode": "support_power",
            "prevalence_mode": "uploaded",
            "shuffle_prototypes": False,
        }
    if mode == "prototype_shuffle":
        return {
            "prototype_mode": "reference_prototype",
            "evidence_mode": "support_power",
            "prevalence_mode": "uploaded",
            "shuffle_prototypes": True,
        }
    if mode == "uniform_client_weight":
        return {
            "prototype_mode": "reference_prototype",
            "evidence_mode": "uniform_present_client",
            "prevalence_mode": "uploaded",
            "shuffle_prototypes": False,
        }
    if mode == "binary_support":
        return {
            "prototype_mode": "reference_prototype",
            "evidence_mode": "uniform_present_client",
            "prevalence_mode": "binary_support",
            "shuffle_prototypes": False,
        }
    if mode == "global_client_size_weight":
        return {
            "prototype_mode": "reference_prototype",
            "evidence_mode": "global_client_size",
            "prevalence_mode": "uploaded",
            "shuffle_prototypes": False,
        }
    if mode == "uniform_prevalence":
        return {
            "prototype_mode": "reference_prototype",
            "evidence_mode": "support_power",
            "prevalence_mode": "uniform",
            "shuffle_prototypes": False,
        }
    if mode == "smoothed_prevalence":
        return {
            "prototype_mode": "reference_prototype",
            "evidence_mode": "support_power",
            "prevalence_mode": "smoothed",
            "shuffle_prototypes": False,
        }
    if mode == "full":
        return {
            "prototype_mode": "reference_prototype",
            "evidence_mode": "support_power",
            "prevalence_mode": "uploaded",
            "shuffle_prototypes": False,
        }
    raise ValueError(f"Unsupported LAMP-Merge ablation mode: {mode}")


def _prototype_stats_path(meta, cfg):
    direct = (
        cfg.get("lamp_merge_prototype_stats_path")
        or cfg.get("lamp_merge_client_prototype_path")
        or cfg.get("lamp_merge_proto_stats_path")
    )
    if direct:
        path = Path(direct)
        return path if path.exists() else None

    root = (
        cfg.get("lamp_merge_prototype_root")
        or cfg.get("lamp_merge_client_prototype_root")
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
        raise ValueError(f"Invalid LAMP-Merge prototype stats payload: {path}")
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


def _is_valid_prevalence_source(source):
    source = str(source or "")
    return source.startswith("client_local_dataset") or source.startswith("client_uploaded_label_counts")


def _validate_prevalence_provenance(proto_stats):
    payload_source = proto_stats.get("prevalence_source")
    client_sources = [
        item.get("prevalence_source")
        for item in proto_stats.get("clients", [])
        if isinstance(item, dict) and item.get("prevalence_source") is not None
    ]
    if _is_valid_prevalence_source(payload_source):
        return
    if client_sources and all(_is_valid_prevalence_source(source) for source in client_sources):
        return
    raise ValueError(
        "LAMP-Merge analysis modes that use prevalence counts require class_prevalence_counts "
        "uploaded by clients from their local D_i. The prototype statistics payload does not "
        "declare compatible prevalence provenance."
    )


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


def _classifier_weight_means(state_dicts, proto_stats, meta):
    num_classes = int(meta["num_classes"])
    num_clients = _num_clients(meta, proto_stats)
    rows = []
    for idx in range(num_clients):
        if idx >= len(state_dicts):
            raise ValueError(f"Missing state_dict for client {idx}.")
        weight_key, _ = _classifier_pair(state_dicts[idx], num_classes)
        if weight_key is None:
            raise ValueError(f"Could not locate classifier weight tensor for client {idx}.")
        weight = state_dicts[idx][weight_key].detach().cpu().float()
        if weight.ndim != 2 or int(weight.shape[0]) != num_classes:
            raise ValueError(f"Invalid classifier weight for client {idx}: {list(weight.shape)}.")
        rows.append(weight)
    return torch.stack(rows, dim=0)


def _global_feature_mean_proxy(means, feature_counts):
    support = feature_counts.clamp_min(0.0)
    total = support.sum(dim=1).clamp_min(EPS)
    client_global = (means * support.unsqueeze(-1)).sum(dim=1) / total.view(-1, 1)
    global_mean = (client_global * total.view(-1, 1)).sum(dim=0) / total.sum().clamp_min(EPS)
    return global_mean.view(1, 1, -1).expand(means.shape[0], means.shape[1], means.shape[2]).clone()


def _support_only_proxy(num_clients, num_classes, feature_dim, cfg):
    seed = int(_cfg_value(cfg, "lamp_merge_ablation_seed", default=1701))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    directions = torch.randn(num_classes, feature_dim, generator=generator, dtype=torch.float32)
    directions = F.normalize(directions, dim=1)
    return directions.view(1, num_classes, feature_dim).expand(num_clients, num_classes, feature_dim).clone()


def _evidence_matrix(feature_counts, proto_stats, meta, cfg, evidence_mode):
    if evidence_mode == "support_power":
        gamma = float(_cfg_value(cfg, "lamp_merge_proto_count_power", default=0.55))
        evidence = (feature_counts + 1.0).pow(gamma)
        return evidence * (feature_counts > 0).to(evidence.dtype), gamma
    if evidence_mode in {"uniform_present_client", "binary_support"}:
        return (feature_counts > 0).to(torch.float32), None
    if evidence_mode == "global_client_size":
        num_classes = int(meta["num_classes"])
        num_clients = _num_clients(meta, proto_stats)
        sample_sizes = []
        clients = proto_stats.get("clients", [])
        meta_clients = meta.get("clients", [])
        for idx in range(num_clients):
            size = 0.0
            if idx < len(meta_clients):
                size = float(meta_clients[idx].get("num_samples", 0) or 0)
            if size <= 0.0 and idx < len(clients):
                item = clients[idx]
                size = float(item.get("num_selected_samples", 0) or 0)
                if size <= 0.0:
                    size = float(torch.as_tensor(item.get("class_prevalence_counts", []), dtype=torch.float32).sum().item())
                if size <= 0.0:
                    size = float(feature_counts[idx].sum().item())
            sample_sizes.append(size)
        client_size = torch.as_tensor(sample_sizes, dtype=torch.float32).view(num_clients, 1)
        evidence = client_size.expand(num_clients, num_classes).clone()
        return evidence * (feature_counts > 0).to(evidence.dtype), None
    raise ValueError(f"Unsupported LAMP-Merge evidence mode: {evidence_mode}")


def _prevalence_counts(proto_stats, feature_counts, meta, cfg, prevalence_mode):
    num_classes = int(meta["num_classes"])
    num_clients = _num_clients(meta, proto_stats)
    prevalence_source = "uploaded_prevalence_counts"
    if prevalence_mode == "disabled":
        return torch.ones(num_clients, num_classes, dtype=torch.float32), "disabled"
    if prevalence_mode == "uniform":
        return torch.ones(num_clients, num_classes, dtype=torch.float32), "uniform_prior_control"
    if prevalence_mode == "binary_support":
        return (feature_counts > 0).to(torch.float32), "binary_support_control"

    if prevalence_mode in {"uploaded", "smoothed"}:
        _validate_prevalence_provenance(proto_stats)

    counts = _client_stat_matrix(
        proto_stats,
        ("class_prevalence_counts",),
        num_clients,
        num_classes,
    )
    missing_clients = torch.nonzero(counts.sum(dim=1) <= 0, as_tuple=False).view(-1).tolist()
    if missing_clients:
        raise ValueError(f"Missing uploaded class_prevalence_counts for clients: {missing_clients}")
    if prevalence_mode == "smoothed":
        smoothing = float(_cfg_value(
            cfg,
            "lamp_merge_prevalence_smoothing",
            default=1.0,
        ))
        counts = counts + smoothing
        prevalence_source = f"uploaded_prevalence_counts_plus_{smoothing:g}"
    elif prevalence_mode != "uploaded":
        raise ValueError(f"Unsupported LAMP-Merge prevalence mode: {prevalence_mode}")

    if float(counts.sum().item()) <= 0:
        counts = torch.zeros_like(feature_counts)
        prevalence_source = "missing"
    return counts, prevalence_source


def _global_prototypes(proto_stats, meta, cfg, state_dicts=None):
    num_classes = int(meta["num_classes"])
    num_clients = _num_clients(meta, proto_stats)
    if num_clients <= 0:
        raise ValueError("LAMP-Merge requires at least one uploaded client prototype payload.")

    components = _ablation_components(_cfg_value(cfg, "lamp_merge_ablation_mode", default="full"))
    prototype_mode = str(_cfg_value(
        cfg,
        "lamp_merge_prototype_mode",
        default=components["prototype_mode"],
    ))
    evidence_mode = str(_cfg_value(
        cfg,
        "lamp_merge_evidence_mode",
        default=components["evidence_mode"],
    ))
    prevalence_mode = str(_cfg_value(
        cfg,
        "lamp_merge_prevalence_mode",
        default=components["prevalence_mode"],
    ))
    shuffle_prototypes = bool(components.get("shuffle_prototypes", False))

    means = _client_feature_means(proto_stats, num_clients, num_classes)
    feature_counts = _client_stat_matrix(
        proto_stats,
        ("class_feature_counts", "class_counts"),
        num_clients,
        num_classes,
    )
    if float(feature_counts.sum().item()) <= 0:
        raise ValueError("Uploaded prototype statistics contain no class support counts.")

    if prototype_mode == "reference_prototype":
        prototype_inputs = means
    elif prototype_mode == "classifier_head_aggregation":
        if state_dicts is None:
            raise ValueError("Classifier-head aggregation ablation requires client state_dicts.")
        prototype_inputs = _classifier_weight_means(state_dicts, proto_stats, meta)
    elif prototype_mode == "global_feature_mean":
        prototype_inputs = _global_feature_mean_proxy(means, feature_counts)
    elif prototype_mode == "support_only_synthetic":
        prototype_inputs = _support_only_proxy(num_clients, num_classes, int(means.shape[-1]), cfg)
    else:
        raise ValueError(f"Unsupported LAMP-Merge prototype mode: {prototype_mode}")

    evidence, gamma = _evidence_matrix(feature_counts, proto_stats, meta, cfg, evidence_mode)
    evidence_per_class = evidence.sum(dim=0)
    missing_classes = torch.nonzero(evidence_per_class <= 0, as_tuple=False).view(-1).tolist()
    if missing_classes:
        raise ValueError(f"Uploaded prototypes do not cover classes: {missing_classes}")

    weights = evidence / evidence_per_class.view(1, -1).clamp_min(EPS)
    prototypes = (prototype_inputs * weights.unsqueeze(-1)).sum(dim=0)
    if shuffle_prototypes:
        seed = int(_cfg_value(cfg, "lamp_merge_ablation_seed", default=1701))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + int(meta.get("seed", 0)) + int(num_classes))
        permutation = torch.randperm(num_classes, generator=generator)
        prototypes = prototypes[permutation]
    else:
        permutation = torch.arange(num_classes)

    prevalence_counts, prevalence_source = _prevalence_counts(proto_stats, feature_counts, meta, cfg, prevalence_mode)

    return {
        "prototypes": prototypes,
        "class_counts": prevalence_counts.sum(dim=0).clamp_min(0.0),
        "class_prevalence_source": prevalence_source,
        "prototype_class_counts": feature_counts.sum(dim=0).clamp_min(0.0),
        "prototype_weights": weights,
        "prototype_mode": prototype_mode,
        "evidence_mode": evidence_mode,
        "prevalence_mode": prevalence_mode,
        "evidence_gamma": gamma,
        "prototype_shuffle_permutation": [int(x) for x in permutation.tolist()],
        "num_clients": num_clients,
    }


def _dominant_prior_threshold(cfg, num_classes):
    imbalance_threshold = float(_cfg_value(
        cfg,
        "lamp_merge_reference_prior_threshold",
        default=2.5,
    ))
    if imbalance_threshold > 0:
        return imbalance_threshold / float(num_classes)
    prevalence_threshold = _cfg_value(cfg, "lamp_merge_prevalence_threshold")
    if prevalence_threshold is not None:
        return float(prevalence_threshold)
    return 0.5


def _prevalence_calibration_strength(class_prior, num_classes, cfg):
    if bool(_cfg_value(
        cfg,
        "lamp_merge_disable_prevalence_calibration",
        default=False,
    )):
        return 0.0
    if float(class_prior.sum().item()) <= 0:
        return 0.0
    threshold = _dominant_prior_threshold(cfg, num_classes)
    dominant_prior = float(class_prior.max().item())
    if dominant_prior <= threshold:
        return 0.0
    max_tau = float(_cfg_value(
        cfg,
        "lamp_merge_reference_prior_max_tau",
        default=4.25,
    ))
    explicit = _cfg_value(cfg, "lamp_merge_reference_prior_tau")
    if explicit is not None:
        explicit_tau = float(explicit)
        if explicit_tau >= 0.0:
            return explicit_tau
    return max_tau


def _centered_log_prior_bias(class_prior, strength):
    centered_log_prior = torch.log(class_prior.clamp_min(EPS))
    centered_log_prior = centered_log_prior - centered_log_prior.mean()
    return float(strength) * centered_log_prior


def _synthesize_reference_prototype_model(base_state, proto_stats, meta, cfg, state_dicts=None):
    proto = _global_prototypes(proto_stats, meta, cfg, state_dicts=state_dicts)
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
        default=18.75,
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
        "prototype_mode": proto["prototype_mode"],
        "evidence_mode": proto["evidence_mode"],
        "prevalence_mode": proto["prevalence_mode"],
        "evidence_gamma": proto["evidence_gamma"],
        "prototype_shuffle_permutation": proto["prototype_shuffle_permutation"],
        "head_scale": scale,
        "prevalence_bias_scale": prevalence_strength,
        "prevalence_threshold": _dominant_prior_threshold(cfg, num_classes),
        "reference_prior_threshold": float(_cfg_value(
            cfg,
            "lamp_merge_reference_prior_threshold",
            default=2.5,
        )),
        "reference_prior_max_tau": float(_cfg_value(
            cfg,
            "lamp_merge_reference_prior_max_tau",
            default=4.25,
        )),
        "imbalance_ratio": imbalance_ratio,
        "num_clients": int(proto["num_clients"]),
        "class_counts": [float(x) for x in class_counts.tolist()],
        "class_prevalence_source": proto["class_prevalence_source"],
        "class_prior": [float(x) for x in class_prior.tolist()],
        "prototype_class_counts": [float(x) for x in proto["prototype_class_counts"].tolist()],
        "prototype_weights": [[float(v) for v in row] for row in proto["prototype_weights"].tolist()],
    }


def _avg_plus_prevalence_model(state_dicts, weights, proto_stats, meta, cfg):
    num_classes = int(meta["num_classes"])
    averaged, normalized_weights = average_state_dicts(state_dicts, weights)
    state = OrderedDict((key, value.detach().cpu().clone()) for key, value in averaged.items())
    local_cfg = dict(cfg)
    local_cfg["lamp_merge_ablation_mode"] = "full"
    proto = _global_prototypes(proto_stats, meta, local_cfg)
    class_counts = proto["class_counts"]
    class_prior = class_counts / class_counts.sum().clamp_min(EPS)
    prevalence_strength = _prevalence_calibration_strength(class_prior, num_classes, cfg)
    imbalance_ratio = float((class_prior.max() * float(num_classes)).item())

    _weight_key, bias_key = _classifier_pair(state, num_classes)
    if bias_key is None:
        raise ValueError("avg+M2 ablation requires a classifier bias tensor.")
    if prevalence_strength != 0.0:
        bias = state[bias_key].detach().cpu().float()
        bias = bias + _centered_log_prior_bias(class_prior, prevalence_strength)
        state[bias_key] = bias.to(dtype=state[bias_key].dtype)

    return state, {
        "used": prevalence_strength != 0.0,
        "bias_key": bias_key,
        "prevalence_bias_scale": prevalence_strength,
        "prevalence_threshold": _dominant_prior_threshold(cfg, num_classes),
        "reference_prior_threshold": float(_cfg_value(
            cfg,
            "lamp_merge_reference_prior_threshold",
            default=2.5,
        )),
        "reference_prior_max_tau": float(_cfg_value(
            cfg,
            "lamp_merge_reference_prior_max_tau",
            default=4.25,
        )),
        "imbalance_ratio": imbalance_ratio,
        "class_counts": [float(x) for x in class_counts.tolist()],
        "class_prevalence_source": proto["class_prevalence_source"],
        "class_prior": [float(x) for x in class_prior.tolist()],
        "avg_weights": [float(x) for x in normalized_weights],
    }


def merge_lamp_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    if meta is None or cfg is None:
        raise ValueError("LAMP-Merge requires task metadata and runtime config.")
    if not _is_medical_image_task(meta):
        raise ValueError(f"LAMP-Merge is defined only for medical image tasks, got dataset={meta.get('dataset')}.")

    cfg = cfg or {}
    client_stats, client_stats_path = _load_prototype_stats(meta, cfg)
    if not _is_reference_prototype_stats(client_stats):
        raise ValueError(
            "LAMP-Merge requires client-side reference prototype statistics. "
            "Please export prototype_stats.pt before running this method."
        )

    ablation_mode = _normalize_ablation_mode(_cfg_value(cfg, "lamp_merge_ablation_mode", default="full"))
    if ablation_mode not in LAMP_MERGE_ABLATION_MODES:
        raise ValueError(f"Unsupported LAMP-Merge ablation mode: {ablation_mode}")

    if ablation_mode == "avg_m2":
        merged, prevalence_trace = _avg_plus_prevalence_model(state_dicts, weights, client_stats, meta, cfg)
        trace = {
            "used": False,
            "reason": "M1 diagnostic prototype head is disabled by avg_m2 ablation.",
        }
    else:
        base_state, _ = build_reference_bundle(meta, device="cpu")
        local_cfg = dict(cfg)
        local_cfg["lamp_merge_ablation_mode"] = ablation_mode
        if ablation_mode in {"m1_only", "no_prevalence"}:
            local_cfg["lamp_merge_disable_prevalence_calibration"] = True
        merged, trace = _synthesize_reference_prototype_model(
            base_state,
            client_stats,
            meta,
            local_cfg,
            state_dicts=state_dicts,
        )
        prevalence_trace = {
            "used": bool(float(trace.get("prevalence_bias_scale", 0.0)) != 0.0),
            "prevalence_bias_scale": float(trace.get("prevalence_bias_scale", 0.0)),
            "imbalance_ratio": float(trace.get("imbalance_ratio", 0.0)),
            "class_prior": trace.get("class_prior", []),
        }
    trace["client_stats_path"] = client_stats_path

    base_input_weights = []
    if weights is not None:
        base_input_weights = [float(x) for x in torch.as_tensor(weights, dtype=torch.float32).view(-1).tolist()]

    return merged, {
        "method_name": "LAMP-Merge",
        "implementation": "lamp_merge_v1",
        "ablation_mode": ablation_mode,
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
