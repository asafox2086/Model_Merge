from collections import OrderedDict
import torch
import torch.nn.functional as F

from utils.runtime import build_reference_bundle, build_runtime
from utils.state_dict import average_state_dicts


EPS = 1e-8
HEAD_SCALE_DEFAULT = 0.08
DEFAULT_STATS_MAX_BATCHES = 4
DEFAULT_EVAL_MAX_BATCHES = 2
DEFAULT_BN_BATCHES = 4
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

MEDICAL_IMAGE_DATASETS = {
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
}

MODULE1_COMPONENTS = {
    "image_space",
    "diagnostic_client_information",
    "diagnostic_evidence",
    "class_rarity",
    "focal_weight",
    "domain_focus",
}

MODULE2_COMPONENTS = {
    "balanced_selection",
    "layerwise_merge",
    "sparse_residual",
    "morph_anchor_candidate",
    "specialist_candidate",
    "reference_delta_candidate",
    "prototype_head_candidate",
    "consensus_candidate",
    "candidate_selection",
    "bn_recalibration",
    "head_temperature",
}

ABLATION_PRESETS = {
    "full": set(),
    "no_diagnostic_evidence": {"diagnostic_evidence", "domain_focus"},
    "no_domain_preprocess": set(MODULE1_COMPONENTS),
    "no_modality_features": set(MODULE1_COMPONENTS),
    "no_medical_preprocess": set(MODULE1_COMPONENTS),
    "no_medical_prior": set(MODULE1_COMPONENTS),
    "no_client_information": set(MODULE1_COMPONENTS),
    "no_diagnostic_information": set(MODULE1_COMPONENTS),
    "no_fusion_selection": set(MODULE2_COMPONENTS),
    "no_medical_fusion_selection": set(MODULE2_COMPONENTS),
    "no_rarity": {"class_rarity"},
    "no_focal": {"focal_weight"},
    "no_domain_focus": {"domain_focus"},
    "no_balanced_selection": {"balanced_selection"},
    "no_layerwise": {"layerwise_merge"},
    "no_residual": {"sparse_residual"},
    "no_candidate_bank": {
        "morph_anchor_candidate",
        "specialist_candidate",
        "reference_delta_candidate",
        "prototype_head_candidate",
        "consensus_candidate",
    },
    "no_specialist": {"specialist_candidate"},
    "no_reference_delta": {"reference_delta_candidate"},
    "no_prototype": {"prototype_head_candidate"},
    "no_calibration": {"bn_recalibration", "head_temperature"},
    "avg_only": {
        *MODULE1_COMPONENTS,
        *MODULE2_COMPONENTS,
    },
}

COMPONENT_ALIASES = {
    "clip_denorm": "image_space",
    "vlm_denorm": "image_space",
}

COMPONENT_GROUP_ALIASES = {
    "modality_features": set(MODULE1_COMPONENTS),
    "medical_preprocess": set(MODULE1_COMPONENTS),
    "medical_prior": set(MODULE1_COMPONENTS),
    "domain_features": set(MODULE1_COMPONENTS),
}

METHOD_MODULES = {
    "module_1": {
        "name": "Diagnostic Evidence Client Information Estimation",
        "purpose": "Estimate medically useful client information from generic image evidence, class rarity, hard cases, and prediction margins.",
    },
    "module_2": {
        "name": "Medical Evidence Guided Fusion and Selection",
        "purpose": "Fuse layers, classifier rows, and candidates using diagnostic client information rather than only sample counts.",
    },
}


def _normalize_scores(values, fallback):
    scores = torch.as_tensor(values, dtype=torch.float32)
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = torch.clamp(scores, min=0.0)
    total = float(scores.sum().item())
    if total <= 0.0:
        return torch.as_tensor(fallback, dtype=torch.float32)
    return scores / total


def _blend_scores(primary, secondary, blend=0.7):
    primary = torch.as_tensor(primary, dtype=torch.float32)
    secondary = torch.as_tensor(secondary, dtype=torch.float32)
    mixed = blend * primary + (1.0 - blend) * secondary
    return _normalize_scores(mixed, fallback=secondary)


def _is_small_med_task(meta):
    return meta.get("task_type") == "small"


def _is_medical_image_task(meta):
    return meta.get("task_type") in {"small", "vlm"} and meta.get("dataset") in MEDICAL_IMAGE_DATASETS


def _model_family(meta):
    if meta.get("task_type") == "vlm":
        return "vlm"
    model = meta.get("model", "")
    if model in {"resnet", "mobilenet", "convnext"}:
        return "cnn"
    if model in {"vit_t", "swin_tiny"}:
        return "transformer"
    return "generic"


def _split_tokens(value):
    if value in (None, "", False):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_tokens = value
    else:
        raw_tokens = str(value).replace(";", ",").split(",")
    return [str(token).strip() for token in raw_tokens if str(token).strip()]


def _parse_bool(value, default=True):
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _ablation_labels(cfg):
    cfg = cfg or {}
    labels = _split_tokens(cfg.get("my_merge_ablation", "full"))
    if not labels:
        labels = ["full"]
    labels.extend(_split_tokens(cfg.get("my_merge_disable", "")))
    return labels


def _disabled_components(cfg):
    disabled = set()

    def add_component(name):
        if name in COMPONENT_GROUP_ALIASES:
            disabled.update(COMPONENT_GROUP_ALIASES[name])
        else:
            disabled.add(COMPONENT_ALIASES.get(name, name))

    for label in _ablation_labels(cfg):
        normalized = label.strip().lower().replace("-", "_")
        if normalized in ABLATION_PRESETS:
            disabled.update(ABLATION_PRESETS[normalized])
        elif normalized in {"candidate_bank", "candidate_pool"}:
            disabled.update(ABLATION_PRESETS["no_candidate_bank"])
        elif normalized.startswith("no_"):
            add_component(normalized[3:])
        elif normalized.startswith("without_"):
            add_component(normalized[8:])
        elif normalized != "full":
            add_component(normalized)
    return disabled


def _component_enabled(cfg, name, default=True):
    cfg = cfg or {}
    explicit_key = f"my_merge_use_{name}"
    if explicit_key in cfg:
        return _parse_bool(cfg.get(explicit_key), default=default)
    return default and name not in _disabled_components(cfg)


def _ablation_config(cfg):
    components = [
        "image_space",
        "diagnostic_evidence",
        "diagnostic_client_information",
        "class_rarity",
        "focal_weight",
        "domain_focus",
        "balanced_selection",
        "layerwise_merge",
        "sparse_residual",
        "morph_anchor_candidate",
        "specialist_candidate",
        "reference_delta_candidate",
        "prototype_head_candidate",
        "consensus_candidate",
        "candidate_selection",
        "bn_recalibration",
        "head_temperature",
    ]
    return {
        "labels": _ablation_labels(cfg),
        "disabled_components": sorted(_disabled_components(cfg)),
        "enabled": {name: _component_enabled(cfg, name) for name in components},
    }


def _spatial_eccentricity(mask):
    side_h, side_w = mask.shape[-2], mask.shape[-1]
    xs = torch.linspace(-1.0, 1.0, side_w, device=mask.device)
    ys = torch.linspace(-1.0, 1.0, side_h, device=mask.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    weight = mask.float()
    mass = weight.sum(dim=(1, 2)) + EPS
    mx = (weight * xx).sum(dim=(1, 2)) / mass
    my = (weight * yy).sum(dim=(1, 2)) / mass
    dx = xx - mx[:, None, None]
    dy = yy - my[:, None, None]
    cov_xx = (weight * dx.square()).sum(dim=(1, 2)) / mass
    cov_yy = (weight * dy.square()).sum(dim=(1, 2)) / mass
    cov_xy = (weight * dx * dy).sum(dim=(1, 2)) / mass
    trace = cov_xx + cov_yy
    det_term = torch.sqrt(torch.clamp((cov_xx - cov_yy).square() + 4.0 * cov_xy.square(), min=0.0))
    eig_1 = 0.5 * (trace + det_term)
    eig_2 = 0.5 * (trace - det_term)
    return eig_2 / (eig_1 + EPS)


def _sobel_edges(gray):
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=gray.device,
    ).view(1, 1, 3, 3)
    gray_map = gray.unsqueeze(1)
    grad_x = F.conv2d(gray_map, sobel_x, padding=1)
    grad_y = F.conv2d(gray_map, sobel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + EPS).squeeze(1)


def _image_space01(meta, x, cfg=None):
    image = x.detach().float()
    if _component_enabled(cfg, "image_space") and meta.get("task_type") == "vlm" and image.shape[1] >= 3:
        mean = torch.tensor(CLIP_IMAGE_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_IMAGE_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        image = image[:, :3] * std + mean
    return torch.clamp(image, min=0.0, max=1.0)


def _local_variance(gray, kernel=9):
    gray_map = gray.unsqueeze(1)
    mean = F.avg_pool2d(gray_map, kernel_size=kernel, stride=1, padding=kernel // 2)
    mean_sq = F.avg_pool2d(gray_map.square(), kernel_size=kernel, stride=1, padding=kernel // 2)
    return (mean_sq - mean.square()).clamp_min(0.0).squeeze(1)


def _safe_quantile(values, q):
    return torch.quantile(values.flatten(1), q=q, dim=1, keepdim=True).view(-1, 1, 1)


def _robust_rescale01(gray, low_q=0.04, high_q=0.96):
    low = _safe_quantile(gray, q=low_q)
    high = _safe_quantile(gray, q=high_q)
    return torch.clamp((gray - low) / (high - low + EPS), min=0.0, max=1.0)


def _generic_medical_features(x):
    gray = x[:, 0] if x.shape[1] == 1 else x.mean(dim=1)
    gray = _robust_rescale01(gray, low_q=0.04, high_q=0.96)
    edge_map = _sobel_edges(gray)
    local_mean = F.avg_pool2d(gray.unsqueeze(1), kernel_size=9, stride=1, padding=4).squeeze(1)
    local_contrast_map = (gray - local_mean).abs()
    texture_map = torch.sqrt(_local_variance(gray, kernel=7) + EPS)
    evidence_map = 0.42 * edge_map + 0.34 * local_contrast_map + 0.24 * texture_map
    fg_mask = evidence_map >= _safe_quantile(evidence_map, q=0.70)
    mask = fg_mask.float()
    mass = mask.sum(dim=(1, 2)) + EPS

    area_ratio = mask.mean(dim=(1, 2))
    boundary_strength = (edge_map * mask).sum(dim=(1, 2)) / mass
    local_contrast = (local_contrast_map * mask).sum(dim=(1, 2)) / mass
    texture_heterogeneity = (texture_map * mask).sum(dim=(1, 2)) / mass
    shape_compactness = _spatial_eccentricity(fg_mask)
    diagnostic_salience = (
        (0.35 + boundary_strength)
        * (0.35 + local_contrast)
        * (0.35 + texture_heterogeneity)
        * (0.65 + area_ratio)
        * (0.75 + shape_compactness)
    )
    return torch.stack(
        [
            area_ratio,
            boundary_strength,
            local_contrast,
            texture_heterogeneity,
            shape_compactness,
            diagnostic_salience,
        ],
        dim=1,
    )


def _neutral_medical_features(x):
    return torch.ones((x.shape[0], 6), dtype=x.dtype, device=x.device)


def _batch_morphology_features(meta, x, cfg=None):
    x = _image_space01(meta, x, cfg=cfg)
    if not _component_enabled(cfg, "diagnostic_evidence"):
        return _neutral_medical_features(x)
    return _generic_medical_features(x)


def _sample_importance(features):
    importance = features[:, -1]
    importance = importance / (importance.mean() + EPS)
    return torch.clamp(importance, min=0.25, max=3.5)


def _class_rarity_weights(labels, num_classes, meta, cfg=None):
    if not _component_enabled(cfg, "class_rarity"):
        return torch.ones(num_classes, dtype=torch.float32, device=labels.device)
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    inv_sqrt = torch.sqrt(counts.sum() / counts)
    inv_sqrt = inv_sqrt / (inv_sqrt.mean() + EPS)
    strength = 0.50
    rarity = 1.0 + strength * (inv_sqrt - 1.0)
    return torch.clamp(rarity, min=0.55, max=3.0)


def _medical_sample_weights(meta, features, labels, num_classes, cfg=None):
    morphology = _sample_importance(features)
    rarity = _class_rarity_weights(labels, num_classes, meta, cfg=cfg)[labels]
    if not _component_enabled(cfg, "domain_focus") or not _component_enabled(cfg, "diagnostic_evidence"):
        domain_focus = torch.ones_like(morphology)
    else:
        focus = 0.45 * features[:, 1] + 0.35 * features[:, 2] + 0.20 * features[:, 3]
        focus = focus / (focus.mean() + EPS)
        domain_focus = torch.clamp(focus, min=0.65, max=1.85)
    weights = morphology * rarity * domain_focus
    weights = weights / (weights.mean() + EPS)
    return torch.clamp(weights, min=0.20, max=5.0)


def _feature_names(meta):
    return [
        "foreground_area",
        "boundary_strength",
        "local_contrast",
        "texture_heterogeneity",
        "shape_compactness",
        "diagnostic_salience",
    ]


def _resolve_max_batches(meta, cfg, cfg_key, default_value):
    raw = cfg.get(cfg_key, None)
    if raw not in (None, ""):
        return int(raw)
    family = _model_family(meta)
    if family in {"transformer", "vlm"}:
        return 0
    return int(default_value)


def _collect_split_batches(meta, cfg, split):
    runtime = build_runtime(
        meta=meta,
        data_root=cfg["data_root"],
        split=split,
        batch_size=int(cfg.get("stats_batch_size", 32) or cfg.get("batch_size", 64)),
        num_workers=int(cfg.get("stats_num_workers", cfg.get("num_workers", 4))),
        device=torch.device("cpu"),
    )
    batches = []
    feature_chunks = []
    labels = []
    max_batches = _resolve_max_batches(meta, cfg, "my_merge_stats_max_batches", DEFAULT_STATS_MAX_BATCHES)
    for batch_idx, (x, y) in enumerate(runtime["loader"]):
        if max_batches and batch_idx >= max_batches:
            break
        batches.append((x.clone(), y.clone()))
        feature_chunks.append(_batch_morphology_features(meta, x, cfg=cfg).cpu())
        labels.append(y.clone())
    return batches, torch.cat(feature_chunks, dim=0), torch.cat(labels, dim=0)


def _collect_diagnostic_evidence(meta, cfg):
    batches, features, labels = _collect_split_batches(meta, cfg, split=cfg.get("stats_split", "val"))
    names = _feature_names(meta)
    feature_summary = {
        names[idx] if idx < len(names) else f"feature_{idx}": float(features[:, idx].mean().item())
        for idx in range(features.shape[1])
    }
    return {
        "module_name": METHOD_MODULES["module_1"]["name"],
        "batches": batches,
        "features": features,
        "labels": labels,
        "feature_names": names,
        "feature_summary": feature_summary,
    }


def _evaluate_client_predictions(meta, checkpoint, batches, cfg):
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    runtime = build_runtime(
        meta=meta,
        data_root=cfg["data_root"],
        split=cfg.get("stats_split", "val"),
        batch_size=max(1, len(batches[0][1])),
        num_workers=0,
        device=device,
    )
    model = runtime["model"]
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    forward_fn = runtime["forward_fn"]

    logits_out = []
    with torch.no_grad():
        for x, _ in batches:
            x = x.to(device, non_blocking=True)
            logits_out.append(forward_fn(model, x).detach().float().cpu())
    return torch.cat(logits_out, dim=0)


def _extract_pooled_features(model, meta, x):
    if meta.get("task_type") != "small":
        raise ValueError("prototype feature extraction currently supports only task_type=small")
    features = model.forward_features(x)
    if isinstance(features, (tuple, list)):
        features = features[-1]
    if hasattr(model, "forward_head"):
        pooled = model.forward_head(features, pre_logits=True)
    elif torch.is_tensor(features) and features.ndim == 4:
        pooled = features.mean(dim=(2, 3))
    elif torch.is_tensor(features) and features.ndim == 3:
        pooled = features[:, 0]
    else:
        pooled = features
    return pooled.detach().float()


def _extract_state_embeddings(meta, merged_state_dict, batches, cfg):
    if meta.get("task_type") != "small":
        return None
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    runtime = build_runtime(
        meta=meta,
        data_root=cfg["data_root"],
        split=cfg.get("stats_split", "val"),
        batch_size=max(1, len(batches[0][1])),
        num_workers=0,
        device=device,
    )
    model = runtime["model"]
    model.load_state_dict(merged_state_dict, strict=True)
    model.eval()
    pooled_out = []
    logits_out = []
    with torch.no_grad():
        for x, _ in batches:
            x = x.to(device, non_blocking=True)
            pooled = _extract_pooled_features(model, meta, x)
            logits = runtime["forward_fn"](model, x).detach().float()
            pooled_out.append(pooled.cpu())
            logits_out.append(logits.cpu())
    return torch.cat(pooled_out, dim=0), torch.cat(logits_out, dim=0)


def _client_scores(meta, checkpoints, batches, features, labels, cfg, base_weights):
    num_clients = len(checkpoints)
    num_classes = int(meta["num_classes"])
    sample_importance = _medical_sample_weights(meta, features, labels, num_classes, cfg=cfg)
    class_rarity = _class_rarity_weights(labels, num_classes, meta, cfg=cfg)
    hard_mask = sample_importance >= torch.quantile(sample_importance, q=0.60)
    use_focal = _component_enabled(cfg, "focal_weight")

    overall_scores = []
    morph_scores = []
    class_scores = torch.zeros(num_classes, num_clients, dtype=torch.float32)
    client_summaries = []

    for client_idx, checkpoint in enumerate(checkpoints):
        logits = _evaluate_client_predictions(meta, checkpoint, batches, cfg)
        preds = logits.argmax(dim=1)
        correct = (preds == labels).float()

        true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        masked_logits = logits.clone()
        masked_logits[torch.arange(logits.size(0)), labels] = float("-inf")
        other_logits = masked_logits.max(dim=1).values
        margin = torch.sigmoid(true_logits - other_logits)

        overall_acc = float(correct.mean().item())
        morph_acc = float((correct * sample_importance).sum().item() / sample_importance.sum().item())
        hard_acc = float(correct[hard_mask].mean().item()) if torch.any(hard_mask) else overall_acc
        margin_score = float((margin * sample_importance).sum().item() / sample_importance.sum().item())
        if use_focal:
            focal_weight = sample_importance * torch.clamp((1.0 - margin).pow(1.35) + 0.25, min=0.25, max=2.5)
        else:
            focal_weight = sample_importance
        focal_acc = float((correct * focal_weight).sum().item() / (focal_weight.sum().item() + EPS))

        prior = float(base_weights[client_idx])
        overall_score = prior * (0.16 + overall_acc + 0.26 * margin_score + 0.16 * hard_acc + 0.18 * focal_acc)
        morph_score = prior * (0.10 + 0.82 * morph_acc + 0.28 * hard_acc + 0.16 * margin_score + 0.24 * focal_acc)
        overall_scores.append(overall_score)
        morph_scores.append(morph_score)
        client_summaries.append(
            {
                "overall_acc": overall_acc,
                "morph_acc": morph_acc,
                "hard_acc": hard_acc,
                "margin_score": margin_score,
                "focal_acc": focal_acc,
                "overall_score": overall_score,
                "morph_score": morph_score,
            }
        )

        seen_classes = set(meta["clients"][client_idx].get("classes", []))
        for cls_idx in range(num_classes):
            cls_mask = labels == cls_idx
            if not torch.any(cls_mask):
                class_scores[cls_idx, client_idx] = prior
                continue
            cls_correct = correct[cls_mask]
            cls_importance = sample_importance[cls_mask]
            cls_margin = margin[cls_mask]
            cls_acc = float(cls_correct.mean().item())
            cls_morph = float((cls_correct * cls_importance).sum().item() / (cls_importance.sum().item() + EPS))
            cls_conf = float((cls_margin * cls_importance).sum().item() / (cls_importance.sum().item() + EPS))
            seen_bonus = 1.22 if cls_idx in seen_classes else 0.58
            rarity_bonus = float(class_rarity[cls_idx].item())
            class_scores[cls_idx, client_idx] = prior * seen_bonus * rarity_bonus * (
                0.10 + 0.74 * cls_acc + 0.42 * cls_morph + 0.24 * cls_conf
            )

    overall_weights = _normalize_scores(overall_scores, fallback=base_weights)
    morph_weights = _normalize_scores(morph_scores, fallback=base_weights)
    class_weights = [_normalize_scores(class_scores[cls_idx], fallback=morph_weights) for cls_idx in range(num_classes)]
    return overall_weights, morph_weights, torch.stack(class_weights, dim=0), client_summaries


def _estimate_diagnostic_client_information(
    meta,
    checkpoints,
    batches,
    features,
    labels,
    cfg,
    base_weights,
):
    if not _component_enabled(cfg, "diagnostic_client_information"):
        base_tensor = torch.as_tensor(base_weights, dtype=torch.float32)
        num_classes = int(meta["num_classes"])
        class_weights = torch.stack([base_tensor.clone() for _ in range(num_classes)], dim=0)
        client_information = []
        for client_idx, weight in enumerate(base_weights):
            client_meta = meta["clients"][client_idx] if client_idx < len(meta.get("clients", [])) else {}
            client_information.append(
                {
                    "client_index": int(client_idx),
                    "client_name": client_meta.get("checkpoint", f"client_{client_idx}.pt"),
                    "seen_classes": [int(cls) for cls in client_meta.get("classes", [])],
                    "base_weight": float(weight),
                    "overall_weight": float(weight),
                    "morphology_weight": float(weight),
                    "ordinary_accuracy": 0.0,
                    "medical_weighted_accuracy": 0.0,
                    "hard_case_accuracy": 0.0,
                    "margin_confidence": 0.0,
                    "focal_hard_case_accuracy": 0.0,
                    "overall_score": float(weight),
                    "morphology_score": float(weight),
                    "diagnostic_information_vector": [0.0, 0.0, 0.0, 0.0, 0.0],
                }
            )
        return {
            "module_name": METHOD_MODULES["module_1"]["name"],
            "overall_weights": base_tensor,
            "morphology_weights": base_tensor,
            "class_weights": class_weights,
            "client_summaries": [
                {
                    "overall_acc": 0.0,
                    "morph_acc": 0.0,
                    "hard_acc": 0.0,
                    "margin_score": 0.0,
                    "focal_acc": 0.0,
                    "overall_score": float(weight),
                    "morph_score": float(weight),
                }
                for weight in base_weights
            ],
            "client_diagnostic_information": client_information,
            "theory": "Diagnostic client information ablated; fusion falls back to prior client weights.",
        }
    overall_weights, morph_weights, class_weights, client_summaries = _client_scores(
        meta,
        checkpoints,
        batches,
        features,
        labels,
        cfg,
        base_weights,
    )
    client_information = []
    for client_idx, summary in enumerate(client_summaries):
        client_meta = meta["clients"][client_idx] if client_idx < len(meta.get("clients", [])) else {}
        row = {
            "client_index": int(client_idx),
            "client_name": client_meta.get("checkpoint", f"client_{client_idx}.pt"),
            "seen_classes": [int(cls) for cls in client_meta.get("classes", [])],
            "base_weight": float(base_weights[client_idx]),
            "overall_weight": float(overall_weights[client_idx].item()),
            "morphology_weight": float(morph_weights[client_idx].item()),
            "ordinary_accuracy": float(summary["overall_acc"]),
            "medical_weighted_accuracy": float(summary["morph_acc"]),
            "hard_case_accuracy": float(summary["hard_acc"]),
            "margin_confidence": float(summary["margin_score"]),
            "focal_hard_case_accuracy": float(summary["focal_acc"]),
            "overall_score": float(summary["overall_score"]),
            "morphology_score": float(summary["morph_score"]),
        }
        row["diagnostic_information_vector"] = [
            row["ordinary_accuracy"],
            row["medical_weighted_accuracy"],
            row["hard_case_accuracy"],
            row["margin_confidence"],
            row["focal_hard_case_accuracy"],
        ]
        client_information.append(row)
    return {
        "module_name": METHOD_MODULES["module_1"]["name"],
        "overall_weights": overall_weights,
        "morphology_weights": morph_weights,
        "class_weights": class_weights,
        "client_summaries": client_summaries,
        "client_diagnostic_information": client_information,
        "theory": "Each client carries different diagnostic information; fusion weights depend on morphology, hard cases, margins, and class-level expertise rather than only sample counts.",
    }


def _module1_diagnostic_client_information_estimation(meta, checkpoints, cfg, base_weights):
    evidence = _collect_diagnostic_evidence(meta, cfg)
    client_info = _estimate_diagnostic_client_information(
        meta,
        checkpoints,
        evidence["batches"],
        evidence["features"],
        evidence["labels"],
        cfg,
        base_weights,
    )
    client_info.update(
        {
            "module_name": METHOD_MODULES["module_1"]["name"],
            "batches": evidence["batches"],
            "features": evidence["features"],
            "labels": evidence["labels"],
            "feature_names": evidence["feature_names"],
            "feature_summary": evidence["feature_summary"],
            "evidence_theory": (
                "Medical specificity comes from generic image evidence that is shared across medical imaging tasks: "
                "foreground structure, boundaries, local contrast, texture heterogeneity, class rarity, and hard-case margins."
            ),
        }
    )
    return client_info


def _find_classifier_keys(state_dict, num_classes):
    weight_keys = []
    bias_keys = []
    for key, value in state_dict.items():
        if not torch.is_floating_point(value):
            continue
        if value.ndim == 2 and value.shape[0] == num_classes and any(token in key for token in ("fc", "classifier", "head")):
            weight_keys.append(key)
        elif value.ndim == 1 and value.shape[0] == num_classes and any(token in key for token in ("fc", "classifier", "head")):
            bias_keys.append(key)
    weight_key = sorted(weight_keys)[-1] if weight_keys else None
    bias_key = None
    if weight_key is not None:
        prefix = weight_key.rsplit(".", 1)[0]
        for key in bias_keys:
            if key.startswith(prefix):
                bias_key = key
                break
    if bias_key is None and bias_keys:
        bias_key = sorted(bias_keys)[-1]
    return weight_key, bias_key


def _is_classifier_tensor(key, tensor, num_classes):
    if not torch.is_floating_point(tensor):
        return False
    if tensor.ndim not in (1, 2):
        return False
    if tensor.shape[0] != num_classes:
        return False
    return any(token in key for token in ("fc", "classifier", "head", "proj"))


def _param_group(key, meta):
    family = _model_family(meta)
    if family == "cnn":
        early_tokens = (
            "conv1",
            "bn1",
            "layer1",
            "layer2",
            "stem",
            "downsample_layers.0",
            "downsample_layers.1",
            "features.0",
            "features.1",
        )
        late_tokens = ("layer4", "fc", "classifier", "head", "norm", "stages.3", "features.16", "features.17")
    else:
        early_tokens = (
            "patch_embed",
            "embeddings",
            "visual.conv1",
            "visual.class_embedding",
            "visual.positional_embedding",
            "visual.ln_pre",
            "blocks.0",
            "blocks.1",
            "layers.0",
            "stages.0",
            "visual.transformer.resblocks.0",
            "visual.transformer.resblocks.1",
        )
        late_tokens = (
            "head",
            "fc",
            "classifier",
            "norm",
            "visual.ln_post",
            "visual.proj",
            "blocks.10",
            "blocks.11",
            "layers.3",
            "stages.3",
            "visual.transformer.resblocks.10",
            "visual.transformer.resblocks.11",
        )
    if any(token in key for token in early_tokens):
        return "early"
    if any(token in key for token in late_tokens):
        return "late"
    return "mid"


def _merge_profile(meta):
    family = _model_family(meta)
    profile = {
        "early_anchor": 0.76,
        "mid_anchor": 0.60,
        "late_anchor": 0.48,
        "class_power": 3.0,
        "class_topk": 2,
        "candidate_alpha": 0.68,
        "early_residual_keep": 0.12,
        "mid_residual_keep": 0.08,
        "late_residual_keep": 0.04,
        "early_residual_scale": 0.24,
        "mid_residual_scale": 0.17,
        "late_residual_scale": 0.09,
    }
    if family in {"transformer", "vlm"}:
        profile["early_anchor"] -= 0.10
        profile["mid_anchor"] -= 0.06
        profile["late_anchor"] -= 0.04
        profile["candidate_alpha"] = 0.64
        profile["early_residual_keep"] += 0.04
        profile["mid_residual_keep"] += 0.04
        profile["late_residual_keep"] += 0.03
        profile["early_residual_scale"] += 0.05
        profile["mid_residual_scale"] += 0.05
        profile["late_residual_scale"] += 0.04
    return profile


def _weighted_average_for_key(values, weight_tensor):
    out = values[0].detach().clone() * float(weight_tensor[0].item())
    for value, weight in zip(values[1:], weight_tensor[1:]):
        out.add_(value.detach(), alpha=float(weight.item()))
    return out


def _sparse_residual_reinjection(base_value, values, primary_weights, secondary_weights, keep_ratio, scale):
    if keep_ratio <= 0.0 or scale <= 0.0 or values[0].ndim < 2:
        return base_value

    stacked = torch.stack([value.detach().float() for value in values], dim=0)
    primary_idx = int(torch.argmax(primary_weights).item())
    secondary_idx = int(torch.argmax(secondary_weights).item())

    primary_residual = stacked[primary_idx] - stacked.mean(dim=0)
    secondary_residual = stacked[secondary_idx] - stacked.mean(dim=0)
    residual = 0.7 * primary_residual + 0.3 * secondary_residual

    flat_abs = residual.abs().flatten()
    if flat_abs.numel() == 0:
        return base_value
    keep_count = max(1, int(flat_abs.numel() * keep_ratio))
    if keep_count >= flat_abs.numel():
        mask = torch.ones_like(residual, dtype=torch.bool)
    else:
        threshold = torch.topk(flat_abs, k=keep_count, largest=True).values[-1]
        mask = residual.abs() >= threshold

    adjusted = base_value.detach().clone().float()
    adjusted = adjusted + scale * residual * mask.float()
    return adjusted.to(dtype=base_value.dtype)


def _merge_classifier_rows(values, class_weights, profile, fallback_weights):
    template = values[0].detach().clone().float().zero_()
    stacked = torch.stack([value.detach().float() for value in values], dim=0)
    anchor_rows = _weighted_average_for_key(list(stacked), fallback_weights)
    sharpened = torch.stack(
        [_normalize_scores(row.pow(profile["class_power"]), fallback=fallback_weights) for row in class_weights],
        dim=0,
    )
    topk = min(int(profile["class_topk"]), sharpened.shape[1])
    top_weights, top_indices = torch.topk(sharpened, k=topk, dim=1)
    top_weights = torch.softmax(top_weights / 0.42, dim=1)

    if values[0].ndim == 1:
        for cls_idx in range(template.shape[0]):
            row = 0.0
            for rank in range(topk):
                client_idx = int(top_indices[cls_idx, rank].item())
                row = row + top_weights[cls_idx, rank] * stacked[client_idx, cls_idx]
            row = 0.78 * row + 0.22 * anchor_rows[cls_idx]
            anchor_scale = anchor_rows[cls_idx].abs() + EPS
            row_scale = row.abs() + EPS
            row = row * torch.clamp(anchor_scale / row_scale, min=0.45, max=1.15)
            template[cls_idx] = row
        return template.to(dtype=values[0].dtype)

    for cls_idx in range(template.shape[0]):
        row = 0.0
        for rank in range(topk):
            client_idx = int(top_indices[cls_idx, rank].item())
            row = row + top_weights[cls_idx, rank] * stacked[client_idx, cls_idx]
        row = 0.78 * row + 0.22 * anchor_rows[cls_idx]
        anchor_norm = anchor_rows[cls_idx].norm() + EPS
        row_norm = row.norm() + EPS
        row = row * torch.clamp(anchor_norm / row_norm, min=0.45, max=1.15)
        template[cls_idx] = row
    return template.to(dtype=values[0].dtype)


def _build_morphology_anchor_candidate(state_dicts, morph_weights, class_weights, num_classes, meta):
    profile = _merge_profile(meta)
    anchor_idx = int(torch.argmax(morph_weights).item())
    anchor_state = OrderedDict((key, value.detach().clone()) for key, value in state_dicts[anchor_idx].items())
    for key, value in list(anchor_state.items()):
        if _is_classifier_tensor(key, value, num_classes):
            values = [sd[key] for sd in state_dicts]
            anchor_state[key] = _merge_classifier_rows(values, class_weights, profile, morph_weights)
    return anchor_state


def _build_specialist_client_candidate(state_dicts, client_summaries, meta):
    family = _model_family(meta)
    if family in {"transformer", "vlm"}:
        key = lambda item: (
            0.45 * item[1]["morph_acc"] + 0.30 * item[1]["overall_acc"] + 0.25 * item[1]["focal_acc"],
            item[1]["hard_acc"],
        )
    else:
        key = lambda item: (
            0.42 * item[1]["morph_acc"] + 0.34 * item[1]["overall_acc"] + 0.24 * item[1]["focal_acc"],
            item[1]["margin_score"],
        )
    best_idx = max(enumerate(client_summaries), key=key)[0]
    state = OrderedDict((k, v.detach().clone()) for k, v in state_dicts[best_idx].items())
    return state, best_idx


def _build_prototype_head_candidate(base_state_dict, meta, batches, labels, features, cfg):
    if meta.get("task_type") != "small" or _model_family(meta) != "transformer":
        return None

    num_classes = int(meta["num_classes"])
    weight_key, bias_key = _find_classifier_keys(base_state_dict, num_classes)
    if weight_key is None:
        return None

    embedding_result = _extract_state_embeddings(meta, base_state_dict, batches, cfg)
    if embedding_result is None:
        return None
    pooled, _ = embedding_result
    pooled = pooled.float()
    sample_importance = _medical_sample_weights(meta, features, labels, num_classes, cfg=cfg)
    class_weight = base_state_dict[weight_key].detach().clone().float()
    proto_weight = class_weight.clone()

    row_norms = class_weight.norm(dim=1)
    target_norm = float(torch.median(row_norms).item()) if row_norms.numel() else 1.0
    bias_value = base_state_dict[bias_key].detach().clone().float() if bias_key is not None else None
    if bias_value is not None:
        prototype_bias = bias_value.clone()

    for cls_idx in range(num_classes):
        cls_mask = labels == cls_idx
        if not torch.any(cls_mask):
            continue
        cls_feat = pooled[cls_mask]
        cls_imp = sample_importance[cls_mask].view(-1, 1)
        proto = (cls_feat * cls_imp).sum(dim=0) / (cls_imp.sum() + EPS)
        proto = F.normalize(proto, dim=0) * target_norm
        proto_weight[cls_idx] = 0.72 * proto + 0.28 * class_weight[cls_idx]
        if bias_value is not None:
            class_prior = float(cls_mask.float().mean().item())
            prototype_bias[cls_idx] = 0.65 * bias_value[cls_idx] + 0.35 * torch.tensor(
                torch.log(torch.tensor(class_prior + EPS)).item(),
                dtype=bias_value.dtype,
            )

    candidate = OrderedDict((k, v.detach().clone()) for k, v in base_state_dict.items())
    candidate[weight_key] = proto_weight.to(dtype=base_state_dict[weight_key].dtype)
    if bias_key is not None:
        candidate[bias_key] = prototype_bias.to(dtype=base_state_dict[bias_key].dtype)
    return candidate


def _layerwise_merge(state_dicts, overall_weights, morph_weights, class_weights, num_classes, meta, cfg=None):
    profile = _merge_profile(meta)
    anchor_idx = int(torch.argmax(morph_weights).item())
    anchor_one_hot = F.one_hot(torch.tensor(anchor_idx), num_classes=len(state_dicts)).float()
    consensus = _blend_scores(morph_weights, overall_weights, blend=0.58)
    use_layerwise = _component_enabled(cfg, "layerwise_merge")
    use_residual = _component_enabled(cfg, "sparse_residual")

    merged = OrderedDict()
    for key in state_dicts[0].keys():
        values = [sd[key] for sd in state_dicts]
        first = values[0]
        if not torch.is_floating_point(first):
            merged[key] = values[anchor_idx].detach().clone()
            continue
        if _is_classifier_tensor(key, first, num_classes):
            merged[key] = _merge_classifier_rows(values, class_weights, profile, consensus)
            continue

        group = _param_group(key, meta)
        if not use_layerwise:
            weights = consensus
            reinject_keep = 0.0
            reinject_scale = 0.0
        elif group == "early":
            weights = _blend_scores(anchor_one_hot, morph_weights, blend=profile["early_anchor"])
            reinject_keep = profile["early_residual_keep"]
            reinject_scale = profile["early_residual_scale"]
        elif group == "late":
            weights = _blend_scores(anchor_one_hot, overall_weights, blend=profile["late_anchor"])
            reinject_keep = profile["late_residual_keep"]
            reinject_scale = profile["late_residual_scale"]
        else:
            weights = _blend_scores(anchor_one_hot, consensus, blend=profile["mid_anchor"])
            reinject_keep = profile["mid_residual_keep"]
            reinject_scale = profile["mid_residual_scale"]
        if not use_residual:
            reinject_keep = 0.0
            reinject_scale = 0.0
        merged_value = _weighted_average_for_key(values, weights)
        merged[key] = _sparse_residual_reinjection(
            merged_value,
            values,
            primary_weights=morph_weights,
            secondary_weights=overall_weights,
            keep_ratio=reinject_keep,
            scale=reinject_scale,
        )
    return merged


def _apply_head_temperature(merged_state_dict, num_classes, scale):
    adjusted = OrderedDict()
    for key, value in merged_state_dict.items():
        out = value
        if _is_classifier_tensor(key, value, num_classes):
            out = value.detach().clone()
            if torch.is_floating_point(out):
                out.mul_(float(scale))
        adjusted[key] = out
    return adjusted


def _evaluate_merged_state(meta, merged_state_dict, cfg, split):
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    runtime = build_runtime(
        meta=meta,
        data_root=cfg["data_root"],
        split=split,
        batch_size=int(cfg.get("stats_batch_size", 32) or cfg.get("batch_size", 64)),
        num_workers=0,
        device=device,
    )
    model = runtime["model"]
    model.load_state_dict(merged_state_dict, strict=True)
    model.eval()
    forward_fn = runtime["forward_fn"]
    total = 0
    correct = 0.0
    weighted_correct = 0.0
    importance_total = 0.0
    hard_correct = 0.0
    hard_total = 0.0
    max_abs = 0.0
    num_classes = int(meta["num_classes"])
    class_correct = torch.zeros(num_classes, dtype=torch.float32)
    class_total = torch.zeros(num_classes, dtype=torch.float32)
    with torch.no_grad():
        max_batches = _resolve_max_batches(meta, cfg, "my_merge_eval_max_batches", DEFAULT_EVAL_MAX_BATCHES)
        for batch_idx, (x, y) in enumerate(runtime["loader"]):
            if max_batches and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = forward_fn(model, x).detach().float()
            pred = logits.argmax(dim=1)
            hit = (pred == y).float().cpu()
            features = _batch_morphology_features(meta, x.detach().float(), cfg=cfg).cpu()
            y_cpu = y.detach().cpu()
            importance = _medical_sample_weights(meta, features, y_cpu, num_classes, cfg=cfg)
            hard_mask = importance >= torch.quantile(importance, q=0.60)
            correct += float(hit.sum().item())
            weighted_correct += float((hit * importance).sum().item())
            importance_total += float(importance.sum().item())
            hard_correct += float(hit[hard_mask].sum().item()) if torch.any(hard_mask) else 0.0
            hard_total += float(hard_mask.float().sum().item()) if torch.any(hard_mask) else 0.0
            class_correct += torch.bincount(y_cpu, weights=hit, minlength=num_classes).float()
            class_total += torch.bincount(y_cpu, minlength=num_classes).float()
            total += int(y.size(0))
            max_abs = max(max_abs, float(logits.abs().max().item()))
    if total <= 0:
        return {"acc": 0.0, "balanced_acc": 0.0, "morph_acc": 0.0, "hard_acc": 0.0, "score": 0.0, "max_abs": max_abs}
    acc = correct / total
    morph_acc = weighted_correct / max(importance_total, EPS)
    hard_acc = hard_correct / max(hard_total, EPS) if hard_total > 0 else acc
    valid_classes = class_total > 0
    balanced_acc = float((class_correct[valid_classes] / (class_total[valid_classes] + EPS)).mean().item()) if torch.any(valid_classes) else acc
    if not _component_enabled(cfg, "balanced_selection"):
        score = 0.70 * acc + 0.20 * morph_acc + 0.10 * hard_acc
    else:
        score = 0.50 * acc + 0.22 * balanced_acc + 0.18 * morph_acc + 0.10 * hard_acc
    return {
        "acc": acc,
        "balanced_acc": balanced_acc,
        "morph_acc": morph_acc,
        "hard_acc": hard_acc,
        "score": score,
        "max_abs": max_abs,
    }


def _recalibrate_batchnorm(meta, merged_state_dict, cfg):
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    runtime = build_runtime(
        meta=meta,
        data_root=cfg["data_root"],
        split=cfg.get("stats_split", "val"),
        batch_size=int(cfg.get("stats_batch_size", 32) or cfg.get("batch_size", 64)),
        num_workers=int(cfg.get("stats_num_workers", cfg.get("num_workers", 4))),
        device=device,
    )
    model = runtime["model"]
    model.load_state_dict(merged_state_dict, strict=True)
    bn_modules = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_modules:
        return merged_state_dict

    for module in bn_modules:
        module.running_mean.zero_()
        module.running_var.fill_(1.0)
        module.num_batches_tracked.zero_()
        module.momentum = None
    model.train()

    max_batches = _resolve_max_batches(meta, cfg, "my_merge_bn_batches", DEFAULT_BN_BATCHES)
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(runtime["loader"]):
            if max_batches and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            _ = runtime["forward_fn"](model, x)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _auto_head_scale(meta, merged_state_dict, cfg):
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    runtime = build_runtime(
        meta=meta,
        data_root=cfg["data_root"],
        split=cfg.get("stats_split", "val"),
        batch_size=int(cfg.get("stats_batch_size", 32) or cfg.get("batch_size", 64)),
        num_workers=0,
        device=device,
    )
    model = runtime["model"]
    model.load_state_dict(merged_state_dict, strict=True)
    model.eval()
    max_abs = 0.0
    max_batches = _resolve_max_batches(meta, cfg, "my_merge_head_scale_batches", 1)
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(runtime["loader"]):
            if max_batches and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            logits = runtime["forward_fn"](model, x)
            max_abs = max(max_abs, float(logits.detach().abs().max().item()))
    if max_abs <= 0.0:
        return HEAD_SCALE_DEFAULT
    safe_cap = 40.0 if _model_family(meta) in {"transformer", "vlm"} else 45.0
    return min(1.0, max(0.01, safe_cap / max_abs))


def _interpolate_state_dicts(base_state_dict, candidate_state_dict, alpha):
    mixed = OrderedDict()
    for key, base_value in base_state_dict.items():
        cand_value = candidate_state_dict[key]
        if torch.is_floating_point(base_value) and torch.is_floating_point(cand_value):
            mixed[key] = (1.0 - alpha) * base_value.detach().clone() + alpha * cand_value.detach().clone()
        else:
            mixed[key] = cand_value.detach().clone()
    return mixed


def _build_reference_delta_candidate(state_dicts, overall_weights, morph_weights, class_weights, num_classes, meta, cfg=None):
    reference_state, _ = build_reference_bundle(meta, device="cpu")
    profile = _merge_profile(meta)
    anchor_idx = int(torch.argmax(morph_weights).item())
    anchor_one_hot = F.one_hot(torch.tensor(anchor_idx), num_classes=len(state_dicts)).float()
    consensus = _blend_scores(morph_weights, overall_weights, blend=0.58)
    use_residual = _component_enabled(cfg, "sparse_residual")

    merged = OrderedDict()
    for key in state_dicts[0].keys():
        values = [sd[key] for sd in state_dicts]
        ref_value = reference_state.get(key, values[0])
        first = values[0]
        if not torch.is_floating_point(first):
            merged[key] = values[anchor_idx].detach().clone()
            continue
        if _is_classifier_tensor(key, first, num_classes):
            merged[key] = _merge_classifier_rows(values, class_weights, profile, morph_weights)
            continue

        group = _param_group(key, meta)
        if group == "early":
            weights = _blend_scores(anchor_one_hot, morph_weights, blend=min(0.92, profile["early_anchor"] + 0.10))
            reinject_keep = profile["early_residual_keep"]
            reinject_scale = profile["early_residual_scale"]
        elif group == "late":
            weights = _blend_scores(anchor_one_hot, overall_weights, blend=min(0.88, profile["late_anchor"] + 0.12))
            reinject_keep = profile["late_residual_keep"]
            reinject_scale = profile["late_residual_scale"]
        else:
            weights = _blend_scores(anchor_one_hot, consensus, blend=min(0.90, profile["mid_anchor"] + 0.10))
            reinject_keep = profile["mid_residual_keep"]
            reinject_scale = profile["mid_residual_scale"]
        if not use_residual:
            reinject_keep = 0.0
            reinject_scale = 0.0

        delta_values = [value.detach() - ref_value.detach() for value in values]
        merged_delta = _weighted_average_for_key(delta_values, weights)
        merged_delta = _sparse_residual_reinjection(
            merged_delta,
            delta_values,
            primary_weights=morph_weights,
            secondary_weights=overall_weights,
            keep_ratio=reinject_keep,
            scale=reinject_scale,
        )
        merged[key] = ref_value.detach().clone() + merged_delta
    return merged


def _prepare_candidate(meta, merged_state_dict, cfg, apply_bn=True, apply_head_temperature=True):
    prepared = merged_state_dict
    if apply_bn and _component_enabled(cfg, "bn_recalibration"):
        prepared = _recalibrate_batchnorm(meta, prepared, cfg)
    if apply_head_temperature and _component_enabled(cfg, "head_temperature"):
        prepared = _apply_head_temperature(
            prepared,
            num_classes=int(meta["num_classes"]),
            scale=_auto_head_scale(meta, prepared, cfg),
        )
    return prepared


def _candidate_priority(meta, candidate_name):
    family = _model_family(meta)
    if family == "vlm":
        order = {
            "specialist_client": 6,
            "reference_delta": 5,
            "morphology": 4,
            "morph_anchor": 3,
            "consensus": 2,
            "avg": 1,
        }
        return order.get(candidate_name, 0)
    if family == "transformer":
        order = {
            "prototype_head": 6,
            "specialist_client": 5,
            "morph_anchor": 4,
            "reference_delta": 3,
            "morphology": 2,
            "consensus": 1,
            "avg": 0,
        }
        return order.get(candidate_name, 0)
    return 0


def _choose_candidate(meta, candidate_metrics, cfg=None):
    if not _component_enabled(cfg, "candidate_selection") and "avg" in candidate_metrics:
        return "avg"
    family = _model_family(meta)
    balanced_weight = 1.0 if _component_enabled(cfg, "balanced_selection") else 0.0
    if family == "vlm":
        def key_fn(item):
            name, metric = item
            return (
                float(metric["score"]),
                float(metric["acc"]),
                balanced_weight * float(metric.get("balanced_acc", 0.0)),
                float(metric["hard_acc"]),
                _candidate_priority(meta, name),
            )
    elif family == "transformer":
        def key_fn(item):
            name, metric = item
            return (
                float(metric["score"]),
                float(metric["acc"]),
                balanced_weight * float(metric.get("balanced_acc", 0.0)),
                float(metric["morph_acc"]),
                _candidate_priority(meta, name),
            )
    else:
        def key_fn(item):
            name, metric = item
            return (
                float(metric["score"]),
                float(metric["acc"]),
                balanced_weight * float(metric.get("balanced_acc", 0.0)),
                float(metric["hard_acc"]),
                _candidate_priority(meta, name),
            )
    return max(candidate_metrics.items(), key=key_fn)[0]


def _module2_medical_evidence_guided_fusion_and_selection(
    state_dicts,
    base_merged,
    meta,
    cfg,
    client_info,
):
    batches = client_info["batches"]
    features = client_info["features"]
    labels = client_info["labels"]
    overall_weights = client_info["overall_weights"]
    morph_weights = client_info["morphology_weights"]
    class_weights = client_info["class_weights"]
    client_summaries = client_info["client_summaries"]

    morphology_merged = _layerwise_merge(
        state_dicts,
        overall_weights=overall_weights,
        morph_weights=morph_weights,
        class_weights=class_weights,
        num_classes=int(meta["num_classes"]),
        meta=meta,
        cfg=cfg,
    )
    specialist_idx = None
    reference_delta = None
    if _component_enabled(cfg, "reference_delta_candidate") and _model_family(meta) in {"transformer", "vlm"}:
        reference_delta = _build_reference_delta_candidate(
            state_dicts,
            overall_weights=overall_weights,
            morph_weights=morph_weights,
            class_weights=class_weights,
            num_classes=int(meta["num_classes"]),
            meta=meta,
            cfg=cfg,
        )

    base_candidate = _prepare_candidate(meta, base_merged, cfg, apply_head_temperature=False)
    morphology_candidate = _prepare_candidate(meta, morphology_merged, cfg, apply_head_temperature=False)
    candidate_pool = {
        "avg": base_candidate,
        "morphology": morphology_candidate,
    }
    if _component_enabled(cfg, "morph_anchor_candidate"):
        morphology_anchor = _build_morphology_anchor_candidate(
            state_dicts,
            morph_weights=morph_weights,
            class_weights=class_weights,
            num_classes=int(meta["num_classes"]),
            meta=meta,
        )
        candidate_pool["morph_anchor"] = _prepare_candidate(meta, morphology_anchor, cfg, apply_head_temperature=False)
    if _component_enabled(cfg, "specialist_candidate"):
        specialist_candidate_state, specialist_idx = _build_specialist_client_candidate(
            state_dicts,
            client_summaries=client_summaries,
            meta=meta,
        )
        candidate_pool["specialist_client"] = _prepare_candidate(
            meta,
            specialist_candidate_state,
            cfg,
            apply_head_temperature=False,
        )
    if _component_enabled(cfg, "consensus_candidate"):
        alpha = _merge_profile(meta)["candidate_alpha"]
        consensus_candidate = _prepare_candidate(
            meta,
            _interpolate_state_dicts(base_candidate, morphology_candidate, alpha=alpha),
            cfg,
            apply_head_temperature=False,
        )
        candidate_pool["consensus"] = consensus_candidate
    if reference_delta is not None:
        candidate_pool["reference_delta"] = _prepare_candidate(meta, reference_delta, cfg, apply_head_temperature=False)
    prototype_candidate_state = None
    if _component_enabled(cfg, "prototype_head_candidate"):
        prototype_candidate_state = _build_prototype_head_candidate(
            base_merged,
            meta=meta,
            batches=batches,
            labels=labels,
            features=features,
            cfg=cfg,
        )
    if prototype_candidate_state is not None:
        candidate_pool["prototype_head"] = _prepare_candidate(
            meta,
            prototype_candidate_state,
            cfg,
            apply_head_temperature=False,
        )

    candidate_metrics = {
        name: _evaluate_merged_state(meta, candidate_state, cfg, split=cfg.get("stats_split", "val"))
        for name, candidate_state in candidate_pool.items()
    }
    selected_name = _choose_candidate(meta, candidate_metrics, cfg=cfg)
    selected_state = _prepare_candidate(
        meta,
        candidate_pool[selected_name],
        cfg,
        apply_bn=False,
        apply_head_temperature=True,
    )
    return selected_state, {
        "module_name": METHOD_MODULES["module_2"]["name"],
        "candidate_pool": sorted(candidate_pool.keys()),
        "candidate_metrics": candidate_metrics,
        "selected_candidate": selected_name,
        "specialist_client_index": None if specialist_idx is None else int(specialist_idx),
    }


def merge_my_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    if meta is None or checkpoints is None or cfg is None or not _is_medical_image_task(meta):
        merged_state_dict, normalized_weights = average_state_dicts(state_dicts, weights)
        return merged_state_dict, {
            "implementation": "medical_image_only_merge_fallback",
            "normalized_weights": normalized_weights,
        }

    base_merged, base_weights = average_state_dicts(state_dicts, weights)
    ablation_config = _ablation_config(cfg)
    ablation_names = {label.lower().replace("-", "_") for label in ablation_config["labels"]}
    if "avg_only" in ablation_names:
        return base_merged, {
            "implementation": "medical_evidence_two_module_merge_v6_ablation_avg_only",
            "medical_only": True,
            "method_modules": METHOD_MODULES,
            "ablation_config": ablation_config,
            "selected_candidate": "avg",
            "base_weights": [float(x) for x in base_weights],
            "normalized_weights": base_weights,
        }
    try:
        client_info = _module1_diagnostic_client_information_estimation(
            meta,
            checkpoints,
            cfg,
            base_weights,
        )
        merged_state_dict, fusion_info = _module2_medical_evidence_guided_fusion_and_selection(
            state_dicts,
            base_merged,
            meta,
            cfg,
            client_info,
        )

        return merged_state_dict, {
            "implementation": "medical_evidence_two_module_posthoc_merge_v7",
            "medical_only": True,
            "method_modules": METHOD_MODULES,
            "ablation_config": ablation_config,
            "modality": meta.get("dataset"),
            "model_family": _model_family(meta),
            "base_weights": [float(x) for x in base_weights],
            "overall_weights": [float(x) for x in client_info["overall_weights"].tolist()],
            "morphology_weights": [float(x) for x in client_info["morphology_weights"].tolist()],
            "class_weights": [[float(v) for v in row] for row in client_info["class_weights"].tolist()],
            "client_summaries": client_info["client_summaries"],
            "client_diagnostic_information": client_info["client_diagnostic_information"],
            "client_information_theory": client_info["theory"],
            "evidence_theory": client_info["evidence_theory"],
            "specialist_client_index": fusion_info["specialist_client_index"],
            "feature_summary": client_info["feature_summary"],
            "feature_names": client_info["feature_names"],
            "candidate_pool": fusion_info["candidate_pool"],
            "candidate_metrics": fusion_info["candidate_metrics"],
            "selected_candidate": fusion_info["selected_candidate"],
        }
    except Exception as exc:
        return base_merged, {
            "implementation": "medical_evidence_two_module_posthoc_merge_fallback",
            "medical_only": True,
            "method_modules": METHOD_MODULES,
            "ablation_config": ablation_config,
            "fallback_reason": str(exc),
            "normalized_weights": base_weights,
        }
