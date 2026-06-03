from collections import OrderedDict
import torch
import torch.nn.functional as F

from model import build_model
from model.clip_model import build_clip_model, encode_image_features, get_clip_base
from utils.runtime import build_reference_bundle, build_runtime, resolve_vlm_max_text_len
from utils.state_dict import average_state_dicts

from .common import build_task_matrix, disjoint_merge, elect_sign, mask_smallest_magnitude, overlay_param_dict, vector_to_param_dict


EPS = 1e-8
DEFAULT_STATS_MAX_BATCHES = 16
DEFAULT_BN_BATCHES = 4
DEFAULT_RELIABILITY_THRESHOLD = 0.10
DEFAULT_SEPARATION_THRESHOLD = 0.05
DEFAULT_SOUP_SCORE_MARGIN = 0.05
DEFAULT_SOUP_MIN_SCORE_DELTA = 0.0
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
    "medical_weighted_fusion",
    "sign_consistent_delta",
    "validated_selection",
    "bn_recalibration",
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
    "no_balanced_selection": set(),
    "no_layerwise": set(),
    "no_residual": set(),
    "no_candidate_bank": set(),
    "no_specialist": set(),
    "no_reference_delta": set(),
    "no_prototype": set(),
    "no_sign_delta": {"sign_consistent_delta"},
    "no_ties_delta": {"sign_consistent_delta"},
    "no_calibration": {"bn_recalibration"},
    "avg_only": {
        *MODULE1_COMPONENTS,
        *MODULE2_COMPONENTS,
    },
}

COMPONENT_ALIASES = {
    "clip_denorm": "image_space",
    "vlm_denorm": "image_space",
    "fusion_selection": "medical_weighted_fusion",
    "candidate_selection": "medical_weighted_fusion",
    "medical_fusion_selection": "medical_weighted_fusion",
    "layerwise_merge": "validated_selection",
    "classwise_head": "validated_selection",
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
        "name": "Validated Conservative Checkpoint Fusion",
        "purpose": "Fuse client deltas with medical evidence, inject M1 consensus into sign-consistent deltas, and use validation-guided top-2 soup when candidates are statistically close.",
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


def _stable_evidence_weights(values, fallback, temperature=0.70, floor=0.015):
    scores = torch.as_tensor(values, dtype=torch.float32)
    fallback = torch.as_tensor(fallback, dtype=torch.float32)
    if scores.numel() == 0:
        return fallback
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = torch.clamp(scores, min=EPS)
    logits = torch.log(scores)
    logits = logits - logits.mean()
    scale = torch.clamp(logits.abs().median(), min=0.25)
    probs = torch.softmax(logits / (float(temperature) * scale + EPS), dim=0)
    if floor > 0.0:
        probs = (1.0 - float(floor)) * probs + float(floor) * fallback
    return _normalize_scores(probs, fallback=fallback)


def _weight_entropy(weights):
    probs = torch.as_tensor(weights, dtype=torch.float32)
    probs = _normalize_scores(probs, fallback=torch.ones_like(probs) / max(1, probs.numel()))
    return -torch.sum(probs * torch.log(probs + EPS))


def _evidence_concentration(weights):
    probs = torch.as_tensor(weights, dtype=torch.float32)
    if probs.numel() <= 1:
        return 0.0
    entropy = _weight_entropy(probs)
    max_entropy = torch.log(torch.tensor(float(probs.numel()), dtype=torch.float32))
    return float(torch.clamp(1.0 - entropy / (max_entropy + EPS), min=0.0, max=1.0).item())


def _evidence_reliability_from_features(features):
    if features is None or features.ndim != 2 or features.shape[1] <= 6:
        return 1.0
    return float(torch.clamp(features[:, 6].mean(), min=0.0, max=1.0).item())


def _score_separation(scores):
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if scores.numel() <= 1:
        return 0.0
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mean = scores.mean()
    if float(mean.item()) <= 0.0:
        return 0.0
    return float(torch.clamp(scores.std(unbiased=False) / (mean + EPS), min=0.0, max=1.0).item())


def _diagnostic_weight_gate(scores, evidence_reliability):
    reliability_denom = max(EPS, 1.0 - DEFAULT_RELIABILITY_THRESHOLD)
    reliability_gate = max(0.0, min(1.0, (float(evidence_reliability) - DEFAULT_RELIABILITY_THRESHOLD) / reliability_denom))
    separation_gate = max(0.0, min(1.0, _score_separation(scores) / max(EPS, DEFAULT_SEPARATION_THRESHOLD)))
    return reliability_gate * separation_gate


def _calibrated_diagnostic_weights(scores, fallback, evidence_reliability, temperature=0.70, floor=0.015):
    fallback = torch.as_tensor(fallback, dtype=torch.float32)
    raw = _stable_evidence_weights(scores, fallback=fallback, temperature=temperature, floor=floor)
    gate = _diagnostic_weight_gate(scores, evidence_reliability)
    mixed = (1.0 - gate) * fallback + gate * raw
    return _normalize_scores(mixed, fallback=fallback)


def _metadata_prior_weights(meta, base_weights):
    base = torch.as_tensor(base_weights, dtype=torch.float32)
    if base.numel() == 0:
        return base
    return _normalize_scores(base, fallback=torch.ones_like(base) / max(1, base.numel()))


def _evidence_support_gate(labels, num_classes):
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1)
    if labels.numel() <= 0:
        return 0.0
    counts = torch.bincount(labels, minlength=max(1, int(num_classes))).float()
    probs = counts / (counts.sum() + EPS)
    entropy = -torch.sum(probs * torch.log(probs + EPS))
    max_entropy = torch.log(torch.tensor(float(max(2, int(num_classes))), dtype=torch.float32))
    balance_gate = float(torch.clamp(entropy / (max_entropy + EPS), min=0.0, max=1.0).item())
    size_gate = float(labels.numel()) / (float(labels.numel()) + 32.0 * float(max(1, int(num_classes))))
    return max(0.0, min(1.0, size_gate * balance_gate))


def _label_coverage_ratio(labels, num_classes):
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1)
    if labels.numel() <= 0:
        return 0.0
    counts = torch.bincount(labels, minlength=max(1, int(num_classes)))
    present = float((counts > 0).sum().item())
    return max(0.0, min(1.0, present / float(max(1, int(num_classes)))))


def _client_specialization_ratio(meta):
    num_classes = max(1, int(meta.get("num_classes", 1)))
    clients = meta.get("clients", [])
    if not clients:
        return 1.0
    counts = [len(set(client.get("classes", []))) for client in clients]
    if not counts:
        return 1.0
    return float(sum(counts)) / (float(len(counts)) * float(num_classes))


def _use_sign_consistent_delta(meta, cfg):
    if not _component_enabled(cfg, "sign_consistent_delta"):
        return False
    return meta.get("task_type") == "small" and _is_medical_image_task(meta)


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


def _build_model_forward_only(meta, cfg, device):
    if meta.get("task_type") == "small":
        model, _, _, _ = build_model(
            name=meta["model"],
            num_classes=int(meta["num_classes"]),
            in_channels=int(meta.get("in_channels", 3)),
            pretrained=False,
        )
        return model.to(device), lambda m, x: m(x)
    if meta.get("task_type") == "vlm":
        max_text_len = resolve_vlm_max_text_len(meta, default=int(meta.get("max_text_len", 32)))
        model, text_encoder = build_clip_model(
            clip_model_name=meta["clip_model"],
            class_names=meta["class_names"],
            text_template=meta.get("text_template", "a medical image of class {}"),
            max_text_len=max_text_len,
            random_init=bool(meta.get("clip_random_init", False)),
            device=device,
        )

        def forward_fn(m, x):
            base = get_clip_base(m)
            image_features = encode_image_features(base, x)
            text_features = text_encoder()
            logit_scale = base.logit_scale.exp()
            return logit_scale * image_features @ text_features.t()

        return model, forward_fn
    raise ValueError(f"Unsupported task_type: {meta.get('task_type')}")


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
        "medical_weighted_fusion",
        "sign_consistent_delta",
        "validated_selection",
        "bn_recalibration",
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
    evidence_reliability = torch.clamp(
        (0.50 * local_contrast + 0.35 * boundary_strength + 0.15 * shape_compactness)
        / (0.50 + EPS),
        min=0.05,
        max=2.0,
    )
    diagnostic_salience = (
        (0.35 + boundary_strength)
        * (0.35 + local_contrast)
        * (0.35 + texture_heterogeneity)
        * (0.65 + area_ratio)
        * (0.75 + shape_compactness)
        * evidence_reliability
    )
    return torch.stack(
        [
            area_ratio,
            boundary_strength,
            local_contrast,
            texture_heterogeneity,
            shape_compactness,
            diagnostic_salience,
            evidence_reliability,
        ],
        dim=1,
    )


def _neutral_medical_features(x):
    return torch.ones((x.shape[0], 7), dtype=x.dtype, device=x.device)


def _batch_morphology_features(meta, x, cfg=None):
    x = _image_space01(meta, x, cfg=cfg)
    if not _component_enabled(cfg, "diagnostic_evidence"):
        return _neutral_medical_features(x)
    return _generic_medical_features(x)


def _sample_importance(features):
    if features.ndim == 2 and features.shape[1] > 5:
        importance = features[:, 5]
    else:
        importance = features[:, -1]
    importance = importance / (importance.mean() + EPS)
    return torch.clamp(importance, min=0.25, max=3.5)


def _class_rarity_weights(labels, num_classes, meta, cfg=None, evidence_reliability=1.0):
    if not _component_enabled(cfg, "class_rarity"):
        return torch.ones(num_classes, dtype=torch.float32, device=labels.device)
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    inv_sqrt = torch.sqrt(counts.sum() / counts)
    inv_sqrt = inv_sqrt / (inv_sqrt.mean() + EPS)
    strength = 0.22 + 0.28 * max(0.0, min(1.0, float(evidence_reliability)))
    rarity = 1.0 + strength * (inv_sqrt - 1.0)
    return torch.clamp(rarity, min=0.55, max=3.0)


def _medical_sample_weights(meta, features, labels, num_classes, cfg=None):
    morphology = _sample_importance(features)
    reliability = torch.clamp(features[:, 6], min=0.15, max=1.5) if features.shape[1] > 6 else torch.ones_like(morphology)
    morphology = (0.55 + 0.45 * reliability) * morphology
    morphology = morphology / (morphology.mean() + EPS)
    evidence_reliability = _evidence_reliability_from_features(features)
    rarity = _class_rarity_weights(labels, num_classes, meta, cfg=cfg, evidence_reliability=evidence_reliability)[labels]
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
        "evidence_reliability",
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
    loader = runtime["loader"]
    for batch_idx, (x, y) in enumerate(loader):
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
    model, forward_fn = _build_model_forward_only(meta, cfg, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    logits_out = []
    with torch.no_grad():
        for x, _ in batches:
            x = x.to(device, non_blocking=True)
            logits_out.append(forward_fn(model, x).detach().float().cpu())
    return torch.cat(logits_out, dim=0)


def _client_scores(meta, checkpoints, batches, features, labels, cfg, base_weights):
    num_clients = len(checkpoints)
    num_classes = int(meta["num_classes"])
    evidence_reliability = _evidence_reliability_from_features(features)
    metadata_prior = _metadata_prior_weights(meta, base_weights)
    support_gate = _evidence_support_gate(labels, num_classes)
    label_coverage = _label_coverage_ratio(labels, num_classes)
    evidence_floor = 0.30 if label_coverage >= 0.50 else 0.0
    class_floor = 0.15 if label_coverage >= 0.50 else 0.0
    evidence_gate = max(evidence_floor, support_gate) if labels.numel() > 0 else 0.0
    class_gate = max(class_floor, support_gate) if labels.numel() > 0 else 0.0
    sample_importance = _medical_sample_weights(meta, features, labels, num_classes, cfg=cfg)
    class_rarity = _class_rarity_weights(labels, num_classes, meta, cfg=cfg, evidence_reliability=evidence_reliability)
    hard_mask = sample_importance >= torch.quantile(sample_importance, q=0.60)
    use_focal = _component_enabled(cfg, "focal_weight")

    overall_scores = []
    morph_scores = []
    coverage_gates = []
    class_scores = torch.zeros(num_classes, num_clients, dtype=torch.float32)
    client_summaries = []

    for client_idx, checkpoint in enumerate(checkpoints):
        seen_classes = set(meta["clients"][client_idx].get("classes", []))
        if seen_classes:
            seen_tensor = torch.tensor(sorted(seen_classes), dtype=labels.dtype, device=labels.device)
            coverage = torch.isin(labels, seen_tensor).float().mean()
            expected_coverage = max(float(len(seen_classes)) / max(1, num_classes), EPS)
            coverage_gate = float(torch.clamp(coverage / expected_coverage, min=0.0, max=1.0).item())
        else:
            coverage_gate = 1.0
        coverage_gates.append(coverage_gate)

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
                "class_coverage_gate": coverage_gate,
            }
        )

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

    if overall_scores:
        mean_overall_score = float(torch.as_tensor(overall_scores, dtype=torch.float32).mean().item())
        mean_morph_score = float(torch.as_tensor(morph_scores, dtype=torch.float32).mean().item())
        overall_scores = [
            gate * score + (1.0 - gate) * mean_overall_score
            for score, gate in zip(overall_scores, coverage_gates)
        ]
        morph_scores = [
            gate * score + (1.0 - gate) * mean_morph_score
            for score, gate in zip(morph_scores, coverage_gates)
        ]

    overall_weights = _calibrated_diagnostic_weights(
        overall_scores,
        fallback=metadata_prior,
        evidence_reliability=evidence_reliability,
        temperature=0.78,
        floor=0.025,
    )
    overall_weights = _normalize_scores(
        (1.0 - evidence_gate) * metadata_prior + evidence_gate * overall_weights,
        fallback=metadata_prior,
    )
    morph_weights = _calibrated_diagnostic_weights(
        morph_scores,
        fallback=metadata_prior,
        evidence_reliability=evidence_reliability,
        temperature=0.72,
        floor=0.025,
    )
    morph_weights = _normalize_scores(
        (1.0 - evidence_gate) * metadata_prior + evidence_gate * morph_weights,
        fallback=metadata_prior,
    )
    class_weights = []
    for cls_idx in range(num_classes):
        cls_weight = _calibrated_diagnostic_weights(
            class_scores[cls_idx],
            fallback=morph_weights,
            evidence_reliability=evidence_reliability,
            temperature=0.84,
            floor=0.05,
        )
        cls_weight = _normalize_scores(
            (1.0 - class_gate) * morph_weights + class_gate * cls_weight,
            fallback=morph_weights,
        )
        class_weights.append(cls_weight)
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
            "base_weights": base_tensor,
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
            "class_coverage_gate": float(summary.get("class_coverage_gate", 1.0)),
            "evidence_reliability": float(_evidence_reliability_from_features(features)),
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
        "base_weights": torch.as_tensor(base_weights, dtype=torch.float32),
        "overall_weights": overall_weights,
        "morphology_weights": morph_weights,
        "class_weights": class_weights,
        "client_summaries": client_summaries,
        "client_diagnostic_information": client_information,
        "metadata_prior_weights": _metadata_prior_weights(meta, base_weights),
        "evidence_support_gate": _evidence_support_gate(labels, int(meta["num_classes"])),
        "label_coverage_ratio": _label_coverage_ratio(labels, int(meta["num_classes"])),
        "theory": "Each client carries different diagnostic information; fusion weights depend on morphology, hard cases, margins, and class-level expertise. When image evidence is unreliable or client scores are weakly separated, M1 keeps weights close to the prior to avoid amplifying acquisition noise.",
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
                "Medical specificity comes from image evidence shared across medical imaging tasks: foreground structure, "
                "boundary strength, local contrast, texture heterogeneity, evidence reliability, class rarity, and hard-case margins."
            ),
        }
    )
    return client_info


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


def _weighted_average_for_key(values, weight_tensor):
    out = values[0].detach().clone() * float(weight_tensor[0].item())
    for value, weight in zip(values[1:], weight_tensor[1:]):
        out.add_(value.detach(), alpha=float(weight.item()))
    return out


def _weighted_delta_for_key(values, reference_value, weight_tensor):
    reference_float = reference_value.detach().float()
    out = torch.zeros_like(reference_float)
    for value, weight in zip(values, weight_tensor):
        out.add_(value.detach().float() - reference_float, alpha=float(weight.item()))
    return out


def _medical_consensus_weights(overall_weights, morph_weights, base_weights):
    base_weights = torch.as_tensor(base_weights, dtype=torch.float32)
    consensus = 0.50 * torch.as_tensor(overall_weights, dtype=torch.float32)
    consensus = consensus + 0.50 * torch.as_tensor(morph_weights, dtype=torch.float32)
    return _normalize_scores(consensus, fallback=base_weights)


def _layer_fusion_weights(group, base_weights, overall_weights, morph_weights, consensus_weights):
    base_weights = torch.as_tensor(base_weights, dtype=torch.float32)
    overall_weights = torch.as_tensor(overall_weights, dtype=torch.float32)
    morph_weights = torch.as_tensor(morph_weights, dtype=torch.float32)
    if group == "early":
        return _normalize_scores(0.70 * base_weights + 0.30 * morph_weights, fallback=base_weights)
    if group == "late":
        return _normalize_scores(0.30 * base_weights + 0.70 * overall_weights, fallback=base_weights)
    return _normalize_scores(0.45 * base_weights + 0.55 * consensus_weights, fallback=base_weights)


def _classifier_fusion_weights(cls_idx, class_weights, consensus_weights):
    row_weights = class_weights[cls_idx]
    return _normalize_scores(0.45 * row_weights + 0.55 * consensus_weights, fallback=consensus_weights)


def _medical_delta_weighted_state_merge(
    state_dicts,
    base_merged,
    reference_state,
    base_weights,
    overall_weights,
    morph_weights,
    class_weights,
    num_classes,
    meta,
):
    consensus = _medical_consensus_weights(overall_weights, morph_weights, base_weights)
    merged = OrderedDict()
    routing_summary = {"early": 0, "mid": 0, "late": 0, "classifier": 0}
    group_weights = {
        group: _layer_fusion_weights(group, base_weights, overall_weights, morph_weights, consensus)
        for group in ("early", "mid", "late")
    }

    for key in state_dicts[0].keys():
        values = [sd[key] for sd in state_dicts]
        first = values[0]
        if not torch.is_floating_point(first):
            merged[key] = base_merged[key].detach().clone()
            continue
        reference_value = reference_state.get(key) if reference_state is not None else None
        use_delta = (
            reference_value is not None
            and torch.is_floating_point(reference_value)
            and tuple(reference_value.shape) == tuple(first.shape)
        )
        if _is_classifier_tensor(key, first, num_classes):
            out = first.detach().clone().float().zero_()
            if use_delta:
                out = reference_value.detach().clone().float()
            for cls_idx in range(first.shape[0]):
                row_weights = _classifier_fusion_weights(cls_idx, class_weights, consensus)
                row_values = [value[cls_idx] for value in values]
                if use_delta:
                    row_ref = reference_value[cls_idx]
                    out[cls_idx] = (row_ref.detach().float() + _weighted_delta_for_key(row_values, row_ref, row_weights)).to(
                        dtype=out.dtype
                    )
                else:
                    out[cls_idx] = _weighted_average_for_key(row_values, row_weights)
            merged[key] = out.to(dtype=first.dtype)
            routing_summary["classifier"] += 1
            continue

        group = _param_group(key, meta)
        if use_delta:
            delta = _weighted_delta_for_key(values, reference_value, group_weights[group])
            merged[key] = (reference_value.detach().float() + delta).to(dtype=first.dtype)
        else:
            merged[key] = _weighted_average_for_key(values, group_weights[group])
        routing_summary[group] += 1

    return merged, {
        "fusion_weights": [float(x) for x in consensus.tolist()],
        "group_weights": {group: [float(x) for x in weights.tolist()] for group, weights in group_weights.items()},
        "routing_summary": routing_summary,
        "delta_space": reference_state is not None,
    }


def _medical_sign_consistent_delta_merge(state_dicts, base_merged, base_state, param_keys, weights):
    task_matrix, base_vector = build_task_matrix(state_dicts, base_state, param_keys)
    trimmed = mask_smallest_magnitude(task_matrix, preserve_density=0.5)
    elected_sign = elect_sign(trimmed)
    merged_delta = disjoint_merge(trimmed, elected_sign, weights=weights)
    merged_vector = base_vector + merged_delta
    merged_params = vector_to_param_dict(merged_vector, base_state, param_keys)
    return overlay_param_dict(base_merged, merged_params)


def _blend_state_dicts(left, right, right_weight):
    right_weight = float(right_weight)
    left_weight = 1.0 - right_weight
    merged = OrderedDict()
    for key, left_value in left.items():
        right_value = right[key]
        if torch.is_floating_point(left_value):
            merged[key] = (left_weight * left_value.detach().float() + right_weight * right_value.detach().float()).to(
                dtype=left_value.dtype
            )
        else:
            merged[key] = left_value.detach().clone()
    return merged


def _evaluate_candidate_on_batches(meta, state_dict, cfg, batches, features, labels):
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    model, forward_fn = _build_model_forward_only(meta, cfg, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    num_classes = int(meta["num_classes"])
    sample_weights = _medical_sample_weights(meta, features, labels, num_classes, cfg=cfg).detach().cpu().float()
    total = 0
    correct = 0.0
    weighted_correct = 0.0
    weighted_total = 0.0
    loss_sum = 0.0
    offset = 0
    with torch.no_grad():
        for x, y in batches:
            batch_size = int(y.numel())
            x = x.to(device, non_blocking=True)
            y_device = y.to(device, non_blocking=True)
            logits = forward_fn(model, x).detach().float()
            loss_vec = F.cross_entropy(logits, y_device, reduction="none").detach().cpu()
            preds = logits.argmax(dim=1).detach().cpu()
            y_cpu = y.detach().cpu()
            batch_correct = (preds == y_cpu).float()
            batch_weights = sample_weights[offset : offset + batch_size]
            correct += float(batch_correct.sum().item())
            weighted_correct += float((batch_correct * batch_weights).sum().item())
            weighted_total += float(batch_weights.sum().item())
            loss_sum += float(loss_vec.sum().item())
            total += batch_size
            offset += batch_size

    if total <= 0:
        return {"val_acc": 0.0, "val_medical_acc": 0.0, "val_loss": float("inf"), "selection_score": -float("inf")}
    val_acc = correct / float(total)
    val_medical_acc = weighted_correct / max(weighted_total, EPS)
    val_loss = loss_sum / float(total)
    selection_score = 0.85 * val_acc + 0.15 * val_medical_acc - 0.001 * val_loss
    return {
        "val_acc": float(val_acc),
        "val_medical_acc": float(val_medical_acc),
        "val_loss": float(val_loss),
        "selection_score": float(selection_score),
    }


def _validated_standard_checkpoint_merge(
    state_dicts,
    base_merged,
    base_weights,
    overall_weights,
    morph_weights,
    class_weights,
    num_classes,
    meta,
    cfg,
    batches,
    features,
    labels,
    reference_state=None,
    reference_param_names=None,
):
    candidates = OrderedDict()
    candidates["avg"] = base_merged
    candidate_traces = {}
    consensus_weights = _medical_consensus_weights(overall_weights, morph_weights, base_weights)

    if _component_enabled(cfg, "medical_weighted_fusion"):
        weighted_state, weighted_trace = _medical_delta_weighted_state_merge(
            state_dicts,
            base_merged,
            reference_state,
            base_weights,
            overall_weights,
            morph_weights,
            class_weights,
            num_classes,
            meta,
        )
        candidates["medical_weighted_fusion"] = weighted_state
        candidate_traces["medical_weighted_fusion"] = weighted_trace

    if _use_sign_consistent_delta(meta, cfg) and reference_state is not None and reference_param_names is not None:
        sign_state = _medical_sign_consistent_delta_merge(
            state_dicts,
            base_merged,
            reference_state,
            reference_param_names,
            consensus_weights.tolist(),
        )
        candidates["sign_consistent_delta"] = sign_state
        candidates["avg_sign_blend_0p25"] = _blend_state_dicts(base_merged, sign_state, right_weight=0.25)
        sign_trace = {
            "fusion_weights": [float(x) for x in consensus_weights.tolist()],
            "group_weights": {},
            "routing_summary": {"sign_delta_weight_source": "medical_consensus"},
        }
        candidate_traces["sign_consistent_delta"] = sign_trace
        candidate_traces["avg_sign_blend_0p25"] = {
            **sign_trace,
            "routing_summary": {"sign_delta_weight_source": "medical_consensus", "blend_right_weight": 0.25},
        }

    if not _component_enabled(cfg, "validated_selection") or len(candidates) == 1:
        return base_merged, {
            "selected_candidate": "avg",
            "candidate_pool": list(candidates.keys()),
            "candidate_metrics": {},
            "fusion_weights": [float(x) for x in torch.as_tensor(base_weights, dtype=torch.float32).tolist()],
            "group_weights": {},
            "routing_summary": {"validated_candidates": len(candidates)},
        }

    candidate_metrics = {}
    prepared_candidates = {}
    best_name = None
    best_key = None
    rank_keys = {}
    for name, candidate_state in candidates.items():
        prepared_state = _prepare_candidate(
            meta,
            candidate_state,
            cfg,
            apply_bn=_component_enabled(cfg, "bn_recalibration"),
            batches=batches,
        )
        prepared_candidates[name] = prepared_state
        metrics = _evaluate_candidate_on_batches(meta, prepared_state, cfg, batches, features, labels)
        candidate_metrics[name] = metrics
        rank_key = (metrics["selection_score"], metrics["val_acc"], -metrics["val_loss"])
        rank_keys[name] = rank_key
        if best_key is None or rank_key > best_key:
            best_key = rank_key
            best_name = name

    selected_name = best_name or "avg"
    soup_trace = {}
    ranked_names = sorted(rank_keys, key=lambda item: rank_keys[item], reverse=True)
    if len(ranked_names) >= 2:
        top_name = ranked_names[0]
        second_name = ranked_names[1]
        score_gap = float(candidate_metrics[top_name]["selection_score"] - candidate_metrics[second_name]["selection_score"])
        soup_margin = float(cfg.get("my_merge_soup_score_margin", DEFAULT_SOUP_SCORE_MARGIN) or DEFAULT_SOUP_SCORE_MARGIN)
        if score_gap <= soup_margin:
            soup_name = f"top2_soup:{top_name}+{second_name}"
            soup_state = _blend_state_dicts(prepared_candidates[top_name], prepared_candidates[second_name], right_weight=0.5)
            prepared_candidates[soup_name] = soup_state
            candidates[soup_name] = soup_state
            soup_metrics = _evaluate_candidate_on_batches(meta, soup_state, cfg, batches, features, labels)
            candidate_metrics[soup_name] = soup_metrics
            top_score = float(candidate_metrics[top_name]["selection_score"])
            soup_score = float(soup_metrics["selection_score"])
            min_delta = float(cfg.get("my_merge_soup_min_score_delta", DEFAULT_SOUP_MIN_SCORE_DELTA) or DEFAULT_SOUP_MIN_SCORE_DELTA)
            soup_accepted = soup_score + EPS >= top_score + min_delta
            if soup_accepted:
                selected_name = soup_name
            soup_trace = {
                "soup_components": [top_name, second_name],
                "soup_score_gap": score_gap,
                "soup_score_margin": soup_margin,
                "soup_right_weight": 0.5,
                "soup_selection_score": soup_score,
                "top_selection_score": top_score,
                "soup_min_score_delta": min_delta,
                "soup_accepted": soup_accepted,
            }
            candidate_traces[soup_name] = {
                "fusion_weights": [
                    float(x)
                    for x in torch.as_tensor(
                        candidate_traces.get(top_name, {}).get("fusion_weights", consensus_weights.tolist()),
                        dtype=torch.float32,
                    ).tolist()
                ],
                "group_weights": candidate_traces.get(top_name, {}).get("group_weights", {}),
                "routing_summary": {"top2_soup": soup_trace},
            }

    selected_trace = candidate_traces.get(selected_name, {})
    routing_summary = {"validated_candidates": len(candidates)}
    routing_summary.update(selected_trace.get("routing_summary", {}))
    if soup_trace:
        routing_summary["selected_by_top2_soup"] = bool(soup_trace.get("soup_accepted", False))
    return prepared_candidates.get(selected_name, candidates[selected_name]), {
        "selected_candidate": selected_name,
        "candidate_pool": list(candidates.keys()),
        "candidate_metrics": candidate_metrics,
        "fusion_weights": selected_trace.get(
            "fusion_weights",
            [float(x) for x in torch.as_tensor(base_weights, dtype=torch.float32).tolist()],
        ),
        "group_weights": selected_trace.get("group_weights", {}),
        "routing_summary": routing_summary,
        "candidates_prepared": True,
    }


def _recalibrate_batchnorm(meta, merged_state_dict, cfg, batches=None):
    max_batches = _resolve_max_batches(meta, cfg, "my_merge_bn_batches", DEFAULT_BN_BATCHES)
    if max_batches <= 0:
        return merged_state_dict
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    if batches is None:
        runtime = build_runtime(
            meta=meta,
            data_root=cfg["data_root"],
            split=cfg.get("stats_split", "val"),
            batch_size=int(cfg.get("stats_batch_size", 32) or cfg.get("batch_size", 64)),
            num_workers=int(cfg.get("stats_num_workers", cfg.get("num_workers", 4))),
            device=device,
        )
        model = runtime["model"]
        forward_fn = runtime["forward_fn"]
        iterator = runtime["loader"]
    else:
        model, forward_fn = _build_model_forward_only(meta, cfg, device)
        iterator = batches
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

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(iterator):
            if max_batches and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            _ = forward_fn(model, x)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _prepare_candidate(meta, merged_state_dict, cfg, apply_bn=True, batches=None):
    prepared = merged_state_dict
    if apply_bn and _component_enabled(cfg, "bn_recalibration"):
        prepared = _recalibrate_batchnorm(meta, prepared, cfg, batches=batches)
    return prepared


def _module2_medical_evidence_guided_fusion_and_selection(
    state_dicts,
    base_merged,
    meta,
    cfg,
    client_info,
    reference_state=None,
    reference_param_names=None,
):
    batches = client_info["batches"]
    features = client_info["features"]
    labels = client_info["labels"]
    overall_weights = client_info["overall_weights"]
    morph_weights = client_info["morphology_weights"]
    class_weights = client_info["class_weights"]
    base_weights = torch.as_tensor(client_info.get("base_weights", []), dtype=torch.float32)
    if base_weights.numel() == 0:
        base_weights = torch.ones(len(state_dicts), dtype=torch.float32) / max(1, len(state_dicts))
    num_classes = int(meta["num_classes"])
    if features.shape[1] > 6:
        evidence_reliability = float(torch.clamp(features[:, 6].mean(), min=0.0, max=1.0).item())
    else:
        evidence_reliability = 1.0

    if not _component_enabled(cfg, "medical_weighted_fusion"):
        selected_state = base_merged
        merge_trace = {"fusion_weights": [float(x) for x in base_weights.tolist()], "group_weights": {}, "routing_summary": {}}
        selected_candidate = "avg"
        fusion_rule = "module2_disabled_average_fallback"
    else:
        selected_state, merge_trace = _validated_standard_checkpoint_merge(
            state_dicts,
            base_merged,
            base_weights,
            overall_weights,
            morph_weights,
            class_weights,
            num_classes,
            meta=meta,
            cfg=cfg,
            batches=batches,
            features=features,
            labels=labels,
            reference_state=reference_state,
            reference_param_names=reference_param_names,
        )
        selected_candidate = merge_trace.get("selected_candidate", "avg")
        fusion_rule = "validated_standard_checkpoint_merge"

    selected_state = _prepare_candidate(
        meta,
        selected_state,
        cfg,
        apply_bn=_component_enabled(cfg, "bn_recalibration") and not merge_trace.get("candidates_prepared", False),
        batches=batches,
    )
    return selected_state, {
        "module_name": METHOD_MODULES["module_2"]["name"],
        "fusion_rule": fusion_rule,
        "candidate_pool": merge_trace.get("candidate_pool", [selected_candidate]),
        "candidate_metrics": merge_trace.get("candidate_metrics", {}),
        "selected_candidate": selected_candidate,
        "specialist_client_index": None,
        "routing_summary": merge_trace.get("routing_summary", {}),
        "evidence_reliability": evidence_reliability,
        "fusion_weights": merge_trace.get("fusion_weights", []),
        "group_weights": merge_trace.get("group_weights", {}),
        "fusion_confidence": None,
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
        reference_state = None
        reference_param_names = None
        if _component_enabled(cfg, "medical_weighted_fusion") or _use_sign_consistent_delta(meta, cfg):
            reference_state, reference_param_names = build_reference_bundle(meta, device="cpu")
        merged_state_dict, fusion_info = _module2_medical_evidence_guided_fusion_and_selection(
            state_dicts,
            base_merged,
            meta,
            cfg,
            client_info,
            reference_state=reference_state,
            reference_param_names=reference_param_names,
        )

        return merged_state_dict, {
            "implementation": "medical_evidence_two_module_posthoc_merge_v8_delta_soup_guarded",
            "medical_only": True,
            "method_modules": METHOD_MODULES,
            "ablation_config": ablation_config,
            "modality": meta.get("dataset"),
            "model_family": _model_family(meta),
            "base_weights": [float(x) for x in base_weights],
            "metadata_prior_weights": [
                float(x)
                for x in torch.as_tensor(client_info.get("metadata_prior_weights", base_weights), dtype=torch.float32).tolist()
            ],
            "evidence_support_gate": float(client_info.get("evidence_support_gate", 1.0)),
            "label_coverage_ratio": float(client_info.get("label_coverage_ratio", 1.0)),
            "client_specialization_ratio": float(_client_specialization_ratio(meta)),
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
            "fusion_rule": fusion_info.get("fusion_rule", ""),
            "fusion_weights": fusion_info.get("fusion_weights", []),
            "group_weights": fusion_info.get("group_weights", {}),
            "routing_summary": fusion_info.get("routing_summary", {}),
            "evidence_reliability": fusion_info.get("evidence_reliability", None),
            "fusion_confidence": fusion_info.get("fusion_confidence", None),
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
