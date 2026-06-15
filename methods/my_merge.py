from collections import OrderedDict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.runtime import build_reference_bundle, build_runtime
from utils.state_dict import average_state_dicts

from .common import (
    build_task_matrix,
    disjoint_merge,
    elect_sign,
    mask_smallest_magnitude,
    overlay_param_dict,
    vector_to_param_dict,
)


EPS = 1e-8
FEATURE_DIM = 5
DEFAULT_STATS_MAX_BATCHES = 16
MIN_SPECIALIST_ANCHOR_WEIGHT = 0.15
SPECIALIST_SEPARATION_START = 0.05
SPECIALIST_SEPARATION_FULL = 0.17
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

MEDICAL_IMAGE_DATASETS = {
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
}

NATURAL_CONTROL_DATASETS = {
    "cifar10_32",
    "cifar100_32",
    "svhn_32",
    "tinyimagenet_64",
}

M1 = {"image_space", "diagnostic_evidence", "diagnostic_client_information", "medical_weighted_fusion", "class_routing"}
M2 = {"checkpoint_consistency", "delta_agreement_weighting", "conflict_stabilization", "specialist_anchor"}
M3 = set()
OPTIONAL_COMPONENTS = {"statistical_alignment"}

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


def _disabled_components(cfg):
    disabled = set(OPTIONAL_COMPONENTS)
    for raw in _ablation_labels(cfg):
        name = raw.lower().replace("-", "_")
        if name == "avg_only":
            disabled.update(M1 | M2 | M3)
        elif name in {"statistical_alignment", "with_statistical_alignment", "bn_recalibration"}:
            disabled.discard("statistical_alignment")
        elif name in {"no_client_information", "no_diagnostic_information", "no_m1"}:
            disabled.update(M1)
        elif name == "no_diagnostic_evidence":
            disabled.add("diagnostic_evidence")
        elif name in {"no_class_routing", "no_classifier_routing"}:
            disabled.add("class_routing")
        elif name in {"no_fusion_selection", "no_medical_fusion_selection", "no_m2"}:
            disabled.update(M2)
        elif name in {
            "no_checkpoint_consistency",
            "no_consistency_guard",
        }:
            disabled.add("checkpoint_consistency")
        elif name in {
            "no_conflict_stabilization",
            "no_sign_delta",
            "no_ties_delta",
            "no_avg_sign_blend",
            "no_sign_blend",
        }:
            disabled.add("conflict_stabilization")
        elif name in {
            "no_adaptive_candidates",
            "no_adaptive_candidate_generation",
            "no_m3",
        }:
            disabled.update(M3)
        elif name == "specialist_client":
            disabled.add("specialist_anchor")
        elif name in {"no_calibration", "no_bn_recalibration", "no_statistical_alignment"}:
            disabled.add("statistical_alignment")
        elif name in {"prototype_head", "validated_selection"}:
            continue
        elif name in M1 or name in M2 or name in M3 or name in OPTIONAL_COMPONENTS:
            disabled.add(name)
        elif name.startswith("no_"):
            component = name[3:]
            disabled.add("specialist_anchor" if component == "specialist_client" else component)
        elif name.startswith("without_"):
            disabled.add(name[8:])
    return disabled


def _enabled(cfg, name):
    return name not in _disabled_components(cfg)


def _ablation_config(cfg):
    components = [
        "image_space",
        "diagnostic_evidence",
        "diagnostic_client_information",
        "medical_weighted_fusion",
        "class_routing",
        "checkpoint_consistency",
        "delta_agreement_weighting",
        "conflict_stabilization",
        "specialist_anchor",
        "statistical_alignment",
    ]
    return {
        "labels": _ablation_labels(cfg),
        "disabled_components": sorted(_disabled_components(cfg)),
        "enabled": {name: _enabled(cfg, name) for name in components},
    }


def _is_medical_image_task(meta):
    return meta.get("task_type") in {"small", "vlm"} and meta.get("dataset") in MEDICAL_IMAGE_DATASETS


def _is_natural_control_task(meta):
    return meta.get("task_type") == "small" and meta.get("dataset") in NATURAL_CONTROL_DATASETS


def _domain_control_enabled(cfg):
    value = (cfg or {}).get("my_merge_domain_control", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _model_family(meta):
    if meta.get("task_type") == "vlm":
        return "vlm"
    return "cnn" if meta.get("model") in {"resnet", "mobilenet", "convnext"} else "transformer"


def _normalize(values, fallback):
    scores = torch.as_tensor(values, dtype=torch.float32)
    fallback = torch.as_tensor(fallback, dtype=torch.float32)
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = scores.sum()
    return fallback if float(total.item()) <= 0 else scores / total


def _stable_weights(scores, fallback, temperature=0.75, floor=0.025):
    fallback = torch.as_tensor(fallback, dtype=torch.float32)
    scores = torch.as_tensor(scores, dtype=torch.float32).clamp_min(EPS)
    logits = torch.log(scores)
    logits = logits - logits.mean()
    scale = torch.clamp(logits.abs().median(), min=0.25)
    probs = torch.softmax(logits / (temperature * scale + EPS), dim=0)
    if floor > 0:
        probs = (1.0 - floor) * probs + floor * fallback
    return _normalize(probs, fallback)


def _coverage_gate(labels, num_classes):
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1)
    if labels.numel() == 0:
        return 0.0, 0.0
    counts = torch.bincount(labels, minlength=max(1, int(num_classes))).float()
    present = float((counts > 0).sum().item()) / float(max(1, int(num_classes)))
    probs = counts / (counts.sum() + EPS)
    entropy = -torch.sum(probs * torch.log(probs + EPS))
    max_entropy = torch.log(torch.tensor(float(max(2, int(num_classes))), dtype=torch.float32))
    balance = float(torch.clamp(entropy / (max_entropy + EPS), min=0.0, max=1.0).item())
    size = float(labels.numel()) / (float(labels.numel()) + 32.0 * float(max(1, int(num_classes))))
    return present, max(0.0, min(1.0, size * balance))


def _score_gate(scores, reliability):
    scores = torch.as_tensor(scores, dtype=torch.float32).clamp_min(0.0)
    if scores.numel() <= 1 or float(scores.mean().item()) <= 0:
        return 0.0
    separation = float(torch.clamp(scores.std(unbiased=False) / (scores.mean() + EPS), min=0.0, max=1.0).item())
    return max(0.0, min(1.0, (reliability - 0.10) / 0.90)) * max(0.0, min(1.0, separation / 0.05))


def _calibrated(scores, fallback, reliability, temperature=0.75, floor=0.025):
    fallback = torch.as_tensor(fallback, dtype=torch.float32)
    raw = _stable_weights(scores, fallback, temperature=temperature, floor=floor)
    gate = _score_gate(scores, reliability)
    return _normalize((1.0 - gate) * fallback + gate * raw, fallback)


def _image01(meta, x, cfg):
    x = x.detach().float()
    if not _enabled(cfg, "image_space"):
        return x
    if meta.get("task_type") == "vlm" and x.shape[1] == 3:
        mean = torch.tensor(CLIP_IMAGE_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_IMAGE_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x = x * std + mean
    return torch.clamp(x, 0.0, 1.0)


def _safe_quantile(values, q):
    return torch.quantile(values.reshape(values.shape[0], -1), q=float(q), dim=1).view(-1, 1, 1)


def _sobel(gray):
    sx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=gray.device).view(1, 1, 3, 3)
    sy = sx.transpose(2, 3)
    g = gray.unsqueeze(1)
    return torch.sqrt(F.conv2d(g, sx, padding=1).square() + F.conv2d(g, sy, padding=1).square() + EPS).squeeze(1)


def _local_var(gray, kernel=7):
    g = gray.unsqueeze(1)
    mean = F.avg_pool2d(g, kernel_size=kernel, stride=1, padding=kernel // 2)
    mean2 = F.avg_pool2d(g * g, kernel_size=kernel, stride=1, padding=kernel // 2)
    return torch.clamp(mean2 - mean.square(), min=0.0).squeeze(1)


def _morph_features(meta, x, cfg):
    x = _image01(meta, x, cfg)
    gray = x[:, 0] if x.shape[1] == 1 else x.mean(dim=1)
    low = _safe_quantile(gray, 0.04)
    high = _safe_quantile(gray, 0.96)
    gray = torch.clamp((gray - low) / (high - low + EPS), 0.0, 1.0)
    if not _enabled(cfg, "diagnostic_evidence"):
        return torch.ones((gray.shape[0], FEATURE_DIM), dtype=gray.dtype, device=gray.device)

    edge = _sobel(gray)
    local_mean = F.avg_pool2d(gray.unsqueeze(1), kernel_size=9, stride=1, padding=4).squeeze(1)
    contrast_map = (gray - local_mean).abs()
    texture_map = torch.sqrt(_local_var(gray, kernel=7) + EPS)
    evidence = 0.42 * edge + 0.34 * contrast_map + 0.24 * texture_map
    mask = (evidence >= _safe_quantile(evidence, 0.70)).float()
    mass = mask.sum(dim=(1, 2)) + EPS
    boundary = (edge * mask).sum(dim=(1, 2)) / mass
    contrast = (contrast_map * mask).sum(dim=(1, 2)) / mass
    texture = (texture_map * mask).sum(dim=(1, 2)) / mass
    reliability = torch.clamp((0.55 * contrast + 0.45 * boundary) / 0.55, 0.05, 2.0)
    salience = (0.35 + boundary) * (0.35 + contrast) * (0.35 + texture) * 0.95 * reliability
    return torch.stack([boundary, contrast, texture, salience, reliability], dim=1)


def _sample_weights(features):
    importance = features[:, 3] / (features[:, 3].mean() + EPS)
    reliability = torch.clamp(features[:, 4], min=0.15, max=1.5)
    weights = importance * (0.65 + 0.35 * reliability)
    return torch.clamp(weights / (weights.mean() + EPS), min=0.20, max=5.0)


def _neutral_feature_summary(num_classes):
    mean = torch.ones(FEATURE_DIM, dtype=torch.float32)
    class_means = torch.stack([mean.clone() for _ in range(max(1, int(num_classes)))], dim=0)
    return {
        "class_means": class_means,
        "class_counts": torch.zeros(max(1, int(num_classes)), dtype=torch.float32),
        "global_mean": mean,
        "global_std": torch.zeros(FEATURE_DIM, dtype=torch.float32),
        "num_samples": 0,
        "label_coverage_ratio": 0.0,
        "source": "neutral_no_raw_data",
    }


def _canonical_feature_summary(class_means, global_mean, global_std):
    class_means = torch.as_tensor(class_means, dtype=torch.float32)
    global_mean = torch.as_tensor(global_mean, dtype=torch.float32)
    global_std = torch.as_tensor(global_std, dtype=torch.float32)
    if class_means.ndim == 2 and class_means.shape[1] == FEATURE_DIM + 1:
        class_means = class_means[:, 1:]
    if global_mean.ndim == 1 and global_mean.numel() == FEATURE_DIM + 1:
        global_mean = global_mean[1:]
    if global_std.ndim == 1 and global_std.numel() == FEATURE_DIM + 1:
        global_std = global_std[1:]
    return class_means, global_mean, global_std


def _feature_summary_path(meta, cfg):
    root = cfg.get("my_merge_feature_summary_root")
    if not root:
        return None
    model_name = meta.get("model") if meta.get("task_type") == "small" else str(meta.get("clip_model", meta.get("model", ""))).split("/")[-1]
    beta = str(meta.get("beta", "0")).replace(".", "p")
    name = f"{meta.get('task_type')}__{meta.get('dataset')}__{model_name}__c{meta.get('num_clients')}__b{beta}__s{meta.get('seed')}.json"
    return Path(root) / name


def _load_feature_summary_file(meta, cfg):
    path = _feature_summary_path(meta, cfg)
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("source", "feature_summary_file")
    return payload


def _collect_feature_summary(meta, cfg):
    num_classes = int(meta["num_classes"])
    uploaded = cfg.get("my_merge_feature_summary") or _load_feature_summary_file(meta, cfg)
    if isinstance(uploaded, dict):
        class_means = torch.as_tensor(uploaded.get("class_means"), dtype=torch.float32)
        class_counts = torch.as_tensor(uploaded.get("class_counts"), dtype=torch.float32)
        global_mean = torch.as_tensor(uploaded.get("global_mean", class_means.mean(dim=0)), dtype=torch.float32)
        global_std = torch.as_tensor(uploaded.get("global_std", torch.zeros_like(global_mean)), dtype=torch.float32)
        class_means, global_mean, global_std = _canonical_feature_summary(class_means, global_mean, global_std)
        return {
            "class_means": class_means,
            "class_counts": class_counts,
            "global_mean": global_mean,
            "global_std": global_std,
            "num_samples": int(uploaded.get("num_samples", int(class_counts.sum().item()))),
            "label_coverage_ratio": float(uploaded.get("label_coverage_ratio", float((class_counts > 0).float().mean().item()))),
            "source": uploaded.get("source", "uploaded_client_summary"),
        }

    require_summary = str(cfg.get("my_merge_require_feature_summary", False)).strip().lower() in {"1", "true", "yes", "on"}
    if require_summary:
        path = _feature_summary_path(meta, cfg)
        raise FileNotFoundError(f"my_merge feature summary is required but missing: {path}")

    runtime = build_runtime(
        meta=meta,
        data_root=cfg["data_root"],
        split=cfg.get("stats_split", "val"),
        batch_size=int(cfg.get("stats_batch_size", 32) or cfg.get("batch_size", 64)),
        num_workers=int(cfg.get("stats_num_workers", cfg.get("num_workers", 4))),
        device=torch.device("cpu"),
    )
    max_batches = int(cfg.get("my_merge_stats_max_batches", DEFAULT_STATS_MAX_BATCHES))
    if _model_family(meta) in {"transformer", "vlm"} and "my_merge_stats_max_batches" not in cfg:
        max_batches = 0
    if max_batches <= 0:
        return _neutral_feature_summary(num_classes)

    features, labels = [], []
    for batch_idx, (x, y) in enumerate(runtime["loader"]):
        if batch_idx >= max_batches:
            break
        features.append(_morph_features(meta, x, cfg).cpu())
        labels.append(y.clone())
    if not features:
        return _neutral_feature_summary(num_classes)

    features = torch.cat(features, dim=0).float()
    labels = torch.cat(labels, dim=0).long().view(-1)
    global_mean = features.mean(dim=0)
    global_std = features.std(dim=0, unbiased=False) if features.shape[0] > 1 else torch.zeros(FEATURE_DIM)
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    class_means = []
    for cls_idx in range(num_classes):
        mask = labels == cls_idx
        class_means.append(features[mask].mean(dim=0) if torch.any(mask) else global_mean)
    return {
        "class_means": torch.stack(class_means, dim=0),
        "class_counts": class_counts,
        "global_mean": global_mean,
        "global_std": global_std,
        "num_samples": int(labels.numel()),
        "label_coverage_ratio": float((class_counts > 0).float().mean().item()),
        "source": "local_feature_summary_simulation",
    }


def _class_prior_from_meta(meta, num_classes):
    priors = torch.zeros(num_classes, dtype=torch.float32)
    for client in meta.get("clients", []):
        samples = float(client.get("num_samples", 1.0))
        classes = [int(x) for x in client.get("classes", [])]
        if not classes:
            continue
        share = samples / float(len(classes))
        for cls_idx in classes:
            if 0 <= cls_idx < num_classes:
                priors[cls_idx] += share
    if float(priors.sum().item()) <= 0:
        priors.fill_(1.0)
    return priors / (priors.sum() + EPS)


def _client_information(meta, feature_summary, cfg, base_weights):
    base = torch.as_tensor(base_weights, dtype=torch.float32)
    num_classes = int(meta["num_classes"])
    class_means = torch.as_tensor(feature_summary["class_means"], dtype=torch.float32)
    class_counts = torch.as_tensor(feature_summary["class_counts"], dtype=torch.float32)
    global_mean = torch.as_tensor(feature_summary["global_mean"], dtype=torch.float32).clamp_min(EPS)
    label_coverage = float(feature_summary.get("label_coverage_ratio", 0.0))
    reliability = float(torch.clamp(global_mean[4] / (global_mean[4] + 0.50), min=0.0, max=1.0).item())
    client_samples = torch.tensor([float(c.get("num_samples", 1.0)) for c in meta.get("clients", [])], dtype=torch.float32)
    if client_samples.numel() != base.numel():
        client_samples = torch.ones_like(base)
    sample_prior = _normalize(torch.sqrt(client_samples.clamp_min(1.0)), fallback=base)
    metadata_prior = _normalize(0.55 * base + 0.45 * sample_prior, fallback=base)

    union_classes = set()
    for client in meta.get("clients", []):
        union_classes.update(int(x) for x in client.get("classes", []))
    metadata_coverage = len([x for x in union_classes if 0 <= x < num_classes]) / float(max(1, num_classes))
    support_gate = max(label_coverage, metadata_coverage)

    if not _enabled(cfg, "diagnostic_client_information"):
        class_weights = torch.stack([base.clone() for _ in range(num_classes)], dim=0)
        rows = []
        for idx, weight in enumerate(base.tolist()):
            client = meta["clients"][idx] if idx < len(meta.get("clients", [])) else {}
            rows.append({
                "client_index": idx,
                "client_name": client.get("checkpoint", f"client_{idx}.pt"),
                "seen_classes": [int(x) for x in client.get("classes", [])],
                "base_weight": float(weight),
                "overall_weight": float(weight),
                "morphology_weight": float(weight),
                "support_score": float(weight),
                "feature_salience": 1.0,
                "feature_reliability": reliability,
            })
        return base, base, class_weights, rows, label_coverage, support_gate, reliability

    class_prior = _class_prior_from_meta(meta, num_classes)
    evidence_gate = 0.30 + 0.45 * support_gate
    class_gate = 0.0 if not _enabled(cfg, "class_routing") else max(0.15 if label_coverage >= 0.50 else 0.0, support_gate)
    overall_scores, morph_scores, summaries = [], [], []
    class_scores = torch.zeros(num_classes, base.numel(), dtype=torch.float32)

    for idx in range(base.numel()):
        client = meta["clients"][idx] if idx < len(meta.get("clients", [])) else {}
        seen = [int(x) for x in client.get("classes", []) if 0 <= int(x) < num_classes]
        if seen:
            counts = class_counts[seen].clamp_min(0.0)
            if float(counts.sum().item()) <= 0:
                counts = torch.ones(len(seen), dtype=torch.float32)
            client_feature = (class_means[seen] * (counts / (counts.sum() + EPS)).view(-1, 1)).sum(dim=0)
            class_coverage = len(seen) / float(max(1, num_classes))
        else:
            client_feature = global_mean
            class_coverage = 1.0 / float(max(1, num_classes))
        feature_ratio = torch.clamp(client_feature / global_mean, min=0.25, max=2.50)
        boundary = float(feature_ratio[0].item())
        contrast = float(feature_ratio[1].item())
        texture = float(feature_ratio[2].item())
        salience = float(feature_ratio[3].item())
        feature_reliability = float(feature_ratio[4].item())
        morph_quality = max(0.10, 0.26 * boundary + 0.24 * contrast + 0.20 * texture + 0.18 * salience + 0.12 * feature_reliability)
        support_score = float(torch.sqrt(torch.tensor(float(client.get("num_samples", 1.0)))).item())
        support_score = support_score / float(torch.sqrt(client_samples).mean().item() + EPS)
        prior = float(metadata_prior[idx].item())
        overall_score = prior * (0.35 + 0.30 * support_score + 0.20 * class_coverage + 0.15 * morph_quality)
        morph_score = prior * (0.25 + 0.55 * morph_quality + 0.12 * class_coverage + 0.08 * support_score)
        summaries.append({
            "support_score": support_score,
            "class_coverage": class_coverage,
            "morph_quality": morph_quality,
            "feature_salience": salience,
            "feature_reliability": feature_reliability,
            "overall_score": overall_score,
            "morph_score": morph_score,
        })
        overall_scores.append(overall_score)
        morph_scores.append(morph_score)

        for cls_idx in range(num_classes):
            cls_feature = torch.clamp(class_means[cls_idx] / global_mean, min=0.25, max=2.50)
            cls_quality = 0.30 * cls_feature[0] + 0.25 * cls_feature[1] + 0.20 * cls_feature[2] + 0.15 * cls_feature[3] + 0.10 * cls_feature[4]
            seen_bonus = 1.35 if cls_idx in seen else 0.42
            rarity = torch.clamp((class_prior.mean() / (class_prior[cls_idx] + EPS)).sqrt(), min=0.75, max=1.40)
            class_scores[cls_idx, idx] = prior * seen_bonus * float(rarity.item()) * float(0.25 + 0.75 * cls_quality.item())

    overall = _normalize((1.0 - evidence_gate) * metadata_prior + evidence_gate * _calibrated(overall_scores, metadata_prior, reliability, 0.78, 0.025), metadata_prior)
    morph = _normalize((1.0 - evidence_gate) * metadata_prior + evidence_gate * _calibrated(morph_scores, metadata_prior, reliability, 0.72, 0.025), metadata_prior)
    class_weights = []
    for cls_idx in range(num_classes):
        cls_w = _calibrated(class_scores[cls_idx], morph, reliability, 0.84, 0.05)
        class_weights.append(_normalize((1.0 - class_gate) * morph + class_gate * cls_w, morph))

    rows = []
    for idx, summary in enumerate(summaries):
        client = meta["clients"][idx] if idx < len(meta.get("clients", [])) else {}
        rows.append({
            "client_index": idx,
            "client_name": client.get("checkpoint", f"client_{idx}.pt"),
            "seen_classes": [int(x) for x in client.get("classes", [])],
            "base_weight": float(base[idx]),
            "overall_weight": float(overall[idx]),
            "morphology_weight": float(morph[idx]),
            "support_score": float(summary["support_score"]),
            "class_coverage": float(summary["class_coverage"]),
            "morph_quality": float(summary["morph_quality"]),
            "feature_salience": float(summary["feature_salience"]),
            "feature_reliability": float(summary["feature_reliability"]),
            "overall_score": float(summary["overall_score"]),
            "morphology_score": float(summary["morph_score"]),
            "evidence_reliability": reliability,
        })
    return overall, morph, torch.stack(class_weights, dim=0), rows, label_coverage, support_gate, reliability


def _module1(meta, cfg, base_weights):
    feature_summary = _collect_feature_summary(meta, cfg)
    overall, morph, class_weights, rows, label_coverage, support_gate, reliability = _client_information(meta, feature_summary, cfg, base_weights)
    return {
        "feature_summary": feature_summary,
        "base_weights": torch.as_tensor(base_weights, dtype=torch.float32),
        "overall_weights": overall,
        "morphology_weights": morph,
        "class_weights": class_weights,
        "client_diagnostic_information": rows,
        "evidence_support_gate": support_gate,
        "label_coverage_ratio": label_coverage,
        "evidence_reliability": reliability,
        "class_morphology_separation": _class_morphology_separation(feature_summary),
    }


def _class_morphology_separation(feature_summary):
    class_means = torch.as_tensor(feature_summary["class_means"], dtype=torch.float32)
    global_mean = torch.as_tensor(feature_summary["global_mean"], dtype=torch.float32)
    if class_means.ndim != 2 or class_means.shape[0] <= 1:
        return 0.0
    selected = class_means
    denom = global_mean[:selected.shape[1]].abs().clamp_min(EPS)
    normalized = selected / denom.view(1, -1)
    return float(torch.clamp(normalized.std(dim=0, unbiased=False).mean(), min=0.0, max=1.0).item())


def _param_group(key, meta):
    if _model_family(meta) == "cnn":
        early = ("conv1", "bn1", "layer1", "layer2", "stem", "features.0", "features.1")
        late = ("layer4", "fc", "classifier", "head", "norm", "features.16", "features.17")
    else:
        early = ("patch_embed", "embeddings", "visual.conv1", "blocks.0", "blocks.1", "layers.0", "stages.0")
        late = ("head", "fc", "classifier", "norm", "visual.ln_post", "visual.proj", "blocks.10", "blocks.11", "layers.3", "stages.3")
    if any(t in key for t in early):
        return "early"
    if any(t in key for t in late):
        return "late"
    return "mid"


def _is_classifier_tensor(key, tensor, num_classes):
    return (
        torch.is_floating_point(tensor)
        and tensor.ndim in (1, 2)
        and tensor.shape[0] == num_classes
        and any(t in key for t in ("fc", "classifier", "head", "proj"))
    )


def _classifier_bias_keys(state_dict, num_classes):
    return [
        k for k, v in state_dict.items()
        if torch.is_floating_point(v) and v.ndim == 1 and v.shape[0] == num_classes and any(t in k for t in ("fc", "classifier", "head", "proj"))
    ]


def _find_classifier_keys(state_dict, num_classes):
    weight_key, bias_key = None, None
    for key, value in state_dict.items():
        if (
            torch.is_floating_point(value)
            and value.ndim == 2
            and value.shape[0] == num_classes
            and any(t in key for t in ("fc", "classifier", "head", "proj"))
        ):
            weight_key = key
            prefix = key.rsplit(".", 1)[0] if "." in key else ""
            for bias in _classifier_bias_keys(state_dict, num_classes):
                if prefix and bias.startswith(prefix):
                    bias_key = bias
                    break
            if bias_key is None:
                biases = _classifier_bias_keys(state_dict, num_classes)
                bias_key = biases[0] if biases else None
            break
    return weight_key, bias_key


def _weighted_delta(values, reference, weights):
    out = torch.zeros_like(reference.detach().float())
    for value, weight in zip(values, weights):
        out.add_(value.detach().float() - reference.detach().float(), alpha=float(weight))
    return out


def _weighted_average(values, weights):
    out = values[0].detach().clone().float() * float(weights[0])
    for value, weight in zip(values[1:], weights[1:]):
        out.add_(value.detach().float(), alpha=float(weight))
    return out


def _consensus(overall, morph, base):
    return _normalize(0.5 * torch.as_tensor(overall) + 0.5 * torch.as_tensor(morph), base)


def _checkpoint_consistency_scores(state_dicts, reference, meta, num_classes):
    num_clients = len(state_dicts)
    if num_clients <= 2 or reference is None:
        return torch.ones(num_clients, dtype=torch.float32), {"enabled": False}

    pair_sums = torch.zeros(num_clients, num_clients, dtype=torch.float64)
    used_tensors, used_params = 0, 0
    for key, ref in reference.items():
        if key not in state_dicts[0]:
            continue
        first = state_dicts[0][key]
        if (
            not torch.is_floating_point(first)
            or not torch.is_floating_point(ref)
            or tuple(first.shape) != tuple(ref.shape)
            or _is_classifier_tensor(key, first, num_classes)
            or _param_group(key, meta) == "early"
        ):
            continue
        values = [sd[key].detach().float() - ref.detach().float() for sd in state_dicts]
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                pair_sums[i, j] += float((values[i] - values[j]).square().sum().item())
        used_tensors += 1
        used_params += int(first.numel())

    if used_tensors == 0:
        return torch.ones(num_clients, dtype=torch.float32), {"enabled": False}

    pair_dist = pair_sums + pair_sums.t()
    mean_dist = torch.sqrt(pair_dist.sum(dim=1) / float(max(1, num_clients - 1))).float()
    median = torch.clamp(mean_dist.median(), min=EPS)
    consistency = torch.exp(-torch.clamp(mean_dist / median - 1.0, min=0.0)).clamp(0.05, 1.0)
    trace = {
        "enabled": True,
        "used_tensors": int(used_tensors),
        "used_params": int(used_params),
        "mean_trunk_distance": [float(x) for x in mean_dist.tolist()],
        "consistency_scores": [float(x) for x in consistency.tolist()],
    }
    return consistency, trace


def _apply_checkpoint_consistency(info, consistency, cfg):
    base = torch.as_tensor(info["base_weights"], dtype=torch.float32)
    consistency = torch.as_tensor(consistency, dtype=torch.float32)
    if consistency.numel() != base.numel() or not _enabled(cfg, "checkpoint_consistency"):
        return info, {"applied": False}

    pre_consensus = _consensus(info["overall_weights"], info["morphology_weights"], base)
    concentration = float(torch.clamp(pre_consensus.max() - pre_consensus.mean(), min=0.0, max=1.0).item())
    outlier = float(torch.clamp(1.0 - consistency.min(), min=0.0, max=1.0).item())
    top_idx = int(torch.argmax(pre_consensus).item())
    outlier_idx = int(torch.argmin(consistency).item())
    top_is_outlier = top_idx == outlier_idx
    top_row = info["client_diagnostic_information"][top_idx] if top_idx < len(info["client_diagnostic_information"]) else {}
    top_class_coverage = float(top_row.get("class_coverage", 0.0))
    top_is_generalist = top_class_coverage >= 0.95
    gate = 1.0 if concentration >= 0.33 and outlier >= 0.45 and top_is_outlier and top_is_generalist else 0.0
    if gate <= 0.0:
        return info, {
            "applied": False,
            "gate": 0.0,
            "outlier_strength": outlier,
            "weight_concentration": concentration,
            "top_weight_client": top_idx,
            "outlier_client": outlier_idx,
            "top_is_outlier": bool(top_is_outlier),
            "top_class_coverage": top_class_coverage,
            "top_is_generalist": bool(top_is_generalist),
        }

    power = 2.0

    def adjust(weights):
        weights = torch.as_tensor(weights, dtype=torch.float32)
        guarded = _normalize(weights * consistency.pow(power), weights)
        return _normalize((1.0 - gate) * weights + gate * guarded, weights)

    adjusted = dict(info)
    adjusted["overall_weights"] = adjust(info["overall_weights"])
    adjusted["morphology_weights"] = adjust(info["morphology_weights"])
    adjusted["class_weights"] = torch.stack([adjust(row) for row in info["class_weights"]], dim=0)
    rows = []
    for idx, row in enumerate(info["client_diagnostic_information"]):
        updated = dict(row)
        updated["overall_weight"] = float(adjusted["overall_weights"][idx])
        updated["morphology_weight"] = float(adjusted["morphology_weights"][idx])
        updated["checkpoint_consistency"] = float(consistency[idx])
        rows.append(updated)
    adjusted["client_diagnostic_information"] = rows
    return adjusted, {
        "applied": True,
        "gate": float(gate),
        "power": float(power),
        "outlier_strength": outlier,
        "weight_concentration": concentration,
        "top_weight_client": top_idx,
        "outlier_client": outlier_idx,
        "top_is_outlier": bool(top_is_outlier),
        "top_class_coverage": top_class_coverage,
        "top_is_generalist": bool(top_is_generalist),
        "pre_consensus": [float(x) for x in pre_consensus.tolist()],
        "post_consensus": [float(x) for x in _consensus(adjusted["overall_weights"], adjusted["morphology_weights"], base).tolist()],
    }


def _layer_weights(group, base, overall, morph, consensus):
    if group == "early":
        return _normalize(0.70 * base + 0.30 * morph, base)
    if group == "late":
        return _normalize(0.30 * base + 0.70 * overall, base)
    return _normalize(0.45 * base + 0.55 * consensus, base)


def _delta_agreement_weights(values, reference, weights, base, strength=0.35):
    if reference is None or reference.numel() < 64 or len(values) <= 1:
        return weights, 0.0
    ref = reference.detach().float()
    deltas = torch.stack([(value.detach().float() - ref).reshape(-1) for value in values], dim=0)
    common = torch.zeros_like(deltas[0])
    for delta, coef in zip(deltas, torch.as_tensor(base, dtype=torch.float32).tolist()):
        common.add_(delta, alpha=float(coef))
    common_norm = common.norm().clamp_min(EPS)
    delta_norms = deltas.norm(dim=1).clamp_min(EPS)
    cos = (deltas @ common) / (delta_norms * common_norm)
    agreement = torch.clamp((cos + 1.0) * 0.5, min=0.05, max=1.0)
    prior = torch.as_tensor(weights, dtype=torch.float32)
    aligned = _normalize(prior * agreement, prior)
    adjusted = _normalize((1.0 - float(strength)) * prior + float(strength) * aligned, prior)
    return adjusted, float(torch.mean(torch.abs(adjusted - prior)).item())


def _medical_weighted_merge(state_dicts, base_merged, reference, base, overall, morph, class_weights, num_classes, meta, cfg):
    consensus = _consensus(overall, morph, base)
    group_weights = {g: _layer_weights(g, base, overall, morph, consensus) for g in ("early", "mid", "late")}
    merged = OrderedDict()
    routed = {"early": 0, "mid": 0, "late": 0, "classifier": 0}
    agreement_layers, agreement_shift = 0, 0.0

    for key in state_dicts[0]:
        values = [sd[key] for sd in state_dicts]
        first = values[0]
        if not torch.is_floating_point(first):
            merged[key] = base_merged[key].detach().clone()
            continue
        ref = reference.get(key) if reference is not None else None
        use_delta = ref is not None and torch.is_floating_point(ref) and tuple(ref.shape) == tuple(first.shape)

        if _is_classifier_tensor(key, first, num_classes):
            out = ref.detach().clone().float() if use_delta else first.detach().clone().float().zero_()
            for cls_idx in range(first.shape[0]):
                weights = _normalize(0.45 * class_weights[cls_idx] + 0.55 * consensus, consensus) if _enabled(cfg, "class_routing") else consensus
                row_values = [v[cls_idx] for v in values]
                out[cls_idx] = (ref[cls_idx].float() + _weighted_delta(row_values, ref[cls_idx], weights)).to(out.dtype) if use_delta else _weighted_average(row_values, weights)
            merged[key] = out.to(first.dtype)
            routed["classifier"] += 1
            continue

        group = _param_group(key, meta)
        weights = group_weights[group]
        if use_delta and _enabled(cfg, "delta_agreement_weighting"):
            adjusted, shift = _delta_agreement_weights(values, ref, weights, base)
            if shift > 0:
                weights = adjusted
                agreement_layers += 1
                agreement_shift += shift
        merged[key] = (ref.float() + _weighted_delta(values, ref, weights)).to(first.dtype) if use_delta else _weighted_average(values, weights).to(first.dtype)
        routed[group] += 1
    return merged, {
        "fusion_weights": [float(x) for x in consensus.tolist()],
        "group_weights": {g: [float(x) for x in w.tolist()] for g, w in group_weights.items()},
        "routing_summary": {
            **routed,
            "delta_space": reference is not None,
            "delta_agreement_layers": int(agreement_layers),
            "delta_agreement_mean_shift": float(agreement_shift / max(1, agreement_layers)),
        },
    }


def _sign_delta_merge(state_dicts, base_merged, reference, param_names, weights, density=0.5):
    task_matrix, base_vector = build_task_matrix(state_dicts, reference, param_names)
    trimmed = mask_smallest_magnitude(task_matrix, preserve_density=density)
    merged_delta = disjoint_merge(trimmed, elect_sign(trimmed), weights=weights)
    params = vector_to_param_dict(base_vector + merged_delta, reference, param_names)
    return overlay_param_dict(base_merged, params)


def _task_conflict_stats(state_dicts, reference, param_names):
    task_matrix, _ = build_task_matrix(state_dicts, reference, param_names)
    abs_mass = task_matrix.abs().sum(dim=0)
    active = abs_mass > EPS
    if torch.any(active):
        sign_agree = task_matrix.sum(dim=0).abs() / (abs_mass + EPS)
        sign_conflict = 1.0 - float((sign_agree[active] * abs_mass[active]).sum().item() / (abs_mass[active].sum().item() + EPS))
    else:
        sign_conflict = 0.0

    flat = task_matrix.float()
    norms = flat.norm(dim=1).clamp_min(EPS)
    if flat.shape[0] > 1:
        unit = flat / norms.view(-1, 1)
        cos = unit @ unit.t()
        pair_mask = ~torch.eye(flat.shape[0], dtype=torch.bool, device=flat.device)
        direction_conflict = float(torch.clamp((1.0 - cos[pair_mask].mean()) * 0.5, min=0.0, max=1.0).item())
    else:
        direction_conflict = 0.0
    norm_dispersion = float(torch.clamp(norms.std(unbiased=False) / (norms.mean() + EPS), min=0.0, max=1.0).item())
    conflict = max(0.0, min(1.0, 0.45 * sign_conflict + 0.35 * direction_conflict + 0.20 * norm_dispersion))
    return {
        "sign_conflict": sign_conflict,
        "direction_conflict": direction_conflict,
        "norm_dispersion": norm_dispersion,
        "conflict_score": conflict,
    }


def _delta_blend_weight(conflict_stats, reliability, meta):
    conflict = float(conflict_stats.get("conflict_score", 0.0))
    reliability = float(max(0.0, min(1.0, reliability)))
    family = _model_family(meta)
    cap = 0.35
    raw = cap * max(0.0, conflict - 0.20) / 0.80
    return max(0.0, min(cap, raw * (0.55 + 0.45 * reliability)))


def _specialist_scores(client_rows, meta):
    family = _model_family(meta)

    def score(row):
        overall = float(row.get("overall_weight", 0.0))
        morph = float(row.get("morphology_weight", overall))
        support = float(row.get("support_score", 1.0))
        coverage = float(row.get("class_coverage", 0.0))
        salience = float(row.get("feature_salience", 1.0))
        reliability = float(row.get("feature_reliability", 1.0))
        if family in {"transformer", "vlm"}:
            return 0.38 * morph + 0.24 * overall + 0.18 * coverage + 0.12 * salience + 0.08 * reliability
        return 0.34 * overall + 0.30 * morph + 0.16 * support + 0.12 * coverage + 0.08 * reliability

    return torch.tensor([score(row) for row in client_rows], dtype=torch.float32)


def _specialist_separation_gate(separation):
    separation = float(max(0.0, min(1.0, separation)))
    return max(0.0, min(1.0, (separation - SPECIALIST_SEPARATION_START) / (SPECIALIST_SEPARATION_FULL - SPECIALIST_SEPARATION_START + EPS)))


def _specialist_anchor_weight(scores, consensus, reliability, meta, class_separation):
    if scores.numel() <= 1:
        return 0.0, 0
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    order = torch.argsort(scores, descending=True)
    best_idx = int(order[0].item())
    best = float(scores[best_idx].item())
    second = float(scores[int(order[1].item())].item())
    dominance = (best - second) / (best + EPS)
    consensus = torch.as_tensor(consensus, dtype=torch.float32)
    concentration = float(torch.clamp(consensus.max() - consensus.mean(), min=0.0, max=1.0).item())
    reliability = float(max(0.0, min(1.0, reliability)))
    family = _model_family(meta)
    cap = 0.55 if family in {"transformer", "vlm"} else 0.35
    gate = max(0.0, min(1.0, (dominance - 0.06) / 0.24))
    gate *= _specialist_separation_gate(class_separation)
    weight = cap * gate * (0.50 + 0.50 * reliability) * (0.70 + 0.30 * min(1.0, concentration * len(consensus)))
    weight = max(0.0, min(cap, weight))
    if weight < MIN_SPECIALIST_ANCHOR_WEIGHT:
        weight = 0.0
    return weight, best_idx


def _blend_state_dicts(left, right, right_weight):
    right_weight = float(right_weight)
    left_weight = 1.0 - right_weight
    merged = OrderedDict()
    for key, left_value in left.items():
        right_value = right[key]
        if torch.is_floating_point(left_value):
            merged[key] = (left_weight * left_value.detach().float() + right_weight * right_value.detach().float()).to(left_value.dtype)
        else:
            merged[key] = left_value.detach().clone()
    return merged


def _blend_label(value):
    return f"{float(value):.2f}".replace(".", "p")


def _statistical_alignment(state_dicts, state, weights):
    aligned = OrderedDict((k, v.detach().clone()) for k, v in state.items())
    weights = torch.as_tensor(weights, dtype=torch.float32)
    aligned_keys = []
    for key, value in state.items():
        if "running_mean" not in key and "running_var" not in key and "num_batches_tracked" not in key:
            continue
        values = [sd[key] for sd in state_dicts if key in sd and tuple(sd[key].shape) == tuple(value.shape)]
        if len(values) != len(state_dicts):
            continue
        if torch.is_floating_point(value):
            aligned[key] = _weighted_average(values, weights).to(value.dtype)
        else:
            aligned[key] = values[int(torch.argmax(weights).item())].detach().clone()
        aligned_keys.append(key)
    return aligned, {
        "bn_stat_aligned": bool(aligned_keys),
        "aligned_bn_stat_tensors": len(aligned_keys),
        "privacy": "client_uploaded_running_statistics_only",
    }


def _module2_module3(state_dicts, base_merged, meta, cfg, info, reference, param_names):
    base = torch.as_tensor(info["base_weights"], dtype=torch.float32)
    consensus = _consensus(info["overall_weights"], info["morphology_weights"], base)
    module2_pool = ["delta_0p00"]
    routing = {}
    fusion_weights = [float(x) for x in base.tolist()]
    group_weights = {}

    if _enabled(cfg, "medical_weighted_fusion"):
        state, medical_trace = _medical_weighted_merge(
            state_dicts,
            base_merged,
            reference,
            base,
            info["overall_weights"],
            info["morphology_weights"],
            info["class_weights"],
            int(meta["num_classes"]),
            meta,
            cfg,
        )
        module2_pool.append("medical_weighted_fusion")
        routing.update(medical_trace.get("routing_summary", {}))
        fusion_weights = medical_trace.get("fusion_weights", fusion_weights)
        group_weights = medical_trace.get("group_weights", {})
        selected_steps = ["medical_weighted_fusion"]
    else:
        state = base_merged
        selected_steps = ["delta_0p00"]

    conflict_stats = {}
    delta_weight = 0.0
    if _enabled(cfg, "conflict_stabilization") and meta.get("task_type") == "small" and reference is not None and param_names is not None:
        conflict_stats = _task_conflict_stats(state_dicts, reference, param_names)
        sign_state = _sign_delta_merge(state_dicts, base_merged, reference, param_names, consensus.tolist(), density=0.50)
        delta_weight = _delta_blend_weight(conflict_stats, info.get("evidence_reliability", 0.0), meta)
        if delta_weight > 0:
            state = _blend_state_dicts(state, sign_state, right_weight=delta_weight)
        module2_pool.append("conflict_delta")
        selected_steps.append(f"conflict_delta_{_blend_label(delta_weight)}")

    specialist_weight, specialist_idx = 0.0, None
    if _enabled(cfg, "specialist_anchor"):
        scores = _specialist_scores(info["client_diagnostic_information"], meta)
        specialist_weight, specialist_idx = _specialist_anchor_weight(
            scores,
            consensus,
            info.get("evidence_reliability", 0.0),
            meta,
            info.get("class_morphology_separation", 0.0),
        )
        if specialist_weight > 0:
            specialist_state = OrderedDict((k, v.detach().clone()) for k, v in state_dicts[specialist_idx].items())
            state = _blend_state_dicts(state, specialist_state, right_weight=specialist_weight)
        module2_pool.append("specialist_anchor")
        selected_steps.append(f"specialist_anchor_{_blend_label(specialist_weight)}")

    alignment_trace = {"bn_stat_aligned": False, "aligned_bn_stat_tensors": 0}
    if _enabled(cfg, "statistical_alignment"):
        state, alignment_trace = _statistical_alignment(state_dicts, state, consensus)
        module2_pool.append("statistical_alignment")
        if alignment_trace.get("bn_stat_aligned"):
            selected_steps.append(f"statistical_alignment_bn_stats{alignment_trace.get('aligned_bn_stat_tensors', 0)}")

    routing.update({
        "selection_rule": "deterministic_medical_evidence_conflict_flow",
        "validated_candidates": 0,
        "conflict_stats": conflict_stats,
        "delta_blend_weight": float(delta_weight),
        "sign_delta_density": 0.50 if conflict_stats else 0.0,
        "specialist_anchor_weight": float(specialist_weight),
        "specialist_separation_gate": float(_specialist_separation_gate(info.get("class_morphology_separation", 0.0))),
        "specialist_client_index": None if specialist_idx is None else int(specialist_idx),
        "classifier_class_weight": 0.45 if _enabled(cfg, "class_routing") else 0.0,
        "statistical_alignment": alignment_trace,
    })
    return state, {
        "fusion_rule": "m1_medical_weighted_m2_conflict_representation_flow",
        "module2_candidate_pool": module2_pool,
        "module3_candidate_pool": [],
        "candidate_pool": module2_pool,
        "candidate_metrics": {},
        "selected_candidate": " + ".join(selected_steps),
        "routing_summary": routing,
        "fusion_weights": fusion_weights,
        "group_weights": group_weights,
        "evidence_reliability": info.get("evidence_reliability"),
        "class_morphology_separation": info.get("class_morphology_separation", 0.0),
    }


def merge_my_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    if meta is None or cfg is None:
        raise ValueError("my_merge requires task metadata and runtime config.")
    natural_control = _is_natural_control_task(meta) and _domain_control_enabled(cfg)
    if not _is_medical_image_task(meta) and not natural_control:
        raise ValueError(f"my_merge is defined only for medical image tasks, got dataset={meta.get('dataset')}.")

    base_merged, base_weights = average_state_dicts(state_dicts, weights)
    ablation = _ablation_config(cfg)
    if "avg_only" in {x.lower().replace("-", "_") for x in ablation["labels"]}:
        return base_merged, {
            "implementation": "medical_evidence_three_module_merge_v12_ablation_avg_only",
            "medical_only": True,
            "natural_domain_control": bool(natural_control),
            "ablation_config": ablation,
            "selected_candidate": "avg",
            "base_weights": [float(x) for x in base_weights],
            "normalized_weights": base_weights,
        }

    info = _module1(meta, cfg, base_weights)
    reference = param_names = None
    if _enabled(cfg, "medical_weighted_fusion") or _enabled(cfg, "conflict_stabilization"):
        reference, param_names = build_reference_bundle(meta, device="cpu")
    consistency_trace = {"enabled": False}
    consistency_guard_trace = {"applied": False}
    if _enabled(cfg, "checkpoint_consistency") and reference is not None:
        consistency_scores, consistency_trace = _checkpoint_consistency_scores(
            state_dicts,
            reference,
            meta,
            int(meta["num_classes"]),
        )
        info, consistency_guard_trace = _apply_checkpoint_consistency(info, consistency_scores, cfg)
    merged, fusion = _module2_module3(state_dicts, base_merged, meta, cfg, info, reference, param_names)

    implementation = "medical_evidence_three_module_posthoc_merge_v12"
    return merged, {
        "implementation": implementation,
        "medical_only": True,
        "natural_domain_control": bool(natural_control),
        "ablation_config": ablation,
        "modality": meta.get("dataset"),
        "model_family": _model_family(meta),
        "base_weights": [float(x) for x in base_weights],
        "overall_weights": [float(x) for x in info["overall_weights"].tolist()],
        "morphology_weights": [float(x) for x in info["morphology_weights"].tolist()],
        "class_weights": [[float(v) for v in row] for row in info["class_weights"].tolist()],
        "client_diagnostic_information": info["client_diagnostic_information"],
        "module2_candidate_pool": fusion["module2_candidate_pool"],
        "module3_candidate_pool": fusion["module3_candidate_pool"],
        "candidate_pool": fusion["candidate_pool"],
        "candidate_metrics": fusion["candidate_metrics"],
        "selected_candidate": fusion["selected_candidate"],
        "fusion_rule": fusion["fusion_rule"],
        "fusion_weights": fusion["fusion_weights"],
        "group_weights": fusion["group_weights"],
        "routing_summary": fusion["routing_summary"],
        "checkpoint_consistency": consistency_trace,
        "checkpoint_consistency_guard": consistency_guard_trace,
        "evidence_reliability": fusion["evidence_reliability"],
        "class_morphology_separation": fusion["class_morphology_separation"],
    }
