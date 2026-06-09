from collections import OrderedDict

import torch
import torch.nn.functional as F

from model import build_model
from model.clip_model import build_clip_model, encode_image_features, get_clip_base
from utils.runtime import build_reference_bundle, build_runtime, resolve_vlm_max_text_len
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
DEFAULT_STATS_MAX_BATCHES = 16
DEFAULT_BN_BATCHES = 4
DEFAULT_SELECTION_MEDICAL_WEIGHT = 0.15
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

M1 = {"image_space", "diagnostic_evidence", "diagnostic_client_information"}
M2 = {"medical_weighted_fusion", "validated_selection", "bn_recalibration"}
M3 = {"conflict_stabilization", "specialist_client", "prototype_head"}

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
    disabled = set()
    for raw in _ablation_labels(cfg):
        name = raw.lower().replace("-", "_")
        if name == "avg_only":
            disabled.update(M1 | M2 | M3)
        elif name in {"no_client_information", "no_diagnostic_information", "no_m1"}:
            disabled.update(M1)
        elif name == "no_diagnostic_evidence":
            disabled.add("diagnostic_evidence")
        elif name in {"no_fusion_selection", "no_medical_fusion_selection", "no_m2"}:
            disabled.update(M2 | M3)
        elif name in {
            "no_adaptive_candidates",
            "no_adaptive_candidate_generation",
            "no_m3",
            "no_conflict_stabilization",
            "no_sign_delta",
            "no_ties_delta",
            "no_avg_sign_blend",
            "no_sign_blend",
        }:
            disabled.update(M3)
        elif name in M1 or name in M2 or name in M3:
            disabled.add(name)
        elif name == "no_calibration":
            disabled.add("bn_recalibration")
        elif name.startswith("no_"):
            disabled.add(name[3:])
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
        "conflict_stabilization",
        "specialist_client",
        "prototype_head",
        "validated_selection",
        "bn_recalibration",
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


def _build_model_forward(meta, cfg, device):
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
            return base.logit_scale.exp() * image_features @ text_features.t()

        return model, forward_fn
    raise ValueError(f"Unsupported task_type: {meta.get('task_type')}")


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
        return torch.ones((gray.shape[0], 6), dtype=gray.dtype, device=gray.device)

    edge = _sobel(gray)
    local_mean = F.avg_pool2d(gray.unsqueeze(1), kernel_size=9, stride=1, padding=4).squeeze(1)
    contrast_map = (gray - local_mean).abs()
    texture_map = torch.sqrt(_local_var(gray, kernel=7) + EPS)
    evidence = 0.42 * edge + 0.34 * contrast_map + 0.24 * texture_map
    mask = (evidence >= _safe_quantile(evidence, 0.70)).float()
    mass = mask.sum(dim=(1, 2)) + EPS
    area = mask.mean(dim=(1, 2))
    boundary = (edge * mask).sum(dim=(1, 2)) / mass
    contrast = (contrast_map * mask).sum(dim=(1, 2)) / mass
    texture = (texture_map * mask).sum(dim=(1, 2)) / mass
    reliability = torch.clamp((0.55 * contrast + 0.45 * boundary) / 0.55, 0.05, 2.0)
    salience = (0.35 + boundary) * (0.35 + contrast) * (0.35 + texture) * (0.65 + area) * reliability
    return torch.stack([area, boundary, contrast, texture, salience, reliability], dim=1)


def _sample_weights(features):
    importance = features[:, 4] / (features[:, 4].mean() + EPS)
    reliability = torch.clamp(features[:, 5], min=0.15, max=1.5)
    weights = importance * (0.65 + 0.35 * reliability)
    return torch.clamp(weights / (weights.mean() + EPS), min=0.20, max=5.0)


def _collect_batches(meta, cfg):
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
    batches, features, labels = [], [], []
    for batch_idx, (x, y) in enumerate(runtime["loader"]):
        if max_batches and batch_idx >= max_batches:
            break
        batches.append((x.clone(), y.clone()))
        features.append(_morph_features(meta, x, cfg).cpu())
        labels.append(y.clone())
    return batches, torch.cat(features, dim=0), torch.cat(labels, dim=0)


def _predict_logits(meta, checkpoint, batches, cfg):
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    model, forward_fn = _build_model_forward(meta, cfg, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in batches:
            out.append(forward_fn(model, x.to(device, non_blocking=True)).detach().float().cpu())
    return torch.cat(out, dim=0)


def _client_information(meta, checkpoints, batches, features, labels, cfg, base_weights):
    base = torch.as_tensor(base_weights, dtype=torch.float32)
    num_classes = int(meta["num_classes"])
    label_coverage, support_gate = _coverage_gate(labels, num_classes)
    reliability = float(torch.clamp(features[:, 5].mean(), min=0.0, max=1.0).item())

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
            })
        return base, base, class_weights, rows, label_coverage, support_gate, reliability

    sample_w = _sample_weights(features)
    metadata_prior = _normalize(base, fallback=torch.ones_like(base) / max(1, base.numel()))
    evidence_gate = max(0.30 if label_coverage >= 0.50 else 0.0, support_gate)
    class_gate = max(0.15 if label_coverage >= 0.50 else 0.0, support_gate)
    overall_scores, morph_scores, summaries = [], [], []
    class_scores = torch.zeros(num_classes, len(checkpoints), dtype=torch.float32)

    for idx, checkpoint in enumerate(checkpoints):
        logits = _predict_logits(meta, checkpoint, batches, cfg)
        preds = logits.argmax(dim=1)
        correct = (preds == labels).float()
        true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        masked = logits.clone()
        masked[torch.arange(logits.shape[0]), labels] = float("-inf")
        margin = torch.sigmoid(true_logits - masked.max(dim=1).values)

        client = meta["clients"][idx] if idx < len(meta.get("clients", [])) else {}
        seen = set(client.get("classes", []))
        acc = float(correct.mean().item())
        morph_acc = float((correct * sample_w).sum().item() / (sample_w.sum().item() + EPS))
        margin_score = float((margin * sample_w).sum().item() / (sample_w.sum().item() + EPS))
        prior = float(metadata_prior[idx].item())
        overall_score = prior * (0.18 + acc + 0.30 * margin_score + 0.22 * morph_acc)
        morph_score = prior * (0.12 + 0.90 * morph_acc + 0.25 * margin_score + 0.15 * acc)
        summaries.append({
            "overall_acc": acc,
            "morph_acc": morph_acc,
            "margin_score": margin_score,
            "overall_score": overall_score,
            "morph_score": morph_score,
        })
        overall_scores.append(overall_score)
        morph_scores.append(morph_score)

        for cls_idx in range(num_classes):
            cls_mask = labels == cls_idx
            if not torch.any(cls_mask):
                class_scores[cls_idx, idx] = prior
                continue
            cls_correct = correct[cls_mask]
            cls_weight = sample_w[cls_mask]
            cls_margin = margin[cls_mask]
            cls_acc = float(cls_correct.mean().item())
            cls_morph = float((cls_correct * cls_weight).sum().item() / (cls_weight.sum().item() + EPS))
            cls_conf = float((cls_margin * cls_weight).sum().item() / (cls_weight.sum().item() + EPS))
            seen_bonus = 1.22 if cls_idx in seen else 0.58
            class_scores[cls_idx, idx] = prior * seen_bonus * (0.10 + 0.74 * cls_acc + 0.42 * cls_morph + 0.24 * cls_conf)

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
            "ordinary_accuracy": float(summary["overall_acc"]),
            "medical_weighted_accuracy": float(summary["morph_acc"]),
            "margin_confidence": float(summary["margin_score"]),
            "overall_score": float(summary["overall_score"]),
            "morphology_score": float(summary["morph_score"]),
            "evidence_reliability": reliability,
        })
    return overall, morph, torch.stack(class_weights, dim=0), rows, label_coverage, support_gate, reliability


def _module1(meta, checkpoints, cfg, base_weights):
    batches, features, labels = _collect_batches(meta, cfg)
    overall, morph, class_weights, rows, label_coverage, support_gate, reliability = _client_information(
        meta, checkpoints, batches, features, labels, cfg, base_weights
    )
    return {
        "batches": batches,
        "features": features,
        "labels": labels,
        "base_weights": torch.as_tensor(base_weights, dtype=torch.float32),
        "overall_weights": overall,
        "morphology_weights": morph,
        "class_weights": class_weights,
        "client_diagnostic_information": rows,
        "evidence_support_gate": support_gate,
        "label_coverage_ratio": label_coverage,
        "evidence_reliability": reliability,
    }


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


def _layer_weights(group, base, overall, morph, consensus):
    if group == "early":
        return _normalize(0.70 * base + 0.30 * morph, base)
    if group == "late":
        return _normalize(0.30 * base + 0.70 * overall, base)
    return _normalize(0.45 * base + 0.55 * consensus, base)


def _medical_weighted_merge(state_dicts, base_merged, reference, base, overall, morph, class_weights, num_classes, meta):
    consensus = _consensus(overall, morph, base)
    group_weights = {g: _layer_weights(g, base, overall, morph, consensus) for g in ("early", "mid", "late")}
    merged = OrderedDict()
    routed = {"early": 0, "mid": 0, "late": 0, "classifier": 0}

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
                weights = _normalize(0.45 * class_weights[cls_idx] + 0.55 * consensus, consensus)
                row_values = [v[cls_idx] for v in values]
                out[cls_idx] = (ref[cls_idx].float() + _weighted_delta(row_values, ref[cls_idx], weights)).to(out.dtype) if use_delta else _weighted_average(row_values, weights)
            merged[key] = out.to(first.dtype)
            routed["classifier"] += 1
            continue

        group = _param_group(key, meta)
        weights = group_weights[group]
        merged[key] = (ref.float() + _weighted_delta(values, ref, weights)).to(first.dtype) if use_delta else _weighted_average(values, weights).to(first.dtype)
        routed[group] += 1
    return merged, {
        "fusion_weights": [float(x) for x in consensus.tolist()],
        "group_weights": {g: [float(x) for x in w.tolist()] for g, w in group_weights.items()},
        "routing_summary": {**routed, "delta_space": reference is not None},
    }


def _sign_delta_merge(state_dicts, base_merged, reference, param_names, weights, density=0.5):
    task_matrix, base_vector = build_task_matrix(state_dicts, reference, param_names)
    trimmed = mask_smallest_magnitude(task_matrix, preserve_density=density)
    merged_delta = disjoint_merge(trimmed, elect_sign(trimmed), weights=weights)
    params = vector_to_param_dict(base_vector + merged_delta, reference, param_names)
    return overlay_param_dict(base_merged, params)


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


def _extract_pooled_features(model, x):
    if not hasattr(model, "forward_features"):
        return None
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
    return pooled.detach().float() if torch.is_tensor(pooled) else None


def _state_embeddings(meta, state_dict, batches, cfg):
    if meta.get("task_type") != "small" or meta.get("model") not in {"vit_t", "swin_tiny"}:
        return None
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    model, _ = _build_model_forward(meta, cfg, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    pooled = []
    with torch.no_grad():
        for x, _ in batches:
            features = _extract_pooled_features(model, x.to(device, non_blocking=True))
            if features is None:
                return None
            pooled.append(features.cpu())
    return torch.cat(pooled, dim=0)


def _build_prototype_head_candidate(base_state_dict, meta, batches, features, labels, cfg):
    if meta.get("task_type") != "small" or meta.get("model") not in {"vit_t", "swin_tiny"}:
        return None
    num_classes = int(meta["num_classes"])
    weight_key, bias_key = _find_classifier_keys(base_state_dict, num_classes)
    if weight_key is None:
        return None
    pooled = _state_embeddings(meta, base_state_dict, batches, cfg)
    if pooled is None:
        return None

    class_weight = base_state_dict[weight_key].detach().clone().float()
    if pooled.shape[1] != class_weight.shape[1]:
        return None
    sample_w = _sample_weights(features).detach().cpu().float()
    labels = labels.detach().cpu()
    proto_weight = class_weight.clone()
    row_norms = class_weight.norm(dim=1)
    target_norm = float(torch.median(row_norms).item()) if row_norms.numel() else 1.0
    bias_value = base_state_dict[bias_key].detach().clone().float() if bias_key is not None else None
    proto_bias = bias_value.clone() if bias_value is not None else None

    for cls_idx in range(num_classes):
        mask = labels == cls_idx
        if not torch.any(mask):
            continue
        cls_feat = pooled[mask]
        cls_w = sample_w[mask].view(-1, 1)
        proto = (cls_feat * cls_w).sum(dim=0) / (cls_w.sum() + EPS)
        proto = F.normalize(proto, dim=0) * target_norm
        proto_weight[cls_idx] = 0.72 * proto + 0.28 * class_weight[cls_idx]
        if proto_bias is not None:
            prior = float(mask.float().mean().item())
            proto_bias[cls_idx] = 0.65 * bias_value[cls_idx] + 0.35 * torch.log(torch.tensor(prior + EPS, dtype=bias_value.dtype))

    candidate = OrderedDict((k, v.detach().clone()) for k, v in base_state_dict.items())
    candidate[weight_key] = proto_weight.to(dtype=base_state_dict[weight_key].dtype)
    if bias_key is not None:
        candidate[bias_key] = proto_bias.to(dtype=base_state_dict[bias_key].dtype)
    return candidate


def _build_specialist_client_candidate(state_dicts, client_rows, meta):
    family = _model_family(meta)
    dataset = meta.get("dataset")

    def score(row):
        overall = float(row.get("ordinary_accuracy", row.get("overall_weight", 0.0)))
        morph = float(row.get("medical_weighted_accuracy", row.get("morphology_weight", overall)))
        margin = float(row.get("margin_confidence", 0.0))
        if family in {"transformer", "vlm"}:
            return 0.45 * morph + 0.30 * overall + 0.25 * margin
        if dataset == "dermamnist_224":
            return 0.42 * morph + 0.32 * margin + 0.26 * overall
        return 0.48 * overall + 0.36 * morph + 0.16 * margin

    best_idx = max(range(len(state_dicts)), key=lambda idx: score(client_rows[idx]) if idx < len(client_rows) else 0.0)
    return OrderedDict((k, v.detach().clone()) for k, v in state_dicts[best_idx].items()), best_idx


def _build_m2_m3_candidates(
    state_dicts,
    base_merged,
    base,
    overall,
    morph,
    class_weights,
    num_classes,
    meta,
    cfg,
    reference,
    param_names,
    batches,
    features,
    labels,
    client_rows,
):
    candidates, traces = OrderedDict(), {}
    consensus = _consensus(overall, morph, base)
    avg_name = "delta_0p00" if _enabled(cfg, "conflict_stabilization") else "avg"
    candidates[avg_name] = base_merged
    traces[avg_name] = {
        "fusion_weights": [float(x) for x in torch.as_tensor(base, dtype=torch.float32).tolist()],
        "group_weights": {},
        "routing_summary": {
            "conflict_stabilization": bool(_enabled(cfg, "conflict_stabilization")),
            "delta_interpolation_weight": 0.0,
            "delta_path_start": bool(_enabled(cfg, "conflict_stabilization")),
        },
    }
    if _enabled(cfg, "medical_weighted_fusion"):
        state, trace = _medical_weighted_merge(state_dicts, base_merged, reference, base, overall, morph, class_weights, num_classes, meta)
        candidates["medical_weighted_fusion"] = state
        traces["medical_weighted_fusion"] = trace
    module2_pool = [name for name in candidates.keys() if name == "medical_weighted_fusion"]
    module3_pool = []
    if _enabled(cfg, "conflict_stabilization"):
        module3_pool.append(avg_name)

    if _enabled(cfg, "conflict_stabilization") and meta.get("task_type") == "small" and reference is not None and param_names is not None:
        sign_state = _sign_delta_merge(state_dicts, base_merged, reference, param_names, consensus.tolist())
        for weight in (0.25, 0.50, 0.75, 1.00):
            name = f"delta_{_blend_label(weight)}"
            candidates[name] = sign_state if weight >= 1.0 else _blend_state_dicts(base_merged, sign_state, right_weight=weight)
            traces[name] = {
                "fusion_weights": [float(x) for x in consensus.tolist()],
                "group_weights": {},
                "routing_summary": {
                    "conflict_stabilization": True,
                    "sign_delta_weight_source": "medical_consensus",
                    "delta_interpolation_weight": float(weight),
                    "delta_path_endpoint": bool(weight >= 1.0),
                },
            }
            module3_pool.append(name)

    if _enabled(cfg, "specialist_client"):
        specialist_state, specialist_idx = _build_specialist_client_candidate(state_dicts, client_rows, meta)
        candidates["specialist_client"] = specialist_state
        traces["specialist_client"] = {
            "fusion_weights": [1.0 if idx == specialist_idx else 0.0 for idx in range(len(state_dicts))],
            "group_weights": {},
            "routing_summary": {
                "representation_preservation": True,
                "preservation_candidate": "specialist_client",
                "specialist_client_index": int(specialist_idx),
            },
        }
        module3_pool.append("specialist_client")

    if _enabled(cfg, "prototype_head"):
        prototype_state = _build_prototype_head_candidate(base_merged, meta, batches, features, labels, cfg)
        if prototype_state is not None:
            candidates["prototype_head"] = prototype_state
            traces["prototype_head"] = {
                "fusion_weights": [float(x) for x in torch.as_tensor(base, dtype=torch.float32).tolist()],
                "group_weights": {},
                "routing_summary": {
                    "representation_preservation": True,
                    "preservation_candidate": "prototype_head",
                    "prototype_source": "delta_0p00",
                },
            }
            module3_pool.append("prototype_head")
    return candidates, traces, module2_pool, module3_pool


def _selection_score(acc, medical_acc, loss):
    w = DEFAULT_SELECTION_MEDICAL_WEIGHT
    return (1.0 - w) * acc + w * medical_acc - 0.001 * loss


def _eval_candidate(meta, state_dict, cfg, batches, features, labels):
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    model, forward_fn = _build_model_forward(meta, cfg, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    sample_w = _sample_weights(features).detach().cpu().float()
    total = correct = weighted_correct = weighted_total = loss_sum = 0.0
    offset = 0
    with torch.no_grad():
        for x, y in batches:
            x = x.to(device, non_blocking=True)
            y_device = y.to(device, non_blocking=True)
            logits = forward_fn(model, x).detach().float()
            preds = logits.argmax(dim=1).cpu()
            y_cpu = y.cpu()
            ok = (preds == y_cpu).float()
            weights = sample_w[offset: offset + y.numel()]
            total += int(y.numel())
            correct += float(ok.sum().item())
            weighted_correct += float((ok * weights).sum().item())
            weighted_total += float(weights.sum().item())
            loss_sum += float(F.cross_entropy(logits, y_device, reduction="sum").item())
            offset += int(y.numel())
    acc = correct / max(total, EPS)
    medical_acc = weighted_correct / max(weighted_total, EPS)
    loss = loss_sum / max(total, EPS)
    return {
        "val_acc": float(acc),
        "val_medical_acc": float(medical_acc),
        "val_loss": float(loss),
        "selection_score": float(_selection_score(acc, medical_acc, loss)),
        "selection_medical_weight": DEFAULT_SELECTION_MEDICAL_WEIGHT,
    }


def _recalibrate_bn(meta, state_dict, cfg, batches):
    max_batches = int(cfg.get("my_merge_bn_batches", DEFAULT_BN_BATCHES))
    if max_batches <= 0:
        return state_dict
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    model, forward_fn = _build_model_forward(meta, cfg, device)
    model.load_state_dict(state_dict, strict=True)
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return state_dict
    for module in bns:
        module.running_mean.zero_()
        module.running_var.fill_(1.0)
        module.num_batches_tracked.zero_()
        module.momentum = None
    model.train()
    with torch.no_grad():
        for idx, (x, _) in enumerate(batches):
            if max_batches and idx >= max_batches:
                break
            forward_fn(model, x.to(device, non_blocking=True))
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _prepare(meta, state_dict, cfg, batches):
    return _recalibrate_bn(meta, state_dict, cfg, batches) if _enabled(cfg, "bn_recalibration") else state_dict


def _select_candidate(candidates, traces, base_merged, base_weights, meta, cfg, batches, features, labels):
    if not _enabled(cfg, "validated_selection") or not candidates:
        return base_merged, {
            "selected_candidate": "avg",
            "candidate_pool": list(candidates.keys()),
            "candidate_metrics": {},
            "fusion_weights": [float(x) for x in torch.as_tensor(base_weights, dtype=torch.float32).tolist()],
            "group_weights": {},
            "routing_summary": {"validated_candidates": len(candidates)},
        }

    metrics, prepared, best_name, best_key = {}, {}, None, None
    for name, state in list(candidates.items()):
        prepared_state = _prepare(meta, state, cfg, batches)
        prepared[name] = prepared_state
        metrics[name] = _eval_candidate(meta, prepared_state, cfg, batches, features, labels)
        key = (metrics[name]["selection_score"], metrics[name]["val_acc"], -metrics[name]["val_loss"])
        if best_key is None or key > best_key:
            best_name, best_key = name, key
    trace = traces.get(best_name, {})
    routing = {"validated_candidates": len(candidates)}
    routing.update(trace.get("routing_summary", {}))
    return prepared.get(best_name, candidates[best_name]), {
        "selected_candidate": best_name,
        "candidate_pool": list(candidates.keys()),
        "candidate_metrics": metrics,
        "fusion_weights": trace.get("fusion_weights", [float(x) for x in torch.as_tensor(base_weights, dtype=torch.float32).tolist()]),
        "group_weights": trace.get("group_weights", {}),
        "routing_summary": routing,
        "candidates_prepared": True,
    }


def _module2_module3(state_dicts, base_merged, meta, cfg, info, reference, param_names):
    base = torch.as_tensor(info["base_weights"], dtype=torch.float32)
    candidates, traces, module2_pool, module3_pool = _build_m2_m3_candidates(
        state_dicts,
        base_merged,
        base,
        info["overall_weights"],
        info["morphology_weights"],
        info["class_weights"],
        int(meta["num_classes"]),
        meta,
        cfg,
        reference,
        param_names,
        info["batches"],
        info["features"],
        info["labels"],
        info["client_diagnostic_information"],
    )
    state, trace = _select_candidate(candidates, traces, base_merged, base, meta, cfg, info["batches"], info["features"], info["labels"])
    if not trace.get("candidates_prepared"):
        state = _prepare(meta, state, cfg, info["batches"])
    return state, {
        "fusion_rule": "m2_medical_fusion_m3_conflict_stabilization_validation",
        "module2_candidate_pool": module2_pool,
        "module3_candidate_pool": module3_pool,
        "candidate_pool": trace.get("candidate_pool", []),
        "candidate_metrics": trace.get("candidate_metrics", {}),
        "selected_candidate": trace.get("selected_candidate", "avg"),
        "routing_summary": trace.get("routing_summary", {}),
        "fusion_weights": trace.get("fusion_weights", []),
        "group_weights": trace.get("group_weights", {}),
        "evidence_reliability": info.get("evidence_reliability"),
    }


def merge_my_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    if meta is None or checkpoints is None or cfg is None:
        raise ValueError("my_merge requires task metadata, client checkpoints, and runtime config.")
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

    info = _module1(meta, checkpoints, cfg, base_weights)
    reference = param_names = None
    if _enabled(cfg, "medical_weighted_fusion") or _enabled(cfg, "conflict_stabilization"):
        reference, param_names = build_reference_bundle(meta, device="cpu")
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
        "evidence_reliability": fusion["evidence_reliability"],
    }
