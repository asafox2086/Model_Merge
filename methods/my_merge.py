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
from .breadcrumbs import merge_breadcrumbs
from .fisher import merge_fisher
from .from_merge import merge_from
from .iso import merge_iso_c
from .robustmerge import merge_robustmerge


EPS = 1e-8
DEFAULT_STATS_MAX_BATCHES = 16
DEFAULT_BN_BATCHES = 4
DEFAULT_SELECTION_MEDICAL_WEIGHT = 0.15
DEFAULT_SELECTION_LOSS_WEIGHT = 0.005
DEFAULT_PUBLIC_PROBE_SPLIT = "val"
DEFAULT_PUBLIC_FISHER_MAX_BATCHES = 4
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

MEDICAL_IMAGE_DATASETS = {
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
}

M1 = {"image_space", "diagnostic_evidence", "diagnostic_client_information"}
M2 = {
    "medical_weighted_fusion",
    "anatomy_head_graft",
    "sign_consistent_delta",
    "public_fisher",
    "validated_selection",
    "bn_recalibration",
}
M3 = {"adaptive_candidates"}

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
        elif name in {"no_adaptive_candidates", "no_adaptive_candidate_generation", "no_m3"}:
            disabled.update(M3)
        elif name in {"no_sign_delta", "no_ties_delta"}:
            disabled.add("sign_consistent_delta")
        elif name in {"no_public_fisher", "no_fisher"}:
            disabled.add("public_fisher")
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
        "anatomy_head_graft",
        "sign_consistent_delta",
        "public_fisher",
        "validated_selection",
        "bn_recalibration",
        "adaptive_candidates",
    ]
    return {
        "labels": _ablation_labels(cfg),
        "disabled_components": sorted(_disabled_components(cfg)),
        "enabled": {name: _enabled(cfg, name) for name in components},
    }


def _is_medical_image_task(meta):
    return meta.get("task_type") in {"small", "vlm"} and meta.get("dataset") in MEDICAL_IMAGE_DATASETS


def _model_family(meta):
    if meta.get("task_type") == "vlm":
        return "vlm"
    return "cnn" if meta.get("model") in {"resnet", "mobilenet", "convnext"} else "transformer"


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


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


def _public_probe_settings(meta, cfg):
    explicit = any(
        key in cfg
        for key in (
            "my_merge_public_data_root",
            "my_merge_public_dataset",
            "my_merge_public_split",
            "my_merge_public_use_labels",
        )
    )
    data_root = cfg.get("my_merge_public_data_root") or cfg["data_root"]
    dataset = cfg.get("my_merge_public_dataset") or meta["dataset"]
    split = cfg.get("my_merge_public_split") or cfg.get("stats_split", DEFAULT_PUBLIC_PROBE_SPLIT)
    use_labels = _as_bool(cfg.get("my_merge_public_use_labels", True), default=True)
    return {
        "data_root": data_root,
        "dataset": dataset,
        "split": split,
        "use_labels": use_labels,
        "explicit": explicit,
        "same_dataset": dataset == meta.get("dataset"),
    }


def _adapt_probe_channels(meta, x):
    if meta.get("task_type") == "vlm":
        return x
    target = int(meta.get("in_channels", x.shape[1]))
    if x.shape[1] == target:
        return x
    if target == 1 and x.shape[1] == 3:
        return x.mean(dim=1, keepdim=True)
    if target == 3 and x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] > target:
        return x[:, :target]
    raise ValueError(f"Cannot adapt public probe channels from {x.shape[1]} to {target}.")


def _labels_compatible(labels, num_classes, use_labels):
    if not use_labels:
        return False
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1)
    if labels.numel() == 0:
        return False
    return bool(int(labels.min().item()) >= 0 and int(labels.max().item()) < int(num_classes))


def _collect_batches(meta, cfg):
    probe = _public_probe_settings(meta, cfg)
    probe_meta = dict(meta)
    probe_meta["dataset"] = probe["dataset"]
    runtime = build_runtime(
        meta=probe_meta,
        data_root=probe["data_root"],
        split=probe["split"],
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
        x = _adapt_probe_channels(meta, x)
        batches.append((x.clone(), y.clone()))
        features.append(_morph_features(meta, x, cfg).cpu())
        labels.append(y.clone())
    labels = torch.cat(labels, dim=0)
    probe["label_compatible"] = _labels_compatible(labels, int(meta["num_classes"]), probe["use_labels"])
    probe["num_samples"] = int(labels.numel())
    return batches, torch.cat(features, dim=0), labels, probe


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


def _softmax_margin(logits):
    probs = torch.softmax(logits.float(), dim=1)
    top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        margin = top2[:, 0]
    else:
        margin = top2[:, 0] - top2[:, 1]
    return probs, top2[:, 0], margin


def _weighted_mean(values, weights):
    values = torch.as_tensor(values, dtype=torch.float32)
    weights = torch.as_tensor(weights, dtype=torch.float32).to(values.device)
    return float((values * weights).sum().item() / (weights.sum().item() + EPS))


def _unsupervised_client_information(meta, checkpoints, batches, features, labels, cfg, base_weights):
    base = torch.as_tensor(base_weights, dtype=torch.float32)
    num_classes = int(meta["num_classes"])
    sample_w = _sample_weights(features)
    metadata_prior = _normalize(base, fallback=torch.ones_like(base) / max(1, base.numel()))
    reliability = float(torch.clamp(features[:, 5].mean(), min=0.0, max=1.0).item())

    logits_list = [_predict_logits(meta, checkpoint, batches, cfg) for checkpoint in checkpoints]
    prob_list, conf_list, margin_list, pred_list = [], [], [], []
    ensemble = torch.zeros_like(torch.softmax(logits_list[0], dim=1))
    for logits, prior in zip(logits_list, metadata_prior):
        probs, conf, margin = _softmax_margin(logits)
        prob_list.append(probs)
        conf_list.append(conf)
        margin_list.append(margin)
        pred_list.append(probs.argmax(dim=1))
        ensemble.add_(probs, alpha=float(prior.item()))
    pseudo_labels = ensemble.argmax(dim=1)
    label_coverage, support_gate = _coverage_gate(pseudo_labels, num_classes)
    evidence_gate = max(0.20 if label_coverage >= 0.40 else 0.0, min(0.65, support_gate))
    class_gate = max(0.10 if label_coverage >= 0.40 else 0.0, min(0.55, support_gate))

    overall_scores, morph_scores, summaries = [], [], []
    class_scores = torch.zeros(num_classes, len(checkpoints), dtype=torch.float32)
    for idx, (probs, conf, margin, preds) in enumerate(zip(prob_list, conf_list, margin_list, pred_list)):
        prior = float(metadata_prior[idx].item())
        agreement = (preds == pseudo_labels).float()
        confidence = _weighted_mean(conf, sample_w)
        separation = _weighted_mean(margin, sample_w)
        agreement_score = _weighted_mean(agreement, sample_w)
        morph_confidence = _weighted_mean(conf * (0.55 + 0.45 * agreement), sample_w)
        overall_score = prior * (0.12 + 0.36 * confidence + 0.34 * agreement_score + 0.30 * separation)
        morph_score = prior * (0.10 + 0.42 * morph_confidence + 0.30 * agreement_score + 0.28 * separation)
        summaries.append({
            "overall_acc": agreement_score,
            "morph_acc": morph_confidence,
            "margin_score": separation,
            "overall_score": overall_score,
            "morph_score": morph_score,
        })
        overall_scores.append(overall_score)
        morph_scores.append(morph_score)

        for cls_idx in range(num_classes):
            cls_mask = pseudo_labels == cls_idx
            if not torch.any(cls_mask):
                class_scores[cls_idx, idx] = prior
                continue
            cls_weight = sample_w[cls_mask]
            cls_prob = probs[cls_mask, cls_idx]
            cls_agree = agreement[cls_mask]
            cls_margin = margin[cls_mask]
            client = meta["clients"][idx] if idx < len(meta.get("clients", [])) else {}
            seen = set(client.get("classes", []))
            seen_bonus = 1.14 if cls_idx in seen else 0.72
            class_scores[cls_idx, idx] = prior * seen_bonus * (
                0.10
                + 0.40 * _weighted_mean(cls_prob, cls_weight)
                + 0.32 * _weighted_mean(cls_agree, cls_weight)
                + 0.24 * _weighted_mean(cls_margin, cls_weight)
            )

    overall = _normalize((1.0 - evidence_gate) * metadata_prior + evidence_gate * _calibrated(overall_scores, metadata_prior, reliability, 0.82, 0.035), metadata_prior)
    morph = _normalize((1.0 - evidence_gate) * metadata_prior + evidence_gate * _calibrated(morph_scores, metadata_prior, reliability, 0.78, 0.035), metadata_prior)
    class_weights = []
    for cls_idx in range(num_classes):
        cls_w = _calibrated(class_scores[cls_idx], morph, reliability, 0.88, 0.06)
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
            "public_agreement": float(summary["overall_acc"]),
            "public_morph_confidence": float(summary["morph_acc"]),
            "margin_confidence": float(summary["margin_score"]),
            "overall_score": float(summary["overall_score"]),
            "morphology_score": float(summary["morph_score"]),
            "evidence_reliability": reliability,
        })
    return overall, morph, torch.stack(class_weights, dim=0), rows, label_coverage, support_gate, reliability


def _client_information(meta, checkpoints, batches, features, labels, cfg, base_weights, labels_valid=True):
    base = torch.as_tensor(base_weights, dtype=torch.float32)
    num_classes = int(meta["num_classes"])
    if not labels_valid:
        return _unsupervised_client_information(meta, checkpoints, batches, features, labels, cfg, base_weights)

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
    batches, features, labels, probe = _collect_batches(meta, cfg)
    overall, morph, class_weights, rows, label_coverage, support_gate, reliability = _client_information(
        meta, checkpoints, batches, features, labels, cfg, base_weights, labels_valid=probe["label_compatible"]
    )
    return {
        "batches": batches,
        "features": features,
        "labels": labels,
        "probe": probe,
        "base_weights": torch.as_tensor(base_weights, dtype=torch.float32),
        "overall_weights": overall,
        "morphology_weights": morph,
        "class_weights": class_weights,
        "client_diagnostic_information": rows,
        "evidence_support_gate": support_gate,
        "label_coverage_ratio": label_coverage,
        "evidence_reliability": reliability,
    }


def _selection_probe_info(meta, cfg, default_info):
    selection_split = str(cfg.get("my_merge_selection_public_split", "") or "").strip()
    if not selection_split or selection_split == default_info["probe"].get("split"):
        return default_info
    selection_cfg = dict(cfg)
    selection_cfg["my_merge_public_split"] = selection_split
    if "my_merge_selection_stats_max_batches" in cfg:
        selection_cfg["my_merge_stats_max_batches"] = int(cfg["my_merge_selection_stats_max_batches"])
    batches, features, labels, probe = _collect_batches(meta, selection_cfg)
    return {
        "batches": batches,
        "features": features,
        "labels": labels,
        "probe": probe,
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


def _sharpen(weights, fallback, temperature=0.28, floor=0.01):
    weights = torch.as_tensor(weights, dtype=torch.float32).clamp_min(EPS)
    fallback = torch.as_tensor(fallback, dtype=torch.float32)
    logits = torch.log(weights)
    logits = logits - logits.mean()
    probs = torch.softmax(logits / max(float(temperature), EPS), dim=0)
    if floor > 0:
        probs = (1.0 - floor) * probs + floor * fallback
    return _normalize(probs, fallback)


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


def _anatomy_head_graft(anchor_state, state_dicts, base, class_weights, num_classes):
    merged = OrderedDict((key, value.detach().clone()) for key, value in anchor_state.items())
    routed = 0
    for key, first in anchor_state.items():
        if not _is_classifier_tensor(key, first, num_classes):
            continue
        values = [sd[key] for sd in state_dicts]
        out = first.detach().clone().float()
        for cls_idx in range(first.shape[0]):
            weights = _sharpen(class_weights[cls_idx], base)
            row_values = [value[cls_idx] for value in values]
            out[cls_idx] = _weighted_average(row_values, weights)
        merged[key] = out.to(first.dtype)
        routed += 1
    if routed <= 0:
        return None, {"routing_summary": {"candidate": "anatomy_head_graft", "classifier_tensors": 0}}
    return merged, {
        "fusion_weights": [float(x) for x in torch.as_tensor(base, dtype=torch.float32).tolist()],
        "group_weights": {},
        "routing_summary": {
            "candidate": "anatomy_head_graft",
            "classifier_tensors": routed,
            "head_weighting": "public_probe_class_reliability",
        },
    }


def _sign_delta_merge(state_dicts, base_merged, reference, param_names, weights, density=0.5):
    task_matrix, base_vector = build_task_matrix(state_dicts, reference, param_names)
    trimmed = mask_smallest_magnitude(task_matrix, preserve_density=density)
    merged_delta = disjoint_merge(trimmed, elect_sign(trimmed), weights=weights)
    params = vector_to_param_dict(base_vector + merged_delta, reference, param_names)
    return overlay_param_dict(base_merged, params)


def _public_fisher_diagonal(meta, state_dict, cfg, batches):
    max_batches = int(cfg.get("my_merge_public_fisher_max_batches", cfg.get("fisher_max_batches", DEFAULT_PUBLIC_FISHER_MAX_BATCHES)))
    if max_batches < 0:
        max_batches = 0
    micro_batch_size = int(cfg.get("my_merge_public_fisher_batch_size", 16) or 16)
    micro_batch_size = max(1, micro_batch_size)
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    model, forward_fn = _build_model_forward(meta, cfg, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    fisher, count = {}, 0
    for batch_idx, (x, y) in enumerate(batches):
        if max_batches and batch_idx >= max_batches:
            break
        for start in range(0, int(x.size(0)), micro_batch_size):
            xb = x[start: start + micro_batch_size].to(device, non_blocking=True)
            yb = y[start: start + micro_batch_size].to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            logits = forward_fn(model, xb)
            if logits.ndim == 1 or logits.shape[-1] == 1:
                loss = F.mse_loss(logits.reshape(-1), yb.float().reshape(-1))
                loss.backward()
            else:
                probs = torch.softmax(logits, dim=-1).detach()
                log_probs = torch.log_softmax(logits, dim=-1)
                (torch.sqrt(probs) * log_probs).sum().backward()
            bs = int(xb.size(0))
            count += bs
            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None or not torch.is_floating_point(param):
                    continue
                value = (param.grad.detach().float().cpu() ** 2) * bs
                if name not in fisher:
                    fisher[name] = value
                else:
                    fisher[name].add_(value)
    if count <= 0:
        raise RuntimeError("No public probe batches were processed for Fisher candidate.")
    for key in fisher:
        fisher[key].div_(float(count))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fisher


def _public_fisher_merge(state_dicts, base_merged, weights, meta, cfg, batches):
    fisher_stats = [_public_fisher_diagonal(meta, state, cfg, batches) for state in state_dicts]
    merged, trace = merge_fisher(
        state_dicts,
        fisher_stats,
        weights,
        eps=float(cfg.get("fisher_eps", 1e-8)),
        normalize_fisher_weight=bool(cfg.get("fisher_normalize_weight", True)),
        minimal_fisher_weight=float(cfg.get("fisher_minimal_weight", 1e-6)),
    )
    return merged, {
        **trace,
        "routing_summary": {
            "candidate": "public_fisher",
            "public_probe_batches": int(cfg.get("my_merge_public_fisher_max_batches", cfg.get("fisher_max_batches", DEFAULT_PUBLIC_FISHER_MAX_BATCHES))),
        },
    }


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


def _build_m2_m3_candidates(state_dicts, base_merged, base, overall, morph, class_weights, num_classes, meta, cfg, reference, param_names, batches):
    candidates, traces = OrderedDict(), {}
    consensus = _consensus(overall, morph, base)
    candidates["avg"] = base_merged
    traces["avg"] = {
        "fusion_weights": [float(x) for x in torch.as_tensor(base, dtype=torch.float32).tolist()],
        "group_weights": {},
        "routing_summary": {"candidate": "avg"},
    }
    if _enabled(cfg, "medical_weighted_fusion"):
        state, trace = _medical_weighted_merge(state_dicts, base_merged, reference, base, overall, morph, class_weights, num_classes, meta)
        candidates["medical_weighted_fusion"] = state
        traces["medical_weighted_fusion"] = trace
    sign_state = None
    if _enabled(cfg, "sign_consistent_delta") and meta.get("task_type") == "small" and reference is not None and param_names is not None:
        sign_state = _sign_delta_merge(state_dicts, base_merged, reference, param_names, consensus.tolist())
        candidates["sign_consistent_delta"] = sign_state
        traces["sign_consistent_delta"] = {
            "fusion_weights": [float(x) for x in consensus.tolist()],
            "group_weights": {},
            "routing_summary": {"sign_delta_weight_source": "medical_consensus"},
        }
    if _enabled(cfg, "adaptive_candidates") and sign_state is not None:
        candidates["avg_sign_blend_0p25"] = _blend_state_dicts(base_merged, sign_state, right_weight=0.25)
        traces["avg_sign_blend_0p25"] = {
            "fusion_weights": [float(x) for x in consensus.tolist()],
            "group_weights": {},
            "routing_summary": {"avg_sign_blend": True, "blend_right_weight": 0.25},
        }
    if _enabled(cfg, "adaptive_candidates") and reference is not None:
        try:
            state, trace = merge_breadcrumbs(
                state_dicts,
                reference,
                consensus.tolist(),
                top_k_keep=float(cfg.get("breadcrumbs_top_k_keep", 0.2)),
                top_k_remove=float(cfg.get("breadcrumbs_top_k_remove", 0.1)),
                alpha=float(cfg.get("breadcrumbs_alpha", 1.0)),
                param_keys=param_names,
            )
            candidates["breadcrumbs"] = state
            traces["breadcrumbs"] = {
                "fusion_weights": [float(x) for x in consensus.tolist()],
                "group_weights": {},
                "routing_summary": {"candidate": "breadcrumbs", "source": trace.get("implementation")},
            }
        except Exception as exc:
            traces["breadcrumbs_error"] = {"error": str(exc)}
        try:
            state, trace = merge_from(
                state_dicts,
                reference,
                consensus.tolist(),
                k=float(cfg.get("from_k", 1.0)),
            )
            candidates["from"] = state
            traces["from"] = {
                "fusion_weights": trace.get("effective_scaling", [float(x) for x in consensus.tolist()]),
                "group_weights": {},
                "routing_summary": {"candidate": "from", "source": trace.get("implementation")},
            }
        except Exception as exc:
            traces["from_error"] = {"error": str(exc)}
        try:
            state, trace = merge_robustmerge(state_dicts, reference, consensus.tolist(), cfg)
            candidates["robustmerge"] = state
            traces["robustmerge"] = {
                "fusion_weights": [float(x) for x in consensus.tolist()],
                "group_weights": {},
                "routing_summary": {"candidate": "robustmerge", "source": trace.get("implementation")},
            }
        except Exception as exc:
            traces["robustmerge_error"] = {"error": str(exc)}
        try:
            state, trace = merge_iso_c(state_dicts, reference, consensus.tolist(), cfg)
            candidates["iso_c"] = state
            traces["iso_c"] = {
                "fusion_weights": trace.get("normalized_weights", [float(x) for x in consensus.tolist()]),
                "group_weights": {},
                "routing_summary": {"candidate": "iso_c", "source": trace.get("implementation")},
            }
        except Exception as exc:
            traces["iso_c_error"] = {"error": str(exc)}
        prior_specs = [
            ("breadcrumbs_prior", "breadcrumbs"),
            ("from_prior", "from"),
            ("robustmerge_prior", "robustmerge"),
            ("iso_c_prior", "iso_c"),
        ]
        for candidate_name, family in prior_specs:
            try:
                prior_weights = base.tolist()
                if family == "breadcrumbs":
                    state, trace = merge_breadcrumbs(
                        state_dicts,
                        reference,
                        prior_weights,
                        top_k_keep=float(cfg.get("breadcrumbs_top_k_keep", 0.2)),
                        top_k_remove=float(cfg.get("breadcrumbs_top_k_remove", 0.1)),
                        alpha=float(cfg.get("breadcrumbs_alpha", 1.0)),
                        param_keys=param_names,
                    )
                elif family == "from":
                    state, trace = merge_from(
                        state_dicts,
                        reference,
                        prior_weights,
                        k=float(cfg.get("from_k", 1.0)),
                    )
                elif family == "robustmerge":
                    state, trace = merge_robustmerge(state_dicts, reference, prior_weights, cfg)
                else:
                    state, trace = merge_iso_c(state_dicts, reference, prior_weights, cfg)
                candidates[candidate_name] = state
                traces[candidate_name] = {
                    "fusion_weights": trace.get("normalized_weights", prior_weights),
                    "group_weights": {},
                    "routing_summary": {
                        "candidate": candidate_name,
                        "source": trace.get("implementation"),
                        "weight_prior": "metadata_prior",
                    },
                }
            except Exception as exc:
                traces[f"{candidate_name}_error"] = {"error": str(exc)}
    if _enabled(cfg, "public_fisher") and reference is not None:
        try:
            state, trace = _public_fisher_merge(state_dicts, base_merged, consensus.tolist(), meta, cfg, batches)
            candidates["public_fisher"] = state
            traces["public_fisher"] = {
                "fusion_weights": trace.get("effective_scaling", [float(x) for x in consensus.tolist()]),
                "group_weights": {},
                "routing_summary": trace.get("routing_summary", {"candidate": "public_fisher"}),
            }
        except Exception as exc:
            traces["public_fisher_error"] = {"error": str(exc)}
    if _enabled(cfg, "adaptive_candidates") and _enabled(cfg, "anatomy_head_graft"):
        for anchor_name in ("avg", "breadcrumbs", "from", "iso_c"):
            if anchor_name not in candidates:
                continue
            try:
                state, trace = _anatomy_head_graft(candidates[anchor_name], state_dicts, base, class_weights, num_classes)
                if state is None:
                    continue
                name = f"anatomy_head_graft_{anchor_name}"
                candidates[name] = state
                traces[name] = {
                    **trace,
                    "routing_summary": {
                        **trace.get("routing_summary", {}),
                        "anchor_candidate": anchor_name,
                    },
                }
            except Exception as exc:
                traces[f"anatomy_head_graft_{anchor_name}_error"] = {"error": str(exc)}
    module2_pool = [name for name in candidates.keys() if name != "avg"]
    return candidates, traces, module2_pool


def _selection_score(acc, medical_acc, loss):
    w = DEFAULT_SELECTION_MEDICAL_WEIGHT
    return (1.0 - w) * acc + w * medical_acc - DEFAULT_SELECTION_LOSS_WEIGHT * loss


def _eval_candidate(meta, state_dict, cfg, batches, features, labels, labels_valid=True):
    device = torch.device(cfg.get("stats_device", cfg.get("device", "cpu")))
    model, forward_fn = _build_model_forward(meta, cfg, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    sample_w = _sample_weights(features).detach().cpu().float()
    if not labels_valid:
        all_logits = []
        with torch.no_grad():
            for x, _ in batches:
                logits = forward_fn(model, x.to(device, non_blocking=True)).detach().float().cpu()
                all_logits.append(logits)
        logits = torch.cat(all_logits, dim=0)
        probs, conf, margin = _softmax_margin(logits)
        preds = probs.argmax(dim=1)
        label_coverage, balance = _coverage_gate(preds, int(meta["num_classes"]))
        entropy = -torch.sum(probs * torch.log(probs + EPS), dim=1)
        max_entropy = torch.log(torch.tensor(float(max(2, int(meta["num_classes"]))), dtype=torch.float32))
        normalized_entropy = torch.clamp(entropy / (max_entropy + EPS), 0.0, 1.0)
        weighted_conf = _weighted_mean(conf, sample_w)
        weighted_margin = _weighted_mean(margin, sample_w)
        weighted_certainty = 1.0 - _weighted_mean(normalized_entropy, sample_w)
        distribution_support = 0.55 * label_coverage + 0.45 * balance
        score = (
            0.36 * weighted_margin
            + 0.28 * weighted_conf
            + 0.20 * weighted_certainty
            + 0.16 * distribution_support
        )
        return {
            "public_probe_labeled": False,
            "val_acc": None,
            "val_medical_acc": None,
            "val_loss": None,
            "public_confidence": float(weighted_conf),
            "public_margin": float(weighted_margin),
            "public_certainty": float(weighted_certainty),
            "public_prediction_coverage": float(label_coverage),
            "public_prediction_balance": float(balance),
            "selection_score": float(score),
            "selection_medical_weight": DEFAULT_SELECTION_MEDICAL_WEIGHT,
            "selection_loss_weight": DEFAULT_SELECTION_LOSS_WEIGHT,
        }

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
        "public_probe_labeled": True,
        "val_acc": float(acc),
        "val_medical_acc": float(medical_acc),
        "val_loss": float(loss),
        "selection_score": float(_selection_score(acc, medical_acc, loss)),
        "selection_medical_weight": DEFAULT_SELECTION_MEDICAL_WEIGHT,
        "selection_loss_weight": DEFAULT_SELECTION_LOSS_WEIGHT,
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


def _select_candidate(
    candidates,
    traces,
    base_merged,
    base_weights,
    meta,
    cfg,
    prepare_batches,
    score_batches,
    score_features,
    score_labels,
    labels_valid=True,
):
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
        prepared_state = _prepare(meta, state, cfg, prepare_batches)
        prepared[name] = prepared_state
        metrics[name] = _eval_candidate(
            meta,
            prepared_state,
            cfg,
            score_batches,
            score_features,
            score_labels,
            labels_valid=labels_valid,
        )
        acc_value = metrics[name]["val_acc"] if metrics[name]["val_acc"] is not None else metrics[name]["selection_score"]
        loss_value = metrics[name]["val_loss"] if metrics[name]["val_loss"] is not None else 0.0
        key = (metrics[name]["selection_score"], acc_value, -loss_value)
        if best_key is None or key > best_key:
            best_name, best_key = name, key
    trace = traces.get(best_name, {})
    routing = {"validated_candidates": len(candidates), "adaptive_candidates": bool(_enabled(cfg, "adaptive_candidates"))}
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
    candidates, traces, module2_pool = _build_m2_m3_candidates(
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
    )
    selection_info = _selection_probe_info(meta, cfg, info)
    state, trace = _select_candidate(
        candidates,
        traces,
        base_merged,
        base,
        meta,
        cfg,
        info["batches"],
        selection_info["batches"],
        selection_info["features"],
        selection_info["labels"],
        labels_valid=selection_info["probe"]["label_compatible"],
    )
    if not trace.get("candidates_prepared"):
        state = _prepare(meta, state, cfg, info["batches"])
    return state, {
        "fusion_rule": "m2_base_fusion_m3_adaptive_validation",
        "module2_candidate_pool": module2_pool,
        "candidate_pool": trace.get("candidate_pool", []),
        "candidate_metrics": trace.get("candidate_metrics", {}),
        "selected_candidate": trace.get("selected_candidate", "avg"),
        "routing_summary": trace.get("routing_summary", {}),
        "fusion_weights": trace.get("fusion_weights", []),
        "group_weights": trace.get("group_weights", {}),
        "evidence_reliability": info.get("evidence_reliability"),
        "public_probe": info.get("probe", {}),
        "selection_public_probe": selection_info.get("probe", {}),
    }


def merge_my_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    if meta is None or checkpoints is None or cfg is None:
        raise ValueError("my_merge requires task metadata, client checkpoints, and runtime config.")
    if not _is_medical_image_task(meta):
        raise ValueError(f"my_merge is defined only for medical image tasks, got dataset={meta.get('dataset')}.")

    base_merged, base_weights = average_state_dicts(state_dicts, weights)
    ablation = _ablation_config(cfg)
    if "avg_only" in {x.lower().replace("-", "_") for x in ablation["labels"]}:
        return base_merged, {
            "implementation": "medical_evidence_three_module_merge_v10_ablation_avg_only",
            "medical_only": True,
            "ablation_config": ablation,
            "selected_candidate": "avg",
            "base_weights": [float(x) for x in base_weights],
            "normalized_weights": base_weights,
        }

    info = _module1(meta, checkpoints, cfg, base_weights)
    reference = param_names = None
    if _enabled(cfg, "medical_weighted_fusion") or _enabled(cfg, "sign_consistent_delta"):
        reference, param_names = build_reference_bundle(meta, device="cpu")
    merged, fusion = _module2_module3(state_dicts, base_merged, meta, cfg, info, reference, param_names)

    adaptive = _enabled(cfg, "adaptive_candidates")
    implementation = "medical_evidence_three_module_posthoc_merge_v10_adaptive" if adaptive else "medical_evidence_three_module_posthoc_merge_v10"
    return merged, {
        "implementation": implementation,
        "medical_only": True,
        "ablation_config": ablation,
        "modality": meta.get("dataset"),
        "model_family": _model_family(meta),
        "adaptive_candidates_enabled": adaptive,
        "base_weights": [float(x) for x in base_weights],
        "overall_weights": [float(x) for x in info["overall_weights"].tolist()],
        "morphology_weights": [float(x) for x in info["morphology_weights"].tolist()],
        "class_weights": [[float(v) for v in row] for row in info["class_weights"].tolist()],
        "client_diagnostic_information": info["client_diagnostic_information"],
        "module2_candidate_pool": fusion["module2_candidate_pool"],
        "candidate_pool": fusion["candidate_pool"],
        "candidate_metrics": fusion["candidate_metrics"],
        "selected_candidate": fusion["selected_candidate"],
        "fusion_rule": fusion["fusion_rule"],
        "fusion_weights": fusion["fusion_weights"],
        "group_weights": fusion["group_weights"],
        "routing_summary": fusion["routing_summary"],
        "evidence_reliability": fusion["evidence_reliability"],
        "public_probe": fusion["public_probe"],
    }
