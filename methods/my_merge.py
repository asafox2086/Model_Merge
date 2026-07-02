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
from .iso import merge_iso_c


EPS = 1e-8
DEFAULT_STATS_MAX_BATCHES = 16
DEFAULT_SELECTION_MEDICAL_WEIGHT = 0.15
DEFAULT_SELECTION_LOSS_WEIGHT = 0.005
DEFAULT_PUBLIC_PROBE_SPLIT = "val"
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

MEDICAL_IMAGE_DATASETS = {
    "bloodmnist_224",
    "dermamnist_224",
    "organcmnist_224",
    "organsmnist_224",
    "chaoshengmnist_224",
}


def _is_medical_image_task(meta):
    return meta.get("task_type") in {"small", "vlm"} and meta.get("dataset") in MEDICAL_IMAGE_DATASETS


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
        )
    )
    data_root = str(cfg.get("my_merge_public_data_root") or "").strip()
    if not data_root:
        raise ValueError("my_merge requires --my-merge-public-data-root with class-compatible labeled medical probe images.")
    if str(data_root).rstrip("/") == str(cfg.get("data_root", "")).rstrip("/"):
        raise ValueError(
            "my_merge public probe root must be independent from the benchmark/client data root; "
            "using Med_data or any benchmark split for public selection would leak data."
        )
    dataset = cfg.get("my_merge_public_dataset") or meta["dataset"]
    if dataset != meta.get("dataset"):
        raise ValueError(
            "my_merge labeled public probe must use the same medical type and class taxonomy as the target task: "
            f"expected dataset={meta.get('dataset')}, got dataset={dataset}."
        )
    split = cfg.get("my_merge_public_split") or cfg.get("stats_split", DEFAULT_PUBLIC_PROBE_SPLIT)
    return {
        "data_root": data_root,
        "dataset": dataset,
        "split": split,
        "explicit": explicit,
        "same_dataset": dataset == meta.get("dataset"),
        "independent_from_data_root": True,
        "use_labels": True,
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
    batches, features, labels = [], [], []
    for batch_idx, (x, y) in enumerate(runtime["loader"]):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        x = _adapt_probe_channels(meta, x)
        y = y.clone().long().view(-1)
        batches.append((x.clone(), y.clone()))
        features.append(_morph_features(meta, x, cfg).cpu())
        labels.append(y)
    if not batches:
        raise RuntimeError("No public probe batches were available for my_merge.")
    labels = torch.cat(labels, dim=0)
    num_classes = int(meta["num_classes"])
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError(
            "my_merge requires class-compatible labeled public probe data; "
            f"observed label range [{int(labels.min().item())}, {int(labels.max().item())}] for num_classes={num_classes}."
        )
    coverage, balance = _coverage_gate(labels, num_classes)
    if coverage < 0.50:
        raise ValueError(
            "my_merge labeled public probe does not cover enough target medical classes: "
            f"coverage={coverage:.3f}, num_classes={num_classes}."
        )
    probe["num_samples"] = int(labels.numel())
    probe["label_coverage"] = float(coverage)
    probe["label_balance"] = float(balance)
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


def _client_information(meta, checkpoints, batches, features, labels, cfg, base_weights):
    base = torch.as_tensor(base_weights, dtype=torch.float32)
    num_classes = int(meta["num_classes"])
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1)
    sample_w = _sample_weights(features)
    metadata_prior = _normalize(base, fallback=torch.ones_like(base) / max(1, base.numel()))
    reliability = float(torch.clamp(features[:, 5].mean(), min=0.0, max=1.0).item())

    logits_list = [_predict_logits(meta, checkpoint, batches, cfg) for checkpoint in checkpoints]
    prob_list, conf_list, margin_list, pred_list = [], [], [], []
    for logits in logits_list:
        probs, conf, margin = _softmax_margin(logits)
        prob_list.append(probs)
        conf_list.append(conf)
        margin_list.append(margin)
        pred_list.append(probs.argmax(dim=1))
    label_coverage, support_gate = _coverage_gate(labels, num_classes)
    evidence_gate = max(0.35 if label_coverage >= 0.50 else 0.0, min(0.80, support_gate))
    class_gate = max(0.25 if label_coverage >= 0.50 else 0.0, min(0.75, support_gate))

    overall_scores, morph_scores, summaries = [], [], []
    class_scores = torch.zeros(num_classes, len(checkpoints), dtype=torch.float32)
    for idx, (probs, conf, margin, preds) in enumerate(zip(prob_list, conf_list, margin_list, pred_list)):
        prior = float(metadata_prior[idx].item())
        correct = (preds == labels).float()
        confidence = _weighted_mean(conf, sample_w)
        separation = _weighted_mean(margin, sample_w)
        accuracy = float(correct.mean().item())
        medical_accuracy = _weighted_mean(correct, sample_w)
        morph_confidence = _weighted_mean(conf * (0.50 + 0.50 * correct), sample_w)
        overall_score = prior * (0.10 + 0.48 * accuracy + 0.22 * confidence + 0.20 * separation)
        morph_score = prior * (0.10 + 0.52 * medical_accuracy + 0.20 * morph_confidence + 0.18 * separation)
        summaries.append({
            "public_accuracy": accuracy,
            "public_medical_accuracy": medical_accuracy,
            "margin_score": separation,
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
            cls_weight = sample_w[cls_mask]
            cls_prob = probs[cls_mask, cls_idx]
            cls_correct = correct[cls_mask]
            cls_margin = margin[cls_mask]
            client = meta["clients"][idx] if idx < len(meta.get("clients", [])) else {}
            seen = set(client.get("classes", []))
            seen_bonus = 1.14 if cls_idx in seen else 0.72
            class_scores[cls_idx, idx] = prior * seen_bonus * (
                0.10
                + 0.40 * _weighted_mean(cls_prob, cls_weight)
                + 0.36 * _weighted_mean(cls_correct, cls_weight)
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
            "public_accuracy": float(summary["public_accuracy"]),
            "public_medical_accuracy": float(summary["public_medical_accuracy"]),
            "margin_confidence": float(summary["margin_score"]),
            "overall_score": float(summary["overall_score"]),
            "morphology_score": float(summary["morph_score"]),
            "evidence_reliability": reliability,
        })
    return overall, morph, torch.stack(class_weights, dim=0), rows, label_coverage, support_gate, reliability


def _module1(meta, checkpoints, cfg, base_weights):
    batches, features, labels, probe = _collect_batches(meta, cfg)
    overall, morph, class_weights, rows, label_coverage, support_gate, reliability = _client_information(
        meta, checkpoints, batches, features, labels, cfg, base_weights
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
        "client_medical_evidence": rows,
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
        key for key, value in state_dict.items()
        if (
            torch.is_floating_point(value)
            and value.ndim == 1
            and value.shape[0] == num_classes
            and any(t in key for t in ("fc", "classifier", "head", "proj"))
        )
    ]


def _find_classifier_keys(state_dict, num_classes):
    weight_key, bias_key = None, None
    bias_keys = _classifier_bias_keys(state_dict, num_classes)
    for key, value in state_dict.items():
        if (
            torch.is_floating_point(value)
            and value.ndim == 2
            and value.shape[0] == num_classes
            and any(t in key for t in ("fc", "classifier", "head", "proj"))
        ):
            weight_key = key
            prefix = key.rsplit(".", 1)[0] if "." in key else ""
            for bias in bias_keys:
                if prefix and bias.startswith(prefix):
                    bias_key = bias
                    break
            if bias_key is None and bias_keys:
                bias_key = bias_keys[0]
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


def _legacy_profile(meta):
    profile = {
        "early_anchor": 0.78,
        "mid_anchor": 0.60,
        "late_anchor": 0.48,
        "class_power": 3.2,
        "class_topk": 3,
        "candidate_alpha": 0.72,
        "early_residual_keep": 0.14,
        "mid_residual_keep": 0.08,
        "late_residual_keep": 0.04,
        "early_residual_scale": 0.28,
        "mid_residual_scale": 0.18,
        "late_residual_scale": 0.09,
    }
    if _model_family(meta) in {"transformer", "vlm"}:
        profile.update({
            "early_anchor": 0.68,
            "mid_anchor": 0.54,
            "late_anchor": 0.44,
            "candidate_alpha": 0.64,
            "early_residual_keep": 0.18,
            "mid_residual_keep": 0.12,
            "late_residual_keep": 0.07,
            "early_residual_scale": 0.33,
            "mid_residual_scale": 0.23,
            "late_residual_scale": 0.13,
        })
    return profile


def _blend_scores(primary, secondary, blend):
    primary = torch.as_tensor(primary, dtype=torch.float32)
    secondary = torch.as_tensor(secondary, dtype=torch.float32)
    return _normalize(float(blend) * primary + (1.0 - float(blend)) * secondary, secondary)


def _sparse_residual_reinjection(base_value, values, primary_weights, secondary_weights, keep_ratio, scale):
    if keep_ratio <= 0.0 or scale <= 0.0 or values[0].ndim < 2:
        return base_value
    stacked = torch.stack([value.detach().float() for value in values], dim=0)
    primary_idx = int(torch.argmax(torch.as_tensor(primary_weights, dtype=torch.float32)).item())
    secondary_idx = int(torch.argmax(torch.as_tensor(secondary_weights, dtype=torch.float32)).item())
    residual = 0.7 * (stacked[primary_idx] - stacked.mean(dim=0)) + 0.3 * (stacked[secondary_idx] - stacked.mean(dim=0))
    flat_abs = residual.abs().flatten()
    if flat_abs.numel() == 0:
        return base_value
    keep_count = max(1, int(flat_abs.numel() * float(keep_ratio)))
    if keep_count >= flat_abs.numel():
        mask = torch.ones_like(residual, dtype=torch.bool)
    else:
        threshold = torch.topk(flat_abs, k=keep_count, largest=True).values[-1]
        mask = residual.abs() >= threshold
    adjusted = base_value.detach().clone().float() + float(scale) * residual * mask.float()
    return adjusted.to(dtype=base_value.dtype)


def _merge_classifier_rows(values, class_weights, profile, fallback_weights):
    template = values[0].detach().clone().float().zero_()
    stacked = torch.stack([value.detach().float() for value in values], dim=0)
    anchor_rows = _weighted_average(list(stacked), fallback_weights)
    sharpened = torch.stack(
        [_normalize(torch.as_tensor(row, dtype=torch.float32).pow(profile["class_power"]), fallback_weights) for row in class_weights],
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
            row_scale = row.abs() + EPS
            anchor_scale = anchor_rows[cls_idx].abs() + EPS
            template[cls_idx] = row * torch.clamp(anchor_scale / row_scale, min=0.45, max=1.15)
        return template.to(dtype=values[0].dtype)

    for cls_idx in range(template.shape[0]):
        row = 0.0
        for rank in range(topk):
            client_idx = int(top_indices[cls_idx, rank].item())
            row = row + top_weights[cls_idx, rank] * stacked[client_idx, cls_idx]
        row = 0.78 * row + 0.22 * anchor_rows[cls_idx]
        row_norm = row.norm() + EPS
        anchor_norm = anchor_rows[cls_idx].norm() + EPS
        template[cls_idx] = row * torch.clamp(anchor_norm / row_norm, min=0.45, max=1.15)
    return template.to(dtype=values[0].dtype)


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


def _morphology_merge(state_dicts, overall, morph, class_weights, num_classes, meta):
    profile = _legacy_profile(meta)
    anchor_idx = int(torch.argmax(torch.as_tensor(morph, dtype=torch.float32)).item())
    anchor_one_hot = F.one_hot(torch.tensor(anchor_idx), num_classes=len(state_dicts)).float()
    consensus = _blend_scores(morph, overall, blend=0.58)
    merged = OrderedDict()
    routed = {"early": 0, "mid": 0, "late": 0, "classifier": 0}
    group_weights = {}

    for key in state_dicts[0]:
        values = [sd[key] for sd in state_dicts]
        first = values[0]
        if not torch.is_floating_point(first):
            merged[key] = values[anchor_idx].detach().clone()
            continue
        if _is_classifier_tensor(key, first, num_classes):
            merged[key] = _merge_classifier_rows(values, class_weights, profile, consensus)
            routed["classifier"] += 1
            continue

        group = _param_group(key, meta)
        if group == "early":
            weights = _blend_scores(anchor_one_hot, morph, blend=profile["early_anchor"])
            keep = profile["early_residual_keep"]
            scale = profile["early_residual_scale"]
        elif group == "late":
            weights = _blend_scores(anchor_one_hot, overall, blend=profile["late_anchor"])
            keep = profile["late_residual_keep"]
            scale = profile["late_residual_scale"]
        else:
            weights = _blend_scores(anchor_one_hot, consensus, blend=profile["mid_anchor"])
            keep = profile["mid_residual_keep"]
            scale = profile["mid_residual_scale"]
        group_weights[group] = weights
        merged_value = _weighted_average(values, weights)
        merged[key] = _sparse_residual_reinjection(
            merged_value,
            values,
            primary_weights=morph,
            secondary_weights=overall,
            keep_ratio=keep,
            scale=scale,
        )
        routed[group] += 1

    return merged, {
        "fusion_weights": [float(x) for x in consensus.tolist()],
        "group_weights": {group: [float(x) for x in weights.tolist()] for group, weights in group_weights.items()},
        "routing_summary": {
            **routed,
            "candidate": "morphology",
            "morphology_anchor_index": anchor_idx,
        },
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


def _morphology_anchor_candidate(state_dicts, morph, class_weights, num_classes, meta):
    profile = _legacy_profile(meta)
    morph = torch.as_tensor(morph, dtype=torch.float32)
    anchor_idx = int(torch.argmax(morph).item())
    state = OrderedDict((key, value.detach().clone()) for key, value in state_dicts[anchor_idx].items())
    routed = 0
    for key, value in list(state.items()):
        if _is_classifier_tensor(key, value, num_classes):
            values = [sd[key] for sd in state_dicts]
            state[key] = _merge_classifier_rows(values, class_weights, profile, morph)
            routed += 1
    return state, {
        "fusion_weights": [1.0 if idx == anchor_idx else 0.0 for idx in range(len(state_dicts))],
        "group_weights": {},
        "routing_summary": {
            "candidate": "morph_anchor",
            "morphology_anchor_index": anchor_idx,
            "classifier_tensors": routed,
        },
    }


def _specialist_client_candidate(state_dicts, client_rows, meta):
    family = _model_family(meta)

    def score(row):
        overall = float(row.get("public_accuracy", row.get("ordinary_accuracy", row.get("overall_weight", 0.0))))
        morph = float(row.get("public_medical_accuracy", row.get("medical_weighted_accuracy", row.get("morphology_weight", overall))))
        margin = float(row.get("margin_confidence", 0.0))
        if family in {"transformer", "vlm"}:
            return 0.45 * morph + 0.30 * overall + 0.25 * margin
        return 0.48 * overall + 0.36 * morph + 0.16 * margin

    best_idx = max(range(len(state_dicts)), key=lambda idx: score(client_rows[idx]) if idx < len(client_rows) else 0.0)
    state = OrderedDict((key, value.detach().clone()) for key, value in state_dicts[best_idx].items())
    return state, {
        "fusion_weights": [1.0 if idx == best_idx else 0.0 for idx in range(len(state_dicts))],
        "group_weights": {},
        "routing_summary": {
            "candidate": "specialist_client",
            "representation_preservation": True,
            "specialist_client_index": best_idx,
        },
    }


def _reference_delta_candidate(state_dicts, reference, overall, morph, class_weights, num_classes, meta):
    if reference is None:
        return None, {"routing_summary": {"candidate": "reference_delta", "available": False}}
    profile = _legacy_profile(meta)
    morph = torch.as_tensor(morph, dtype=torch.float32)
    overall = torch.as_tensor(overall, dtype=torch.float32)
    anchor_idx = int(torch.argmax(morph).item())
    anchor_one_hot = F.one_hot(torch.tensor(anchor_idx), num_classes=len(state_dicts)).float()
    consensus = _blend_scores(morph, overall, blend=0.58)
    merged = OrderedDict()
    routed = {"early": 0, "mid": 0, "late": 0, "classifier": 0, "missing_reference": 0}
    group_weights = {}

    for key in state_dicts[0]:
        values = [sd[key] for sd in state_dicts]
        first = values[0]
        ref = reference.get(key)
        if not torch.is_floating_point(first):
            merged[key] = values[anchor_idx].detach().clone()
            continue
        if ref is None or not torch.is_floating_point(ref) or tuple(ref.shape) != tuple(first.shape):
            merged[key] = _weighted_average(values, consensus).to(first.dtype)
            routed["missing_reference"] += 1
            continue
        if _is_classifier_tensor(key, first, num_classes):
            merged[key] = _merge_classifier_rows(values, class_weights, profile, morph)
            routed["classifier"] += 1
            continue

        group = _param_group(key, meta)
        if group == "early":
            weights = _blend_scores(anchor_one_hot, morph, blend=min(0.92, profile["early_anchor"] + 0.10))
            keep = profile["early_residual_keep"]
            scale = profile["early_residual_scale"]
        elif group == "late":
            weights = _blend_scores(anchor_one_hot, overall, blend=min(0.88, profile["late_anchor"] + 0.12))
            keep = profile["late_residual_keep"]
            scale = profile["late_residual_scale"]
        else:
            weights = _blend_scores(anchor_one_hot, consensus, blend=min(0.90, profile["mid_anchor"] + 0.10))
            keep = profile["mid_residual_keep"]
            scale = profile["mid_residual_scale"]
        group_weights[group] = weights
        deltas = [value.detach().float() - ref.detach().float() for value in values]
        merged_delta = _weighted_average(deltas, weights)
        merged_delta = _sparse_residual_reinjection(
            merged_delta,
            deltas,
            primary_weights=morph,
            secondary_weights=overall,
            keep_ratio=keep,
            scale=scale,
        )
        merged[key] = (ref.detach().float() + merged_delta).to(first.dtype)
        routed[group] += 1

    return merged, {
        "fusion_weights": [float(x) for x in consensus.tolist()],
        "group_weights": {group: [float(x) for x in weights.tolist()] for group, weights in group_weights.items()},
        "routing_summary": {
            **routed,
            "candidate": "reference_delta",
            "morphology_anchor_index": anchor_idx,
            "delta_space": True,
        },
    }


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
    if meta.get("task_type") != "small" or _model_family(meta) != "transformer":
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


def _prototype_head_candidate(base_state_dict, meta, batches, features, labels, cfg):
    if meta.get("task_type") != "small" or _model_family(meta) != "transformer":
        return None, {"routing_summary": {"candidate": "prototype_head", "available": False}}
    num_classes = int(meta["num_classes"])
    weight_key, bias_key = _find_classifier_keys(base_state_dict, num_classes)
    if weight_key is None:
        return None, {"routing_summary": {"candidate": "prototype_head", "available": False, "reason": "no_classifier"}}
    pooled = _state_embeddings(meta, base_state_dict, batches, cfg)
    if pooled is None:
        return None, {"routing_summary": {"candidate": "prototype_head", "available": False, "reason": "no_embeddings"}}
    class_weight = base_state_dict[weight_key].detach().clone().float()
    if pooled.shape[1] != class_weight.shape[1]:
        return None, {
            "routing_summary": {
                "candidate": "prototype_head",
                "available": False,
                "reason": "embedding_head_shape_mismatch",
            }
        }

    sample_w = _sample_weights(features).detach().cpu().float()
    labels = labels.detach().cpu().long().view(-1)
    proto_weight = class_weight.clone()
    row_norms = class_weight.norm(dim=1)
    target_norm = float(torch.median(row_norms).item()) if row_norms.numel() else 1.0
    bias_value = base_state_dict[bias_key].detach().clone().float() if bias_key is not None else None
    proto_bias = bias_value.clone() if bias_value is not None else None

    updated = 0
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
        updated += 1

    candidate = OrderedDict((key, value.detach().clone()) for key, value in base_state_dict.items())
    candidate[weight_key] = proto_weight.to(dtype=base_state_dict[weight_key].dtype)
    if bias_key is not None:
        candidate[bias_key] = proto_bias.to(dtype=base_state_dict[bias_key].dtype)
    return candidate, {
        "fusion_weights": [],
        "group_weights": {},
        "routing_summary": {
            "candidate": "prototype_head",
            "representation_preservation": True,
            "prototype_classes": updated,
            "prototype_weight_key": weight_key,
            "prototype_bias_key": bias_key,
        },
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


def _build_core_candidates(
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
    candidates["avg"] = base_merged
    traces["avg"] = {
        "fusion_weights": [float(x) for x in torch.as_tensor(base, dtype=torch.float32).tolist()],
        "group_weights": {},
        "routing_summary": {"candidate": "avg", "role": "typed_public_validation_anchor"},
    }

    try:
        state, trace = _morphology_merge(state_dicts, overall, morph, class_weights, num_classes, meta)
        candidates["morphology"] = state
        traces["morphology"] = trace
    except Exception as exc:
        traces["morphology_error"] = {"error": str(exc)}

    try:
        state, trace = _morphology_anchor_candidate(state_dicts, morph, class_weights, num_classes, meta)
        candidates["morph_anchor"] = state
        traces["morph_anchor"] = trace
    except Exception as exc:
        traces["morph_anchor_error"] = {"error": str(exc)}

    try:
        state, trace = _specialist_client_candidate(state_dicts, client_rows, meta)
        candidates["specialist_client"] = state
        traces["specialist_client"] = trace
    except Exception as exc:
        traces["specialist_client_error"] = {"error": str(exc)}

    if "morphology" in candidates:
        alpha = _legacy_profile(meta)["candidate_alpha"]
        candidates["consensus"] = _blend_state_dicts(base_merged, candidates["morphology"], right_weight=alpha)
        traces["consensus"] = {
            "fusion_weights": [float(x) for x in consensus.tolist()],
            "group_weights": traces.get("morphology", {}).get("group_weights", {}),
            "routing_summary": {
                "candidate": "consensus",
                "source": "avg_morphology_interpolation",
                "morphology_right_weight": float(alpha),
            },
        }

    if reference is not None:
        try:
            state, trace = _reference_delta_candidate(state_dicts, reference, overall, morph, class_weights, num_classes, meta)
            if state is not None:
                candidates["reference_delta"] = state
                traces["reference_delta"] = trace
        except Exception as exc:
            traces["reference_delta_error"] = {"error": str(exc)}

    try:
        state, trace = _prototype_head_candidate(base_merged, meta, batches, features, labels, cfg)
        if state is not None:
            candidates["prototype_head"] = state
            if not trace.get("fusion_weights"):
                trace["fusion_weights"] = [float(x) for x in torch.as_tensor(base, dtype=torch.float32).tolist()]
            traces["prototype_head"] = trace
    except Exception as exc:
        traces["prototype_head_error"] = {"error": str(exc)}

    state, trace = _medical_weighted_merge(state_dicts, base_merged, reference, base, overall, morph, class_weights, num_classes, meta)
    candidates["medical_weighted_fusion"] = state
    traces["medical_weighted_fusion"] = trace
    if meta.get("task_type") == "small" and reference is not None and param_names is not None:
        try:
            sign_state = _sign_delta_merge(state_dicts, base_merged, reference, param_names, consensus.tolist())
            candidates["sign_consistent_delta"] = sign_state
            traces["sign_consistent_delta"] = {
                "fusion_weights": [float(x) for x in consensus.tolist()],
                "group_weights": {},
                "routing_summary": {
                    "candidate": "sign_consistent_delta",
                    "sign_delta_weight_source": "type_labeled_medical_consensus",
                    "blend_right_weight": 1.0,
                },
            }
            candidates["avg_sign_blend_0p25"] = _blend_state_dicts(base_merged, sign_state, right_weight=0.25)
            traces["avg_sign_blend_0p25"] = {
                "fusion_weights": [float(x) for x in consensus.tolist()],
                "group_weights": {},
                "routing_summary": {
                    "candidate": "avg_sign_blend_0p25",
                    "sign_delta_weight_source": "type_labeled_medical_consensus",
                    "blend_right_weight": 0.25,
                },
            }
        except Exception as exc:
            traces["sign_delta_error"] = {"error": str(exc)}
    try:
        state, trace = _anatomy_head_graft(base_merged, state_dicts, base, class_weights, num_classes)
        if state is not None:
            candidates["anatomy_head_graft_avg"] = state
            traces["anatomy_head_graft_avg"] = {
                **trace,
                "routing_summary": {
                    **trace.get("routing_summary", {}),
                    "anchor_candidate": "avg",
                },
            }
    except Exception as exc:
        traces["anatomy_head_graft_avg_error"] = {"error": str(exc)}
    if reference is not None:
        try:
            state, trace = merge_iso_c(state_dicts, reference, consensus.tolist(), cfg)
            candidates["medical_iso_c"] = state
            traces["medical_iso_c"] = {
                "fusion_weights": trace.get("normalized_weights", [float(x) for x in consensus.tolist()]),
                "group_weights": {},
                "routing_summary": {
                    "candidate": "medical_iso_c",
                    "source": trace.get("implementation"),
                    "weight_source": "medical_public_probe_consensus",
                },
            }
        except Exception as exc:
            traces["medical_iso_c_error"] = {"error": str(exc)}
    return candidates, traces


def _candidate_ranking(metrics, selected_name):
    ranked = []
    for name, item in metrics.items():
        score = float(item.get("selection_score", 0.0))
        ranked.append((score, _candidate_priority(name), name))
    ranked.sort(reverse=True)
    best_score = ranked[0][0] if ranked else 0.0
    return [
        {
            "rank": rank,
            "candidate": name,
            "selected": name == selected_name,
            "selection_score": float(score),
            "gap_to_best": float(score - best_score),
            "priority": int(priority),
        }
        for rank, (score, priority, name) in enumerate(ranked, start=1)
    ]


def _candidate_priority(name):
    order = {
        "prototype_head": 80,
        "specialist_client": 75,
        "reference_delta": 70,
        "morphology": 65,
        "morph_anchor": 60,
        "consensus": 55,
        "medical_weighted_fusion": 50,
        "sign_consistent_delta": 45,
        "medical_iso_c": 40,
        "anatomy_head_graft_avg": 35,
        "avg_sign_blend_0p25": 30,
        "avg": 10,
    }
    return order.get(name, 0)


def _selection_score(acc, medical_acc, loss):
    return (
        float(acc)
        + DEFAULT_SELECTION_MEDICAL_WEIGHT * (float(medical_acc) - float(acc))
        - DEFAULT_SELECTION_LOSS_WEIGHT * float(loss)
    )


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
        "public_probe_labeled": True,
        "public_acc": float(acc),
        "public_medical_acc": float(medical_acc),
        "public_loss": float(loss),
        "selection_score": float(_selection_score(acc, medical_acc, loss)),
        "selection_medical_weight": DEFAULT_SELECTION_MEDICAL_WEIGHT,
        "selection_loss_weight": DEFAULT_SELECTION_LOSS_WEIGHT,
        "selection_rule": "type_matched_labeled_public_accuracy",
    }


def _select_candidate(
    candidates,
    traces,
    base_merged,
    base_weights,
    meta,
    cfg,
    score_batches,
    score_features,
    score_labels,
):
    if not candidates:
        raise RuntimeError("my_merge did not construct any formal fusion candidates.")

    metrics, best_name, best_key = {}, None, None
    for name, state in list(candidates.items()):
        metrics[name] = _eval_candidate(
            meta,
            state,
            cfg,
            score_batches,
            score_features,
            score_labels,
        )
        key = (metrics[name]["selection_score"], _candidate_priority(name), name)
        if best_key is None or key > best_key:
            best_name, best_key = name, key
    trace = traces.get(best_name, {})
    routing = {"validated_candidates": len(candidates)}
    routing.update(trace.get("routing_summary", {}))
    ranking = _candidate_ranking(metrics, best_name)
    return candidates[best_name], {
        "selected_candidate": best_name,
        "candidate_pool": list(candidates.keys()),
        "candidate_metrics": metrics,
        "candidate_ranking": ranking,
        "selected_candidate_metrics": metrics.get(best_name, {}),
        "fusion_weights": trace.get("fusion_weights", [float(x) for x in torch.as_tensor(base_weights, dtype=torch.float32).tolist()]),
        "group_weights": trace.get("group_weights", {}),
        "routing_summary": routing,
    }


def _module2_fusion_selection(state_dicts, base_merged, meta, cfg, info, reference, param_names):
    base = torch.as_tensor(info["base_weights"], dtype=torch.float32)
    candidates, traces = _build_core_candidates(
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
        info["client_medical_evidence"],
    )
    state, trace = _select_candidate(
        candidates,
        traces,
        base_merged,
        base,
        meta,
        cfg,
        info["batches"],
        info["features"],
        info["labels"],
    )
    return state, {
        "fusion_rule": "medical_evidence_guided_fusion_selection",
        "candidate_pool": trace.get("candidate_pool", []),
        "candidate_metrics": trace.get("candidate_metrics", {}),
        "candidate_ranking": trace.get("candidate_ranking", []),
        "selected_candidate_metrics": trace.get("selected_candidate_metrics", {}),
        "selected_candidate": trace.get("selected_candidate", "avg"),
        "routing_summary": trace.get("routing_summary", {}),
        "fusion_weights": trace.get("fusion_weights", []),
        "group_weights": trace.get("group_weights", {}),
        "evidence_reliability": info.get("evidence_reliability"),
        "public_probe": info.get("probe", {}),
    }


def merge_my_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    if meta is None or checkpoints is None or cfg is None:
        raise ValueError("my_merge requires task metadata, client checkpoints, and runtime config.")
    if not _is_medical_image_task(meta):
        raise ValueError(f"my_merge is defined only for medical image tasks, got dataset={meta.get('dataset')}.")

    base_merged, base_weights = average_state_dicts(state_dicts, weights)
    info = _module1(meta, checkpoints, cfg, base_weights)
    reference, param_names = build_reference_bundle(meta, device="cpu")
    merged, fusion = _module2_fusion_selection(state_dicts, base_merged, meta, cfg, info, reference, param_names)

    return merged, {
        "implementation": "type_matched_labeled_medical_posthoc_merge_v3",
        "medical_only": True,
        "modality": meta.get("dataset"),
        "model_family": _model_family(meta),
        "base_weights": [float(x) for x in base_weights],
        "overall_weights": [float(x) for x in info["overall_weights"].tolist()],
        "morphology_weights": [float(x) for x in info["morphology_weights"].tolist()],
        "class_weights": [[float(v) for v in row] for row in info["class_weights"].tolist()],
        "client_medical_evidence": info["client_medical_evidence"],
        "candidate_pool": fusion["candidate_pool"],
        "candidate_metrics": fusion["candidate_metrics"],
        "candidate_ranking": fusion["candidate_ranking"],
        "selected_candidate_metrics": fusion["selected_candidate_metrics"],
        "selected_candidate": fusion["selected_candidate"],
        "fusion_rule": fusion["fusion_rule"],
        "fusion_weights": fusion["fusion_weights"],
        "group_weights": fusion["group_weights"],
        "routing_summary": fusion["routing_summary"],
        "evidence_reliability": fusion["evidence_reliability"],
        "public_probe": fusion["public_probe"],
    }
