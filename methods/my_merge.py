from collections import OrderedDict

import torch
import torch.nn.functional as F

from utils.runtime import build_runtime
from utils.state_dict import average_state_dicts


EPS = 1e-8
HEAD_SCALE_DEFAULT = 0.05
DEFAULT_STATS_MAX_BATCHES = 1
DEFAULT_EVAL_MAX_BATCHES = 1
DEFAULT_BN_BATCHES = 1


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
    return meta.get('task_type') == 'small'


def _is_blood_morphology_task(meta):
    return (
        _is_small_med_task(meta)
        and meta.get('dataset') == 'bloodmnist_224'
        and meta.get('model') in {'resnet', 'convnext', 'mobilenet'}
    )


def _spatial_eccentricity(mask):
    side = mask.shape[-1]
    coords = torch.linspace(-1.0, 1.0, side, device=mask.device)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
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
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=gray.device).view(1, 1, 3, 3)
    gray_map = gray.unsqueeze(1)
    grad_x = F.conv2d(gray_map, sobel_x, padding=1)
    grad_y = F.conv2d(gray_map, sobel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + EPS).squeeze(1)


def _blood_morphology_features(x):
    red = x[:, 0]
    green = x[:, 1]
    blue = x[:, 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    purple = 0.5 * (red + blue) - 0.7 * green
    cell_score = (1.0 - gray) + 0.35 * torch.relu(purple)

    flat_cell = cell_score.flatten(1)
    cell_thresh = torch.quantile(flat_cell, q=0.60, dim=1, keepdim=True).view(-1, 1, 1)
    cell_mask = cell_score >= cell_thresh

    masked_purple = purple.masked_fill(~cell_mask, float('-inf'))
    safe_masked_purple = torch.where(torch.isfinite(masked_purple), masked_purple, torch.zeros_like(masked_purple))
    flat_nucleus = safe_masked_purple.flatten(1)
    nucleus_thresh = torch.quantile(flat_nucleus, q=0.78, dim=1, keepdim=True).view(-1, 1, 1)
    nucleus_mask = cell_mask & (safe_masked_purple >= nucleus_thresh)

    cell_area = cell_mask.float().mean(dim=(1, 2))
    nucleus_area = nucleus_mask.float().mean(dim=(1, 2))
    nc_ratio = nucleus_area / (cell_area + EPS)
    boundary_strength = (_sobel_edges(purple) * cell_mask.float()).sum(dim=(1, 2)) / (cell_mask.float().sum(dim=(1, 2)) + EPS)
    roundness = _spatial_eccentricity(nucleus_mask)
    cell_only = cell_mask & (~nucleus_mask)
    cyto_intensity = (gray * cell_only.float()).sum(dim=(1, 2)) / (cell_only.float().sum(dim=(1, 2)) + EPS)
    boundary_importance = nc_ratio * (0.65 + boundary_strength) * (0.5 + (1.0 - cyto_intensity))
    return torch.stack([nc_ratio, boundary_strength, roundness, cyto_intensity, boundary_importance], dim=1)


def _generic_morphology_features(x):
    if x.shape[1] == 1:
        gray = x[:, 0]
    else:
        gray = x.mean(dim=1)
    center = gray[:, gray.shape[-2] // 4: 3 * gray.shape[-2] // 4, gray.shape[-1] // 4: 3 * gray.shape[-1] // 4]
    center_mean = center.mean(dim=(1, 2), keepdim=True)
    foreground = (gray - center_mean).abs()
    flat = foreground.flatten(1)
    thresh = torch.quantile(flat, q=0.70, dim=1, keepdim=True).view(-1, 1, 1)
    fg_mask = foreground >= thresh
    area_ratio = fg_mask.float().mean(dim=(1, 2))
    edge_map = _sobel_edges(gray)
    edge_strength = (edge_map * fg_mask.float()).sum(dim=(1, 2)) / (fg_mask.float().sum(dim=(1, 2)) + EPS)
    eccentricity = _spatial_eccentricity(fg_mask)
    intensity_contrast = torch.abs(gray.mean(dim=(1, 2)) - center_mean.view(-1))
    compactness = area_ratio / (edge_strength + EPS)
    morphology_salience = area_ratio * (0.5 + edge_strength) * (0.5 + eccentricity)
    return torch.stack([area_ratio, edge_strength, eccentricity, intensity_contrast, morphology_salience, compactness], dim=1)


def _batch_morphology_features(meta, x):
    if meta.get('dataset') == 'bloodmnist_224' and x.shape[1] >= 3:
        return _blood_morphology_features(x)
    return _generic_morphology_features(x)


def _collect_split_batches(meta, cfg, split):
    runtime = build_runtime(
        meta=meta,
        data_root=cfg['data_root'],
        split=split,
        batch_size=int(cfg.get('stats_batch_size', 32) or cfg.get('batch_size', 64)),
        num_workers=int(cfg.get('stats_num_workers', cfg.get('num_workers', 4))),
        device=torch.device('cpu'),
    )
    batches = []
    feature_chunks = []
    labels = []
    max_batches = int(cfg.get('my_merge_stats_max_batches', DEFAULT_STATS_MAX_BATCHES) or 0)
    for batch_idx, (x, y) in enumerate(runtime['loader']):
        if max_batches and batch_idx >= max_batches:
            break
        batches.append((x.clone(), y.clone()))
        feature_chunks.append(_batch_morphology_features(meta, x).cpu())
        labels.append(y.clone())
    return batches, torch.cat(feature_chunks, dim=0), torch.cat(labels, dim=0)


def _evaluate_client_predictions(meta, checkpoint, batches, cfg):
    device = torch.device(cfg.get('stats_device', cfg.get('device', 'cpu')))
    runtime = build_runtime(
        meta=meta,
        data_root=cfg['data_root'],
        split=cfg.get('stats_split', 'val'),
        batch_size=max(1, len(batches[0][1])),
        num_workers=0,
        device=device,
    )
    model = runtime['model']
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    forward_fn = runtime['forward_fn']

    logits_out = []
    with torch.no_grad():
        for x, _ in batches:
            x = x.to(device, non_blocking=True)
            logits_out.append(forward_fn(model, x).detach().float().cpu())
    return torch.cat(logits_out, dim=0)


def _client_scores(meta, checkpoints, batches, features, labels, cfg, base_weights):
    num_clients = len(checkpoints)
    num_classes = int(meta['num_classes'])
    sample_importance = features[:, 4]
    sample_importance = sample_importance / (sample_importance.mean() + EPS)
    sample_importance = torch.clamp(sample_importance, min=0.25, max=3.0)

    overall_scores = []
    morph_scores = []
    class_scores = torch.zeros(num_classes, num_clients, dtype=torch.float32)

    for client_idx, checkpoint in enumerate(checkpoints):
        logits = _evaluate_client_predictions(meta, checkpoint, batches, cfg)
        preds = logits.argmax(dim=1)
        correct = (preds == labels).float()

        true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        masked_logits = logits.clone()
        masked_logits[torch.arange(logits.size(0)), labels] = float('-inf')
        other_logits = masked_logits.max(dim=1).values
        margin = torch.sigmoid(true_logits - other_logits)

        overall_acc = float(correct.mean().item())
        morph_acc = float((correct * sample_importance).sum().item() / sample_importance.sum().item())
        margin_score = float((margin * sample_importance).sum().item() / sample_importance.sum().item())

        prior = float(base_weights[client_idx])
        overall_scores.append(prior * (0.35 + overall_acc + 0.45 * margin_score))
        morph_scores.append(prior * (0.25 + 0.90 * morph_acc + 0.25 * margin_score))

        seen_classes = set(meta['clients'][client_idx].get('classes', []))
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
            seen_bonus = 1.18 if cls_idx in seen_classes else 0.62
            class_scores[cls_idx, client_idx] = prior * seen_bonus * (0.15 + 0.70 * cls_acc + 0.35 * cls_morph + 0.20 * cls_conf)

    overall_weights = _normalize_scores(overall_scores, fallback=base_weights)
    morph_weights = _normalize_scores(morph_scores, fallback=base_weights)
    class_weights = [_normalize_scores(class_scores[cls_idx], fallback=morph_weights) for cls_idx in range(num_classes)]
    return overall_weights, morph_weights, torch.stack(class_weights, dim=0)


def _is_classifier_tensor(key, tensor, num_classes):
    if not torch.is_floating_point(tensor):
        return False
    if tensor.ndim not in (1, 2):
        return False
    if tensor.shape[0] != num_classes:
        return False
    return any(token in key for token in ('fc', 'classifier', 'head'))


def _param_group(key):
    early_tokens = ('conv1', 'bn1', 'layer1', 'layer2', 'stem', 'patch_embed', 'downsample_layers.0', 'downsample_layers.1', 'features.0', 'features.1')
    late_tokens = ('layer4', 'fc', 'classifier', 'head', 'norm', 'blocks.10', 'blocks.11', 'stages.3')
    if any(token in key for token in early_tokens):
        return 'early'
    if any(token in key for token in late_tokens):
        return 'late'
    return 'mid'


def _weighted_average_for_key(values, weight_tensor):
    out = values[0].detach().clone() * float(weight_tensor[0].item())
    for value, weight in zip(values[1:], weight_tensor[1:]):
        out.add_(value.detach(), alpha=float(weight.item()))
    return out


def _merge_classifier_rows(values, class_weights):
    template = values[0].detach().clone().float().zero_()
    stacked = torch.stack([value.detach().float() for value in values], dim=0)
    anchor_rows = stacked.mean(dim=0)
    topk = min(2, class_weights.shape[1])
    top_weights, top_indices = torch.topk(class_weights, k=topk, dim=1)
    top_weights = torch.softmax(top_weights / 0.40, dim=1)
    if values[0].ndim == 1:
        for cls_idx in range(template.shape[0]):
            row = 0.0
            for rank in range(topk):
                client_idx = int(top_indices[cls_idx, rank].item())
                row = row + top_weights[cls_idx, rank] * stacked[client_idx, cls_idx]
            row = 0.82 * row + 0.18 * anchor_rows[cls_idx]
            anchor_scale = anchor_rows[cls_idx].abs() + EPS
            row_scale = row.abs() + EPS
            row = row * torch.clamp(anchor_scale / row_scale, min=0.35, max=1.10)
            template[cls_idx] = row
        return template.to(dtype=values[0].dtype)

    for cls_idx in range(template.shape[0]):
        row = 0.0
        for rank in range(topk):
            client_idx = int(top_indices[cls_idx, rank].item())
            row = row + top_weights[cls_idx, rank] * stacked[client_idx, cls_idx]
        row = 0.82 * row + 0.18 * anchor_rows[cls_idx]
        anchor_norm = anchor_rows[cls_idx].norm() + EPS
        row_norm = row.norm() + EPS
        row = row * torch.clamp(anchor_norm / row_norm, min=0.35, max=1.10)
        template[cls_idx] = row
    return template.to(dtype=values[0].dtype)


def _layerwise_merge(state_dicts, overall_weights, morph_weights, class_weights, num_classes):
    anchor_idx = int(torch.argmax(morph_weights).item())
    anchor_one_hot = F.one_hot(torch.tensor(anchor_idx), num_classes=len(state_dicts)).float()
    sharpened_class_weights = torch.stack([
        _normalize_scores(row.pow(3.0), fallback=morph_weights) for row in class_weights
    ], dim=0)

    merged = OrderedDict()
    for key in state_dicts[0].keys():
        values = [sd[key] for sd in state_dicts]
        first = values[0]
        if not torch.is_floating_point(first):
            merged[key] = values[anchor_idx].detach().clone()
            continue
        if _is_classifier_tensor(key, first, num_classes):
            merged[key] = _merge_classifier_rows(values, sharpened_class_weights)
            continue

        group = _param_group(key)
        if group == 'early':
            weights = _blend_scores(anchor_one_hot, morph_weights, blend=0.82)
        elif group == 'late':
            weights = _blend_scores(anchor_one_hot, overall_weights, blend=0.68)
        else:
            weights = _blend_scores(anchor_one_hot, _blend_scores(morph_weights, overall_weights, blend=0.55), blend=0.58)
        merged[key] = _weighted_average_for_key(values, weights)
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
    device = torch.device(cfg.get('stats_device', cfg.get('device', 'cpu')))
    runtime = build_runtime(
        meta=meta,
        data_root=cfg['data_root'],
        split=split,
        batch_size=int(cfg.get('stats_batch_size', 32) or cfg.get('batch_size', 64)),
        num_workers=0,
        device=device,
    )
    model = runtime['model']
    model.load_state_dict(merged_state_dict, strict=True)
    model.eval()
    forward_fn = runtime['forward_fn']
    total = 0
    correct = 0.0
    weighted_correct = 0.0
    importance_total = 0.0
    max_abs = 0.0
    with torch.no_grad():
        max_batches = int(cfg.get('my_merge_eval_max_batches', DEFAULT_EVAL_MAX_BATCHES) or 0)
        for batch_idx, (x, y) in enumerate(runtime['loader']):
            if max_batches and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = forward_fn(model, x).detach().float()
            pred = logits.argmax(dim=1)
            hit = (pred == y).float().cpu()
            importance = _batch_morphology_features(meta, x.detach().float()).cpu()[:, 4]
            importance = importance / (importance.mean() + EPS)
            importance = torch.clamp(importance, min=0.25, max=3.0)
            correct += float(hit.sum().item())
            weighted_correct += float((hit * importance).sum().item())
            importance_total += float(importance.sum().item())
            total += int(y.size(0))
            max_abs = max(max_abs, float(logits.abs().max().item()))
    if total <= 0:
        return {'acc': 0.0, 'morph_acc': 0.0, 'score': 0.0, 'max_abs': max_abs}
    acc = correct / total
    morph_acc = weighted_correct / max(importance_total, EPS)
    score = acc + 0.18 * morph_acc
    return {'acc': acc, 'morph_acc': morph_acc, 'score': score, 'max_abs': max_abs}


def _recalibrate_batchnorm(meta, merged_state_dict, cfg):
    device = torch.device(cfg.get('stats_device', cfg.get('device', 'cpu')))
    runtime = build_runtime(
        meta=meta,
        data_root=cfg['data_root'],
        split=cfg.get('stats_split', 'val'),
        batch_size=int(cfg.get('stats_batch_size', 32) or cfg.get('batch_size', 64)),
        num_workers=int(cfg.get('stats_num_workers', cfg.get('num_workers', 4))),
        device=device,
    )
    model = runtime['model']
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

    max_batches = int(cfg.get('my_merge_bn_batches', DEFAULT_BN_BATCHES))
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(runtime['loader']):
            if max_batches and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            _ = runtime['forward_fn'](model, x)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _auto_head_scale(meta, merged_state_dict, cfg):
    device = torch.device(cfg.get('stats_device', cfg.get('device', 'cpu')))
    runtime = build_runtime(
        meta=meta,
        data_root=cfg['data_root'],
        split=cfg.get('stats_split', 'val'),
        batch_size=int(cfg.get('stats_batch_size', 32) or cfg.get('batch_size', 64)),
        num_workers=0,
        device=device,
    )
    model = runtime['model']
    model.load_state_dict(merged_state_dict, strict=True)
    model.eval()
    max_abs = 0.0
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(runtime['loader']):
            if batch_idx >= 4:
                break
            x = x.to(device, non_blocking=True)
            logits = runtime['forward_fn'](model, x)
            max_abs = max(max_abs, float(logits.detach().abs().max().item()))
    if max_abs <= 0.0:
        return HEAD_SCALE_DEFAULT
    safe_cap = 40.0 if meta.get('dataset') == 'bloodmnist_224' else 60.0
    return min(1.0, max(0.005, safe_cap / max_abs))


def merge_my_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    if meta is None or checkpoints is None or cfg is None or not _is_blood_morphology_task(meta):
        merged_state_dict, normalized_weights = average_state_dicts(state_dicts, weights)
        return merged_state_dict, {
            'implementation': 'plain_weighted_average_baseline' if meta is None or not _is_small_med_task(meta) else 'blood_specialized_average_fallback',
            'normalized_weights': normalized_weights,
        }

    base_merged, base_weights = average_state_dicts(state_dicts, weights)
    try:
        batches, features, labels = _collect_split_batches(meta, cfg, split=cfg.get('stats_split', 'val'))
        overall_weights, morph_weights, class_weights = _client_scores(
            meta,
            checkpoints,
            batches,
            features,
            labels,
            cfg,
            base_weights,
        )
        morphology_merged = _layerwise_merge(
            state_dicts,
            overall_weights=overall_weights,
            morph_weights=morph_weights,
            class_weights=class_weights,
            num_classes=int(meta['num_classes']),
        )
        morphology_merged = _recalibrate_batchnorm(meta, morphology_merged, cfg)
        head_scale = _auto_head_scale(meta, morphology_merged, cfg)
        morphology_merged = _apply_head_temperature(
            morphology_merged,
            num_classes=int(meta['num_classes']),
            scale=head_scale,
        )
        avg_candidate = _recalibrate_batchnorm(meta, base_merged, cfg)
        avg_candidate = _apply_head_temperature(
            avg_candidate,
            num_classes=int(meta['num_classes']),
            scale=_auto_head_scale(meta, avg_candidate, cfg),
        )

        avg_metrics = _evaluate_merged_state(meta, avg_candidate, cfg, split=cfg.get('stats_split', 'val'))
        morph_metrics = _evaluate_merged_state(meta, morphology_merged, cfg, split=cfg.get('stats_split', 'val'))
        use_morphology = morph_metrics['score'] >= avg_metrics['score']
        merged_state_dict = morphology_merged if use_morphology else avg_candidate

        return merged_state_dict, {
            'implementation': 'morphology_guided_boundary_cascade_merge_v3',
            'base_weights': [float(x) for x in base_weights],
            'overall_weights': [float(x) for x in overall_weights.tolist()],
            'morphology_weights': [float(x) for x in morph_weights.tolist()],
            'class_weights': [[float(v) for v in row] for row in class_weights.tolist()],
            'feature_summary': {f'feature_{idx}': float(features[:, idx].mean().item()) for idx in range(features.shape[1])},
            'head_scale': float(head_scale),
            'bn_recalibrated': True,
            'candidate_metrics': {
                'avg_candidate': avg_metrics,
                'morphology_candidate': morph_metrics,
            },
            'selected_candidate': 'morphology' if use_morphology else 'avg',
        }
    except Exception as exc:
        return base_merged, {
            'implementation': 'morphology_guided_boundary_cascade_merge_fallback',
            'fallback_reason': str(exc),
            'normalized_weights': base_weights,
        }
