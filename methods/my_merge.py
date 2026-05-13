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


def _image_space01(meta, x):
    image = x.detach().float()
    if meta.get("task_type") == "vlm" and image.shape[1] >= 3:
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


def _max_pool_map(value, kernel):
    return F.max_pool2d(value.unsqueeze(1), kernel_size=kernel, stride=1, padding=kernel // 2).squeeze(1)


def _min_pool_map(value, kernel):
    return -_max_pool_map(-value, kernel=kernel)


def _morph_close(value, kernel=9):
    return _min_pool_map(_max_pool_map(value, kernel=kernel), kernel=kernel)


def _shades_of_gray_color_constancy(x, power=6.0):
    image = torch.clamp(x, min=0.0, max=1.0)
    illuminant = (image.clamp_min(EPS).pow(power).mean(dim=(2, 3), keepdim=True) + EPS).pow(1.0 / power)
    target = illuminant.mean(dim=1, keepdim=True)
    return torch.clamp(image / (illuminant + EPS) * target, min=0.0, max=1.0)


def _dullrazor_clean_gray(gray):
    black_hat = torch.clamp(_morph_close(gray, kernel=9) - gray, min=0.0)
    edges = _sobel_edges(gray)
    hair_mask = (black_hat >= _safe_quantile(black_hat, q=0.93)) & (edges >= _safe_quantile(edges, q=0.62))
    local_mean = F.avg_pool2d(gray.unsqueeze(1), kernel_size=9, stride=1, padding=4).squeeze(1)
    cleaned = torch.where(hair_mask, local_mean, gray)
    hair_density = hair_mask.float().mean(dim=(1, 2))
    return cleaned, hair_density, black_hat


def _soft_tissue_window(gray):
    central_window = torch.clamp((gray - 0.50) / 0.36 + 0.50, min=0.0, max=1.0)
    percentile_window = _robust_rescale01(gray, low_q=0.08, high_q=0.92)
    return 0.62 * central_window + 0.38 * percentile_window


def _anisotropic_diffusion(gray, iterations=3, kappa=0.08, gamma=0.18):
    diffused = gray
    for _ in range(iterations):
        padded = F.pad(diffused.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
        north = padded[:, :-2, 1:-1] - diffused
        south = padded[:, 2:, 1:-1] - diffused
        west = padded[:, 1:-1, :-2] - diffused
        east = padded[:, 1:-1, 2:] - diffused
        update = 0.0
        for delta in (north, south, west, east):
            conductance = torch.exp(-((delta / kappa).square()))
            update = update + conductance * delta
        diffused = torch.clamp(diffused + gamma * update, min=0.0, max=1.0)
    return diffused


def _blood_morphology_features(x):
    red = x[:, 0]
    green = x[:, 1]
    blue = x[:, 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    purple = 0.5 * (red + blue) - 0.7 * green
    cell_score = (1.0 - gray) + 0.35 * torch.relu(purple)

    cell_mask = cell_score >= _safe_quantile(cell_score, q=0.60)
    masked_purple = torch.where(cell_mask, purple, torch.zeros_like(purple))
    nucleus_mask = cell_mask & (masked_purple >= _safe_quantile(masked_purple, q=0.78))

    cell_area = cell_mask.float().mean(dim=(1, 2))
    nucleus_area = nucleus_mask.float().mean(dim=(1, 2))
    nc_ratio = nucleus_area / (cell_area + EPS)
    boundary_strength = (_sobel_edges(purple) * cell_mask.float()).sum(dim=(1, 2)) / (cell_mask.float().sum(dim=(1, 2)) + EPS)
    roundness = _spatial_eccentricity(nucleus_mask)
    cell_only = cell_mask & (~nucleus_mask)
    cyto_intensity = (gray * cell_only.float()).sum(dim=(1, 2)) / (cell_only.float().sum(dim=(1, 2)) + EPS)
    chromatin_contrast = (purple * nucleus_mask.float()).sum(dim=(1, 2)) / (nucleus_mask.float().sum(dim=(1, 2)) + EPS)
    diagnostic_salience = nc_ratio * (0.65 + boundary_strength) * (0.55 + chromatin_contrast.abs()) * (0.5 + (1.0 - cyto_intensity))
    return torch.stack(
        [cell_area, nc_ratio, boundary_strength, roundness, chromatin_contrast, diagnostic_salience],
        dim=1,
    )


def _derma_morphology_features(x):
    corrected = _shades_of_gray_color_constancy(x)
    red = corrected[:, 0]
    green = corrected[:, 1]
    blue = corrected[:, 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    clean_gray, hair_density, black_hat = _dullrazor_clean_gray(gray)

    center = clean_gray[
        :,
        clean_gray.shape[-2] // 5: 4 * clean_gray.shape[-2] // 5,
        clean_gray.shape[-1] // 5: 4 * clean_gray.shape[-1] // 5,
    ]
    center_ref = center.mean(dim=(1, 2), keepdim=True)
    lesion_signal = (
        (clean_gray - center_ref).abs()
        + 0.36 * (red - green).abs()
        + 0.26 * (red - blue).abs()
        + 0.12 * _sobel_edges(clean_gray)
    )
    lesion_mask = lesion_signal >= _safe_quantile(lesion_signal, q=0.70)

    area_ratio = lesion_mask.float().mean(dim=(1, 2))
    lesion_boundary = _sobel_edges(lesion_mask.float())
    edge_strength = (_sobel_edges(clean_gray) * lesion_mask.float()).sum(dim=(1, 2)) / (
        lesion_mask.float().sum(dim=(1, 2)) + EPS
    )
    border_irregularity = lesion_boundary.sum(dim=(1, 2)) / (
        torch.sqrt(lesion_mask.float().sum(dim=(1, 2)) + EPS) + EPS
    )
    lr_asym = (lesion_mask.float() - lesion_mask.flip(-1).float()).abs().mean(dim=(1, 2))
    tb_asym = (lesion_mask.float() - lesion_mask.flip(-2).float()).abs().mean(dim=(1, 2))
    asymmetry = 0.5 * (lr_asym + tb_asym)

    color_dev = []
    for channel in (red, green, blue):
        lesion_mean = (channel * lesion_mask.float()).sum(dim=(1, 2)) / (lesion_mask.float().sum(dim=(1, 2)) + EPS)
        channel_var = (((channel - lesion_mean[:, None, None]) * lesion_mask.float()) ** 2).sum(dim=(1, 2))
        channel_var = channel_var / (lesion_mask.float().sum(dim=(1, 2)) + EPS)
        color_dev.append(torch.sqrt(channel_var + EPS))
    color_variegation = torch.stack(color_dev, dim=1).mean(dim=1)

    centroid_y = torch.linspace(-1.0, 1.0, gray.shape[-2], device=gray.device).view(1, -1, 1)
    centroid_x = torch.linspace(-1.0, 1.0, gray.shape[-1], device=gray.device).view(1, 1, -1)
    mass = lesion_mask.float().sum(dim=(1, 2)) + EPS
    cx = (lesion_mask.float() * centroid_x).sum(dim=(1, 2)) / mass
    cy = (lesion_mask.float() * centroid_y).sum(dim=(1, 2)) / mass
    center_bias = torch.sqrt(cx.square() + cy.square() + EPS)

    artifact_load = torch.clamp(hair_density + 0.35 * black_hat.mean(dim=(1, 2)), min=0.0, max=1.0)
    diagnostic_salience = (
        area_ratio
        * (0.50 + edge_strength + 0.08 * border_irregularity)
        * (0.45 + asymmetry)
        * (0.45 + color_variegation)
        * (0.85 + 0.60 * artifact_load)
    )
    return torch.stack(
        [area_ratio, border_irregularity, asymmetry, color_variegation, artifact_load + 0.25 * center_bias, diagnostic_salience],
        dim=1,
    )


def _organ_morphology_features(meta, x):
    gray = x[:, 0] if x.shape[1] == 1 else x.mean(dim=1)
    windowed = _soft_tissue_window(gray)
    edge_map = _sobel_edges(windowed)
    local_texture = _local_variance(windowed, kernel=7)
    contrast = (windowed - windowed.mean(dim=(1, 2), keepdim=True)).abs() + 0.30 * edge_map + 0.15 * local_texture
    tissue_mask = contrast >= _safe_quantile(contrast, q=0.68)

    area_ratio = tissue_mask.float().mean(dim=(1, 2))
    boundary_strength = (edge_map * tissue_mask.float()).sum(dim=(1, 2)) / (tissue_mask.float().sum(dim=(1, 2)) + EPS)
    symmetry = 1.0 - (tissue_mask.float() - tissue_mask.flip(-1).float()).abs().mean(dim=(1, 2))
    symmetry = torch.clamp(symmetry, min=0.0)
    eccentricity = _spatial_eccentricity(tissue_mask)
    intensity_band = (gray * tissue_mask.float()).sum(dim=(1, 2)) / (tissue_mask.float().sum(dim=(1, 2)) + EPS)

    coords_y = torch.linspace(-1.0, 1.0, gray.shape[-2], device=gray.device).view(1, -1, 1)
    coords_x = torch.linspace(-1.0, 1.0, gray.shape[-1], device=gray.device).view(1, 1, -1)
    mass = tissue_mask.float().sum(dim=(1, 2)) + EPS
    cx = (tissue_mask.float() * coords_x).sum(dim=(1, 2)) / mass
    cy = (tissue_mask.float() * coords_y).sum(dim=(1, 2)) / mass
    centerline_bias = torch.clamp(1.0 - torch.sqrt(cx.square() + cy.square() + EPS), min=0.0)

    left_band = tissue_mask[:, :, : tissue_mask.shape[-1] // 5].float().mean(dim=(1, 2))
    right_band = tissue_mask[:, :, -tissue_mask.shape[-1] // 5 :].float().mean(dim=(1, 2))
    lateral_structure = torch.clamp((left_band - right_band).abs() + cx.abs(), min=0.0, max=1.0)
    vertical_elongation = torch.clamp(1.0 - eccentricity, min=0.0, max=1.0)
    spine_band = torch.clamp(torch.maximum(left_band, right_band) / (tissue_mask.float().mean(dim=(1, 2)) + EPS), min=0.0, max=2.0)
    if meta.get("dataset") == "organsmnist_224":
        view_prior = 0.45 + 0.35 * vertical_elongation + 0.20 * spine_band
    else:
        view_prior = 0.45 + 0.35 * lateral_structure + 0.20 * symmetry

    soft_tissue_contrast = (contrast * tissue_mask.float()).sum(dim=(1, 2)) / (tissue_mask.float().sum(dim=(1, 2)) + EPS)
    diagnostic_salience = area_ratio * (0.50 + soft_tissue_contrast) * (0.50 + boundary_strength) * view_prior
    return torch.stack(
        [area_ratio, soft_tissue_contrast, view_prior, eccentricity, intensity_band + 0.25 * centerline_bias, diagnostic_salience],
        dim=1,
    )


def _ultrasound_morphology_features(x):
    gray = x[:, 0] if x.shape[1] == 1 else x.mean(dim=1)
    diffused = _anisotropic_diffusion(_robust_rescale01(gray, low_q=0.03, high_q=0.97))
    edge_map = _sobel_edges(diffused)
    speckle = _local_variance(gray, kernel=9)
    residual_speckle = torch.clamp(speckle - _local_variance(diffused, kernel=9), min=0.0)
    tissue_signal = 0.48 * residual_speckle + 0.52 * edge_map
    tissue_mask = tissue_signal >= _safe_quantile(tissue_signal, q=0.74)

    area_ratio = tissue_mask.float().mean(dim=(1, 2))
    edge_coherence = (edge_map * tissue_mask.float()).sum(dim=(1, 2)) / (tissue_mask.float().sum(dim=(1, 2)) + EPS)
    speckle_strength = (residual_speckle * tissue_mask.float()).sum(dim=(1, 2)) / (tissue_mask.float().sum(dim=(1, 2)) + EPS)

    top = diffused[:, : diffused.shape[-2] // 3]
    bottom = diffused[:, -diffused.shape[-2] // 3 :]
    posterior_shadow = torch.clamp((top.mean(dim=(1, 2)) - bottom.mean(dim=(1, 2))) / (top.mean(dim=(1, 2)) + EPS), min=0.0, max=2.0)

    gray_map = diffused.unsqueeze(1)
    grad_x = F.conv2d(gray_map, torch.tensor([[-1.0, 0.0, 1.0]], device=gray.device).view(1, 1, 1, 3), padding=(0, 1)).squeeze(1)
    grad_y = F.conv2d(gray_map, torch.tensor([[-1.0], [0.0], [1.0]], device=gray.device).view(1, 1, 3, 1), padding=(1, 0)).squeeze(1)
    anisotropy = (grad_x.abs().mean(dim=(1, 2)) - grad_y.abs().mean(dim=(1, 2))).abs() / (
        grad_x.abs().mean(dim=(1, 2)) + grad_y.abs().mean(dim=(1, 2)) + EPS
    )

    diagnostic_salience = area_ratio * (0.55 + edge_coherence) * (0.55 + speckle_strength) * (0.50 + posterior_shadow + 0.35 * anisotropy)
    return torch.stack(
        [area_ratio, edge_coherence, speckle_strength, posterior_shadow, anisotropy, diagnostic_salience],
        dim=1,
    )


def _generic_medical_features(x):
    gray = x[:, 0] if x.shape[1] == 1 else x.mean(dim=1)
    edge_map = _sobel_edges(gray)
    contrast = (gray - gray.mean(dim=(1, 2), keepdim=True)).abs()
    fg_mask = (contrast + 0.25 * edge_map) >= _safe_quantile(contrast + 0.25 * edge_map, q=0.70)
    area_ratio = fg_mask.float().mean(dim=(1, 2))
    edge_strength = (edge_map * fg_mask.float()).sum(dim=(1, 2)) / (fg_mask.float().sum(dim=(1, 2)) + EPS)
    eccentricity = _spatial_eccentricity(fg_mask)
    intensity = (gray * fg_mask.float()).sum(dim=(1, 2)) / (fg_mask.float().sum(dim=(1, 2)) + EPS)
    salience = area_ratio * (0.5 + edge_strength) * (0.5 + eccentricity)
    return torch.stack([area_ratio, edge_strength, eccentricity, intensity, salience, salience], dim=1)


def _batch_morphology_features(meta, x):
    x = _image_space01(meta, x)
    dataset = meta.get("dataset")
    if dataset == "bloodmnist_224" and x.shape[1] >= 3:
        return _blood_morphology_features(x)
    if dataset == "dermamnist_224" and x.shape[1] >= 3:
        return _derma_morphology_features(x)
    if dataset in {"organcmnist_224", "organsmnist_224"}:
        return _organ_morphology_features(meta, x)
    if dataset == "chaoshengmnist_224":
        return _ultrasound_morphology_features(x)
    return _generic_medical_features(x)


def _sample_importance(features):
    importance = features[:, -1]
    importance = importance / (importance.mean() + EPS)
    return torch.clamp(importance, min=0.25, max=3.5)


def _class_rarity_weights(labels, num_classes, meta):
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    inv_sqrt = torch.sqrt(counts.sum() / counts)
    inv_sqrt = inv_sqrt / (inv_sqrt.mean() + EPS)
    strength = 0.85 if meta.get("dataset") == "dermamnist_224" else 0.35
    rarity = 1.0 + strength * (inv_sqrt - 1.0)
    return torch.clamp(rarity, min=0.55, max=3.0)


def _medical_sample_weights(meta, features, labels, num_classes):
    morphology = _sample_importance(features)
    rarity = _class_rarity_weights(labels, num_classes, meta)[labels]
    dataset = meta.get("dataset")
    if dataset == "dermamnist_224":
        artifact_or_color = 0.55 * features[:, 3] + 0.45 * features[:, 4]
        domain_focus = 1.0 + torch.clamp(artifact_or_color, min=0.0, max=1.5)
    elif dataset in {"organcmnist_224", "organsmnist_224"}:
        domain_focus = 0.75 + torch.clamp(features[:, 2], min=0.0, max=2.0)
    elif dataset == "chaoshengmnist_224":
        domain_focus = 0.80 + torch.clamp(features[:, 1] + features[:, 3], min=0.0, max=2.0)
    else:
        domain_focus = torch.ones_like(morphology)
    weights = morphology * rarity * domain_focus
    weights = weights / (weights.mean() + EPS)
    return torch.clamp(weights, min=0.20, max=5.0)


def _feature_names(meta):
    dataset = meta.get("dataset")
    if dataset == "bloodmnist_224":
        return ["cell_area", "nc_ratio", "boundary_strength", "nucleus_roundness", "chromatin_contrast", "diagnostic_salience"]
    if dataset == "dermamnist_224":
        return ["lesion_area", "border_irregularity", "asymmetry", "color_variegation", "artifact_load", "diagnostic_salience"]
    if dataset in {"organcmnist_224", "organsmnist_224"}:
        return ["tissue_area", "soft_tissue_contrast", "view_spatial_prior", "shape_eccentricity", "windowed_intensity", "diagnostic_salience"]
    if dataset == "chaoshengmnist_224":
        return ["lesion_area", "edge_coherence", "speckle_strength", "posterior_shadow", "anisotropy", "diagnostic_salience"]
    return [f"feature_{idx}" for idx in range(6)]


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
        feature_chunks.append(_batch_morphology_features(meta, x).cpu())
        labels.append(y.clone())
    return batches, torch.cat(feature_chunks, dim=0), torch.cat(labels, dim=0)


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
    sample_importance = _medical_sample_weights(meta, features, labels, num_classes)
    class_rarity = _class_rarity_weights(labels, num_classes, meta)
    hard_mask = sample_importance >= torch.quantile(sample_importance, q=0.60)

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
        focal_weight = sample_importance * torch.clamp((1.0 - margin).pow(1.35) + 0.25, min=0.25, max=2.5)
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
    dataset = meta.get("dataset")
    family = _model_family(meta)
    profile = {
        "early_anchor": 0.78,
        "mid_anchor": 0.60,
        "late_anchor": 0.48,
        "class_power": 3.2,
        "class_topk": 3,
        "candidate_alpha": 0.72,
        "early_residual_keep": 0.10,
        "mid_residual_keep": 0.06,
        "late_residual_keep": 0.03,
        "early_residual_scale": 0.22,
        "mid_residual_scale": 0.15,
        "late_residual_scale": 0.08,
    }
    if dataset == "bloodmnist_224":
        profile.update(
            {
                "early_anchor": 0.82,
                "mid_anchor": 0.66,
                "late_anchor": 0.50,
                "class_power": 3.4,
                "early_residual_keep": 0.14,
                "mid_residual_keep": 0.08,
                "early_residual_scale": 0.28,
                "mid_residual_scale": 0.18,
            }
        )
    elif dataset == "dermamnist_224":
        profile.update(
            {
                "early_anchor": 0.80,
                "mid_anchor": 0.62,
                "late_anchor": 0.48,
                "class_power": 3.8,
                "class_topk": 2,
                "early_residual_keep": 0.20,
                "mid_residual_keep": 0.12,
                "late_residual_keep": 0.06,
                "early_residual_scale": 0.34,
                "mid_residual_scale": 0.24,
                "late_residual_scale": 0.12,
            }
        )
    elif dataset in {"organcmnist_224", "organsmnist_224"}:
        profile.update(
            {
                "early_anchor": 0.76,
                "mid_anchor": 0.60,
                "late_anchor": 0.45,
                "class_power": 3.0,
                "class_topk": 2,
                "early_residual_keep": 0.16,
                "mid_residual_keep": 0.10,
                "late_residual_keep": 0.05,
                "early_residual_scale": 0.28,
                "mid_residual_scale": 0.20,
                "late_residual_scale": 0.10,
            }
        )
        if dataset == "organsmnist_224":
            profile["mid_anchor"] = 0.64
            profile["mid_residual_keep"] = 0.13
            profile["mid_residual_scale"] = 0.24
    elif dataset == "chaoshengmnist_224":
        profile.update(
            {
                "early_anchor": 0.84,
                "mid_anchor": 0.68,
                "late_anchor": 0.52,
                "class_power": 3.3,
                "class_topk": 2,
                "early_residual_keep": 0.22,
                "mid_residual_keep": 0.15,
                "late_residual_keep": 0.08,
                "early_residual_scale": 0.38,
                "mid_residual_scale": 0.28,
                "late_residual_scale": 0.14,
            }
        )
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
    elif meta.get("dataset") == "dermamnist_224":
        key = lambda item: (
            0.42 * item[1]["focal_acc"] + 0.32 * item[1]["morph_acc"] + 0.26 * item[1]["overall_acc"],
            item[1]["margin_score"],
        )
    else:
        key = lambda item: (
            0.48 * item[1]["overall_acc"] + 0.36 * item[1]["morph_acc"] + 0.16 * item[1]["focal_acc"],
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
    sample_importance = _medical_sample_weights(meta, features, labels, num_classes)
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


def _layerwise_merge(state_dicts, overall_weights, morph_weights, class_weights, num_classes, meta):
    profile = _merge_profile(meta)
    anchor_idx = int(torch.argmax(morph_weights).item())
    anchor_one_hot = F.one_hot(torch.tensor(anchor_idx), num_classes=len(state_dicts)).float()
    consensus = _blend_scores(morph_weights, overall_weights, blend=0.58)

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
        if group == "early":
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
            features = _batch_morphology_features(meta, x.detach().float()).cpu()
            y_cpu = y.detach().cpu()
            importance = _medical_sample_weights(meta, features, y_cpu, num_classes)
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
    dataset = meta.get("dataset")
    if dataset == "dermamnist_224":
        score = 0.42 * acc + 0.34 * balanced_acc + 0.16 * morph_acc + 0.08 * hard_acc
    elif dataset in {"organcmnist_224", "organsmnist_224"}:
        score = 0.48 * acc + 0.18 * balanced_acc + 0.24 * morph_acc + 0.10 * hard_acc
    elif dataset == "chaoshengmnist_224":
        score = 0.44 * acc + 0.14 * balanced_acc + 0.22 * morph_acc + 0.20 * hard_acc
    else:
        score = 0.55 * acc + 0.12 * balanced_acc + 0.23 * morph_acc + 0.10 * hard_acc
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
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(runtime["loader"]):
            if batch_idx >= 4:
                break
            x = x.to(device, non_blocking=True)
            logits = runtime["forward_fn"](model, x)
            max_abs = max(max_abs, float(logits.detach().abs().max().item()))
    if max_abs <= 0.0:
        return HEAD_SCALE_DEFAULT
    dataset = meta.get("dataset")
    if dataset == "bloodmnist_224":
        safe_cap = 40.0
    elif dataset == "chaoshengmnist_224":
        safe_cap = 32.0
    else:
        safe_cap = 50.0
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


def _build_reference_delta_candidate(state_dicts, overall_weights, morph_weights, class_weights, num_classes, meta):
    reference_state, _ = build_reference_bundle(meta, device="cpu")
    profile = _merge_profile(meta)
    anchor_idx = int(torch.argmax(morph_weights).item())
    anchor_one_hot = F.one_hot(torch.tensor(anchor_idx), num_classes=len(state_dicts)).float()
    consensus = _blend_scores(morph_weights, overall_weights, blend=0.58)

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


def _prepare_candidate(meta, merged_state_dict, cfg):
    prepared = _recalibrate_batchnorm(meta, merged_state_dict, cfg)
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


def _choose_candidate(meta, candidate_metrics):
    family = _model_family(meta)
    if family == "vlm":
        def key_fn(item):
            name, metric = item
            return (
                float(metric["score"]),
                float(metric["acc"]),
                float(metric.get("balanced_acc", 0.0)),
                float(metric["hard_acc"]),
                _candidate_priority(meta, name),
            )
    elif family == "transformer":
        def key_fn(item):
            name, metric = item
            return (
                float(metric["score"]),
                float(metric["acc"]),
                float(metric.get("balanced_acc", 0.0)),
                float(metric["morph_acc"]),
                _candidate_priority(meta, name),
            )
    else:
        def key_fn(item):
            name, metric = item
            return (
                float(metric["score"]),
                float(metric["acc"]),
                float(metric.get("balanced_acc", 0.0)),
                float(metric["hard_acc"]),
                _candidate_priority(meta, name),
            )
    return max(candidate_metrics.items(), key=key_fn)[0]


def merge_my_merge(state_dicts, weights, meta=None, checkpoints=None, cfg=None):
    if meta is None or checkpoints is None or cfg is None or not _is_medical_image_task(meta):
        merged_state_dict, normalized_weights = average_state_dicts(state_dicts, weights)
        return merged_state_dict, {
            "implementation": "medical_image_only_merge_fallback",
            "normalized_weights": normalized_weights,
        }

    base_merged, base_weights = average_state_dicts(state_dicts, weights)
    try:
        batches, features, labels = _collect_split_batches(meta, cfg, split=cfg.get("stats_split", "val"))
        overall_weights, morph_weights, class_weights, client_summaries = _client_scores(
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
            num_classes=int(meta["num_classes"]),
            meta=meta,
        )
        morphology_anchor = _build_morphology_anchor_candidate(
            state_dicts,
            morph_weights=morph_weights,
            class_weights=class_weights,
            num_classes=int(meta["num_classes"]),
            meta=meta,
        )
        specialist_candidate_state, specialist_idx = _build_specialist_client_candidate(
            state_dicts,
            client_summaries=client_summaries,
            meta=meta,
        )
        reference_delta = None
        if _model_family(meta) in {"transformer", "vlm"}:
            reference_delta = _build_reference_delta_candidate(
                state_dicts,
                overall_weights=overall_weights,
                morph_weights=morph_weights,
                class_weights=class_weights,
                num_classes=int(meta["num_classes"]),
                meta=meta,
            )

        base_candidate = _prepare_candidate(meta, base_merged, cfg)
        morphology_candidate = _prepare_candidate(meta, morphology_merged, cfg)
        anchor_candidate = _prepare_candidate(meta, morphology_anchor, cfg)
        specialist_candidate = _prepare_candidate(meta, specialist_candidate_state, cfg)
        prototype_candidate_state = _build_prototype_head_candidate(
            base_merged,
            meta=meta,
            batches=batches,
            labels=labels,
            features=features,
            cfg=cfg,
        )
        alpha = _merge_profile(meta)["candidate_alpha"]
        consensus_candidate = _prepare_candidate(
            meta,
            _interpolate_state_dicts(base_candidate, morphology_candidate, alpha=alpha),
            cfg,
        )

        candidate_pool = {
            "avg": base_candidate,
            "morphology": morphology_candidate,
            "morph_anchor": anchor_candidate,
            "specialist_client": specialist_candidate,
            "consensus": consensus_candidate,
        }
        if reference_delta is not None:
            candidate_pool["reference_delta"] = _prepare_candidate(meta, reference_delta, cfg)
        if prototype_candidate_state is not None:
            candidate_pool["prototype_head"] = _prepare_candidate(meta, prototype_candidate_state, cfg)
        candidate_metrics = {
            name: _evaluate_merged_state(meta, candidate_state, cfg, split=cfg.get("stats_split", "val"))
            for name, candidate_state in candidate_pool.items()
        }
        selected_name = _choose_candidate(meta, candidate_metrics)
        merged_state_dict = candidate_pool[selected_name]

        names = _feature_names(meta)
        feature_summary = {
            names[idx] if idx < len(names) else f"feature_{idx}": float(features[:, idx].mean().item())
            for idx in range(features.shape[1])
        }

        return merged_state_dict, {
            "implementation": "medical_modality_specific_boundary_cascade_merge_v5",
            "medical_only": True,
            "modality": meta.get("dataset"),
            "model_family": _model_family(meta),
            "base_weights": [float(x) for x in base_weights],
            "overall_weights": [float(x) for x in overall_weights.tolist()],
            "morphology_weights": [float(x) for x in morph_weights.tolist()],
            "class_weights": [[float(v) for v in row] for row in class_weights.tolist()],
            "client_summaries": client_summaries,
            "specialist_client_index": int(specialist_idx),
            "feature_summary": feature_summary,
            "candidate_metrics": candidate_metrics,
            "selected_candidate": selected_name,
        }
    except Exception as exc:
        return base_merged, {
            "implementation": "medical_modality_specific_boundary_cascade_merge_fallback",
            "medical_only": True,
            "fallback_reason": str(exc),
            "normalized_weights": base_weights,
        }
