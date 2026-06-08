from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ClipImageDataset, NpzTensorDataset, build_clip_transform, load_npz_splits
from model import build_model
from model.clip_model import build_clip_model, encode_image_features, get_clip_base
from utils.reference_cache import load_reference_bundle, save_reference_bundle
from utils.seed import set_seed


def validate_loader_settings(batch_size, num_workers, *, context):
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{context}: batch_size must be an integer, got {batch_size!r}') from exc
    try:
        num_workers = int(num_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{context}: num_workers must be an integer, got {num_workers!r}') from exc
    if batch_size <= 0:
        raise ValueError(f'{context}: batch_size must be a positive integer, got {batch_size}')
    if num_workers < 0:
        raise ValueError(f'{context}: num_workers must be a non-negative integer, got {num_workers}')
    return batch_size, num_workers


def resolve_vlm_max_text_len(meta, default=32):
    raw = meta.get('max_text_len', default)
    try:
        max_text_len = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vlm max_text_len must be an integer, got {raw!r}") from exc
    if max_text_len <= 0:
        raise ValueError(f'vlm max_text_len must be a positive integer, got {max_text_len}')
    return max_text_len


def _image_hw(images):
    if images.ndim == 3:
        return int(images.shape[1]), int(images.shape[2])
    if images.ndim >= 4:
        return int(images.shape[1]), int(images.shape[2])
    raise ValueError(f'Unexpected image shape: {images.shape}')


def _build_small_transform(meta, images):
    source_h, source_w = _image_hw(images)
    target = int(meta.get('image_size') or 0)
    if target > 0 and (source_h != target or source_w != target):
        return transforms.Resize((target, target), antialias=True)
    return None


def build_small_runtime(meta, data_root, split, batch_size, num_workers, device):
    batch_size, num_workers = validate_loader_settings(batch_size, num_workers, context='small runtime')
    npz_path = f"{data_root}/{meta['dataset']}.npz"
    splits = load_npz_splits(npz_path)
    transform = _build_small_transform(meta, splits[split].images)
    ds = NpzTensorDataset(splits[split].images, splits[split].labels, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(str(device).startswith('cuda')))
    model, _, _, _ = build_model(
        name=meta['model'],
        num_classes=int(meta['num_classes']),
        in_channels=int(meta.get('in_channels', 3)),
        pretrained=False,
    )
    model = model.to(device)
    return {
        'model': model,
        'loader': loader,
        'forward_fn': lambda m, x: m(x),
        'text_encoder': None,
    }


def build_vlm_runtime(meta, data_root, split, batch_size, num_workers, device, max_text_len=32):
    batch_size, num_workers = validate_loader_settings(batch_size, num_workers, context='vlm runtime')
    npz_path = f"{data_root}/{meta['dataset']}.npz"
    splits = load_npz_splits(npz_path)
    tf = build_clip_transform(image_size=int(meta.get('image_size', 224)))
    ds = ClipImageDataset(splits[split].images, splits[split].labels, tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(str(device).startswith('cuda')))
    max_text_len = resolve_vlm_max_text_len(meta, default=max_text_len)
    model, text_encoder = build_clip_model(
        clip_model_name=meta['clip_model'],
        class_names=meta['class_names'],
        text_template=meta.get('text_template', 'a medical image of class {}'),
        max_text_len=int(meta.get('max_text_len', max_text_len)),
        random_init=bool(meta.get('clip_random_init', False)),
        device=device,
    )

    def forward_fn(m, x):
        base = get_clip_base(m)
        image_features = encode_image_features(base, x)
        text_features = text_encoder()
        logit_scale = base.logit_scale.exp()
        return logit_scale * image_features @ text_features.t()

    return {
        'model': model,
        'loader': loader,
        'forward_fn': forward_fn,
        'text_encoder': text_encoder,
    }


def build_runtime(meta, data_root, split, batch_size, num_workers, device):
    if meta['task_type'] == 'small':
        return build_small_runtime(meta, data_root, split, batch_size, num_workers, device)
    if meta['task_type'] == 'vlm':
        return build_vlm_runtime(meta, data_root, split, batch_size, num_workers, device)
    raise ValueError(f"Unsupported task_type: {meta['task_type']}")


def build_reference_bundle(meta, device='cpu'):
    cached, cache_path = load_reference_bundle(meta)
    if cached is not None:
        return cached['state_dict'], cached['param_names']

    set_seed(int(meta.get('seed', 42)))

    if meta['task_type'] == 'small':
        model, _, _, _ = build_model(
            name=meta['model'],
            num_classes=int(meta['num_classes']),
            in_channels=int(meta.get('in_channels', 3)),
            pretrained=bool(meta.get('pretrained', False)),
        )
    elif meta['task_type'] == 'vlm':
        max_text_len = resolve_vlm_max_text_len(meta, default=32)
        model, _ = build_clip_model(
            clip_model_name=meta['clip_model'],
            class_names=meta['class_names'],
            text_template=meta.get('text_template', 'a medical image of class {}'),
            max_text_len=max_text_len,
            random_init=bool(meta.get('clip_random_init', False)),
            device=device,
        )
    else:
        raise ValueError(f"Unsupported task_type: {meta['task_type']}")

    param_names = [name for name, _ in model.named_parameters()]
    state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    saved_path = save_reference_bundle(meta, state_dict, param_names)
    if saved_path == cache_path:
        print(f'[INFO] cached reference bundle saved to: {saved_path}')
    return state_dict, param_names


def build_reference_state(meta, device='cpu'):
    state_dict, _ = build_reference_bundle(meta, device=device)
    return state_dict
