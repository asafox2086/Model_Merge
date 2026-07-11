from typing import Tuple

from utils.hf_env import configure_hf_endpoint

MODEL_ALIASES = {
    'resnet': 'resnet18',
    'resnet18': 'resnet18',
    'resnet34': 'resnet34',
    'vit-t': 'vit_tiny_patch16_224',
    'vit_t': 'vit_tiny_patch16_224',
    'vit_tiny': 'vit_tiny_patch16_224',
    'convnext': 'convnext_tiny',
    'convnext_t': 'convnext_tiny',
    'convnext_tiny': 'convnext_tiny',
    'densenet': 'densenet121',
    'densenet121': 'densenet121',
    'efficientnet': 'efficientnet_b0',
    'efficientnet_b0': 'efficientnet_b0',
    'mobilenet': 'mobilenetv3_small_100',
    'mobilenetv3_small': 'mobilenetv3_small_100',
    'mobilenetv3_small_100': 'mobilenetv3_small_100',
    'swin-tiny': 'swin_tiny_patch4_window7_224',
    'swin_tiny': 'swin_tiny_patch4_window7_224',
    'swin_tiny_patch4_window7_224': 'swin_tiny_patch4_window7_224',
}


def resolve_model_name(name: str) -> str:
    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f'Unsupported model: {name}. Supported: {sorted(MODEL_ALIASES.keys())}')
    return MODEL_ALIASES[key]


def build_model(name: str, num_classes: int, in_channels: int, pretrained: bool = False) -> Tuple[object, str, int, bool]:
    try:
        import timm
    except ImportError as exc:
        raise ImportError('timm is required. Install it with: pip install timm') from exc
    resolved = resolve_model_name(name)
    used_pretrained = bool(pretrained)
    if pretrained:
        configure_hf_endpoint()
        try:
            model = timm.create_model(resolved, pretrained=True, num_classes=num_classes, in_chans=in_channels)
        except Exception as exc:
            print(f'[WARN] pretrained load failed for model={resolved}, in_chans={in_channels}: {exc}. Fallback to random initialization.')
            model = timm.create_model(resolved, pretrained=False, num_classes=num_classes, in_chans=in_channels)
            used_pretrained = False
    else:
        model = timm.create_model(resolved, pretrained=False, num_classes=num_classes, in_chans=in_channels)
    default_img_size = 224
    if hasattr(model, 'default_cfg') and isinstance(model.default_cfg, dict):
        input_size = model.default_cfg.get('input_size')
        if input_size and len(input_size) == 3:
            default_img_size = int(input_size[-1])
    return model, resolved, default_img_size, used_pretrained
