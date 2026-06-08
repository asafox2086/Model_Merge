from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import NpzTensorDataset, load_npz_splits
from model import build_model
from utils.metrics import evaluate_classification


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
        return transforms.Resize((target, target), antialias=True), (source_h, source_w), (target, target)
    return None, (source_h, source_w), (source_h, source_w)


def evaluate_small_checkpoint(meta, checkpoint, data_root, split='test', device='cpu', batch_size=64, num_workers=4, amp=False):
    npz_path = f"{data_root}/{meta['dataset']}.npz"
    splits = load_npz_splits(npz_path)
    transform, source_size, eval_size = _build_small_transform(meta, splits[split].images)
    ds = NpzTensorDataset(splits[split].images, splits[split].labels, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(str(device).startswith('cuda')))

    model, _, _, _ = build_model(
        name=meta['model'],
        num_classes=int(meta['num_classes']),
        in_channels=int(meta['in_channels']),
        pretrained=False,
    )
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(device)

    result = evaluate_classification(
        model=model,
        loader=loader,
        device=device,
        forward_fn=lambda m, x: m(x),
        amp_enabled=bool(amp),
    )
    result.update({
        'source_image_size': list(source_size),
        'eval_image_size': list(eval_size),
        'image_resize': source_size != eval_size,
    })
    return result
