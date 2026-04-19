from torch.utils.data import DataLoader

from dataset import ClipImageDataset, build_clip_transform, load_npz_splits
from model.clip_model import build_clip_model, encode_image_features, get_clip_base
from utils.runtime import resolve_vlm_max_text_len, validate_loader_settings
from utils.metrics import evaluate_classification


def build_vlm_forward(text_encoder):
    def _forward(model, x):
        base = get_clip_base(model)
        image_features = encode_image_features(base, x)
        text_features = text_encoder()
        logit_scale = base.logit_scale.exp()
        return logit_scale * image_features @ text_features.t()
    return _forward


def evaluate_vlm_checkpoint(meta, checkpoint, data_root, split='test', device='cpu', batch_size=64, num_workers=4, amp=False):
    batch_size, num_workers = validate_loader_settings(batch_size, num_workers, context='vlm evaluation')
    npz_path = f"{data_root}/{meta['dataset']}.npz"
    splits = load_npz_splits(npz_path)
    tf = build_clip_transform(image_size=int(meta.get('image_size', 224)))
    ds = ClipImageDataset(splits[split].images, splits[split].labels, tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(str(device).startswith('cuda')))
    max_text_len = resolve_vlm_max_text_len(meta, default=32)

    model, text_encoder = build_clip_model(
        clip_model_name=meta['clip_model'],
        class_names=meta['class_names'],
        text_template=meta.get('text_template', 'a medical image of class {}'),
        max_text_len=max_text_len,
        random_init=bool(meta.get('clip_random_init', False)),
        device=device,
    )
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(device)
    text_encoder = text_encoder.to(device)

    return evaluate_classification(
        model=model,
        loader=loader,
        device=device,
        forward_fn=build_vlm_forward(text_encoder),
        amp_enabled=bool(amp),
    )
