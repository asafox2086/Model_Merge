from torch.utils.data import DataLoader

from dataset import NpzTensorDataset, load_npz_splits
from model import build_model
from utils.metrics import evaluate_classification


def evaluate_small_checkpoint(meta, checkpoint, data_root, split='test', device='cpu', batch_size=64, num_workers=4, amp=False):
    npz_path = f"{data_root}/{meta['dataset']}.npz"
    splits = load_npz_splits(npz_path)
    ds = NpzTensorDataset(splits[split].images, splits[split].labels, transform=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(str(device).startswith('cuda')))

    model, _, _, _ = build_model(
        name=meta['model'],
        num_classes=int(meta['num_classes']),
        in_channels=int(meta['in_channels']),
        pretrained=False,
    )
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(device)

    return evaluate_classification(
        model=model,
        loader=loader,
        device=device,
        forward_fn=lambda m, x: m(x),
        amp_enabled=bool(amp),
    )
