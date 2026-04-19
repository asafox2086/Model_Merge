from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitData:
    images: np.ndarray
    labels: np.ndarray


class NpzTensorDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        img = self.images[idx]
        y = int(self.labels[idx])
        if img.ndim == 2:
            img = img[:, :, None]
        x = torch.from_numpy(img).float()
        if x.ndim == 3:
            x = x.permute(2, 0, 1)
        x = x / 255.0
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def _to_single_label(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1).astype(np.int64)
    raise ValueError('Only single-label datasets are supported.')


def load_npz_splits(npz_path: str) -> Dict[str, SplitData]:
    arr = np.load(npz_path, allow_pickle=False)
    out = {}
    for split in ('train', 'val', 'test'):
        ik = f'{split}_images'
        lk = f'{split}_labels'
        if ik not in arr or lk not in arr:
            raise KeyError(f'Missing keys {ik}/{lk} in {npz_path}')
        out[split] = SplitData(images=arr[ik], labels=_to_single_label(arr[lk]))
    return out


def infer_data_info(train_images: np.ndarray, train_labels: np.ndarray) -> Tuple[int, int]:
    if train_images.ndim == 3:
        in_channels = 1
    elif train_images.ndim >= 4:
        in_channels = int(train_images.shape[-1])
    else:
        raise ValueError(f'Unexpected image shape: {train_images.shape}')
    classes = np.unique(train_labels)
    num_classes = int(classes.max()) + 1
    return in_channels, num_classes
