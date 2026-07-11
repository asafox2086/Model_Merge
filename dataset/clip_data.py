from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class SplitData:
    images: np.ndarray
    labels: np.ndarray


class ClipImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
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
        x = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        x = self.transform(x)
        return x, y


def build_clip_transform(image_size: int):
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1) if x.shape[0] == 1 else x[:3]),
        transforms.Normalize(mean=mean, std=std),
    ])


def make_class_names(num_classes: int, raw_names: str):
    if raw_names.strip():
        names = [x.strip() for x in raw_names.split(',') if x.strip()]
        if len(names) != num_classes:
            raise ValueError(f'class-names count ({len(names)}) != num_classes ({num_classes})')
        return names
    return [f'class_{i}' for i in range(num_classes)]
