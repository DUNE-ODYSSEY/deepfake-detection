"""Frame-level dataset over preprocessed FF++ face crops.

Index layout produced by preprocess.py (JPEG) or preprocess_reference.py (PNG):
  <root>/<comp>/<real|Method>/<video_id>/frame_xxxxx.{jpg,png}
Labels: real=0, fake=1.
"""
from pathlib import Path


def _list_frames(vdir: Path):
    """Crops may be .jpg (preprocess.py) or .png (preprocess_reference.py) —
    glob both so the same dataset class serves either pipeline's output."""
    return sorted(list(vdir.glob("*.jpg")) + list(vdir.glob("*.png")))

import random

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

from .splits import load_split_ids

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomBlurSharpen:
    """Randomly blur or sharpen, independent of the label, so high-frequency
    energy stops being a usable real/fake cue (Phase 3 found DF/NT fakes
    blurrier and FS fakes sharper than real faces)."""

    def __init__(self, p: float = 0.9):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        if random.random() < 0.5:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
        return img.filter(ImageFilter.UnsharpMask(radius=random.uniform(1.0, 2.0),
                                                  percent=int(random.uniform(50, 200)),
                                                  threshold=0))


def build_transforms(img_size: int, train: bool, hf_aug: bool = False):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            *([RandomBlurSharpen()] if hf_aug else []),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FFPPDataset(Dataset):
    def __init__(self, root: str, splits_dir: str, split: str, compression: str,
                 methods, img_size: int = 299, train_aug: bool = False,
                 hf_aug: bool = False):
        """methods: list of manipulation names, e.g. ["Deepfakes"] or all four."""
        self.samples = []  # (path, label, video_key)
        real_ids, fake_ids = load_split_ids(splits_dir, split)
        comp_dir = Path(root) / compression

        real_dir = comp_dir / "real"
        for vdir in sorted(real_dir.iterdir()) if real_dir.exists() else []:
            if vdir.name in real_ids:
                for f in _list_frames(vdir):
                    self.samples.append((f, 0, f"real/{vdir.name}"))

        for method in methods:
            mdir = comp_dir / method
            if not mdir.exists():
                raise FileNotFoundError(f"Missing preprocessed dir: {mdir}")
            for vdir in sorted(mdir.iterdir()):
                if vdir.name in fake_ids:
                    for f in _list_frames(vdir):
                        self.samples.append((f, 1, f"{method}/{vdir.name}"))

        if not self.samples:
            raise RuntimeError(
                f"No samples for split={split} comp={compression} methods={methods}")
        self.transform = build_transforms(img_size, train_aug, hf_aug)

    def class_weights(self):
        """Per-sample weights for WeightedRandomSampler (balances real vs fake)."""
        n_fake = sum(lbl for _, lbl, _ in self.samples)
        n_real = len(self.samples) - n_fake
        w = {0: 1.0 / max(n_real, 1), 1: 1.0 / max(n_fake, 1)}
        return torch.tensor([w[lbl] for _, lbl, _ in self.samples], dtype=torch.double)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, video = self.samples[i]
        img = self.transform(Image.open(path).convert("RGB"))
        return img, torch.tensor(label, dtype=torch.float32), video
