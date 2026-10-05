"""Load the dog images of one split as (image_tensor, class_index) pairs."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dogbreeds.breeds import CLASS_NAMES
from dogbreeds.paths import RAW_DIR
from dogbreeds.split import SPLITS_CSV

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Eval images are resized a bit larger than the crop, then centre-cropped (the usual 256 -> 224).
RESIZE_RATIO = 1.14


def read_split(split: str, splits_csv: Path) -> list[dict]:
    """Return the rows of splits.csv that belong to one split."""
    rows = []
    with open(splits_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                rows.append(row)
    return rows


class DogBreedDataset(Dataset):
    """Images of one split ("train", "val" or "test"), labelled by position in CLASS_NAMES."""

    def __init__(
        self,
        split: str,
        transform,
        splits_csv: Path = SPLITS_CSV,
        raw_dir: Path = RAW_DIR,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.transform = transform
        self.raw_dir = Path(raw_dir)
        self.images = []
        self.labels = []
        for row in read_split(split, splits_csv):
            self.images.append(row["image"])
            self.labels.append(CLASS_NAMES.index(row["breed"]))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.raw_dir / self.images[index]) as img:
            image = img.convert("RGB")
        return self.transform(image), self.labels[index]


def build_transform(image_size: int, augment: bool, normalize: str):
    """Build the image transform.

    augment=True is for training only; val/test should always use augment=False.
    normalize is "imagenet" (for pretrained timm models) or "none" (pixels in [0, 1], like v1).
    """
    if augment:
        steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    else:
        steps = [
            transforms.Resize(round(image_size * RESIZE_RATIO)),
            transforms.CenterCrop(image_size),
        ]

    steps.append(transforms.ToTensor())  # also scales pixels to [0, 1]
    if normalize == "imagenet":
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif normalize != "none":
        raise ValueError(f"Unknown normalize mode: {normalize}")
    return transforms.Compose(steps)


def count_train_images_per_class(splits_csv: Path = SPLITS_CSV) -> list[int]:
    """Return the number of training images for each class, in CLASS_NAMES order."""
    counts = [0] * len(CLASS_NAMES)
    for row in read_split("train", splits_csv):
        counts[CLASS_NAMES.index(row["breed"])] += 1
    return counts
