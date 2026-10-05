"""Build the classifiers: the reconstructed v1 CNN or any timm model."""

from pathlib import Path

import timm
import torch
from torch import nn

from dogbreeds.breeds import CLASS_NAMES

# The v1 CNN flattens its last feature map, so it only accepts this input size (as in v1).
V1_IMAGE_SIZE = 200


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """3x3 conv -> ReLU -> 2x2 max-pool, the Keras-style block v1 used."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class V1BaselineCNN(nn.Module):
    """Small from-scratch CNN reconstructing v1 (see legacy/v1/NOTES.md)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 128),
        )
        # 200 -> conv 198 -> pool 99 -> 97 -> 48 -> 46 -> 23 -> 21 -> 10
        flat_size = 128 * 10 * 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Return "v1_baseline" or the timm model called `name`, with a fresh num_classes head."""
    if name == "v1_baseline":
        return V1BaselineCNN(num_classes)
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def get_head(model: nn.Module) -> nn.Module:
    """Return the classifier head of a timm model."""
    # getattr + isinstance: timm models have get_classifier(), plain nn.Modules don't.
    get_classifier = getattr(model, "get_classifier", None)
    head = get_classifier() if callable(get_classifier) else None
    if not isinstance(head, nn.Module):
        raise TypeError(f"{type(model).__name__} has no get_classifier(); use a timm model")
    return head


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze every parameter except the classifier head (timm models only)."""
    for param in model.parameters():
        param.requires_grad = trainable
    for param in get_head(model).parameters():
        param.requires_grad = True


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild the model from a best.pt file and return it (eval mode) with its training config."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("Checkpoint was trained on different classes than CLASS_NAMES")
    config = checkpoint["config"]
    # pretrained=False: the trained weights come from the checkpoint, nothing to download.
    model = build_model(config["model"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), config
