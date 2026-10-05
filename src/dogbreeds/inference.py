"""
Classify a single photo, for the demo app.

Run with: python -m dogbreeds.inference --checkpoint <best.pt> --image <photo>
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps

from dogbreeds.breeds import CLASS_NAMES, DISPLAY_NAMES
from dogbreeds.dataset import build_transform
from dogbreeds.model import load_checkpoint
from dogbreeds.train import get_device

# Chosen in docs/decisions.md: below this confidence the model should say "not sure"
NOT_SURE_THRESHOLD = 0.8


@dataclass
class Classifier:
    """Everything needed to classify a photo: the model, its eval transform and device."""

    model: torch.nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    device: torch.device


def load_classifier(checkpoint_path: Path, device: torch.device | None = None) -> Classifier:
    """Load a best.pt file in eval mode, with the eval transform from its training config."""
    if device is None:
        device = get_device()
    model, config = load_checkpoint(Path(checkpoint_path), device)
    transform = build_transform(config["image_size"], False, config["normalize"])
    return Classifier(model, transform, device)


@torch.no_grad()
def predict_image(
    classifier: Classifier,
    image: Image.Image,
    top_k: int = 3,
    threshold: float = NOT_SURE_THRESHOLD,
) -> dict:
    """Classify one PIL image and return the top_k breeds and all 12 probabilities."""
    # Phone photos are often stored sideways with an EXIF rotation tag; RGB handles PNGs
    # with transparency and grayscale images.
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = classifier.transform(image).unsqueeze(0).to(classifier.device)
    logits = classifier.model(tensor)
    probs = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()

    ranked = sorted(range(len(CLASS_NAMES)), key=lambda i: probs[i], reverse=True)
    top = []
    for i in ranked[:top_k]:
        top.append((CLASS_NAMES[i], DISPLAY_NAMES[CLASS_NAMES[i]], probs[i]))

    probabilities = {}
    for name, prob in zip(CLASS_NAMES, probs, strict=True):
        probabilities[DISPLAY_NAMES[name]] = prob

    breed, display_name, confidence = top[0]
    return {
        "top": top,
        "breed": breed,
        "display_name": display_name,
        "confidence": confidence,
        "not_sure": confidence < threshold,
        "probabilities": probabilities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one dog photo.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to best.pt")
    parser.add_argument("--image", type=Path, required=True, help="photo to classify")
    args = parser.parse_args()

    classifier = load_classifier(args.checkpoint)
    with Image.open(args.image) as image:
        result = predict_image(classifier, image)

    for _, display_name, prob in result["top"]:
        print(f"{display_name:20} {prob:6.1%}")
    if result["not_sure"]:
        print("Not sure: probably not one of the 12 breeds")
    else:
        print(f"Breed: {result['display_name']}")


if __name__ == "__main__":
    main()
