"""
Run a trained model on any folder of images, (other Stanford breeds as unknown dogs).

An image's true breed is its parent folder name if that is one of our 12 breeds, otherwise it
is left empty (an "unknown" dog). The confidence summary helps choose a threshold for
"not one of our breeds".

Run with: python -m dogbreeds.predict --run <run folder> --images <folder> --out <csv>
          [--per-folder N] [--skip-our-breeds]
"""

import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps

from dogbreeds.breeds import BREED_WN_IDS, CLASS_NAMES
from dogbreeds.dataset import build_transform
from dogbreeds.evaluate import load_run
from dogbreeds.train import get_device

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SEED = 42
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def find_images(images_dir: Path) -> list[Path]:
    """Return every image under images_dir (recursively), relative to it, skipping hidden ones."""
    found = []
    for path in sorted(images_dir.rglob("*")):
        relative = path.relative_to(images_dir)
        # Skips hidden files and anything inside hidden folders (e.g. .ipynb_checkpoints).
        is_hidden = any(part.startswith(".") for part in relative.parts)
        if path.is_file() and not is_hidden and path.suffix.lower() in IMAGE_SUFFIXES:
            found.append(relative)
    return found


def is_our_breed_folder(folder_name: str) -> bool:
    """True for Stanford folders of our 12 breeds, e.g. 'n02110958-pug'."""
    for wnid in BREED_WN_IDS.values():
        if folder_name.startswith(wnid):
            return True
    return False


def select_images(images: list[Path], per_folder: int | None, skip_our_breeds: bool) -> list[Path]:
    """Apply --skip-our-breeds and --per-folder, keeping the result sorted."""
    by_folder = {}
    for image in images:
        if skip_our_breeds and is_our_breed_folder(image.parent.name):
            continue
        by_folder.setdefault(image.parent, []).append(image)

    selected = []
    for folder_images in by_folder.values():
        if per_folder is not None and len(folder_images) > per_folder:
            # A fresh seeded generator per folder, so each folder's pick is repeatable and
            # doesn't change when other folders are added or removed.
            folder_images = random.Random(SEED).sample(folder_images, per_folder)
        selected.extend(folder_images)
    return sorted(selected)


def load_image(path: Path, transform) -> torch.Tensor | None:
    """Open and transform one image, or return None (with a warning) if it can't be read."""
    try:
        with Image.open(path) as img:
            upright = ImageOps.exif_transpose(img)
            return transform(upright.convert("RGB"))
    except OSError as error:
        print(f"Warning: skipping {path}: {error}")
        return None


def make_row(relative: Path, probs: list[float]) -> dict:
    """One CSV row: path, folder, true and predicted breed, confidence and all probabilities."""
    folder_name = relative.parent.name
    true_breed = folder_name if folder_name in CLASS_NAMES else ""
    predicted = probs.index(max(probs))
    row = {
        "path": relative.as_posix(),
        "folder": relative.parent.as_posix(),
        "true_breed": true_breed,
        "predicted_breed": CLASS_NAMES[predicted],
        "confidence": round(probs[predicted], 4),
    }
    for name, prob in zip(CLASS_NAMES, probs, strict=True):
        row[f"prob_{name}"] = round(prob, 4)
    return row


@torch.no_grad()
def predict_batch(model, batch: list[torch.Tensor], paths: list[Path], device) -> list[dict]:
    """Predict a batch of image tensors and return one CSV row per image."""
    logits = model(torch.stack(batch).to(device))
    all_probs = torch.softmax(logits.float(), dim=1).cpu().tolist()
    rows = []
    for path, probs in zip(paths, all_probs, strict=True):
        rows.append(make_row(path, probs))
    return rows


def write_rows(out_csv: Path, rows: list[dict]) -> None:
    columns = ["path", "folder", "true_breed", "predicted_breed", "confidence"]
    columns += [f"prob_{name}" for name in CLASS_NAMES]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def print_summary(rows: list[dict]) -> None:
    """Accuracy on labelled images, mean confidence per group and low-confidence fractions."""
    labelled = [row for row in rows if row["true_breed"]]
    unlabelled = [row for row in rows if not row["true_breed"]]
    print(f"Images predicted: {len(rows)} ({len(labelled)} labelled, {len(unlabelled)} unlabelled)")
    if not rows:
        return

    if labelled:
        correct = 0
        for row in labelled:
            if row["predicted_breed"] == row["true_breed"]:
                correct += 1
        print(f"Accuracy on {len(labelled)} labelled images: {correct / len(labelled):.4f}")
        labelled_confidence = mean([row["confidence"] for row in labelled])
        print(f"Mean confidence, labelled:   {labelled_confidence:.4f}")
    if unlabelled:
        unlabelled_confidence = mean([row["confidence"] for row in unlabelled])
        print(f"Mean confidence, unlabelled: {unlabelled_confidence:.4f}")

    print("Fraction of images with confidence below:")
    for threshold in THRESHOLDS:
        below = 0
        for row in rows:
            if row["confidence"] < threshold:
                below += 1
        print(f"  {threshold:.1f}: {below / len(rows):.3f}")


def predict_folder(
    run_dir: Path,
    images_dir: Path,
    out_csv: Path,
    per_folder: int | None = None,
    skip_our_breeds: bool = False,
) -> list[dict]:
    """Predict every selected image under images_dir, write out_csv and return the rows."""
    images_dir = Path(images_dir)
    device = get_device()
    model, config = load_run(Path(run_dir), device)
    transform = build_transform(config["image_size"], False, config["normalize"])

    images = select_images(find_images(images_dir), per_folder, skip_our_breeds)
    rows = []
    batch = []
    batch_paths = []
    for relative in images:
        tensor = load_image(images_dir / relative, transform)
        if tensor is None:
            continue
        batch.append(tensor)
        batch_paths.append(relative)
        # Images are loaded batch by batch so large folders don't fill memory.
        if len(batch) == config["batch_size"]:
            rows.extend(predict_batch(model, batch, batch_paths, device))
            batch = []
            batch_paths = []
    if batch:
        rows.extend(predict_batch(model, batch, batch_paths, device))

    write_rows(Path(out_csv), rows)
    print_summary(rows)
    print(f"Wrote {out_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict breeds for a folder of images.")
    parser.add_argument("--run", type=Path, required=True, help="run folder with best.pt")
    parser.add_argument("--images", type=Path, required=True, help="folder to search for images")
    parser.add_argument("--out", type=Path, required=True, help="CSV file to write")
    parser.add_argument("--per-folder", type=int, default=None, help="max images per folder")
    parser.add_argument(
        "--skip-our-breeds",
        action="store_true",
        help="skip Stanford folders of our 12 breeds (to test on unknown dogs only)",
    )
    args = parser.parse_args()
    predict_folder(args.run, args.images, args.out, args.per_folder, args.skip_our_breeds)


if __name__ == "__main__":
    main()
