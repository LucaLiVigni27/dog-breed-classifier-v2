"""Create the fixed train/val/test split, save it to data/splits.csv.

Stratified Method: each breed is split 70/15/15 on its own, so every split has the same
mix of breeds. Duplicate groups (found in duplicates.py) stay together, so no photo can
appear in two splits. A fixed seed makes the result the same on every machine.

Run with: python -m dogbreeds.split
"""

import csv
import random
from pathlib import Path

from dogbreeds.duplicates import MAX_DISTANCE, compute_hashes, find_pairs, group_duplicates
from dogbreeds.paths import DATA_DIR, RAW_DIR

SPLITS_CSV = DATA_DIR / "splits.csv"
SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15  # the remaining 15% goes to test


def make_units(images: list[str], groups: dict[str, int]) -> list[list[str]]:
    """Bundle images into units that must stay together.

    Each duplicate group becomes one unit; every other image is a unit on its own.
    """
    units = {}
    for image in images:
        # Images in the same duplicate group share one key; all others get their own.
        if image in groups:
            key = f"group_{groups[image]}"
        else:
            key = image
        units.setdefault(key, []).append(image)
    return list(units.values())


def split_breed(images: list[str], groups: dict[str, int], rng: random.Random) -> dict[str, str]:
    """Assign one breed's images to train/val/test, keeping duplicate groups together."""
    units = make_units(images, groups)
    rng.shuffle(units)
    train_target = round(len(images) * TRAIN_FRACTION)
    val_target = round(len(images) * VAL_FRACTION)

    assignment = {}
    train_count = 0
    val_count = 0
    for unit in units:
        # Fill train first, then val; whatever is left goes to test.
        if train_count < train_target:
            split = "train"
            train_count += len(unit)
        elif val_count < val_target:
            split = "val"
            val_count += len(unit)
        else:
            split = "test"
        for image in unit:
            assignment[image] = split
    return assignment


def list_breed_images(breed_dir: Path) -> list[str]:
    """Return 'breed/file.jpg' for every image in a breed folder, sorted."""
    images = []
    for path in sorted(breed_dir.glob("*.jpg")):
        images.append(f"{breed_dir.name}/{path.name}")
    return images


def print_counts(rows: list[dict]) -> None:
    """Print a table of images per breed and split."""
    print(f"\n  {'breed':<22} {'train':>5} {'val':>5} {'test':>5}")
    totals = {"train": 0, "val": 0, "test": 0}
    breeds = sorted({row["breed"] for row in rows})
    for breed in breeds:
        counts = {"train": 0, "val": 0, "test": 0}
        for row in rows:
            if row["breed"] == breed:
                counts[row["split"]] += 1
                totals[row["split"]] += 1
        print(f"  {breed:<22} {counts['train']:>5} {counts['val']:>5} {counts['test']:>5}")
    print(f"  {'total':<22} {totals['train']:>5} {totals['val']:>5} {totals['test']:>5}")


def main() -> None:
    if SPLITS_CSV.exists():
        # The split must never change by accident.
        raise SystemExit(f"{SPLITS_CSV} already exists. Delete it only on purpose.")

    print("Finding duplicate groups...")
    hashes = compute_hashes(RAW_DIR)
    groups = group_duplicates(find_pairs(hashes, MAX_DISTANCE))

    rng = random.Random(SEED)
    rows = []
    for breed_dir in sorted(RAW_DIR.iterdir()):
        if not breed_dir.is_dir():
            continue
        images = list_breed_images(breed_dir)
        assignment = split_breed(images, groups, rng)
        for image in images:
            row = {
                "image": image,
                "breed": breed_dir.name,
                "split": assignment[image],
                "group": groups.get(image, ""),
            }
            rows.append(row)

    with open(SPLITS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "breed", "split", "group"])
        writer.writeheader()
        writer.writerows(rows)

    print_counts(rows)
    print(f"\nWrote {len(rows)} rows to {SPLITS_CSV}")


if __name__ == "__main__":
    main()
