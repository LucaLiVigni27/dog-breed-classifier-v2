"""
Build data/raw/ from the original Stanford Dogs download.

Copies every Stanford Dogs image of v1 12 breeds into data/raw/<breed>/.
Images aren't on Git, use this to recreate the raw dataset.

Needs Stanford Dogs extracted to data/stanford_dogs/Images/.
Run with: python -m dogbreeds.build_dataset
"""

import shutil
from pathlib import Path

from dogbreeds.breeds import BREED_WN_IDS
from dogbreeds.paths import DATA_DIR, RAW_DIR

STANFORD_IMAGES_DIR = DATA_DIR / "stanford_dogs" / "Images"


def find_breed_folder(stanford_dir: Path, wnid: str) -> Path:
    """Return the Stanford folder for a WordNet ID."""
    matches = list(stanford_dir.glob(f"{wnid}-*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one folder for {wnid} in {stanford_dir}")
    return matches[0]


def copy_breed(source_dir: Path, target_dir: Path) -> int:
    """Copy one breed's images and return how many were copied."""
    target_dir.mkdir(parents=True)
    copied = 0
    for path in sorted(source_dir.glob("*.jpg")):
        shutil.copy2(path, target_dir / path.name)
        copied += 1
    return copied


def main() -> None:
    if RAW_DIR.exists():
        raise SystemExit(f"{RAW_DIR} already exists. Move or delete it first.")

    total = 0
    for breed, wnid in BREED_WN_IDS.items():
        source_dir = find_breed_folder(STANFORD_IMAGES_DIR, wnid)
        count = copy_breed(source_dir, RAW_DIR / breed)
        print(f"  {breed:<22} {count:>4}")
        total += count
    print(f"\nCopied {total} images to {RAW_DIR}")


if __name__ == "__main__":
    main()
