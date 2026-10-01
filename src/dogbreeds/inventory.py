"""
Inventory of the raw dataset: which files exist and whether they are usable.

Run with: python -m dogbreeds.inventory
"""

import csv
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image

from dogbreeds.paths import OUTPUTS_DIR, RAW_DIR

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
CSV_PATH = OUTPUTS_DIR / "inventory.csv"
CSV_COLUMNS = ["breed", "file", "status", "format", "mode", "width", "height"]

SMALL_SIDE = 224


def read_image_info(path: Path) -> dict:
    """
    Decode an image and return its format, colour mode and size.

    Returns an empty dict if the file can't be decoded.
    """
    try:
        with Image.open(path) as img:
            img.load()
            return {
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
            }
    except OSError:
        return {}


def scan_breed(breed_dir: Path) -> list[dict]:
    """Return one row per file inside a breed folder."""
    rows = []
    for path in sorted(breed_dir.iterdir()):
        row = {"breed": breed_dir.name, "file": path.name}
        is_hidden = path.name.startswith(".")
        if is_hidden or path.suffix.lower() not in IMAGE_SUFFIXES:
            row["status"] = "not_image"
        else:
            info = read_image_info(path)
            if info:
                row["status"] = "ok"
                row.update(info)
            else:
                row["status"] = "unreadable"
        rows.append(row)
    return rows


def scan_dataset(raw_dir: Path) -> list[dict]:
    """Return one row per file in the dataset, including stray top-level files."""
    rows = []
    for path in sorted(raw_dir.iterdir()):
        if path.is_dir():
            rows.extend(scan_breed(path))
        else:
            rows.append({"breed": "", "file": path.name, "status": "not_image"})
    return rows


def print_size_stats(label: str, values: list[int]) -> None:
    """Print min / median / max of a list of pixel sizes."""
    median = int(statistics.median(values))
    print(f"{label}: min {min(values)}, median {median}, max {max(values)}")


def print_summary(rows: list[dict]) -> None:
    """Print counts per breed, formats, sizes and any problem files."""
    breed_counts = Counter()
    formats = Counter()
    modes = Counter()
    widths = []
    heights = []
    small_count = 0
    problems = []

    for row in rows:
        if row["status"] != "ok":
            problems.append(row)
            continue
        breed_counts[row["breed"]] += 1
        formats[row["format"]] += 1
        modes[row["mode"]] += 1
        widths.append(row["width"])
        heights.append(row["height"])
        if min(row["width"], row["height"]) < SMALL_SIDE:
            small_count += 1

    print(f"Usable images: {len(widths)}\n")
    print("Images per breed:")
    for breed, count in sorted(breed_counts.items()):
        print(f"  {breed:<22} {count:>4}")

    print(f"\nFormats: {dict(formats)}")
    print(f"Colour modes: {dict(modes)}")
    print_size_stats("Width", widths)
    print_size_stats("Height", heights)
    print(f"Images with shortest side < {SMALL_SIDE}px: {small_count}")

    print(f"\nProblem files: {len(problems)}")
    for row in problems:
        print(f"  [{row['status']}] {row['breed']}/{row['file']}")


def write_csv(rows: list[dict], csv_path: Path) -> None:
    """Write the rows to a CSV file (missing fields are left blank)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = scan_dataset(RAW_DIR)
    print_summary(rows)
    write_csv(rows, CSV_PATH)
    print(f"\nWrote {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
