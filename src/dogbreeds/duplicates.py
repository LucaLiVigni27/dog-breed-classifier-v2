"""Find exact and near-duplicate images with perceptual hashing.

Copies must never end up in different splits because a test image that also sits in the
training set measures memory, not generalization. This file finds the copies and the
split script keeps each group of copies in the same split.

Run with: python -m dogbreeds.duplicates
Output: outputs/duplicate_pairs.csv and outputs/duplicate_pairs.jpg (for review)
"""

import csv
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageFont

from dogbreeds.paths import OUTPUTS_DIR, RAW_DIR

# 0 means identical; up to about 8 catches resized or re-saved copies.
MAX_DISTANCE = 8

PAIRS_CSV = OUTPUTS_DIR / "duplicate_pairs.csv"
PAIRS_IMAGE = OUTPUTS_DIR / "duplicate_pairs.jpg"
THUMB_SIZE = 160
MAX_SHEET_ROWS = 300


def compute_hashes(raw_dir: Path) -> dict[str, imagehash.ImageHash]:
    """Return a perceptual hash for every image, keyed by 'breed/file.jpg'."""
    hashes = {}
    for breed_dir in sorted(raw_dir.iterdir()):
        if not breed_dir.is_dir():
            continue
        for path in sorted(breed_dir.glob("*.jpg")):
            with Image.open(path) as img:
                hashes[f"{breed_dir.name}/{path.name}"] = imagehash.phash(img)
    return hashes


def find_pairs(
    hashes: dict[str, imagehash.ImageHash], max_distance: int
) -> list[tuple[str, str, int]]:
    """Return (image_a, image_b, distance) for every pair within max_distance bits."""
    keys = list(hashes)
    pairs = []
    # Compare every image with every later one.
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            distance = hashes[keys[i]] - hashes[keys[j]]
            if distance <= max_distance:
                pairs.append((keys[i], keys[j], int(distance)))
    return pairs


def group_duplicates(pairs: list[tuple[str, str, int]]) -> dict[str, int]:
    """Give every image that has a copy a group number; copies share the same number.

    Groups are chained: if A matches B and B matches C, all three share a group.
    """
    groups = {}
    next_group = 0
    for image_a, image_b, _ in pairs:
        if image_a not in groups and image_b not in groups:
            groups[image_a] = next_group
            groups[image_b] = next_group
            next_group += 1
        elif image_b not in groups:
            groups[image_b] = groups[image_a]
        elif image_a not in groups:
            groups[image_a] = groups[image_b]
        else:
            # Both already have a group: merge image_b's group into image_a's.
            old_group = groups[image_b]
            new_group = groups[image_a]
            for key in groups:
                if groups[key] == old_group:
                    groups[key] = new_group
    return groups


def load_thumbnail(path: Path) -> Image.Image:
    """Open an image as RGB and shrink it to fit THUMB_SIZE, keeping its shape."""
    with Image.open(path) as img:
        thumb = img.convert("RGB")
    thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
    return thumb


def make_pairs_sheet(pairs: list[tuple[str, str, int]], raw_dir: Path) -> Image.Image:
    """Draw each pair side by side with its distance and file names, one pair per row."""
    row_height = THUMB_SIZE + 10
    width = 2 * THUMB_SIZE + 420
    sheet = Image.new("RGB", (width, len(pairs) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)

    for index, (image_a, image_b, distance) in enumerate(pairs):
        y = index * row_height
        sheet.paste(load_thumbnail(raw_dir / image_a), (0, y))
        sheet.paste(load_thumbnail(raw_dir / image_b), (THUMB_SIZE, y))
        text = f"distance {distance}\n{image_a}\n{image_b}"
        draw.text((2 * THUMB_SIZE + 10, y + 10), text, fill="black", font=font)
    return sheet


def write_pairs_csv(pairs: list[tuple[str, str, int]], csv_path: Path) -> None:
    """Save the pairs, marking pairs whose two images are filed under different breeds."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_a", "image_b", "distance", "same_breed"])
        for image_a, image_b, distance in pairs:
            same_breed = image_a.split("/")[0] == image_b.split("/")[0]
            writer.writerow([image_a, image_b, distance, same_breed])


def main() -> None:
    print("Hashing images...")
    hashes = compute_hashes(RAW_DIR)
    pairs = find_pairs(hashes, MAX_DISTANCE)
    # Closest pairs first, so the most certain copies are at the top of the review sheet.
    pairs.sort(key=lambda pair: pair[2])
    groups = group_duplicates(pairs)

    cross_breed = 0
    for image_a, image_b, _ in pairs:
        if image_a.split("/")[0] != image_b.split("/")[0]:
            cross_breed += 1

    print(f"Images hashed: {len(hashes)}")
    print(f"Duplicate pairs (distance <= {MAX_DISTANCE}): {len(pairs)}")
    print(f"  of which across different breeds: {cross_breed}")
    print(f"Images in duplicate groups: {len(groups)} in {len(set(groups.values()))} groups")

    write_pairs_csv(pairs, PAIRS_CSV)
    print(f"\nWrote {PAIRS_CSV}")
    if pairs:
        if len(pairs) > MAX_SHEET_ROWS:
            print(f"Review sheet shows only the closest {MAX_SHEET_ROWS} pairs.")
        make_pairs_sheet(pairs[:MAX_SHEET_ROWS], RAW_DIR).save(PAIRS_IMAGE, quality=90)
        print(f"Wrote {PAIRS_IMAGE}")


if __name__ == "__main__":
    main()
