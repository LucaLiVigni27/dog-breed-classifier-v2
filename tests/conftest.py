import csv

import pytest
from PIL import Image

from dogbreeds.breeds import CLASS_NAMES

# Images per breed in each split of the fake dataset.
FAKE_SPLIT_SIZES = {"train": 2, "val": 1, "test": 1}


@pytest.fixture
def fake_data(tmp_path):
    """A tiny dataset in tmp_path: every breed, small JPEGs of uneven sizes.

    Returns (splits_csv, raw_dir).
    """
    raw_dir = tmp_path / "raw"
    rows = []
    for breed_index, breed in enumerate(CLASS_NAMES):
        (raw_dir / breed).mkdir(parents=True)
        image_number = 0
        for split, count in FAKE_SPLIT_SIZES.items():
            for _ in range(count):
                name = f"{breed}/{image_number}.jpg"
                color = (breed_index * 20, image_number * 60, 100)
                Image.new("RGB", (48 + image_number * 8, 40), color).save(raw_dir / name)
                rows.append({"image": name, "breed": breed, "split": split, "group": ""})
                image_number += 1

    splits_csv = tmp_path / "splits.csv"
    with open(splits_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "breed", "split", "group"])
        writer.writeheader()
        writer.writerows(rows)
    return splits_csv, raw_dir
