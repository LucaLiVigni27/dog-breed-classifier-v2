import random

import imagehash
from PIL import Image

from dogbreeds.duplicates import compute_hashes, find_pairs, group_duplicates


def make_random_image(seed: int) -> Image.Image:
    """A blocky random picture; different seeds give clearly different pictures."""
    rng = random.Random(seed)
    small = Image.new("L", (16, 16))
    small.putdata([rng.randint(0, 255) for _ in range(16 * 16)])
    return small.resize((200, 200)).convert("RGB")


def test_resized_copy_is_found_and_different_image_is_not(tmp_path):
    breed_dir = tmp_path / "pug"
    breed_dir.mkdir()
    original = make_random_image(seed=1)
    original.save(breed_dir / "original.jpg")
    original.resize((120, 120)).save(breed_dir / "smaller_copy.jpg")
    make_random_image(seed=2).save(breed_dir / "different.jpg")

    pairs = find_pairs(compute_hashes(tmp_path), max_distance=8)

    assert len(pairs) == 1
    assert {pairs[0][0], pairs[0][1]} == {"pug/original.jpg", "pug/smaller_copy.jpg"}


def test_find_pairs_respects_max_distance():
    hashes = {
        "a.jpg": imagehash.hex_to_hash("0000000000000000"),
        "b.jpg": imagehash.hex_to_hash("0000000000000003"),  # 2 bits from a
        "c.jpg": imagehash.hex_to_hash("ffffffffffffffff"),  # 64 bits from a
    }
    assert find_pairs(hashes, max_distance=2) == [("a.jpg", "b.jpg", 2)]


def test_chained_pairs_share_one_group():
    pairs = [("a", "b", 1), ("c", "d", 1), ("b", "c", 3)]
    groups = group_duplicates(pairs)
    assert groups["a"] == groups["b"] == groups["c"] == groups["d"]
