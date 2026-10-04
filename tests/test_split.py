import random

from dogbreeds.split import make_units, split_breed


def make_images(count: int) -> list[str]:
    images = []
    for i in range(count):
        images.append(f"pug/{i}.jpg")
    return images


def test_duplicate_group_becomes_one_unit():
    images = ["pug/a.jpg", "pug/b.jpg", "pug/c.jpg"]
    groups = {"pug/a.jpg": 0, "pug/c.jpg": 0}
    assert make_units(images, groups) == [["pug/a.jpg", "pug/c.jpg"], ["pug/b.jpg"]]


def test_split_sizes_are_70_15_15():
    images = make_images(100)
    assignment = split_breed(images, {}, random.Random(0))
    counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1
    assert counts == {"train": 70, "val": 15, "test": 15}


def test_duplicate_groups_stay_in_one_split():
    images = make_images(100)
    groups = {}
    for i in range(0, 40, 2):  # 20 pairs of copies: (0, 1), (2, 3), ...
        groups[f"pug/{i}.jpg"] = i
        groups[f"pug/{i + 1}.jpg"] = i
    assignment = split_breed(images, groups, random.Random(0))
    for i in range(0, 40, 2):
        assert assignment[f"pug/{i}.jpg"] == assignment[f"pug/{i + 1}.jpg"]


def test_same_seed_gives_same_split():
    images = make_images(50)
    first = split_breed(images, {}, random.Random(42))
    second = split_breed(images, {}, random.Random(42))
    assert first == second
