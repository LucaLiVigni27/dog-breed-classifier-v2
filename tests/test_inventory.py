from pathlib import Path

from PIL import Image

from dogbreeds.inventory import scan_dataset


def make_fake_dataset(root: Path) -> None:
    """Build a tiny dataset with one good, one broken and two junk files."""
    breed_dir = root / "Pug"
    breed_dir.mkdir()
    Image.new("RGB", (40, 30)).save(breed_dir / "good.jpg")
    (breed_dir / "broken.jpg").write_bytes(b"not really a jpeg")
    (breed_dir / ".DS_Store").write_bytes(b"")
    (root / ".DS_Store").write_bytes(b"")


def test_each_file_gets_the_right_status(tmp_path):
    make_fake_dataset(tmp_path)
    statuses = {}
    for row in scan_dataset(tmp_path):
        statuses[row["breed"] + "/" + row["file"]] = row["status"]

    assert statuses == {
        "/.DS_Store": "not_image",
        "Pug/.DS_Store": "not_image",
        "Pug/broken.jpg": "unreadable",
        "Pug/good.jpg": "ok",
    }


def test_good_image_details_are_recorded(tmp_path):
    make_fake_dataset(tmp_path)
    good_rows = []
    for row in scan_dataset(tmp_path):
        if row["status"] == "ok":
            good_rows.append(row)

    assert len(good_rows) == 1
    assert good_rows[0]["format"] == "JPEG"
    assert good_rows[0]["mode"] == "RGB"
    assert (good_rows[0]["width"], good_rows[0]["height"]) == (40, 30)
