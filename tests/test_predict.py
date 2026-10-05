import csv

import pytest
import torch
from PIL import Image

from dogbreeds.breeds import CLASS_NAMES
from dogbreeds.model import build_model
from dogbreeds.paths import PROJECT_ROOT
from dogbreeds.predict import predict_folder
from dogbreeds.train import load_config


@pytest.fixture
def fake_run(tmp_path):
    """A run folder with an untrained resnet18 best.pt, built from the smoke config."""
    config = load_config(PROJECT_ROOT / "configs" / "smoke.yaml")
    model = build_model(config["model"], len(CLASS_NAMES), pretrained=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = {"model_state": model.state_dict(), "class_names": CLASS_NAMES, "config": config}
    torch.save(checkpoint, run_dir / "best.pt")
    return run_dir


def save_image(path, size=(40, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 80, 40)).save(path)


@pytest.fixture
def image_tree(tmp_path):
    """Labelled, unlabelled, nested, Stanford-style, hidden, corrupt and non-image files."""
    root = tmp_path / "images"
    for i in range(4):
        save_image(root / "pug" / f"{i}.jpg")
    save_image(root / "pug" / "extra.png")
    save_image(root / "pug" / ".hidden.jpg")
    (root / "pug" / "broken.jpg").write_bytes(b"not an image")
    save_image(root / "n02110958-pug" / "a.jpg")  # one of our breeds, Stanford-style
    save_image(root / "n02085620-Chihuahua" / "b.jpeg")
    save_image(root / "street" / "night" / "c.jpg")  # nested, unknown dog
    (root / "notes.txt").write_text("ignore me")
    return root


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_predict_writes_one_row_per_readable_image(fake_run, image_tree, tmp_path, capsys):
    out_csv = tmp_path / "out" / "predictions.csv"
    predict_folder(fake_run, image_tree, out_csv)

    rows = read_csv(out_csv)
    expected_columns = ["path", "folder", "true_breed", "predicted_breed", "confidence"]
    expected_columns += [f"prob_{name}" for name in CLASS_NAMES]
    assert list(rows[0].keys()) == expected_columns

    paths = [row["path"] for row in rows]
    assert paths == [
        "n02085620-Chihuahua/b.jpeg",
        "n02110958-pug/a.jpg",
        "pug/0.jpg",
        "pug/1.jpg",
        "pug/2.jpg",
        "pug/3.jpg",
        "pug/extra.png",
        "street/night/c.jpg",
    ]
    assert "Warning: skipping" in capsys.readouterr().out  # broken.jpg

    for row in rows:
        expected_breed = "pug" if row["folder"] == "pug" else ""
        assert row["true_breed"] == expected_breed
        assert row["predicted_breed"] in CLASS_NAMES
    nested = rows[-1]
    assert nested["folder"] == "street/night"


def test_per_folder_limit_is_repeatable(fake_run, image_tree, tmp_path):
    first = predict_folder(fake_run, image_tree, tmp_path / "a.csv", per_folder=2)
    second = predict_folder(fake_run, image_tree, tmp_path / "b.csv", per_folder=2)

    pug_paths = [row["path"] for row in first if row["folder"] == "pug"]
    assert len(pug_paths) == 2
    assert [row["path"] for row in first] == [row["path"] for row in second]


def test_skip_our_breeds_drops_their_stanford_folders(fake_run, image_tree, tmp_path):
    rows = predict_folder(fake_run, image_tree, tmp_path / "out.csv", skip_our_breeds=True)
    folders = {row["folder"] for row in rows}
    assert "n02110958-pug" not in folders
    assert "n02085620-Chihuahua" in folders
