import pytest
import torch
from PIL import Image

from dogbreeds.breeds import CLASS_NAMES, DISPLAY_NAMES
from dogbreeds.inference import load_classifier, predict_image
from dogbreeds.model import build_model
from dogbreeds.paths import PROJECT_ROOT
from dogbreeds.train import load_config

CPU = torch.device("cpu")


@pytest.fixture
def classifier(tmp_path):
    """A classifier from an untrained resnet18 best.pt, saved like train.py saves it."""
    config = load_config(PROJECT_ROOT / "configs" / "smoke.yaml")
    model = build_model(config["model"], len(CLASS_NAMES), pretrained=False)
    checkpoint = {"model_state": model.state_dict(), "class_names": CLASS_NAMES, "config": config}
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    return load_classifier(checkpoint_path, CPU)


def rgb_image(size=(60, 40)):
    return Image.new("RGB", size, (120, 80, 40))


def test_display_names_cover_every_class():
    assert sorted(DISPLAY_NAMES) == CLASS_NAMES


def test_classifier_is_in_eval_mode_on_requested_device(classifier):
    assert not classifier.model.training
    assert classifier.device == CPU


def test_result_has_sorted_top_k_and_all_probabilities(classifier):
    result = predict_image(classifier, rgb_image(), top_k=3)

    assert len(result["top"]) == 3
    top_probs = [prob for _, _, prob in result["top"]]
    assert top_probs == sorted(top_probs, reverse=True)
    for breed, display_name, _ in result["top"]:
        assert DISPLAY_NAMES[breed] == display_name

    assert (result["breed"], result["display_name"], result["confidence"]) == result["top"][0]
    assert sorted(result["probabilities"]) == sorted(DISPLAY_NAMES.values())
    for prob in result["probabilities"].values():
        assert 0.0 <= prob <= 1.0
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_not_sure_follows_threshold(classifier):
    assert not predict_image(classifier, rgb_image(), threshold=0.0)["not_sure"]
    assert predict_image(classifier, rgb_image(), threshold=1.01)["not_sure"]


def test_rgba_and_grayscale_images_work(classifier):
    rgba = Image.new("RGBA", (50, 50), (120, 80, 40, 128))
    gray = Image.new("L", (50, 50), 100)
    for image in (rgba, gray):
        assert len(predict_image(classifier, image)["probabilities"]) == len(CLASS_NAMES)


def test_exif_rotated_image_is_turned_upright(classifier, tmp_path):
    upright = Image.new("RGB", (80, 40), (255, 255, 255))
    upright.paste((0, 0, 0), (0, 0, 40, 40))  # left half black, so rotation is visible
    # Stored sideways with EXIF orientation 6 ("rotate 90 degrees clockwise to view"),
    # like many phone photos.
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / "phone.png"
    upright.transpose(Image.Transpose.ROTATE_90).save(path, exif=exif)

    # Record what reaches the transform: an untrained model's probabilities barely change
    # with rotation, so comparing outputs would not catch a missing exif_transpose.
    seen = []
    original_transform = classifier.transform

    def recording_transform(image):
        seen.append(image.copy())
        return original_transform(image)

    classifier.transform = recording_transform
    with Image.open(path) as image:
        predict_image(classifier, image)

    assert seen[0].size == (80, 40)
    assert seen[0].tobytes() == upright.tobytes()
