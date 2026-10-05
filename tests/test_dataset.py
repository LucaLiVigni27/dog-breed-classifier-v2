import torch
from conftest import FAKE_SPLIT_SIZES
from torchvision.transforms.functional import to_pil_image

from dogbreeds.breeds import CLASS_NAMES
from dogbreeds.dataset import DogBreedDataset, build_transform, count_train_images_per_class


def test_length_per_split(fake_data):
    splits_csv, raw_dir = fake_data
    transform = build_transform(32, False, "none")
    for split, count in FAKE_SPLIT_SIZES.items():
        dataset = DogBreedDataset(split, transform, splits_csv, raw_dir)
        assert len(dataset) == count * len(CLASS_NAMES)


def test_label_is_position_in_class_names(fake_data):
    splits_csv, raw_dir = fake_data
    dataset = DogBreedDataset("val", build_transform(32, False, "none"), splits_csv, raw_dir)
    for image, label in zip(dataset.images, dataset.labels, strict=True):
        assert CLASS_NAMES[label] == image.split("/")[0]
    _, label = dataset[0]
    assert label == dataset.labels[0]


def test_output_shape(fake_data):
    splits_csv, raw_dir = fake_data
    for augment in (True, False):
        transform = build_transform(32, augment, "imagenet")
        dataset = DogBreedDataset("train", transform, splits_csv, raw_dir)
        image, _ = dataset[0]
        assert image.shape == (3, 32, 32)
        assert image.dtype == torch.float32


def test_normalize_none_keeps_pixels_in_0_1(fake_data):
    splits_csv, raw_dir = fake_data
    dataset = DogBreedDataset("train", build_transform(32, False, "none"), splits_csv, raw_dir)
    for index in range(len(dataset)):
        image, _ = dataset[index]
        assert image.min() >= 0.0
        assert image.max() <= 1.0


def test_normalize_imagenet_shifts_pixels(fake_data):
    splits_csv, raw_dir = fake_data
    plain = DogBreedDataset("train", build_transform(32, False, "none"), splits_csv, raw_dir)
    normed = DogBreedDataset("train", build_transform(32, False, "imagenet"), splits_csv, raw_dir)
    # Black pixels (the first breed's first image has red = 0) become negative.
    assert plain[0][0][0].min() == 0.0
    assert normed[0][0][0].max() < 0.0


def test_eval_transform_is_deterministic_and_train_is_random():
    # A gradient image so random crops and flips actually change the pixels.
    gradient = torch.linspace(0, 1, 64).repeat(3, 64, 1)
    image = to_pil_image(gradient)
    eval_transform = build_transform(32, False, "none")
    assert torch.equal(eval_transform(image), eval_transform(image))

    train_transform = build_transform(32, True, "none")
    torch.manual_seed(0)
    outputs = [train_transform(image) for _ in range(5)]
    assert any(not torch.equal(outputs[0], other) for other in outputs[1:])


def test_count_train_images_per_class(fake_data):
    splits_csv, _ = fake_data
    assert count_train_images_per_class(splits_csv) == [FAKE_SPLIT_SIZES["train"]] * 12
