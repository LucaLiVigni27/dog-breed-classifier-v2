from dogbreeds.build_dataset import copy_breed, find_breed_folder


def test_find_breed_folder_matches_wnid(tmp_path):
    (tmp_path / "n02110958-pug").mkdir()
    (tmp_path / "n02088238-basset").mkdir()
    assert find_breed_folder(tmp_path, "n02110958").name == "n02110958-pug"


def test_copy_breed_copies_every_jpg(tmp_path):
    source = tmp_path / "n02110958-pug"
    source.mkdir()
    (source / "n02110958_1.jpg").write_bytes(b"fake")
    (source / "n02110958_2.jpg").write_bytes(b"fake")

    target = tmp_path / "raw" / "pug"
    copied = copy_breed(source, target)

    assert copied == 2
    assert (target / "n02110958_1.jpg").exists()
    assert (target / "n02110958_2.jpg").exists()
