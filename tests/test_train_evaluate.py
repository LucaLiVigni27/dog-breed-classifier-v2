import json

import pytest

from dogbreeds.evaluate import evaluate
from dogbreeds.paths import PROJECT_ROOT
from dogbreeds.train import load_config, train


def make_test_config():
    """The real smoke config, with 2 epochs to cover freezing, unfreezing and class weights."""
    config = load_config(PROJECT_ROOT / "configs" / "smoke.yaml")
    config["run_name"] = "test_run"
    config["epochs"] = 2
    config["freeze_backbone_epochs"] = 1
    config["class_weighted_loss"] = True
    config["label_smoothing"] = 0.1
    return config


def test_train_then_evaluate_writes_all_files(fake_data, tmp_path):
    splits_csv, raw_dir = fake_data
    runs_dir = tmp_path / "runs"

    run_dir = train(make_test_config(), splits_csv, raw_dir, runs_dir)
    for name in ["config.yaml", "history.csv", "best.pt", "summary.json"]:
        assert (run_dir / name).exists(), name
    history_lines = (run_dir / "history.csv").read_text().strip().splitlines()
    assert len(history_lines) == 3  # header + 2 epochs
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["best_epoch"] in (1, 2)

    metrics = evaluate(run_dir, "val", splits_csv, raw_dir)
    for name in ["predictions_val.csv", "metrics_val.json", "confusion_val.csv"]:
        assert (run_dir / name).exists(), name
    assert (run_dir / "confusion_val.png").exists()
    assert metrics["num_images"] == 12
    assert len(metrics["per_class"]) == 12
    prediction_lines = (run_dir / "predictions_val.csv").read_text().strip().splitlines()
    assert len(prediction_lines) == 13  # header + 12 images
    assert not (run_dir / "metrics_test.json").exists()


def test_train_refuses_to_overwrite_a_run(fake_data, tmp_path):
    splits_csv, raw_dir = fake_data
    (tmp_path / "runs" / "test_run").mkdir(parents=True)
    with pytest.raises(SystemExit):
        train(make_test_config(), splits_csv, raw_dir, tmp_path / "runs")
