"""
Train a model on the train split and pick the best epoch on the val split.

The best epoch is the one with the lowest val loss, not the highest val accuracy: at ~99%
accuracy epochs differ by a few images, and val loss also rewards well-calibrated confidence.

The test split is never loaded here; use evaluate.py --split test once, at the end.

Run with: python -m dogbreeds.train --config configs/<name>.yaml
Output: outputs/runs/<run_name>/ (config.yaml, history.csv, best.pt, summary.json)
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from dogbreeds.breeds import CLASS_NAMES
from dogbreeds.dataset import DogBreedDataset, build_transform, count_train_images_per_class
from dogbreeds.model import build_model, set_backbone_trainable
from dogbreeds.paths import OUTPUTS_DIR, RAW_DIR
from dogbreeds.split import SPLITS_CSV

RUNS_DIR = OUTPUTS_DIR / "runs"

DEFAULT_CONFIG = {
    "run_name": None,  # required
    "model": None,  # required: "v1_baseline" or a timm model name
    "pretrained": True,
    "image_size": 224,
    "augment": True,
    "normalize": "imagenet",
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 32,
    "epochs": 10,
    "freeze_backbone_epochs": 0,
    "class_weighted_loss": False,
    "label_smoothing": 0.0,
    "early_stopping_patience": 0,
    "seed": 42,
    "num_workers": 2,
    "max_batches": None,
}

HISTORY_COLUMNS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"]


def load_config(path: Path) -> dict:
    """Read a YAML config, fill in defaults and check the values."""
    with open(path) as f:
        user_config = yaml.safe_load(f) or {}

    unknown = set(user_config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **user_config}

    for key in ("run_name", "model"):
        if not config[key]:
            raise ValueError(f"Config needs a '{key}'")
    if config["optimizer"] not in ("adamw", "rmsprop"):
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    if config["freeze_backbone_epochs"] > 0 and config["model"] == "v1_baseline":
        raise ValueError("v1_baseline has no pretrained backbone to freeze")
    # PyYAML reads "1e-3" (no dot) as a string, so make the float settings floats.
    for key in ("lr", "weight_decay", "label_smoothing"):
        config[key] = float(config[key])
    return config


def get_device() -> torch.device:
    """Prefer CUDA (Colab), then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA


def make_class_weights(splits_csv: Path) -> torch.Tensor:
    """Inverse class frequency of the train split, scaled so the weights average 1."""
    counts = count_train_images_per_class(splits_csv)
    if 0 in counts:
        raise ValueError("Every class needs at least one training image for class weights")
    weights = torch.tensor([1.0 / count for count in counts])
    return weights / weights.mean()


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create the optimizer over the parameters that are currently trainable."""
    params = [p for p in model.parameters() if p.requires_grad]
    if config["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])
    # alpha=0.9 matches the Keras RMSprop default (rho=0.9) that v1 would have used.
    return torch.optim.RMSprop(
        params, lr=config["lr"], alpha=0.9, weight_decay=config["weight_decay"]
    )


def cosine_lr(base_lr: float, epoch: int, total_epochs: int) -> float:
    """Cosine annealing from base_lr towards 0, one step per epoch (epoch starts at 1).

    Computed by hand instead of with a scheduler object, because the optimizer is rebuilt
    when the backbone is unfrozen and the schedule must carry on across that.
    """
    return base_lr * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / total_epochs))


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, max_batches):
    """Train for one pass over the loader; return (mean loss, accuracy)."""
    model.train()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    correct = 0
    seen = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate_loss_acc(model, loader, loss_fn, device, max_batches):
    """Return (mean loss, accuracy) over the loader without updating the model."""
    model.eval()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    correct = 0
    seen = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


def make_loaders(config: dict, splits_csv: Path, raw_dir: Path, device: torch.device):
    """Build the train and val loaders. The test split is deliberately not loaded."""
    train_transform = build_transform(config["image_size"], config["augment"], config["normalize"])
    eval_transform = build_transform(config["image_size"], False, config["normalize"])
    train_set = DogBreedDataset("train", train_transform, splits_csv, raw_dir)
    val_set = DogBreedDataset("val", eval_transform, splits_csv, raw_dir)

    pin_memory = device.type == "cuda"
    # A seeded generator makes the shuffle order repeatable.
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def append_history_row(history_path: Path, row: dict) -> None:
    """Add one epoch to history.csv, writing the header first if the file is new."""
    is_new = not history_path.exists()
    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def train(
    config: dict,
    splits_csv: Path = SPLITS_CSV,
    raw_dir: Path = RAW_DIR,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Run one training experiment and return its run folder."""
    run_dir = Path(runs_dir) / config["run_name"]
    if run_dir.exists():
        # Never overwrite a finished experiment.
        raise SystemExit(f"{run_dir} already exists. Pick a new run_name.")
    run_dir.mkdir(parents=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    set_seed(config["seed"])
    device = get_device()
    print(f"Run {config['run_name']} on {device}")

    train_loader, val_loader = make_loaders(config, splits_csv, raw_dir, device)
    model = build_model(config["model"], len(CLASS_NAMES), config["pretrained"]).to(device)

    class_weights = None
    if config["class_weighted_loss"]:
        class_weights = make_class_weights(splits_csv).to(device)
    train_loss_fn = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=config["label_smoothing"]
    )
    # Plain val loss (no weights or smoothing), so it compares across runs.
    val_loss_fn = nn.CrossEntropyLoss()

    freeze_epochs = config["freeze_backbone_epochs"]
    if freeze_epochs > 0:
        set_backbone_trainable(model, False)
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_val_loss = math.inf
    val_acc_at_best = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_run = 0
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            # Head is warmed up: train everything, with a fresh optimizer for the new params.
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, config)
            print("Backbone unfrozen")

        lr = cosine_lr(config["lr"], epoch, config["epochs"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        epochs_run = epoch
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, train_loss_fn, optimizer, scaler, device, config["max_batches"]
        )
        val_loss, val_acc = evaluate_loss_acc(
            model, val_loader, val_loss_fn, device, config["max_batches"]
        )
        seconds = time.time() - epoch_start

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            val_acc_at_best = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "class_names": CLASS_NAMES,
                "config": config,
            }
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        append_history_row(
            run_dir / "history.csv",
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "lr": lr,
                "seconds": round(seconds, 1),
            },
        )
        marker = "  *best" if improved else ""
        print(
            f"epoch {epoch:>3}/{config['epochs']}  "
            f"train loss {train_loss:.4f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}  "
            f"lr {lr:.2e}  {seconds:.0f}s{marker}"
        )

        patience = config["early_stopping_patience"]
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping: no new val loss minimum for {patience} epochs")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "val_acc_at_best": round(val_acc_at_best, 4),
        "epochs_run": epochs_run,
        "total_seconds": round(time.time() - start_time, 1),
        "device": device.type,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Best epoch {best_epoch}: val loss {best_val_loss:.4f}, val acc {val_acc_at_best:.3f}. "
        f"Saved to {run_dir}"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dog breed classifier.")
    parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
