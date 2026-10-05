"""
Evaluate a trained run's best checkpoint on the val (default) or test split.

Run with: python -m dogbreeds.evaluate --run outputs/runs/<run_name> [--split val|test]
Output (in the run folder): predictions_<split>.csv, metrics_<split>.json,
confusion_<split>.csv and confusion_<split>.png
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from dogbreeds.breeds import CLASS_NAMES
from dogbreeds.dataset import DogBreedDataset, build_transform
from dogbreeds.model import load_checkpoint
from dogbreeds.paths import RAW_DIR
from dogbreeds.split import SPLITS_CSV
from dogbreeds.train import get_device


def load_run(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model from run_dir/best.pt and return it with the config it was trained with."""
    return load_checkpoint(run_dir / "best.pt", device)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    """Return softmax probabilities, one row per image, in loader order."""
    all_probs = []
    for images, _ in loader:
        logits = model(images.to(device))
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def write_predictions(path: Path, images: list[str], labels: list[int], probs: np.ndarray):
    """One row per image: true and predicted breed, confidence and every class probability."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        prob_columns = [f"prob_{name}" for name in CLASS_NAMES]
        writer.writerow(["image", "true_breed", "predicted_breed", "confidence"] + prob_columns)
        for image, label, row in zip(images, labels, probs, strict=True):
            predicted = int(row.argmax())
            probs_rounded = [round(float(p), 4) for p in row]
            writer.writerow(
                [image, CLASS_NAMES[label], CLASS_NAMES[predicted], probs_rounded[predicted]]
                + probs_rounded
            )


def safe_divide(numerator: float, denominator: float) -> float:
    """Return 0 instead of failing when a class was never predicted or never present."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(matrix: np.ndarray) -> dict:
    """Accuracy, macro F1 and per-class precision/recall/F1 from the confusion matrix.

    Rows are true classes, columns predicted. Matches scikit-learn with zero_division=0.
    """
    per_class = {}
    f1_scores = []
    for i, name in enumerate(CLASS_NAMES):
        correct = int(matrix[i, i])
        precision = safe_divide(correct, int(matrix[:, i].sum()))
        recall = safe_divide(correct, int(matrix[i, :].sum()))
        f1 = safe_divide(2 * precision * recall, precision + recall)
        f1_scores.append(f1)
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(matrix[i, :].sum()),
        }
    num_images = int(matrix.sum())
    return {
        "accuracy": round(safe_divide(int(np.trace(matrix)), num_images), 4),
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "num_images": num_images,
        "per_class": per_class,
    }


def write_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    """Rows = true breed, columns = predicted breed."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ predicted"] + CLASS_NAMES)
        for name, row in zip(CLASS_NAMES, matrix, strict=True):
            writer.writerow([name] + [int(count) for count in row])


def save_confusion_png(path: Path, matrix: np.ndarray, title: str) -> None:
    """Heatmap of the confusion matrix with the count written in each cell."""
    fig = Figure(figsize=(9, 8))
    ax = fig.subplots()
    ax.imshow(matrix, cmap="Blues")
    ticks = range(len(CLASS_NAMES))
    ax.set_xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(ticks, CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    # Light text on dark cells, dark text on light cells.
    threshold = matrix.max() / 2
    for i in ticks:
        for j in ticks:
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def top_confusions(matrix: np.ndarray, count: int) -> list[tuple[str, str, int]]:
    """Return the most common (true, predicted, count) mistakes, largest first."""
    mistakes = []
    for i, true_name in enumerate(CLASS_NAMES):
        for j, predicted_name in enumerate(CLASS_NAMES):
            if i != j and matrix[i, j] > 0:
                mistakes.append((true_name, predicted_name, int(matrix[i, j])))
    mistakes.sort(key=lambda mistake: mistake[2], reverse=True)
    return mistakes[:count]


def evaluate(
    run_dir: Path,
    split: str = "val",
    splits_csv: Path = SPLITS_CSV,
    raw_dir: Path = RAW_DIR,
) -> dict:
    """Evaluate run_dir/best.pt on one split, write the output files and return the metrics."""
    run_dir = Path(run_dir)
    if split == "test":
        print("Evaluating on TEST. These results are final: do not tune anything after this.")

    device = get_device()
    model, config = load_run(run_dir, device)
    transform = build_transform(config["image_size"], False, config["normalize"])
    dataset = DogBreedDataset(split, transform, splits_csv, raw_dir)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    probs = predict(model, loader, device)
    predicted = probs.argmax(axis=1).tolist()
    labels = dataset.labels

    write_predictions(run_dir / f"predictions_{split}.csv", dataset.images, labels, probs)
    matrix = confusion_matrix(labels, predicted, labels=list(range(len(CLASS_NAMES))))
    metrics = compute_metrics(matrix)
    with open(run_dir / f"metrics_{split}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    write_confusion_csv(run_dir / f"confusion_{split}.csv", matrix)
    save_confusion_png(run_dir / f"confusion_{split}.png", matrix, f"{run_dir.name} ({split})")

    print(f"{split}: {metrics['num_images']} images on {device}")
    print(f"Accuracy {metrics['accuracy']:.4f}   Macro F1 {metrics['macro_f1']:.4f}")
    print("Most common confusions (true -> predicted):")
    for true_name, predicted_name, count in top_confusions(matrix, 5):
        print(f"  {true_name} -> {predicted_name}: {count}")
    print(f"Wrote results to {run_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained run.")
    parser.add_argument("--run", type=Path, required=True, help="run folder with best.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    evaluate(args.run, args.split)


if __name__ == "__main__":
    main()
