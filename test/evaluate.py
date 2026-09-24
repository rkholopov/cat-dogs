import argparse
import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.transform import create_dataset
from models.models import create_model


def evaluate(model, loader, device):
    model.eval()
    matrix = [[0, 0], [0, 0]]

    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[actual][predicted] += 1

    correct = matrix[0][0] + matrix[1][1]
    f1_scores = []
    for class_index in range(2):
        true_positive = matrix[class_index][class_index]
        false_positive = matrix[1 - class_index][class_index]
        false_negative = matrix[class_index][1 - class_index]
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        f1_scores.append(f1)

    return {
        "test_size": len(loader.dataset),
        "crops_per_image": 1,
        "accuracy": correct / len(loader.dataset),
        "macro_f1": sum(f1_scores) / 2,
        "classes": ["cats", "dogs"],
        "confusion_matrix": matrix,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Размер батча должен быть положительным")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = create_dataset(args.data_dir / "test")
    if dataset.class_to_idx != checkpoint["class_to_idx"]:
        raise ValueError("Классы тестовой выборки отличаются от классов при обучении")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    metrics = evaluate(model, loader, device)
    metrics["model"] = checkpoint["model_name"]
    metrics["epoch"] = checkpoint["epoch"]
    result_file = args.checkpoint.parent / "metrics.json"
    result_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Accuracy: {metrics['accuracy']:.4f}; macro F1: {metrics['macro_f1']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Results: {result_file}")


if __name__ == "__main__":
    main()
