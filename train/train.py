import argparse
import sys
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.data import create_dataset
from models.models import create_model


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0

    for batch_number, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (predictions.argmax(dim=1) == labels).sum().item()
        if batch_number % 50 == 0:
            print(f"  Batch {batch_number}/{len(loader)}", flush=True)

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    project_folder = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "alexnet"], default="cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=Path, default=project_folder / "data")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Количество эпох и размер батча должны быть положительными")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_folder = args.output_dir or project_folder / "artifacts" / args.model
    if output_folder.exists():
        raise FileExistsError(f"Папка уже существует, укажите новую --output-dir: {output_folder}")

    dataset = create_dataset(args.data_dir / "train", training=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = create_model(args.model, pretrained=True).to(device)

    if args.model == "alexnet":
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=0.0001, weight_decay=0.0001,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    output_folder.mkdir(parents=True)
    print(f"Device: {device}; images: {len(dataset)}; model: {args.model}", flush=True)
    print(f"Results: {output_folder}", flush=True)

    with (output_folder / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy"])

        for epoch in range(1, args.epochs + 1):
            loss, accuracy = train_one_epoch(model, loader, optimizer, device)

            checkpoint = {
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "class_to_idx": dataset.class_to_idx,
                "epoch": epoch,
            }
            temporary_file = output_folder / "last.tmp"
            torch.save(checkpoint, temporary_file)
            temporary_file.replace(output_folder / "last.pt")

            writer.writerow([epoch, loss, accuracy])
            file.flush()
            print(f"Epoch {epoch}/{args.epochs}: loss={loss:.4f}, accuracy={accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
