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

from preprocessing.transform import create_dataset
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
    parser.add_argument("--model", choices=["cnn", "alexnet"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--data-dir", type=Path, default=project_folder / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("Количество эпох и размер батча должны быть положительными")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    start_epoch = 1
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "optimizer_state_dict" not in checkpoint:
            parser.error("Старый checkpoint содержит только веса. Его можно оценить, но возобновление требует нового checkpoint с состоянием оптимизатора.")
        if args.model is not None and args.model != checkpoint["model_name"]:
            parser.error("Модель в --model отличается от модели в checkpoint")
        args.model = checkpoint["model_name"]
        if args.batch_size is None:
            args.batch_size = checkpoint["batch_size"]
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        if start_epoch > args.epochs:
            parser.error("--epochs должно быть больше числа уже завершенных эпох")

    args.model = args.model or "cnn"
    args.batch_size = args.batch_size or 32
    if args.output_dir:
        output_folder = args.output_dir.resolve()
    elif args.resume:
        output_folder = args.resume.resolve().parent
    else:
        output_folder = project_folder / "artifacts" / args.model
    same_run = args.resume is not None and output_folder == args.resume.resolve().parent
    if output_folder.exists() and not same_run:
        raise FileExistsError(f"Папка уже существует, укажите новую --output-dir: {output_folder}")

    dataset = create_dataset(args.data_dir / "train", training=True)
    if checkpoint and dataset.class_to_idx != checkpoint["class_to_idx"]:
        parser.error("Классы датасета отличаются от классов в checkpoint")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = create_model(args.model, pretrained=checkpoint is None).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if args.model == "alexnet":
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=0.0001, weight_decay=0.0001,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and len(checkpoint["cuda_rng_states"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])

    output_folder.mkdir(parents=True, exist_ok=same_run)
    print(f"Device: {device}; images: {len(dataset)}; model: {args.model}", flush=True)
    print(f"Results: {output_folder}", flush=True)

    with (output_folder / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy"])
        writer.writerows(history)
        file.flush()

        for epoch in range(start_epoch, args.epochs + 1):
            loss, accuracy = train_one_epoch(model, loader, optimizer, device)
            history.append([epoch, loss, accuracy])

            checkpoint = {
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": dataset.class_to_idx,
                "epoch": epoch,
                "batch_size": args.batch_size,
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            }
            temporary_file = output_folder / "last.tmp"
            torch.save(checkpoint, temporary_file)
            temporary_file.replace(output_folder / "last.pt")

            writer.writerow([epoch, loss, accuracy])
            file.flush()
            print(f"Epoch {epoch}/{args.epochs}: loss={loss:.4f}, accuracy={accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
