import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.predict import load_model, predict, prepare_image
from preprocessing.transform import load_rgb


def main():
    parser = argparse.ArgumentParser(description="Вероятности классов для новых изображений")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Изображение или папка с изображениями")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "predictions.csv")
    args = parser.parse_args()

    extensions = {".jpg", ".jpeg", ".png"}
    if args.input.is_file() and args.input.suffix.lower() in extensions:
        paths = [args.input]
    elif args.input.is_dir():
        paths = sorted(path for path in args.input.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    else:
        parser.error("Укажите существующий JPG, JPEG, PNG или папку с изображениями")
    if not paths:
        parser.error("В папке не найдены JPG, JPEG или PNG")
    if args.output.exists():
        parser.error("Выходной файл уже существует. Укажите новый путь через --output")

    model, name = load_model(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "model", "probability_cat", "probability_dog"])
        for path in paths:
            image = prepare_image(load_rgb(path))
            cat, dog = predict(model, image)
            writer.writerow([str(path.resolve()), name, cat, dog])
            print(f"{path.name}: cat={cat:.6f}, dog={dog:.6f}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
