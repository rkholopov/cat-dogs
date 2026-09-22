import argparse
import csv
import random
import shutil
from pathlib import Path


def plan_split(source, seed):
    groups = {"cats": [], "dogs": []}
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        prefix = path.name.split(".", 1)[0].lower()
        if path.suffix.lower() != ".jpg" or prefix not in ("cat", "dog"):
            raise ValueError(f"Unexpected file: {path.name}")
        groups["cats" if prefix == "cat" else "dogs"].append(path)

    rows = []
    rng = random.Random(seed)
    for label, paths in groups.items():
        if len(paths) < 2:
            raise ValueError(f"At least two images required for {label}")
        rng.shuffle(paths)
        boundary = len(paths) * 80 // 100
        print(f"{label}: train={boundary}, test={len(paths) - boundary}")
        rows.extend((path, "train" if i < boundary else "test", label)
                    for i, path in enumerate(paths))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the split without writing files")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir():
        parser.error(f"Source directory does not exist: {source}")
    if source == output or source in output.parents or output in source.parents:
        parser.error("Source and output must be separate, non-nested directories")
    if output.exists() and not args.dry_run:
        parser.error(f"Refusing to overwrite existing output: {output}")
    try:
        rows = plan_split(source, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(f"Seed: {args.seed}\nOutput: {output}")
    if args.dry_run:
        print("Dry run: no files written.")
        return

    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "test"):
        for label in ("cats", "dogs"):
            (output / split / label).mkdir(parents=True)
    for path, split, label in rows:
        shutil.copy2(path, output / split / label / path.name)
    with (output / "split.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "split", "class", "seed"))
        writer.writerows((p.name, s, label, args.seed) for p, s, label in rows)
    print(f"Copied {len(rows)} images. Original files unchanged.")


if __name__ == "__main__":
    main()
