"""Train the custom football detector from the prepared YOLO dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "yolov8n.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "detect"


def count_files(directory: Path, suffix: str) -> int:
    return sum(1 for path in directory.glob(f"*.{suffix}") if path.is_file())


def validate_dataset(data_path: Path) -> None:
    dataset_dir = data_path.parent
    required_dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
    ]
    missing_dirs = [directory for directory in required_dirs if not directory.is_dir()]
    if missing_dirs:
        missing = ", ".join(str(directory) for directory in missing_dirs)
        raise FileNotFoundError(f"Dataset directories are missing: {missing}")

    for split in ("train", "val"):
        image_count = sum(
            count_files(dataset_dir / "images" / split, extension)
            for extension in ("jpg", "jpeg", "png")
        )
        label_count = count_files(dataset_dir / "labels" / split, "txt")
        if image_count == 0 or label_count == 0:
            raise ValueError(
                f"The {split} split is empty (images={image_count}, labels={label_count}). "
                "Run dataset/split_by_clip.py after adding labeled frames."
            )
        print(f"{split}: {image_count} images, {label_count} labels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    data_path = args.data.resolve()
    model_path = args.model.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Base model not found: {model_path}")

    validate_dataset(data_path)
    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(DEFAULT_OUTPUT),
        name="train",
        exist_ok=True,
        pretrained=True,
    )
    print(f"Custom weights written to {DEFAULT_OUTPUT / 'train' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()