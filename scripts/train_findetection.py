"""Train a YOLO fin detector from the local cropping dataset.

Expected source layout:
    cropping_dataset/
        original/      image files
        yolo_labels/   YOLO txt labels with matching image stems

The script creates a YOLO-compatible split dataset under runs/, trains from a
pretrained Ultralytics model, and writes simple learning plots plus a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = ["fin"]



@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    label_path: Path
    has_object: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model to detect fins from cropping_dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("rissos_cropping_dataset_jpeg"),
        help="Directory containing original/ images and yolo_labels/ labels.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("runs/findetection_dataset"),
        help="Where the generated YOLO split dataset should be written.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/findetection_training"),
        help="Ultralytics project output directory.",
    )
    parser.add_argument(
        "--name",
        default="fin_yolo",
        help="Ultralytics run name inside --project.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Pretrained YOLO weights to fine tune, for example yolo11n.pt or yolov8n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default=None, help="Device passed to Ultralytics, e.g. cpu, 0, mps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers passed to Ultralytics.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="symlink",
        help="How to place files in the generated YOLO dataset.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only prepare the dataset and write dataset summary.",
    )
    return parser.parse_args()


def find_dataset_items(dataset_root: Path) -> list[DatasetItem]:
    image_dir = dataset_root / "original"
    label_dir = dataset_root / "yolo_labels"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    label_by_stem = {path.stem: path for path in label_dir.glob("*.txt")}
    items: list[DatasetItem] = []
    missing_labels: list[Path] = []

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_by_stem.get(image_path.stem)
        if label_path is None:
            missing_labels.append(image_path)
            continue
        has_object = bool(label_path.read_text(encoding="utf-8").strip())
        items.append(DatasetItem(image_path=image_path, label_path=label_path, has_object=has_object))

    if missing_labels:
        preview = ", ".join(path.name for path in missing_labels[:5])
        raise ValueError(
            f"{len(missing_labels)} images do not have matching labels in {label_dir}: {preview}"
        )
    if not items:
        raise ValueError(f"No images found in {image_dir}")

    return items


def stratified_split(
    items: list[DatasetItem],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[DatasetItem]]:
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("--val-ratio and --test-ratio must be >= 0 and sum to less than 1.")

    rng = random.Random(seed)
    positives = [item for item in items if item.has_object]
    negatives = [item for item in items if not item.has_object]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    splits = {"train": [], "val": [], "test": []}
    for group in (positives, negatives):
        total = len(group)
        test_count = round(total * test_ratio)
        val_count = round(total * val_ratio)
        splits["test"].extend(group[:test_count])
        splits["val"].extend(group[test_count : test_count + val_count])
        splits["train"].extend(group[test_count + val_count :])

    for split_items in splits.values():
        rng.shuffle(split_items)

    if not splits["train"] or not splits["val"]:
        raise ValueError("Split produced empty train or val data. Reduce --val-ratio/--test-ratio.")
    return splits


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def place_file(source: Path, target: Path, copy_mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def write_split_dataset(
    splits: dict[str, list[DatasetItem]],
    work_dir: Path,
    copy_mode: str,
) -> Path:
    reset_directory(work_dir)

    for split_name, split_items in splits.items():
        image_out = work_dir / "images" / split_name
        label_out = work_dir / "labels" / split_name
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for item in split_items:
            place_file(item.image_path, image_out / item.image_path.name, copy_mode)
            place_file(item.label_path, label_out / item.label_path.name, copy_mode)

    data_yaml = work_dir / "fin_detection.yaml"
    data = {
        "path": str(work_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml


def count_boxes(label_paths: Iterable[Path]) -> int:
    total = 0
    for label_path in label_paths:
        with label_path.open("r", encoding="utf-8") as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def write_dataset_summary(
    items: list[DatasetItem],
    splits: dict[str, list[DatasetItem]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_images": len(items),
        "positive_images": sum(item.has_object for item in items),
        "negative_images": sum(not item.has_object for item in items),
        "total_boxes": count_boxes(item.label_path for item in items),
        "splits": {},
    }

    for split_name, split_items in splits.items():
        summary["splits"][split_name] = {
            "images": len(split_items),
            "positive_images": sum(item.has_object for item in split_items),
            "negative_images": sum(not item.has_object for item in split_items),
            "boxes": count_boxes(item.label_path for item in split_items),
        }

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (output_dir / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "images", "positive_images", "negative_images", "boxes"])
        writer.writerow(
            [
                "all",
                summary["total_images"],
                summary["positive_images"],
                summary["negative_images"],
                summary["total_boxes"],
            ]
        )
        for split_name, split_summary in summary["splits"].items():
            writer.writerow(
                [
                    split_name,
                    split_summary["images"],
                    split_summary["positive_images"],
                    split_summary["negative_images"],
                    split_summary["boxes"],
                ]
            )


def compact_metric_name(column: str) -> str:
    return column.replace("metrics/", "").replace("(B)", "").replace("_", " ").strip()


def plot_metrics(results_csv: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    if not results_csv.exists():
        return

    results = pd.read_csv(results_csv)
    results.columns = [column.strip() for column in results.columns]
    if "epoch" not in results.columns:
        return

    loss_columns = [
        column
        for column in results.columns
        if column.endswith("loss") and results[column].notna().any()
    ]
    metric_columns = [
        column
        for column in results.columns
        if column.startswith("metrics/") and results[column].notna().any()
    ]

    if loss_columns:
        plt.figure(figsize=(10, 6))
        for column in loss_columns:
            plt.plot(results["epoch"], results[column], label=compact_metric_name(column))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("YOLO Training and Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "learning_losses.png", dpi=160)
        plt.close()

    if metric_columns:
        plt.figure(figsize=(10, 6))
        for column in metric_columns:
            plt.plot(results["epoch"], results[column], label=compact_metric_name(column))
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("YOLO Validation Metrics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "validation_metrics.png", dpi=160)
        plt.close()


def best_metric_row(results):
    map_column = "metrics/mAP50-95(B)"
    if map_column in results.columns:
        return results.loc[results[map_column].idxmax()]
    return results.iloc[-1]


def write_training_summary(results_csv: Path, output_dir: Path, metrics: dict) -> None:
    import pandas as pd

    summary = {"validation": metrics}

    if results_csv.exists():
        results = pd.read_csv(results_csv)
        results.columns = [column.strip() for column in results.columns]
        if not results.empty:
            row = best_metric_row(results)
            summary["best_epoch"] = int(row["epoch"]) if "epoch" in row else None
            for column in results.columns:
                if column.startswith("metrics/") or column.endswith("loss"):
                    value = row[column]
                    if pd.notna(value):
                        summary[compact_metric_name(column)] = float(value)

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = ["Fin detection training summary", ""]
    if "best_epoch" in summary:
        lines.append(f"Best epoch: {summary['best_epoch']}")
    for key, value in summary.items():
        if key in {"validation", "best_epoch"}:
            continue
        lines.append(f"{key}: {value:.4f}")
    lines.append("")
    lines.append("Final validation metrics:")
    for key, value in metrics.items():
        lines.append(f"{key}: {value}")
    (output_dir / "training_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def train(args: argparse.Namespace, data_yaml: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
        "workers": args.workers,
        "seed": args.seed,
    }
    if args.device:
        train_kwargs["device"] = args.device

    train_result = model.train(**train_kwargs)
    save_dir_value = getattr(train_result, "save_dir", None) or getattr(model.trainer, "save_dir", None)
    if save_dir_value is None:
        raise RuntimeError("Could not determine Ultralytics training output directory.")
    save_dir = Path(save_dir_value)

    best_weights = save_dir / "weights" / "best.pt"
    validation_model = YOLO(str(best_weights if best_weights.exists() else save_dir / "weights" / "last.pt"))
    validation_metrics = validation_model.val(data=str(data_yaml), split="test")
    write_training_summary(save_dir / "results.csv", save_dir, validation_metrics.results_dict)
    plot_metrics(save_dir / "results.csv", save_dir)
    return save_dir


def main() -> None:
    args = parse_args()
    items = find_dataset_items(args.dataset_root)
    splits = stratified_split(items, args.val_ratio, args.test_ratio, args.seed)
    data_yaml = write_split_dataset(splits, args.work_dir, args.copy_mode)
    write_dataset_summary(items, splits, args.work_dir)

    print(f"Prepared YOLO dataset: {args.work_dir}")
    print(f"Dataset YAML: {data_yaml}")

    if args.skip_train:
        print("Skipping training because --skip-train was set.")
        return

    save_dir = train(args, data_yaml)
    print(f"Training outputs: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"Learning graphs: {save_dir / 'learning_losses.png'} and {save_dir / 'validation_metrics.png'}")
    print(f"Summary: {save_dir / 'training_summary.txt'}")


if __name__ == "__main__":
    main()
