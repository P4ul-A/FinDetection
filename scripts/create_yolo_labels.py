"""Generate YOLO fin labels from manual crop images.

Expected dataset layout:
    <dataset-root>/
        original/      source images
        cropped/       manually cropped fin images
        yolo_labels/   generated YOLO labels

The script finds each cropped fin inside its matching original image using
template matching, then writes standard YOLO detection labels for class 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


# Defaults: edit these values when you want to change normal script behavior.
DEFAULT_DATASET_ROOT = Path("rissos_cropping_dataset")
DEFAULT_ORIGINAL_DIR: Path | None = None
DEFAULT_CROPPED_DIR: Path | None = None
DEFAULT_LABEL_DIR: Path | None = None
DEFAULT_MIN_SCORE = 0.80
DEFAULT_COARSE_MAX_DIM = 1600
DEFAULT_REFINE_MARGIN = 100
DEFAULT_NO_EMPTY_LABELS = False
DEFAULT_DRY_RUN = False

DEFAULT_CLASS_ID = 0
DEFAULT_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".cr2",
}


@dataclass(frozen=True)
class CropMatch:
    crop_path: Path
    original_path: Path
    score: float
    x: int
    y: int
    width: int
    height: int
    original_width: int
    original_height: int

    def to_yolo_line(self) -> str:
        center_x = (self.x + self.width / 2) / self.original_width
        center_y = (self.y + self.height / 2) / self.original_height
        width = self.width / self.original_width
        height = self.height / self.original_height
        return f"{DEFAULT_CLASS_ID} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create YOLO labels by matching manual fin crops back to original images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset directory containing original/ and cropped/.",
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=DEFAULT_ORIGINAL_DIR,
        help="Override original image directory. Defaults to <dataset-root>/original.",
    )
    parser.add_argument(
        "--cropped-dir",
        type=Path,
        default=DEFAULT_CROPPED_DIR,
        help="Override cropped image directory. Defaults to <dataset-root>/cropped.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="Output label directory. Defaults to <dataset-root>/yolo_labels.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum final template-match score required to write a box.",
    )
    parser.add_argument(
        "--coarse-max-dim",
        type=int,
        default=DEFAULT_COARSE_MAX_DIM,
        help="Maximum image dimension used for the coarse search. Use 0 for full-resolution search.",
    )
    parser.add_argument(
        "--refine-margin",
        type=int,
        default=DEFAULT_REFINE_MARGIN,
        help="Full-resolution pixels to search around the coarse match.",
    )
    parser.add_argument(
        "--no-empty-labels",
        action="store_true",
        default=DEFAULT_NO_EMPTY_LABELS,
        help="Do not create empty label files for originals that have no matched crop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DEFAULT_DRY_RUN,
        help="Run matching and reporting without writing labels.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if is_image_file(path))


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return np.array(image.convert("RGB"))


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def original_key_from_crop_stem(crop_stem: str, original_by_stem: dict[str, Path]) -> str | None:
    if crop_stem in original_by_stem:
        return crop_stem

    fin_match = re.search(r"_fin\d+_(.+)$", crop_stem)
    if fin_match and fin_match.group(1) in original_by_stem:
        return fin_match.group(1)

    suffix_matches = [
        stem for stem in original_by_stem if crop_stem.endswith(f"_{stem}") or crop_stem.endswith(stem)
    ]
    if not suffix_matches:
        return None
    return max(suffix_matches, key=len)


def build_crop_groups(
    original_paths: list[Path],
    crop_paths: list[Path],
) -> tuple[dict[Path, list[Path]], list[Path]]:
    original_by_stem: dict[str, Path] = {}
    duplicate_stems: set[str] = set()
    for original_path in original_paths:
        if original_path.stem in original_by_stem:
            duplicate_stems.add(original_path.stem)
        else:
            original_by_stem[original_path.stem] = original_path

    if duplicate_stems:
        stems = ", ".join(sorted(duplicate_stems)[:5])
        raise ValueError(f"Duplicate original stems are ambiguous: {stems}")

    groups: dict[Path, list[Path]] = defaultdict(list)
    unmatched: list[Path] = []
    for crop_path in crop_paths:
        original_stem = original_key_from_crop_stem(crop_path.stem, original_by_stem)
        if original_stem is None:
            unmatched.append(crop_path)
            continue
        groups[original_by_stem[original_stem]].append(crop_path)
    return dict(groups), unmatched


def resize_for_coarse(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1:
        return gray
    width = max(1, round(gray.shape[1] * scale))
    height = max(1, round(gray.shape[0] * scale))
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def match_crop(
    original_gray: np.ndarray,
    crop_gray: np.ndarray,
    crop_path: Path,
    original_path: Path,
    coarse_max_dim: int,
    refine_margin: int,
) -> CropMatch:
    original_height, original_width = original_gray.shape
    crop_height, crop_width = crop_gray.shape
    if crop_width > original_width or crop_height > original_height:
        raise ValueError(
            f"Crop is larger than original: {crop_path.name} ({crop_width}x{crop_height}) "
            f"> {original_path.name} ({original_width}x{original_height})"
        )

    scale = 1.0
    max_dim = max(original_width, original_height)
    if coarse_max_dim > 0 and max_dim > coarse_max_dim:
        scale = coarse_max_dim / max_dim

    if scale < 1.0:
        coarse_original = resize_for_coarse(original_gray, scale)
        coarse_crop = resize_for_coarse(crop_gray, scale)
        if coarse_crop.shape[0] < 2 or coarse_crop.shape[1] < 2:
            coarse_x = 0
            coarse_y = 0
        else:
            coarse_result = cv2.matchTemplate(coarse_original, coarse_crop, cv2.TM_CCOEFF_NORMED)
            _, _, _, coarse_loc = cv2.minMaxLoc(coarse_result)
            coarse_x = round(coarse_loc[0] / scale)
            coarse_y = round(coarse_loc[1] / scale)

        x1 = max(0, coarse_x - refine_margin)
        y1 = max(0, coarse_y - refine_margin)
        x2 = min(original_width, coarse_x + crop_width + refine_margin)
        y2 = min(original_height, coarse_y + crop_height + refine_margin)
        search = original_gray[y1:y2, x1:x2]
        result = cv2.matchTemplate(search, crop_gray, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        x = x1 + loc[0]
        y = y1 + loc[1]
    else:
        result = cv2.matchTemplate(original_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        x, y = loc

    return CropMatch(
        crop_path=crop_path,
        original_path=original_path,
        score=float(score),
        x=int(x),
        y=int(y),
        width=int(crop_width),
        height=int(crop_height),
        original_width=int(original_width),
        original_height=int(original_height),
    )


def progress_line(done: int, total: int, detail: str = "") -> None:
    width = 30
    filled = round(width * done / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {detail}" if detail else ""
    print(f"\r[{bar}] {done}/{total}{suffix}", end="", flush=True)
    if done == total:
        print()


def write_report(
    report_dir: Path,
    matches: list[CropMatch],
    low_score: list[CropMatch],
    unmatched_crops: list[Path],
    errors: list[tuple[Path, str]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "matched_boxes": len(matches),
        "low_score_skipped": len(low_score),
        "unmatched_crops": len(unmatched_crops),
        "errors": len(errors),
        "min_match_score": min((match.score for match in matches), default=None),
    }
    (report_dir / "label_generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (report_dir / "label_generation_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "status",
                "crop",
                "original",
                "score",
                "x",
                "y",
                "width",
                "height",
                "message",
            ]
        )
        for match in matches:
            writer.writerow(
                [
                    "matched",
                    match.crop_path.name,
                    match.original_path.name,
                    f"{match.score:.6f}",
                    match.x,
                    match.y,
                    match.width,
                    match.height,
                    "",
                ]
            )
        for match in low_score:
            writer.writerow(
                [
                    "low_score",
                    match.crop_path.name,
                    match.original_path.name,
                    f"{match.score:.6f}",
                    match.x,
                    match.y,
                    match.width,
                    match.height,
                    "Skipped because score is below --min-score.",
                ]
            )
        for crop_path in unmatched_crops:
            writer.writerow(["unmatched_crop", crop_path.name, "", "", "", "", "", "", "No matching original stem."])
        for crop_path, message in errors:
            writer.writerow(["error", crop_path.name, "", "", "", "", "", "", message])


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    original_dir = args.original_dir or dataset_root / "original"
    cropped_dir = args.cropped_dir or dataset_root / "cropped"
    label_dir = args.label_dir or dataset_root / "yolo_labels"

    if not original_dir.is_dir():
        raise FileNotFoundError(f"Missing original directory: {original_dir}")
    if not cropped_dir.is_dir():
        raise FileNotFoundError(f"Missing cropped directory: {cropped_dir}")

    original_paths = image_files(original_dir)
    crop_paths = image_files(cropped_dir)
    if not original_paths:
        raise ValueError(f"No original images found in {original_dir}")
    if not crop_paths:
        raise ValueError(f"No cropped images found in {cropped_dir}")

    crop_groups, unmatched_crops = build_crop_groups(original_paths, crop_paths)
    total_crops = sum(len(paths) for paths in crop_groups.values())
    print(f"Original images: {len(original_paths)}")
    print(f"Cropped images: {len(crop_paths)}")
    print(f"Crops queued for matching: {total_crops}")
    print(f"Unmatched crop filenames: {len(unmatched_crops)}")
    progress_line(0, total_crops)

    matches: list[CropMatch] = []
    low_score: list[CropMatch] = []
    errors: list[tuple[Path, str]] = []
    labels_by_original: dict[Path, list[str]] = defaultdict(list)
    done = 0

    for original_path in sorted(crop_groups):
        try:
            original_gray = to_gray(load_rgb(original_path))
        except Exception as exc:
            for crop_path in crop_groups[original_path]:
                errors.append((crop_path, f"Could not load original {original_path.name}: {exc}"))
                done += 1
                progress_line(done, total_crops, crop_path.name)
            continue

        for crop_path in crop_groups[original_path]:
            try:
                crop_gray = to_gray(load_rgb(crop_path))
                match = match_crop(
                    original_gray=original_gray,
                    crop_gray=crop_gray,
                    crop_path=crop_path,
                    original_path=original_path,
                    coarse_max_dim=args.coarse_max_dim,
                    refine_margin=args.refine_margin,
                )
                if match.score >= args.min_score:
                    matches.append(match)
                    labels_by_original[original_path].append(match.to_yolo_line())
                else:
                    low_score.append(match)
            except Exception as exc:
                errors.append((crop_path, str(exc)))
            done += 1
            progress_line(done, total_crops, crop_path.name)

    if not args.dry_run:
        label_dir.mkdir(parents=True, exist_ok=True)
        originals_to_write = original_paths if not args.no_empty_labels else sorted(labels_by_original)
        for original_path in originals_to_write:
            label_path = label_dir / f"{original_path.stem}.txt"
            lines = labels_by_original.get(original_path, [])
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    report_dir = label_dir if not args.dry_run else dataset_root
    write_report(report_dir, matches, low_score, unmatched_crops, errors)

    print(f"Matched boxes written: {len(matches)}")
    print(f"Low-score crops skipped: {len(low_score)}")
    print(f"Unmatched crop filenames: {len(unmatched_crops)}")
    print(f"Errors: {len(errors)}")
    if not args.dry_run:
        print(f"YOLO labels: {label_dir}")
    print(f"Report: {report_dir / 'label_generation_report.csv'}")

    if unmatched_crops or low_score or errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
