import argparse
import gc
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

try:
    import rawpy
except ImportError:  # rawpy is only needed for camera RAW formats.
    rawpy = None


# User settings
INPUT_DIR = "/Users/paul/Desktop/NOS/FinDetection/rissos_cropping_dataset"
OUTPUT_DIR = "/Users/paul/Desktop/NOS/FinDetection/rissos_cropping_dataset_jpeg"
JPEG_QUALITY = 75
GC_INTERVAL = 500
JPEG_COPY_THRESHOLD_MB = 5


RAW_EXTENSIONS = {
    ".3fr",
    ".arw",
    ".cr2",
    ".cr3",
    ".crw",
    ".dcr",
    ".dng",
    ".erf",
    ".fff",
    ".iiq",
    ".k25",
    ".kdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".raw",
    ".rw2",
    ".rwl",
    ".sr2",
    ".srf",
    ".srw",
    ".x3f",
}


def convert_directory_to_jpeg(input_dir, output_dir, jpeg_quality, gc_interval=GC_INTERVAL):
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")

    jpeg_quality = validate_jpeg_quality(jpeg_quality)
    gc_interval = validate_gc_interval(gc_interval)
    converted = 0
    skipped = 0
    total_output_size = 0
    failed_files = []
    target_path_counts = {}
    total_files, total_input_size = scan_input_files(input_path, output_path)
    started_at = datetime.now()
    start_time = time.monotonic()

    print(f"Images queued for conversion: {total_files}")
    print(f"Total input size: {format_file_size(total_input_size)}")
    print(f"Started: {format_datetime(started_at)}")

    if not total_files:
        print(f"Finished: {format_datetime(datetime.now())}")
        print("No files found to convert.")
        return

    for index, source_path in enumerate(iter_source_files(input_path, output_path), start=1):
        relative_path = source_path.relative_to(input_path)
        target_path = get_target_path(
            output_path / relative_path.with_suffix(".jpg"),
            target_path_counts,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            save_as_jpeg(source_path, target_path, jpeg_quality)
            os.utime(target_path, (source_path.stat().st_atime, source_path.stat().st_mtime))
            converted += 1
            total_output_size += target_path.stat().st_size
        except Exception as exc:
            skipped += 1
            failed_files.append((relative_path, str(exc)))
        finally:
            if gc_interval and index % gc_interval == 0:
                gc.collect()
            print_progress_bar(index, total_files, start_time)

    print()
    print(f"Finished: {format_datetime(datetime.now())}")
    print("Final report")
    print(f"Converted: {converted}")
    print(f"Failed: {skipped}")
    print(f"Original total size: {format_file_size(total_input_size)}")
    print(f"New total size: {format_file_size(total_output_size)}")

    if failed_files:
        print("Failed files:")
        for failed_path, error_message in failed_files:
            print(f"- {failed_path}: {error_message}")
    else:
        print("Failed files: None")


def scan_input_files(input_path, output_path):
    total_files = 0
    total_size = 0

    for path in iter_source_files(input_path, output_path):
        total_files += 1
        total_size += path.stat().st_size

    return total_files, total_size


def iter_source_files(input_path, output_path):
    for path in input_path.rglob("*"):
        if path.is_file() and not is_relative_to(path, output_path):
            yield path


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def print_progress_bar(current, total, start_time, width=40):
    completed = int(width * current / total)
    bar = "#" * completed + "-" * (width - completed)
    percent = current / total * 100
    elapsed = time.monotonic() - start_time
    average_seconds_per_file = elapsed / current
    remaining_seconds = average_seconds_per_file * (total - current)
    eta = format_duration(remaining_seconds)
    sys.stdout.write(f"\rProgress: [{bar}] {current}/{total} ({percent:5.1f}%) ETA: {eta}")
    sys.stdout.flush()


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def format_datetime(value):
    return value.strftime("%Y-%m-%d %H:%M:%S")


def format_file_size(size_bytes):
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024


def get_target_path(target_path, target_path_counts):
    count = target_path_counts.get(target_path, 0) + 1
    target_path_counts[target_path] = count

    if count == 1:
        return target_path

    return target_path.with_name(f"{target_path.stem}_{count}{target_path.suffix}")


def validate_jpeg_quality(jpeg_quality):
    try:
        jpeg_quality = int(jpeg_quality)
    except (TypeError, ValueError) as exc:
        raise ValueError("JPEG quality must be an integer from 1 to 95") from exc

    if not 1 <= jpeg_quality <= 95:
        raise ValueError("JPEG quality must be between 1 and 95")

    return jpeg_quality


def validate_gc_interval(gc_interval):
    try:
        gc_interval = int(gc_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError("GC interval must be a whole number") from exc

    if gc_interval < 0:
        raise ValueError("GC interval must be 0 or greater")

    return gc_interval


def save_as_jpeg(source_path, target_path, jpeg_quality):
    if (
        source_path.suffix.lower() in {".jpg", ".jpeg"}
        and source_path.stat().st_size < JPEG_COPY_THRESHOLD_MB * 1024 * 1024
    ):
        shutil.copy2(source_path, target_path)
        return

    if source_path.suffix.lower() in RAW_EXTENSIONS:
        image = open_raw_image(source_path)
        try:
            save_image(image, target_path, jpeg_quality)
        finally:
            image.close()
        return

    try:
        with Image.open(source_path) as image:
            image.load()
            exif = image.info.get("exif")
            save_image(image, target_path, jpeg_quality, exif=exif)
    except UnidentifiedImageError:
        image = open_raw_image(source_path)
        try:
            save_image(image, target_path, jpeg_quality)
        finally:
            image.close()


def save_image(image, target_path, jpeg_quality, exif=None):
    jpeg_image = prepare_for_jpeg(image)
    try:
        save_kwargs = {
            "format": "JPEG",
            "quality": jpeg_quality,
            "optimize": True,
            "progressive": True,
        }
        if exif:
            save_kwargs["exif"] = exif
        jpeg_image.save(target_path, **save_kwargs)
    finally:
        if jpeg_image is not image:
            jpeg_image.close()


def open_raw_image(source_path):
    if rawpy is None:
        raise RuntimeError("RAW image support requires rawpy. Install it with: pip install rawpy")

    with rawpy.imread(str(source_path)) as raw:
        rgb = raw.postprocess()

    image = Image.fromarray(rgb)
    image.load()
    del rgb

    return image


def prepare_for_jpeg(image):
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and image.info.get("transparency") is not None
    ):
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha_image = image.convert("RGBA")
        alpha_channel = alpha_image.getchannel("A")
        try:
            background.paste(alpha_image, mask=alpha_channel)
        finally:
            alpha_channel.close()
            alpha_image.close()
        return background

    image = image.convert("RGB")

    return image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert all images in a directory tree to JPEG while preserving folder structure."
    )
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Directory containing images to convert.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory where JPEG files will be written.")
    parser.add_argument(
        "--quality",
        type=int,
        default=JPEG_QUALITY,
        help="JPEG quality from 1 to 95. Higher means larger files.",
    )
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=GC_INTERVAL,
        help="Run forced cleanup every N files. Use 0 to disable. Lower values use less memory but may run slower.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_directory_to_jpeg(args.input_dir, args.output_dir, args.quality, args.gc_interval)
