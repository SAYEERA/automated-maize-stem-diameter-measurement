"""
Image preprocessing pipeline for maize/corn stem phenotyping.

This script prepares field or laboratory stem images for downstream diameter
analysis. Raw Canon CR3 images are converted to standardized JPG files, the
stem foreground is isolated from the background, and processed black- and
white-background images are exported as JPG and TIFF files.

Expected input structure
------------------------
INPUT_BASE_DIR/
    batch_1/
        image_001.CR3
        image_002.CR3
    batch_2/
        image_003.CR3

Output structure
----------------
OUTPUT_BASE_DIR/
    batch_1_YYYYMMDD_HHMMSS/
        raw_to_jpg/
        processed_blackbg/jpg/
        processed_blackbg/tif/
        processed_whitebg/jpg/
        processed_whitebg/tif/
        processing_log.csv
        failed_log.csv

Notes
-----
The background-removal endpoint and API key are read from environment variables:

    BACKGROUND_REMOVAL_API_URL
    BACKGROUND_REMOVAL_API_KEY
    BACKGROUND_REMOVAL_API_KEY_HEADER

Keeping these values outside the source code prevents private credentials or
vendor-specific service details from being committed to a public repository.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable

import rawpy
import requests
from PIL import Image


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """User-editable paths and preprocessing options."""

    input_base_dir: Path = Path(
        r"test_images/sample_inputs"
    )
    output_base_dir: Path = Path(
        r"test_images/sample_outputs"
    )

    processed_batch_log_name: str = "processed_batches_log.txt"

    # The API URL and header name are intentionally supplied through environment
    # variables so the repository remains service-neutral and safe to share.
    background_removal_api_url: str = os.getenv("BACKGROUND_REMOVAL_API_URL", "").strip()
    api_key_env_name: str = "BACKGROUND_REMOVAL_API_KEY"
    api_key_header_name: str = os.getenv("BACKGROUND_REMOVAL_API_KEY_HEADER", "").strip()

    # Canon CR3 images are standardized to landscape orientation before
    # foreground extraction so all downstream measurements use the same layout.
    target_image_size: tuple[int, int] = (6000, 4000)
    raw_brightness: float = 4.5

    # Output image formats used by the downstream segmentation pipeline.
    save_jpg: bool = True
    save_tif: bool = True


CONFIG = PipelineConfig()


# -----------------------------------------------------------------------------
# General file utilities
# -----------------------------------------------------------------------------

def read_completed_batches(log_path: Path) -> set[str]:
    """Return batch names already processed in previous runs."""
    if not log_path.exists():
        return set()

    with log_path.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def append_completed_batch(log_path: Path, batch_name: str) -> None:
    """Record one successfully processed batch."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(batch_name + "\n")


def list_batch_folders(input_base_dir: Path) -> list[Path]:
    """Return immediate subfolders that represent image batches."""
    if not input_base_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_base_dir}")

    return sorted(path for path in input_base_dir.iterdir() if path.is_dir())


def list_raw_images(batch_dir: Path) -> list[Path]:
    """Return Canon CR3 files from a batch folder."""
    return sorted(path for path in batch_dir.iterdir() if path.suffix.lower() == ".cr3")


def create_session_directories(output_base_dir: Path, batch_name: str) -> dict[str, Path]:
    """Create the folder structure for one preprocessing session."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_base_dir / f"{batch_name}_{timestamp}"

    directories = {
        "session": session_dir,
        "converted": session_dir / "raw_to_jpg",
        "black_jpg": session_dir / "processed_blackbg" / "jpg",
        "black_tif": session_dir / "processed_blackbg" / "tif",
        "white_jpg": session_dir / "processed_whitebg" / "jpg",
        "white_tif": session_dir / "processed_whitebg" / "tif",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


# -----------------------------------------------------------------------------
# Raw image conversion
# -----------------------------------------------------------------------------

def standardize_orientation_and_size(
    image: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    """Convert an image to the expected landscape orientation and size.

    Images acquired in portrait orientation are rotated into landscape format.
    The additional 180-degree rotation matches the acquisition layout used in
    this project and keeps the basal side consistently positioned for later
    analysis.
    """
    if image.size == target_size:
        return image

    if image.size == (target_size[1], target_size[0]):
        image = image.rotate(90, expand=True)
        return image.rotate(180)

    if image.height > image.width:
        image = image.rotate(90, expand=True)

    image = image.resize(target_size)
    return image.rotate(180)


def convert_raw_to_jpg(
    raw_path: Path,
    jpg_path: Path,
    config: PipelineConfig,
) -> tuple[bool, str]:
    """Convert one CR3 image to a standardized JPG image."""
    try:
        with rawpy.imread(str(raw_path)) as raw_file:
            rgb_image = raw_file.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                bright=config.raw_brightness,
            )

        image = Image.fromarray(rgb_image)
        image = standardize_orientation_and_size(image, config.target_image_size)
        image.save(jpg_path)

        return True, "CR3 converted to standardized JPG"

    except Exception as error:
        return False, f"CR3 conversion failed: {error}"


def convert_batch_raw_images(
    raw_images: Iterable[Path],
    converted_dir: Path,
    config: PipelineConfig,
) -> list[Path]:
    """Convert all CR3 files in a batch and return successful JPG paths."""
    converted_jpgs: list[Path] = []

    for raw_path in raw_images:
        jpg_path = converted_dir / f"{raw_path.stem}.jpg"
        success, message = convert_raw_to_jpg(raw_path, jpg_path, config)

        if success:
            converted_jpgs.append(jpg_path)
            print(f"Converted: {raw_path.name}")
        else:
            print(f"Conversion skipped for {raw_path.name}: {message}")

    return converted_jpgs


# -----------------------------------------------------------------------------
# Foreground extraction and processed image export
# -----------------------------------------------------------------------------

def get_api_headers(config: PipelineConfig) -> dict[str, str]:
    """Create request headers using credentials stored outside the source code."""
    api_key = os.getenv(config.api_key_env_name, "").strip()
    if not api_key:
        raise RuntimeError(
            f"{config.api_key_env_name} is not set. Set it in your environment before running."
        )

    if not config.api_key_header_name:
        raise RuntimeError(
            "BACKGROUND_REMOVAL_API_KEY_HEADER is not set. "
            "Set the header name required by your background-removal service."
        )

    return {config.api_key_header_name: api_key}


def request_foreground_extraction(image_path: Path, config: PipelineConfig) -> Image.Image:
    """Submit one image for foreground extraction and return the RGBA cutout."""
    if not config.background_removal_api_url:
        raise RuntimeError(
            "BACKGROUND_REMOVAL_API_URL is not set. "
            "Set it in your environment before running this script."
        )

    headers = get_api_headers(config)

    with image_path.open("rb") as image_file:
        files = {"image": image_file}
        data = {"output_type": "cutout", "format": "PNG"}
        response = requests.post(
            config.background_removal_api_url,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        raise RuntimeError(f"Unexpected API response type: {content_type}")

    result = response.json()
    cutout_url = result.get("data", {}).get("url")
    if not cutout_url:
        raise RuntimeError("Foreground extraction response did not include a cutout URL.")

    cutout_response = requests.get(cutout_url, timeout=120)
    cutout_response.raise_for_status()

    return Image.open(BytesIO(cutout_response.content)).convert("RGBA")


def place_foreground_on_background(
    foreground: Image.Image,
    background_color: tuple[int, int, int, int],
) -> Image.Image:
    """Composite a transparent foreground image onto a solid background."""
    background = Image.new("RGBA", foreground.size, background_color)
    background.paste(foreground, (0, 0), foreground)
    return background.convert("RGB")


def save_processed_background_versions(
    foreground: Image.Image,
    image_stem: str,
    directories: dict[str, Path],
    config: PipelineConfig,
) -> None:
    """Save black- and white-background processed images as JPG and TIFF."""
    black_background_image = place_foreground_on_background(foreground, (0, 0, 0, 255))
    white_background_image = place_foreground_on_background(foreground, (255, 255, 255, 255))

    black_jpg_path = directories["black_jpg"] / f"{image_stem}_blackbg.jpg"
    white_jpg_path = directories["white_jpg"] / f"{image_stem}_whitebg.jpg"
    black_tif_path = directories["black_tif"] / f"{image_stem}_blackbg.tif"
    white_tif_path = directories["white_tif"] / f"{image_stem}_whitebg.tif"

    if config.save_jpg:
        black_background_image.save(black_jpg_path)
        white_background_image.save(white_jpg_path)

    if config.save_tif:
        black_background_image.save(black_tif_path)
        white_background_image.save(white_tif_path)


def extract_foreground_and_save_outputs(
    jpg_path: Path,
    directories: dict[str, Path],
    config: PipelineConfig,
) -> tuple[bool, str]:
    """Run foreground extraction for one JPG and save processed outputs."""
    start_time = time.perf_counter()

    try:
        foreground = request_foreground_extraction(jpg_path, config)
        save_processed_background_versions(foreground, jpg_path.stem, directories, config)
        elapsed = time.perf_counter() - start_time
        return True, f"Processed images saved in {elapsed:.2f} s"

    except Exception as error:
        return False, str(error)


# -----------------------------------------------------------------------------
# Logging and batch execution
# -----------------------------------------------------------------------------

def write_log_header(success_writer: csv.writer, failed_writer: csv.writer) -> None:
    """Write consistent CSV headers for success and failure logs."""
    success_writer.writerow(["Filename", "Status"])
    failed_writer.writerow(["Filename", "Error"])


def process_converted_images(
    converted_jpgs: list[Path],
    directories: dict[str, Path],
    config: PipelineConfig,
) -> int:
    """Run foreground extraction for converted JPG images and log outcomes."""
    session_dir = directories["session"]
    success_log_path = session_dir / "processing_log.csv"
    failed_log_path = session_dir / "failed_log.csv"

    successful_images = 0

    with success_log_path.open("w", newline="", encoding="utf-8") as success_log, failed_log_path.open(
        "w", newline="", encoding="utf-8"
    ) as failed_log:
        success_writer = csv.writer(success_log)
        failed_writer = csv.writer(failed_log)
        write_log_header(success_writer, failed_writer)

        for image_index, jpg_path in enumerate(converted_jpgs, start=1):
            success, message = extract_foreground_and_save_outputs(jpg_path, directories, config)

            if success:
                successful_images += 1
                success_writer.writerow([jpg_path.name, message])
                print(f"[{image_index}] Processed: {jpg_path.name}")
            else:
                failed_writer.writerow([jpg_path.name, message])
                print(f"[{image_index}] Failed: {jpg_path.name} ({message})")

    return successful_images


def process_batch(batch_dir: Path, config: PipelineConfig) -> int:
    """Process one input batch from raw images through processed-image export."""
    raw_images = list_raw_images(batch_dir)
    if not raw_images:
        print(f"No CR3 images found in {batch_dir.name}; skipping.")
        return 0

    directories = create_session_directories(config.output_base_dir, batch_dir.name)
    converted_jpgs = convert_batch_raw_images(raw_images, directories["converted"], config)

    if not converted_jpgs:
        print(f"No images were successfully converted in {batch_dir.name}.")
        return 0

    return process_converted_images(converted_jpgs, directories, config)


def run_pipeline(config: PipelineConfig = CONFIG) -> None:
    """Process all unprocessed batches in the input directory."""
    print(f"Input directory: {config.input_base_dir}")
    print(f"Output directory: {config.output_base_dir}")

    config.output_base_dir.mkdir(parents=True, exist_ok=True)
    completed_log_path = config.output_base_dir / config.processed_batch_log_name
    completed_batches = read_completed_batches(completed_log_path)

    for batch_dir in list_batch_folders(config.input_base_dir):
        batch_name = batch_dir.name

        if batch_name in completed_batches:
            print(f"Skipping completed batch: {batch_name}")
            continue

        print(f"Processing batch: {batch_name}")
        successful_images = process_batch(batch_dir, config)

        if successful_images > 0:
            append_completed_batch(completed_log_path, batch_name)
            print(f"Batch completed: {batch_name} ({successful_images} images)")
        else:
            print(f"Batch not logged because no image completed successfully: {batch_name}")

    print("All available batches processed.")


if __name__ == "__main__":
    run_pipeline()
