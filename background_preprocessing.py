"""
Local black-background preprocessing for maize stem images.

This script provides a fully local preprocessing option for stem images acquired
on a light background. The workflow segments the stem using grayscale
thresholding, refines the foreground mask with morphology, composites the stem on
a black background, and applies mild contrast/color enhancement for downstream
phenotyping analysis.

The script does not use any external background-removal service. It is intended
as a reproducible image-processing alternative for repository release and method
comparison.

Outputs
-------
For each input image, the script can save:
1. Initial black-background image after mask-based compositing.
2. Contrast-enhanced black-background image.
3. Final cleaned black-background image.

Example
-------
Update INPUT_IMAGE and OUTPUT_DIR at the bottom of the file, then run:

    python local_black_background_preprocessing_publishable.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalPreprocessingConfig:
    """Parameters used by the local preprocessing workflow."""

    blur_kernel_size: int = 9
    otsu_threshold_offset: int = 10
    closing_iterations: int = 4
    feather_pixels: int = 31

    contrast_factor_first_pass: float = 1.25
    color_factor_first_pass: float = 1.20
    contrast_factor_final: float = 1.10
    color_factor_final: float = 1.08

    top_strip_cleanup_px: int = 10
    top_background_cleanup_px: int = 50

    residual_background_value_threshold: int = 150
    residual_background_saturation_threshold: int = 50

    save_intermediate_outputs: bool = True
    final_jpeg_quality: int = 100


CONFIG = LocalPreprocessingConfig()


# -----------------------------------------------------------------------------
# Image loading and saving
# -----------------------------------------------------------------------------

def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image with OpenCV and return it in RGB channel order."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb_image(image: np.ndarray, output_path: Path, quality: int = 100) -> None:
    """Save an RGB NumPy image using Pillow."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path, quality=quality)


# -----------------------------------------------------------------------------
# Foreground segmentation and compositing
# -----------------------------------------------------------------------------

def create_stem_foreground_mask(
    rgb_image: np.ndarray,
    blur_kernel_size: int = CONFIG.blur_kernel_size,
    otsu_threshold_offset: int = CONFIG.otsu_threshold_offset,
    closing_iterations: int = CONFIG.closing_iterations,
) -> np.ndarray:
    """Create a binary mask for stems photographed on a light background.

    The image is converted to grayscale and smoothed before Otsu thresholding.
    The threshold is shifted slightly downward to retain darker stem-edge pixels.
    Morphological closing fills small gaps, and opening removes isolated noise.
    """
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)

    otsu_threshold, _ = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    adjusted_threshold = max(0, int(otsu_threshold) - int(otsu_threshold_offset))

    _, foreground_mask = cv2.threshold(
        gray_image,
        adjusted_threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )

    cleanup_kernel = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_CLOSE,
        cleanup_kernel,
        iterations=closing_iterations,
    )
    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        cleanup_kernel,
        iterations=1,
    )

    return foreground_mask


def apply_soft_mask_to_black_background(
    rgb_image: np.ndarray,
    foreground_mask: np.ndarray,
    feather_pixels: int = CONFIG.feather_pixels,
) -> np.ndarray:
    """Composite the stem onto a black background using a softened mask.

    Feathering reduces harsh mask edges while keeping the non-stem background
    black. The output remains an RGB image for compatibility with downstream
    analysis scripts.
    """
    feather_pixels = max(3, int(feather_pixels) | 1)
    soft_mask = cv2.GaussianBlur(foreground_mask, (feather_pixels, feather_pixels), 0)
    alpha = (soft_mask.astype(np.float32) / 255.0)[..., None]

    black_background_image = rgb_image.astype(np.float32) * alpha
    return np.clip(black_background_image, 0, 255).astype(np.uint8)


def convert_light_background_to_black(
    rgb_image: np.ndarray,
    config: LocalPreprocessingConfig = CONFIG,
) -> np.ndarray:
    """Segment the stem and return a black-background image."""
    foreground_mask = create_stem_foreground_mask(
        rgb_image,
        blur_kernel_size=config.blur_kernel_size,
        otsu_threshold_offset=config.otsu_threshold_offset,
        closing_iterations=config.closing_iterations,
    )
    return apply_soft_mask_to_black_background(
        rgb_image,
        foreground_mask,
        feather_pixels=config.feather_pixels,
    )


# -----------------------------------------------------------------------------
# Enhancement and residual background cleanup
# -----------------------------------------------------------------------------

def enhance_stem_contrast_and_sharpness(
    rgb_image: np.ndarray,
    contrast_factor: float = CONFIG.contrast_factor_first_pass,
    color_factor: float = CONFIG.color_factor_first_pass,
    top_strip_cleanup_px: int = CONFIG.top_strip_cleanup_px,
) -> np.ndarray:
    """Apply mild color, contrast, and sharpness enhancement.

    The enhancement is intentionally conservative so that stem measurements are
    not dominated by artificial edge sharpening. A small top strip is set to
    black to remove occasional acquisition-border artifacts.
    """
    enhanced = Image.fromarray(rgb_image)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast_factor)
    enhanced = ImageEnhance.Color(enhanced).enhance(color_factor)

    enhanced_array = np.array(enhanced)
    sharpen_kernel = np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=np.float32,
    )
    enhanced_array = cv2.filter2D(enhanced_array, -1, sharpen_kernel)

    if top_strip_cleanup_px > 0:
        enhanced_array[:top_strip_cleanup_px, :, :] = 0

    return enhanced_array


def remove_residual_light_background(
    rgb_image: np.ndarray,
    config: LocalPreprocessingConfig = CONFIG,
) -> np.ndarray:
    """Remove remaining light gray or white background pixels.

    Pixels with high value and low saturation are treated as residual background
    and set to black. This step is useful when the thresholded mask leaves small
    light patches around the stem.
    """
    cleaned = rgb_image.copy()

    if config.top_background_cleanup_px > 0:
        cleaned[:config.top_background_cleanup_px, :, :] = 0

    hsv_image = cv2.cvtColor(cleaned, cv2.COLOR_RGB2HSV)
    _, saturation, value = cv2.split(hsv_image)

    residual_background = (
        (value > config.residual_background_value_threshold)
        & (saturation < config.residual_background_saturation_threshold)
    )
    cleaned[residual_background] = [0, 0, 0]

    cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)
    cleaned_pil = Image.fromarray(cleaned)
    cleaned_pil = ImageEnhance.Contrast(cleaned_pil).enhance(config.contrast_factor_final)
    cleaned_pil = ImageEnhance.Color(cleaned_pil).enhance(config.color_factor_final)

    return np.array(cleaned_pil)


# -----------------------------------------------------------------------------
# Pipeline execution
# -----------------------------------------------------------------------------

def preprocess_single_image(
    input_path: Path,
    output_dir: Path,
    config: LocalPreprocessingConfig = CONFIG,
) -> Path:
    """Run the complete local preprocessing workflow for one image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = input_path.stem

    print(f"Processing image: {image_stem}")

    original_rgb = load_rgb_image(input_path)

    black_background = convert_light_background_to_black(original_rgb, config)
    enhanced_image = enhance_stem_contrast_and_sharpness(
        black_background,
        contrast_factor=config.contrast_factor_first_pass,
        color_factor=config.color_factor_first_pass,
        top_strip_cleanup_px=config.top_strip_cleanup_px,
    )
    final_image = remove_residual_light_background(enhanced_image, config)

    if config.save_intermediate_outputs:
        save_rgb_image(
            black_background,
            output_dir / f"{image_stem}_black_background_initial.jpg",
            quality=config.final_jpeg_quality,
        )
        save_rgb_image(
            enhanced_image,
            output_dir / f"{image_stem}_black_background_enhanced.jpg",
            quality=config.final_jpeg_quality,
        )

    final_output_path = output_dir / f"{image_stem}_black_background_cleaned.jpg"
    save_rgb_image(final_image, final_output_path, quality=config.final_jpeg_quality)

    print(f"Saved final image: {final_output_path}")
    return final_output_path


def preprocess_image_folder(
    input_dir: Path,
    output_dir: Path,
    config: LocalPreprocessingConfig = CONFIG,
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> list[Path]:
    """Preprocess all supported images in a folder."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )

    if not image_paths:
        print(f"No supported image files found in: {input_dir}")
        return []

    output_paths: list[Path] = []
    for image_path in image_paths:
        try:
            output_paths.append(preprocess_single_image(image_path, output_dir, config))
        except Exception as error:
            print(f"Skipped {image_path.name}: {error}")

    return output_paths


if __name__ == "__main__":
    # Choose either a single image or a folder.
    INPUT_IMAGE = Path("images/")
    OUTPUT_DIR = Path("images/output")

    preprocess_single_image(INPUT_IMAGE, OUTPUT_DIR)

    # For folder processing, comment the line above and use:
    # INPUT_DIR = Path("images")
    # preprocess_image_folder(INPUT_DIR, OUTPUT_DIR)
