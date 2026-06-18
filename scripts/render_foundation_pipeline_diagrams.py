#!/usr/bin/env python3
"""Restore the three foundation-pipeline presentation photos.

The public foundation-direction visuals intentionally use the real presentation
photos provided by the project owner, not generated concept art. This script
keeps the asset names stable for the website/README while rebuilding enhanced
high-resolution PNGs from the committed source photos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/assets/foundation-pipelines"
SOURCE_DIR = OUT_DIR / "source-photos"

TARGET_WIDTH = 2560


@dataclass(frozen=True)
class PhotoAsset:
    source: str
    output: str
    title: str
    brightness: float
    contrast: float
    color: float
    sharpness: float


PHOTOS = [
    PhotoAsset(
        source="spatial-intelligence-source.jpg",
        output="spatial-intelligence-pipeline.png",
        title="Spatial intelligence presentation photo",
        brightness=1.04,
        contrast=1.18,
        color=1.08,
        sharpness=1.36,
    ),
    PhotoAsset(
        source="human-video-world-model-source.jpg",
        output="human-video-world-model-pipeline.png",
        title="Human-video world-model presentation photo",
        brightness=1.05,
        contrast=1.20,
        color=1.08,
        sharpness=1.34,
    ),
    PhotoAsset(
        source="vision-language-action-source.jpg",
        output="vision-language-action-pipeline.png",
        title="Vision-language-action presentation photo",
        brightness=1.06,
        contrast=1.18,
        color=1.09,
        sharpness=1.34,
    ),
]


def enhance(asset: PhotoAsset) -> Image.Image:
    source_path = SOURCE_DIR / asset.source
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source photo: {source_path}")

    img = Image.open(source_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = ImageOps.autocontrast(img, cutoff=0.35)
    img = ImageEnhance.Brightness(img).enhance(asset.brightness)
    img = ImageEnhance.Contrast(img).enhance(asset.contrast)
    img = ImageEnhance.Color(img).enhance(asset.color)

    if img.width != TARGET_WIDTH:
        scale = TARGET_WIDTH / img.width
        target_size = (TARGET_WIDTH, round(img.height * scale))
        img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Gentle deblur/edge recovery without hallucinating slide text.
    smooth = img.filter(ImageFilter.GaussianBlur(radius=0.55))
    img = Image.blend(smooth, img, 0.68)
    img = ImageEnhance.Sharpness(img).enhance(asset.sharpness)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.15, percent=135, threshold=3))
    return img


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for asset in PHOTOS:
        output = OUT_DIR / asset.output
        image = enhance(asset)
        image.save(output, optimize=True, compress_level=9)
        print(f"{asset.title}: {output} {image.width}x{image.height} {output.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
