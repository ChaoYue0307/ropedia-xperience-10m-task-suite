#!/usr/bin/env python3
"""Build deterministic brand assets from the generated logo mark."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = ROOT / "docs/assets/brand"
SOURCE_MARK = BRAND_DIR / "xperience10m-logo-mark.png"
OUTPUT_JSON = ROOT / "docs/data/brand_assets.json"

OUTPUTS = {
    "mark_512": BRAND_DIR / "xperience10m-logo-mark-512.png",
    "mark_192": BRAND_DIR / "xperience10m-logo-mark-192.png",
    "favicon_64": BRAND_DIR / "xperience10m-logo-favicon-64.png",
    "favicon_32": BRAND_DIR / "xperience10m-logo-favicon-32.png",
    "apple_touch": BRAND_DIR / "xperience10m-logo-apple-touch.png",
    "social_card": BRAND_DIR / "xperience10m-logo-social-card.png",
    "root_favicon": ROOT / "docs/favicon.png",
    "root_apple_touch": ROOT / "docs/apple-touch-icon.png",
}

INK = (244, 248, 239)
MUTED = (190, 202, 184)
GREEN = (167, 240, 120)
CYAN = (122, 229, 195)
BG = (2, 5, 2)
PANEL = (5, 14, 8)
LINE = (43, 92, 41)


def resample():
    return getattr(Image, "Resampling", Image).LANCZOS


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_record(name: str, path: Path, role: str) -> dict:
    with Image.open(path) as image:
        return {
            "name": name,
            "path": path.relative_to(ROOT).as_posix(),
            "role": role,
            "exists": path.exists(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "format": image.format,
            "width": int(image.width),
            "height": int(image.height),
            "mode": image.mode,
        }


def alpha_crop(image: Image.Image, padding_ratio: float = 0.08) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError(f"No visible pixels in {SOURCE_MARK}")
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    pad = int(max(width, height) * padding_ratio)
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(rgba.width, right + pad)
    bottom = min(rgba.height, bottom + pad)
    return rgba.crop((left, top, right, bottom))


def fit_on_canvas(image: Image.Image, size: int, *, scale: float = 0.88) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    cropped = alpha_crop(image)
    max_side = int(size * scale)
    cropped.thumbnail((max_side, max_side), resample())
    x = (size - cropped.width) // 2
    y = (size - cropped.height) // 2
    canvas.alpha_composite(cropped, (x, y))
    return canvas


def make_dark_tile(mark: Image.Image, size: int) -> Image.Image:
    tile = Image.new("RGBA", (size, size), (*BG, 255))
    draw = ImageDraw.Draw(tile)
    draw.rounded_rectangle(
        (1, 1, size - 2, size - 2),
        radius=max(3, size // 8),
        fill=(*PANEL, 255),
        outline=(*GREEN, 190),
        width=max(1, size // 40),
    )
    fitted = fit_on_canvas(mark, size, scale=0.82)
    tile.alpha_composite(fitted)
    return tile


def draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    step = 34
    for x in range(0, width, step):
        for y in range(0, height, step):
            if (x // step + y // step) % 3 == 0:
                draw.ellipse((x, y, x + 2, y + 2), fill=(35, 72, 34))


def make_social_card(mark: Image.Image) -> Image.Image:
    width, height = 1200, 630
    card = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(card)
    draw_grid(draw, width, height)

    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse((38, 66, 548, 576), fill=(38, 108, 42, 78))
    glow_draw.ellipse((120, 148, 466, 494), fill=(122, 229, 195, 34))
    glow = glow.filter(ImageFilter.GaussianBlur(34))
    card = Image.alpha_composite(card.convert("RGBA"), glow)

    panel = Image.new("RGBA", (420, 420), (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rounded_rectangle(
        (0, 0, 419, 419),
        radius=34,
        fill=(5, 14, 8, 226),
        outline=(83, 155, 71, 210),
        width=2,
    )
    mark_fit = fit_on_canvas(mark, 390, scale=0.9)
    panel.alpha_composite(mark_fit, (15, 15))
    card.alpha_composite(panel, (86, 105))

    title_font = load_font(64, bold=True)
    subtitle_font = load_font(36, bold=True)
    body_font = load_font(25)
    small_font = load_font(22)
    mono_font = load_font(20, bold=True)

    x = 570
    draw = ImageDraw.Draw(card)
    draw.text((x, 145), "Ropedia", font=title_font, fill=INK)
    draw.text((x, 218), "Xperience-10M", font=title_font, fill=GREEN)
    draw.text((x, 308), "Task Suite", font=subtitle_font, fill=CYAN)
    draw.text(
        (x, 370),
        "Auditable multimodal embodied-AI baselines",
        font=body_font,
        fill=MUTED,
    )

    badge_y = 448
    badges = ["video", "audio", "depth", "pose", "mocap", "IMU", "language"]
    cursor = x
    row = 0
    max_x = width - 86
    for badge in badges:
        label_width = int(draw.textlength(badge, font=mono_font))
        if cursor + label_width + 28 > max_x:
            row += 1
            cursor = x
        y = badge_y + row * 46
        box = (cursor, y, cursor + label_width + 28, y + 36)
        draw.rounded_rectangle(box, radius=9, fill=(7, 22, 12), outline=LINE, width=1)
        draw.text((cursor + 14, y + 8), badge, font=mono_font, fill=INK)
        cursor += label_width + 40

    draw.line((x, 555, width - 86, 555), fill=(78, 151, 72), width=1)
    draw.text((x, 573), "single-sample evidence now | multi-episode fine-tuning next", font=small_font, fill=(155, 170, 149))
    return card.convert("RGB")


def main() -> int:
    if not SOURCE_MARK.exists():
        raise FileNotFoundError(f"Missing source logo mark: {SOURCE_MARK}")
    BRAND_DIR.mkdir(parents=True, exist_ok=True)
    mark = Image.open(SOURCE_MARK).convert("RGBA")

    fit_on_canvas(mark, 512).save(OUTPUTS["mark_512"])
    fit_on_canvas(mark, 192).save(OUTPUTS["mark_192"])
    make_dark_tile(mark, 64).save(OUTPUTS["favicon_64"])
    make_dark_tile(mark, 32).save(OUTPUTS["favicon_32"])
    make_dark_tile(mark, 180).save(OUTPUTS["apple_touch"])
    make_dark_tile(mark, 64).save(OUTPUTS["root_favicon"])
    make_dark_tile(mark, 180).save(OUTPUTS["root_apple_touch"])
    make_social_card(mark).save(OUTPUTS["social_card"], optimize=True, quality=92)

    manifest = {
        "title": "Ropedia Xperience-10M Brand Assets",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": SOURCE_MARK.relative_to(ROOT).as_posix(),
            "kind": "ChatGPT-image-generated logo mark with chroma-key background removed locally",
            "prompt_summary": "X-shaped multimodal camera mark with near-black, lime, cyan, trajectory, and point-cloud styling.",
        },
        "assets": [
            image_record("logo_mark", SOURCE_MARK, "Transparent source logo mark."),
            image_record("logo_mark_512", OUTPUTS["mark_512"], "512px transparent logo mark."),
            image_record("logo_mark_192", OUTPUTS["mark_192"], "192px transparent logo mark for app manifest use."),
            image_record("favicon_64", OUTPUTS["favicon_64"], "64px dark-tile favicon and navigation logo."),
            image_record("favicon_32", OUTPUTS["favicon_32"], "32px dark-tile favicon fallback."),
            image_record("apple_touch", OUTPUTS["apple_touch"], "180px apple-touch icon."),
            image_record("social_card", OUTPUTS["social_card"], "1200x630 Open Graph, Twitter, README, and HF-card logo card."),
            image_record("root_favicon", OUTPUTS["root_favicon"], "Root website favicon PNG."),
            image_record("root_apple_touch", OUTPUTS["root_apple_touch"], "Root website apple-touch icon."),
        ],
        "boundary": "Brand assets are generated presentation artifacts. They do not contain raw Xperience-10M video, HDF5, RRD data, or model weights.",
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for name, path in OUTPUTS.items():
        print(f"{name}: {path} ({path.stat().st_size} bytes)")
    print(f"manifest: {OUTPUT_JSON} ({OUTPUT_JSON.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
