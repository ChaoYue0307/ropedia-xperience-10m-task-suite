#!/usr/bin/env python3
"""Build the three foundation-pipeline slide diagrams.

The public foundation-direction visuals intentionally use the direction-slide
sources provided by the project owner, not generated concept art. Clean slide
PNGs are used directly when available; older photo sources are restored only as
fallbacks. The VLA clean slide is a deterministic redraw from the supplied
presentation photo because the latest third clean PNG duplicated the Spatial
slide. The output asset names stay stable for the website, README, and HF
mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/assets/foundation-pipelines"
SOURCE_DIR = OUT_DIR / "source-photos"
SOURCE_SLIDE_DIR = OUT_DIR / "source-slides"

TARGET_WIDTH = 2560
TARGET_HEIGHT = 1920
LIME = (142, 255, 45)
LIME_SOFT = (190, 255, 126)
WHITE = (246, 248, 244)
MUTED = (205, 211, 207)
BLUE = (71, 178, 255)
BG = (0, 0, 0)


@dataclass(frozen=True)
class PhotoAsset:
    source: str
    slide_source: str | None
    output: str
    title: str
    brightness: float
    contrast: float
    color: float
    sharpness: float
    clean_vla_redraw: bool = False


PHOTOS = [
    PhotoAsset(
        source="spatial-intelligence-source.jpg",
        slide_source="spatial-intelligence-slide.png",
        output="spatial-intelligence-pipeline.png",
        title="Spatial intelligence slide diagram",
        brightness=1.04,
        contrast=1.18,
        color=1.08,
        sharpness=1.36,
    ),
    PhotoAsset(
        source="human-video-world-model-source.jpg",
        slide_source="human-video-world-model-slide.png",
        output="human-video-world-model-pipeline.png",
        title="Human-video world-model slide diagram",
        brightness=1.05,
        contrast=1.20,
        color=1.08,
        sharpness=1.34,
    ),
    PhotoAsset(
        source="vision-language-action-source.jpg",
        slide_source=None,
        output="vision-language-action-pipeline.png",
        title="Vision-language-action slide diagram",
        brightness=1.06,
        contrast=1.18,
        color=1.09,
        sharpness=1.34,
        clean_vla_redraw=True,
    ),
]


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    candidates = {
        "regular": [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ],
        "bold": [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
        ],
        "black": [
            "/System/Library/Fonts/Supplemental/Arial Black.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ],
        "mono": [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ],
    }[weight]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, fill=WHITE, weight: str = "regular") -> None:
    draw.text(xy, value, font=font(size, weight), fill=fill)


def fitted_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    max_width: int,
    size: int,
    fill=WHITE,
    weight: str = "regular",
    min_size: int = 36,
) -> None:
    chosen = size
    while chosen > min_size:
        fnt = font(chosen, weight)
        bbox = draw.textbbox((0, 0), value, font=fnt)
        if bbox[2] - bbox[0] <= max_width:
            break
        chosen -= 2
    draw.text(xy, value, font=font(chosen, weight), fill=fill)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    size: int,
    fill=WHITE,
    weight: str = "regular",
) -> None:
    x0, y0, x1, y1 = box
    fnt = font(size, weight)
    bbox = draw.textbbox((0, 0), value, font=fnt)
    x = x0 + (x1 - x0 - (bbox[2] - bbox[0])) / 2
    y = y0 + (y1 - y0 - (bbox[3] - bbox[1])) / 2 - 2
    draw.text((x, y), value, font=fnt, fill=fill)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill=LIME, width: int = 7) -> None:
    draw.line([start, end], fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    head = 34
    spread = 0.55
    points = [
        end,
        (ex - head * math.cos(angle - spread), ey - head * math.sin(angle - spread)),
        (ex - head * math.cos(angle + spread), ey - head * math.sin(angle + spread)),
    ]
    draw.polygon(points, fill=fill)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], outline=LIME, width: int = 3, radius: int = 22) -> None:
    draw.rounded_rectangle(box, radius=radius, outline=outline, width=width)


def render_ropedia_header(draw: ImageDraw.ImageDraw) -> None:
    draw.rounded_rectangle((58, 64, 112, 118), radius=8, fill=WHITE)
    draw.ellipse((77, 82, 93, 98), fill=BG)
    text(draw, (132, 62), "Ropedia", 56, WHITE, "bold")
    draw.line((58, 168, 2502, 168), fill=(190, 196, 190), width=3)


def render_vla_clean_slide() -> Image.Image:
    image = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    render_ropedia_header(draw)

    fitted_text(draw, (58, 240), "Train Vision-Language-Action Models", 2440, 128, WHITE, "black", 84)
    text(draw, (62, 410), "What the robot sees and reads becomes what it does.", 48, MUTED, "regular")

    # Data card.
    card = (60, 540, 790, 1394)
    rounded(draw, card, LIME, 3, 32)
    text(draw, (130, 618), "OUR DATA", 34, LIME, "mono")
    text(draw, (130, 700), "Xperience-10M", 72, WHITE, "black")

    rows = [
        ("video", "Egocentric video"),
        ("body", "Hand & body motion"),
        ("caption", "Language captions"),
    ]
    y = 910
    for kind, label in rows:
        x = 132
        if kind == "video":
            draw.rounded_rectangle((x, y - 16, x + 64, y + 32), radius=8, outline=LIME, width=4)
            draw.polygon([(x + 25, y - 6), (x + 25, y + 22), (x + 48, y + 8)], outline=LIME, fill=None)
        elif kind == "body":
            draw.ellipse((x + 25, y - 28, x + 43, y - 10), outline=LIME, width=4)
            draw.line((x + 34, y - 8, x + 34, y + 30), fill=LIME, width=5)
            draw.line((x + 10, y + 4, x + 58, y + 4), fill=LIME, width=5)
            draw.line((x + 34, y + 30, x + 12, y + 62), fill=LIME, width=5)
            draw.line((x + 34, y + 30, x + 58, y + 62), fill=LIME, width=5)
        else:
            for offset in (0, 13, 26):
                draw.line((x, y - 14 + offset, x + 58, y - 14 + offset), fill=LIME, width=4)
            draw.line((x, y + 26, x + 36, y + 26), fill=LIME, width=4)
        text(draw, (260, y - 30), label, 44, WHITE, "regular")
        y += 145

    arrow(draw, (840, 960), (910, 960), LIME, 7)

    # Model/action flow.
    text(draw, (1038, 556), "VISION + LANGUAGE -> ACTION", 34, LIME, "mono")
    vision_box = (1085, 725, 1238, 805)
    language_box = (1085, 902, 1238, 982)
    rounded(draw, vision_box, WHITE, 3, 12)
    rounded(draw, language_box, WHITE, 3, 12)
    centered_text(draw, vision_box, "Vision", 32, WHITE, "bold")
    centered_text(draw, language_box, "Language", 32, WHITE, "bold")
    text(draw, (1145, 832), "+", 42, WHITE, "bold")
    arrow(draw, (1268, 848), (1355, 848), WHITE, 5)

    # Robot action chunk.
    path = [(1548, 1052), (1662, 946), (1794, 902), (1902, 934), (2010, 858)]
    for i in range(len(path) - 1):
        draw.line((path[i], path[i + 1]), fill=(208, 208, 208), width=3)
    for px, py in path[:-1]:
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=WHITE)
    for px, py in [(1662, 946), (1794, 902), (1902, 934)]:
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=LIME_SOFT)
    gripper_x, gripper_y = path[-1]
    draw.line((gripper_x - 18, gripper_y + 4, gripper_x + 18, gripper_y - 18), fill=WHITE, width=6)
    draw.line((gripper_x - 3, gripper_y - 6, gripper_x - 3, gripper_y - 46), fill=WHITE, width=6)
    draw.line((gripper_x - 3, gripper_y - 20, gripper_x - 24, gripper_y - 44), fill=WHITE, width=5)
    draw.line((gripper_x - 3, gripper_y - 20, gripper_x + 20, gripper_y - 42), fill=WHITE, width=5)
    text(draw, (1500, 1108), "Robot action chunk", 36, WHITE, "regular")

    # Bottom cards.
    left = (60, 1472, 1195, 1788)
    right = (1230, 1472, 2502, 1788)
    rounded(draw, left, LIME, 3, 22)
    rounded(draw, right, LIME, 3, 22)
    draw.ellipse((110, 1545, 230, 1665), outline=(74, 83, 76), width=2)
    text(draw, (255, 1570), "pi_0.7", 58, WHITE, "black")
    draw.line((600, 1528, 600, 1732), fill=(144, 150, 145), width=2)
    text(draw, (655, 1530), "Physical intelligence", 36, WHITE, "regular")
    text(draw, (655, 1604), "generalist", 34, MUTED, "regular")
    text(draw, (655, 1658), "manipulation policy", 34, MUTED, "regular")
    text(draw, (860, 1712), "arXiv:2604.15483", 34, LIME_SOFT, "regular")

    draw.ellipse((1280, 1545, 1400, 1665), outline=(74, 83, 76), width=2)
    for px, py in [(1315, 1614), (1338, 1572), (1365, 1606), (1339, 1641)]:
        draw.ellipse((px - 9, py - 9, px + 9, py + 9), outline=LIME, width=4)
    draw.line((1315, 1614, 1338, 1572, 1365, 1606, 1339, 1641, 1315, 1614), fill=LIME, width=3)
    text(draw, (1448, 1572), "Qwen-VLA", 52, WHITE, "black")
    draw.line((1860, 1528, 1860, 1732), fill=(144, 150, 145), width=2)
    text(draw, (1918, 1530), "Alibaba Qwen", 36, WHITE, "regular")
    text(draw, (1918, 1604), "robot + human-ego", 34, MUTED, "regular")
    text(draw, (1918, 1664), "co-training", 34, MUTED, "regular")
    text(draw, (2210, 1664), "arXiv:2605.30280", 34, LIME_SOFT, "regular")

    return image


def enhance(asset: PhotoAsset) -> Image.Image:
    if asset.clean_vla_redraw:
        return render_vla_clean_slide()

    if asset.slide_source:
        slide_path = SOURCE_SLIDE_DIR / asset.slide_source
        if slide_path.is_file():
            img = Image.open(slide_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            if img.width != TARGET_WIDTH:
                scale = TARGET_WIDTH / img.width
                target_size = (TARGET_WIDTH, round(img.height * scale))
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            return img

    source_path = SOURCE_DIR / asset.source
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source slide/photo for {asset.output}: {source_path}")

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
