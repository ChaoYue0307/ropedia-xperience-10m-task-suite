#!/usr/bin/env python3
"""Build deterministic high-contrast brand assets for the project."""

from __future__ import annotations

import hashlib
import json
import math
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
WHITE = (246, 250, 240)
DEEP = (5, 12, 9)
DEEP_2 = (8, 24, 18)
NEON_GREEN = (205, 255, 105)
NEON_CYAN = (99, 242, 229)
YELLOW = (245, 255, 112)


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


def unit(x: float, y: float) -> tuple[float, float]:
    length = math.hypot(x, y)
    return (x / length, y / length)


def point(
    center: tuple[float, float],
    u: tuple[float, float],
    p: tuple[float, float],
    distance: float,
    offset: float,
) -> tuple[float, float]:
    return (
        center[0] + u[0] * distance + p[0] * offset,
        center[1] + u[1] * distance + p[1] * offset,
    )


def arm_polygon(
    center: tuple[float, float],
    u: tuple[float, float],
    p: tuple[float, float],
    *,
    inner: float,
    shoulder: float,
    outer: float,
    tip: float,
    inner_width: float,
    shoulder_width: float,
    outer_width: float,
) -> list[tuple[float, float]]:
    return [
        point(center, u, p, inner, -inner_width / 2),
        point(center, u, p, shoulder, -shoulder_width / 2),
        point(center, u, p, outer, -outer_width / 2),
        point(center, u, p, tip, 0),
        point(center, u, p, outer, outer_width / 2),
        point(center, u, p, shoulder, shoulder_width / 2),
        point(center, u, p, inner, inner_width / 2),
    ]


def draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], *, fill: tuple[int, int, int, int], width: int) -> None:
    draw.line(points, fill=fill, width=width, joint="curve")


def draw_ring(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: float,
    *,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=fill, width=width)


def make_source_mark(size: int = 1254) -> Image.Image:
    """Create a crisp, small-size-first X/sensor mark.

    The image-generation pass established the visual direction: a bold X-shaped
    sensor mark with lime/cyan accents. This deterministic renderer keeps the
    final logo sharp and reproducible across tiny favicons and large cards.
    """
    aa = 3
    s = size * aa
    canvas = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    center = (s / 2, s / 2)

    # Thin outer orbits are deliberately secondary; the heavy X is the logo.
    orbit_box = (s * 0.18, s * 0.18, s * 0.82, s * 0.82)
    draw.arc(orbit_box, 18, 158, fill=(*NEON_GREEN, 185), width=14 * aa)
    draw.arc(orbit_box, 198, 338, fill=(*NEON_CYAN, 175), width=14 * aa)
    inner_orbit = (s * 0.27, s * 0.27, s * 0.73, s * 0.73)
    draw.arc(inner_orbit, 70, 250, fill=(*YELLOW, 150), width=9 * aa)
    draw.arc(inner_orbit, 250, 430, fill=(*NEON_CYAN, 145), width=9 * aa)

    directions = [unit(-1, -1), unit(1, -1), unit(-1, 1), unit(1, 1)]
    for u in directions:
        p = (-u[1], u[0])
        outline = arm_polygon(
            center,
            u,
            p,
            inner=96 * aa,
            shoulder=326 * aa,
            outer=494 * aa,
            tip=574 * aa,
            inner_width=142 * aa,
            shoulder_width=196 * aa,
            outer_width=172 * aa,
        )
        base = arm_polygon(
            center,
            u,
            p,
            inner=112 * aa,
            shoulder=322 * aa,
            outer=474 * aa,
            tip=538 * aa,
            inner_width=112 * aa,
            shoulder_width=154 * aa,
            outer_width=134 * aa,
        )
        facet = [
            point(center, u, p, 148 * aa, -40 * aa),
            point(center, u, p, 318 * aa, -68 * aa),
            point(center, u, p, 454 * aa, -56 * aa),
            point(center, u, p, 320 * aa, -6 * aa),
        ]

        draw.polygon(outline, fill=(*WHITE, 250))
        draw.polygon(base, fill=(*DEEP, 255))
        draw.polygon(facet, fill=(*DEEP_2, 255))

        # Structural splits make the camera-arm metaphor visible at large sizes
        # without turning into visual noise at favicon scale.
        draw_polyline(
            draw,
            [
                point(center, u, p, 126 * aa, 0),
                point(center, u, p, 305 * aa, 0),
                point(center, u, p, 492 * aa, 0),
            ],
            fill=(*WHITE, 225),
            width=8 * aa,
        )
        draw_polyline(
            draw,
            [point(center, u, p, 176 * aa, 48 * aa), point(center, u, p, 424 * aa, 58 * aa)],
            fill=(*NEON_GREEN, 255),
            width=20 * aa,
        )
        draw_polyline(
            draw,
            [point(center, u, p, 188 * aa, -48 * aa), point(center, u, p, 388 * aa, -60 * aa)],
            fill=(*NEON_CYAN, 230),
            width=12 * aa,
        )

        lens_center = point(center, u, p, 448 * aa, 0)
        draw.ellipse(
            (
                lens_center[0] - 54 * aa,
                lens_center[1] - 54 * aa,
                lens_center[0] + 54 * aa,
                lens_center[1] + 54 * aa,
            ),
            fill=(2, 7, 8, 255),
            outline=(*WHITE, 245),
            width=8 * aa,
        )
        draw_ring(draw, lens_center, 38 * aa, fill=(*NEON_CYAN, 255), width=9 * aa)
        draw_ring(draw, lens_center, 23 * aa, fill=(*NEON_GREEN, 220), width=5 * aa)
        draw.ellipse(
            (
                lens_center[0] - 10 * aa,
                lens_center[1] - 10 * aa,
                lens_center[0] + 10 * aa,
                lens_center[1] + 10 * aa,
            ),
            fill=(*INK, 255),
        )

    # Central data core: large, bright, and readable even after downsampling.
    core_shadow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(core_shadow)
    shadow_draw.ellipse(
        (center[0] - 135 * aa, center[1] - 135 * aa, center[0] + 135 * aa, center[1] + 135 * aa),
        fill=(*NEON_GREEN, 72),
    )
    core_shadow = core_shadow.filter(ImageFilter.GaussianBlur(18 * aa))
    canvas.alpha_composite(core_shadow)
    draw = ImageDraw.Draw(canvas)
    draw.ellipse(
        (center[0] - 104 * aa, center[1] - 104 * aa, center[0] + 104 * aa, center[1] + 104 * aa),
        fill=(*DEEP, 255),
        outline=(*WHITE, 255),
        width=9 * aa,
    )
    draw.ellipse(
        (center[0] - 76 * aa, center[1] - 76 * aa, center[0] + 76 * aa, center[1] + 76 * aa),
        fill=(*NEON_GREEN, 255),
    )
    draw.ellipse(
        (center[0] - 42 * aa, center[1] - 42 * aa, center[0] + 42 * aa, center[1] + 42 * aa),
        fill=(*DEEP, 255),
    )
    draw.ellipse(
        (center[0] - 19 * aa, center[1] - 19 * aa, center[0] + 19 * aa, center[1] + 19 * aa),
        fill=(*INK, 255),
    )

    for angle in range(0, 360, 45):
        radians = math.radians(angle)
        u = (math.cos(radians), math.sin(radians))
        draw_polyline(
            draw,
            [
                (center[0] + u[0] * 118 * aa, center[1] + u[1] * 118 * aa),
                (center[0] + u[0] * 168 * aa, center[1] + u[1] * 168 * aa),
            ],
            fill=(*NEON_GREEN, 220),
            width=8 * aa,
        )

    return canvas.resize((size, size), resample())


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


def add_contrast_halo(image: Image.Image) -> Image.Image:
    """Keep transparent logo marks readable on white README backgrounds."""
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    max_filter_size = max(5, (rgba.width // 24) | 1)
    soft_filter_size = max(3, (rgba.width // 42) | 1)

    hard_alpha = alpha.filter(ImageFilter.MaxFilter(max_filter_size)).point(lambda value: int(value * 0.62))
    soft_alpha = alpha.filter(ImageFilter.MaxFilter(max_filter_size + 4)).filter(
        ImageFilter.GaussianBlur(max(1, rgba.width // 42))
    ).point(lambda value: int(value * 0.36))
    rim_alpha = alpha.filter(ImageFilter.MaxFilter(soft_filter_size)).point(lambda value: int(value * 0.34))

    backing = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    backing.alpha_composite(Image.new("RGBA", rgba.size, (0, 8, 6, 0)))
    shadow = Image.new("RGBA", rgba.size, (0, 8, 6, 255))
    shadow.putalpha(soft_alpha)
    backing.alpha_composite(shadow)
    core = Image.new("RGBA", rgba.size, (1, 12, 8, 255))
    core.putalpha(hard_alpha)
    backing.alpha_composite(core)
    rim = Image.new("RGBA", rgba.size, (78, 180, 124, 255))
    rim.putalpha(rim_alpha)
    backing.alpha_composite(rim)
    backing.alpha_composite(rgba)
    return backing


def fit_on_canvas(image: Image.Image, size: int, *, scale: float = 0.88, contrast_halo: bool = False) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    cropped = alpha_crop(image)
    max_side = int(size * scale)
    cropped.thumbnail((max_side, max_side), resample())
    x = (size - cropped.width) // 2
    y = (size - cropped.height) // 2
    canvas.alpha_composite(cropped, (x, y))
    if contrast_halo:
        canvas = add_contrast_halo(canvas)
    return canvas


def make_dark_tile(mark: Image.Image, size: int) -> Image.Image:
    tile = Image.new("RGBA", (size, size), (*BG, 255))

    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_inset = max(2, size // 18)
    glow_draw.rounded_rectangle(
        (glow_inset, glow_inset, size - glow_inset - 1, size - glow_inset - 1),
        radius=max(5, size // 7),
        fill=(*GREEN, 82),
    )
    glow_draw.ellipse(
        (size * 0.22, size * 0.22, size * 0.78, size * 0.78),
        fill=(*CYAN, 42),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(max(2, size // 10)))
    tile = Image.alpha_composite(tile, glow)

    draw = ImageDraw.Draw(tile)
    border_width = max(2, size // 30)
    draw.rounded_rectangle(
        (1, 1, size - 2, size - 2),
        radius=max(4, size // 8),
        fill=(2, 15, 7, 250),
        outline=(*GREEN, 245),
        width=border_width,
    )
    inner_inset = max(4, border_width + size // 16)
    draw.rounded_rectangle(
        (inner_inset, inner_inset, size - inner_inset - 1, size - inner_inset - 1),
        radius=max(2, size // 10),
        outline=(*CYAN, 92),
        width=max(1, size // 80),
    )
    if size <= 80:
        tile.alpha_composite(make_compact_symbol(size))
    else:
        fitted = fit_on_canvas(mark, size, scale=0.84)
        tile.alpha_composite(fitted)
    return tile


def make_compact_symbol(size: int) -> Image.Image:
    aa = 4
    s = size * aa
    icon = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)

    def sx(value: float) -> float:
        return value / 64 * s

    x_points = [
        (17, 16),
        (32, 26),
        (47, 16),
        (38, 31),
        (47, 48),
        (32, 38),
        (17, 48),
        (26, 31),
    ]
    outer = [(sx(x), sx(y)) for x, y in x_points]
    inner = [(sx(x), sx(y)) for x, y in [(20, 19), (32, 28), (44, 19), (36, 31), (44, 45), (32, 36), (20, 45), (28, 31)]]
    draw.polygon(outer, fill=(*WHITE, 255))
    draw.polygon(inner, fill=(3, 12, 8, 255))
    draw.line([(sx(20), sx(18)), (sx(32), sx(28)), (sx(44), sx(18))], fill=(*NEON_GREEN, 255), width=max(2, int(3.1 * aa)), joint="curve")
    draw.line([(sx(20), sx(46)), (sx(32), sx(36)), (sx(44), sx(46))], fill=(*NEON_GREEN, 255), width=max(2, int(3.1 * aa)), joint="curve")
    draw.line([(sx(18), sx(20)), (sx(28), sx(32)), (sx(18), sx(44))], fill=(*NEON_CYAN, 255), width=max(2, int(2.5 * aa)), joint="curve")
    draw.line([(sx(46), sx(20)), (sx(36), sx(32)), (sx(46), sx(44))], fill=(*NEON_CYAN, 255), width=max(2, int(2.5 * aa)), joint="curve")
    draw.ellipse((sx(26.2), sx(26.2), sx(37.8), sx(37.8)), fill=(*WHITE, 255))
    draw.ellipse((sx(28.2), sx(28.2), sx(35.8), sx(35.8)), fill=(*NEON_GREEN, 255))
    draw.ellipse((sx(30.8), sx(30.8), sx(33.2), sx(33.2)), fill=(*BG, 255))
    return icon.resize((size, size), resample())


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

    panel_glow = Image.new("RGBA", (470, 470), (0, 0, 0, 0))
    panel_glow_draw = ImageDraw.Draw(panel_glow)
    panel_glow_draw.rounded_rectangle((24, 24, 446, 446), radius=38, fill=(*GREEN, 56))
    panel_glow_draw.rounded_rectangle((54, 54, 416, 416), radius=32, fill=(*CYAN, 28))
    panel_glow = panel_glow.filter(ImageFilter.GaussianBlur(22))
    card.alpha_composite(panel_glow, (61, 80))

    panel = Image.new("RGBA", (420, 420), (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rounded_rectangle(
        (0, 0, 419, 419),
        radius=34,
        fill=(3, 15, 8, 238),
        outline=(*GREEN, 238),
        width=3,
    )
    panel_draw.rounded_rectangle(
        (16, 16, 403, 403),
        radius=26,
        outline=(*CYAN, 92),
        width=1,
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
    draw.text((x, 218), "Xperience", font=title_font, fill=GREEN)
    xp_width = int(draw.textlength("Xperience", font=title_font))
    dash_x = x + xp_width + 8
    draw.rounded_rectangle((dash_x, 257, dash_x + 28, 265), radius=4, fill=GREEN)
    draw.text((dash_x + 38, 218), "10M", font=title_font, fill=GREEN)
    draw.text((x, 308), "Task Suite", font=subtitle_font, fill=CYAN)
    draw.text(
        (x, 370),
        "Multimodal embodied-AI task baselines",
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
    BRAND_DIR.mkdir(parents=True, exist_ok=True)
    make_source_mark().save(SOURCE_MARK)
    mark = Image.open(SOURCE_MARK).convert("RGBA")

    fit_on_canvas(mark, 512, contrast_halo=True).save(OUTPUTS["mark_512"])
    fit_on_canvas(mark, 192, contrast_halo=True).save(OUTPUTS["mark_192"])
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
            "kind": "deterministic high-contrast logo mark inspired by a ChatGPT Image direction",
            "prompt_summary": "Small-size-first X-shaped multimodal sensor/camera mark with strong white edges, lime/cyan accents, thick geometry, and reduced micro-detail for favicon clarity.",
        },
        "assets": [
            image_record("logo_mark", SOURCE_MARK, "Transparent source logo mark."),
            image_record("logo_mark_512", OUTPUTS["mark_512"], "512px transparent logo mark."),
            image_record("logo_mark_192", OUTPUTS["mark_192"], "192px transparent logo mark for app manifest use."),
            image_record("favicon_64", OUTPUTS["favicon_64"], "64px dark-tile favicon."),
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
