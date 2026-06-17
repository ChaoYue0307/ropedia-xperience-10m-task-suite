#!/usr/bin/env python3
"""Render clear task/training diagrams for the three foundation pipeline tracks."""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/assets/foundation-pipelines"

W, H = 1800, 1012
SCALE = 2
FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size * SCALE)


def sc(v: int | float) -> int:
    return int(round(v * SCALE))


def rounded(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    box = tuple(sc(x) for x in box)
    draw.rounded_rectangle(box, radius=sc(radius), fill=fill, outline=outline, width=sc(width))


def line(draw: ImageDraw.ImageDraw, xy, fill, width=1):
    draw.line(tuple(sc(x) for p in xy for x in p), fill=fill, width=sc(width))


def text(draw: ImageDraw.ImageDraw, xy, value, size, fill, bold=False, anchor=None):
    draw.text((sc(xy[0]), sc(xy[1])), value, font=font(size, bold), fill=fill, anchor=anchor)


def multiline(draw, xy, value, size, fill, bold=False, max_chars=28, leading=1.24):
    y = xy[1]
    for raw in value.split("\n"):
        lines = wrap(raw, max_chars) or [""]
        for line_value in lines:
            text(draw, (xy[0], y), line_value, size, fill, bold)
            y += size * leading
    return y


def arrow(draw, start, end, color):
    line(draw, [start, end], color, 4)
    x1, y1 = start
    x2, y2 = end
    head = 16
    if x2 >= x1:
        pts = [(x2, y2), (x2 - head, y2 - 10), (x2 - head, y2 + 10)]
    else:
        pts = [(x2, y2), (x2 + head, y2 - 10), (x2 + head, y2 + 10)]
    draw.polygon([(sc(x), sc(y)) for x, y in pts], fill=color)


def draw_frame_stack(draw, x, y, accent):
    for i in range(3):
        rounded(draw, (x + i * 18, y + i * 13, x + 150 + i * 18, y + 92 + i * 13), 10, (10, 18, 18, 230), accent, 2)
        line(draw, [(x + 12 + i * 18, y + 70 + i * 13), (x + 138 + i * 18, y + 34 + i * 13)], accent, 2)
        rounded(draw, (x + 22 + i * 18, y + 20 + i * 13, x + 64 + i * 18, y + 52 + i * 13), 5, (80, 160, 140, 170))


def draw_depth_pose(draw, x, y, accent):
    rounded(draw, (x, y, x + 170, y + 105), 12, (8, 14, 20, 235), accent, 2)
    for i in range(8):
        c = (30 + i * 18, 80 + i * 12, 160 + i * 8)
        draw.rectangle((sc(x + 12 + i * 18), sc(y + 14), sc(x + 30 + i * 18), sc(y + 92)), fill=c)
    line(draw, [(x + 108, y + 86), (x + 142, y + 36)], (230, 240, 255), 3)
    line(draw, [(x + 142, y + 36), (x + 156, y + 62)], (230, 240, 255), 3)
    line(draw, [(x + 142, y + 36), (x + 119, y + 48)], (230, 240, 255), 3)


def draw_world_frames(draw, x, y, accent):
    for i, lab in enumerate(["t", "t+1", "t+2"]):
        bx = x + i * 100
        rounded(draw, (bx, y, bx + 78, y + 78), 10, (12, 18, 18, 230), accent if i == 0 else (150, 170, 170), 2)
        if i > 0:
            line(draw, [(bx + 14, y + 52), (bx + 62, y + 26)], accent, 2)
        text(draw, (bx + 16, y + 14), lab, 18, (240, 248, 238), True)
    arrow(draw, (x + 82, y + 39), (x + 96, y + 39), accent)
    arrow(draw, (x + 182, y + 39), (x + 196, y + 39), accent)


def draw_action_tokens(draw, x, y, accent):
    labels = ["look", "reach", "grasp", "move", "place"]
    for i, lab in enumerate(labels):
        bx = x + (i % 3) * 94
        by = y + (i // 3) * 54
        rounded(draw, (bx, by, bx + 82, by + 38), 8, (12, 18, 15, 235), accent, 2)
        text(draw, (bx + 41, by + 11), lab, 16, (238, 248, 232), True, "ma")


def draw_column(draw, box, title, items, accent, icon_kind):
    x1, y1, x2, y2 = box
    rounded(draw, box, 18, (7, 12, 13, 230), (62, 82, 76), 1)
    text(draw, (x1 + 22, y1 + 22), title.upper(), 19, accent, True)

    icon_y = y1 + 62
    if icon_kind == "frames":
        draw_frame_stack(draw, x1 + 22, icon_y, accent)
    elif icon_kind == "depth":
        draw_depth_pose(draw, x1 + 22, icon_y, accent)
    elif icon_kind == "world":
        draw_world_frames(draw, x1 + 22, icon_y + 10, accent)
    elif icon_kind == "tokens":
        draw_action_tokens(draw, x1 + 22, icon_y + 8, accent)
    elif icon_kind == "model":
        cx, cy = x1 + 112, icon_y + 54
        for r in [64, 46, 28]:
            draw.ellipse((sc(cx - r), sc(cy - r), sc(cx + r), sc(cy + r)), outline=accent, width=sc(2))
        for dx, dy in [(-62, -18), (-44, 48), (0, -58), (46, -36), (58, 36), (4, 60)]:
            draw.ellipse((sc(cx + dx - 5), sc(cy + dy - 5), sc(cx + dx + 5), sc(cy + dy + 5)), fill=accent)
    else:
        rounded(draw, (x1 + 22, icon_y, x1 + 178, icon_y + 96), 14, (10, 20, 18, 230), accent, 2)
        for i in range(5):
            line(draw, [(x1 + 42, icon_y + 78 - i * 12), (x1 + 162, icon_y + 30 + i * 5)], accent, 2)

    y = y1 + 198
    for item in items:
        rounded(draw, (x1 + 22, y + 2, x1 + 34, y + 14), 3, accent)
        y = multiline(draw, (x1 + 48, y - 4), item, 18, (224, 232, 224), False, max_chars=27)
        y += 10


def background(accent_a, accent_b):
    img = Image.new("RGB", (W * SCALE, H * SCALE), (3, 7, 7))
    draw = ImageDraw.Draw(img, "RGBA")
    for y in range(H * SCALE):
        ratio = y / (H * SCALE)
        r = int(4 + ratio * 9)
        g = int(8 + ratio * 18)
        b = int(9 + ratio * 15)
        draw.line((0, y, W * SCALE, y), fill=(r, g, b, 255))

    for i in range(0, W, 72):
        line(draw, [(i, 0), (i + 400, H)], (40, 65, 58, 38), 1)
    for j in range(100, H, 115):
        line(draw, [(0, j), (W, j - 90)], (52, 83, 74, 22), 1)

    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    gd.ellipse((sc(1100), sc(-260), sc(2180), sc(760)), fill=accent_a + (34,))
    gd.ellipse((sc(-320), sc(480), sc(600), sc(1300)), fill=accent_b + (26,))
    glow = glow.filter(ImageFilter.GaussianBlur(sc(56)))
    img = Image.alpha_composite(img.convert("RGBA"), glow)
    return img


TRACKS = [
    {
        "file": "spatial-intelligence-pipeline.png",
        "accent": (120, 244, 214),
        "accent2": (204, 255, 160),
        "title": "Spatial Intelligence",
        "subtitle": "Train models to convert video, depth, and pose into scene memory and spatial reasoning.",
        "status": "Pipeline contract: ready. Strong claims need raw depth/pose artifacts plus held-out metrics.",
        "columns": [
            ("inputs", "Inputs", ["multiview RGB + egocentric video", "metric depth and confidence", "camera pose, calibration, SLAM", "object, contact, and language cues"], "depth"),
            ("targets", "Tasks / targets", ["spatial QA and object count", "object permanence across windows", "relative location and retrieval", "pose-aware 3D consistency"], "frames"),
            ("train", "Train models", ["export scene/object memory records", "train spatial-memory encoder", "add geometry-aware QA/retrieval heads", "keep episode-level split discipline"], "model"),
            ("eval", "Evaluate gates", ["held-out episode spatial metrics", "count/relation accuracy", "retrieval rank and consistency", "saved predictions before public claim"], "metrics"),
        ],
    },
    {
        "file": "human-video-world-model-pipeline.png",
        "accent": (255, 196, 116),
        "accent2": (112, 226, 240),
        "title": "Human-Video World Model",
        "subtitle": "Train models to predict future interaction state from observed human video windows.",
        "status": "Partially evidenced by future probes and Cosmos-style branches; visual/latent future metrics remain gated.",
        "columns": [
            ("inputs", "Inputs", ["observed video/audio/sensor window", "hand/body motion and camera pose", "object/contact state", "action and subtask labels"], "world"),
            ("targets", "Tasks / targets", ["next action and next subtask", "future object set", "contact transition", "camera-motion delta or latent future"], "frames"),
            ("train", "Train models", ["Qwen structured future probes", "Cosmos/dynamics branch separately", "latent rollout or reconstruction loss", "no target-side future leakage"], "model"),
            ("eval", "Evaluate gates", ["held-out future-task metrics", "contact and object-set F1", "rollout or latent consistency", "per-episode breakdown and examples"], "metrics"),
        ],
    },
    {
        "file": "vision-language-action-pipeline.png",
        "accent": (164, 255, 159),
        "accent2": (178, 142, 255),
        "title": "Vision-Language-Action",
        "subtitle": "Train models that map egocentric video and language into traceable action chunks.",
        "status": "Feasible but gated by action-token conversion, normalization, retargeting, and policy metrics.",
        "columns": [
            ("inputs", "Inputs", ["egocentric video and captions", "objects, contacts, and procedures", "hand/body motion windows", "subtask labels and language context"], "tokens"),
            ("targets", "Tasks / targets", ["action-token vocabulary", "next action and action chunks", "object-conditioned actions", "contact state and subtask transition"], "frames"),
            ("train", "Train models", ["build action-space converter", "normalize and audit action chunks", "train VLA/policy-compatible head", "track leakage and retargeting reports"], "model"),
            ("eval", "Evaluate gates", ["held-out action metrics", "chunk and next-action accuracy", "object/contact-conditioned scores", "policy card before robot-policy claim"], "metrics"),
        ],
    },
]


def render(track):
    img = background(track["accent"], track["accent2"])
    draw = ImageDraw.Draw(img, "RGBA")
    accent = track["accent"] + (255,)
    accent2 = track["accent2"] + (255,)

    rounded(draw, (42, 38, W - 42, H - 38), 28, (3, 8, 8, 182), (84, 112, 104), 2)
    text(draw, (82, 74), "Ropedia Xperience-10M", 19, (222, 232, 222), True)
    text(draw, (82, 110), track["title"], 48, (248, 252, 246), True)
    multiline(draw, (84, 172), track["subtitle"], 24, (194, 211, 202), False, max_chars=82)
    rounded(draw, (1140, 72, 1688, 152), 18, (10, 18, 16, 220), accent2, 2)
    text(draw, (1165, 98), "Direction -> task targets -> model training -> evaluation", 22, (246, 255, 239), True)

    col_w = 390
    gap = 34
    start_x = 82
    y1 = 250
    y2 = 808
    centers = []
    for i, (_, title_value, items, icon_kind) in enumerate(track["columns"]):
        x1 = start_x + i * (col_w + gap)
        draw_column(draw, (x1, y1, x1 + col_w, y2), title_value, items, accent, icon_kind)
        centers.append((x1 + col_w, (y1 + y2) / 2))
        if i > 0:
            prev_x = start_x + (i - 1) * (col_w + gap) + col_w
            arrow(draw, (prev_x + 10, (y1 + y2) / 2), (x1 - 12, (y1 + y2) / 2), accent)

    rounded(draw, (82, 846, W - 82, 930), 18, (8, 14, 12, 230), accent2, 2)
    text(draw, (110, 872), "Claim boundary", 21, accent2, True)
    multiline(draw, (310, 868), track["status"], 21, (232, 240, 232), False, max_chars=104)

    out = img.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.save(OUT_DIR / track["file"], optimize=True, quality=95)


def main():
    for track in TRACKS:
        render(track)
    print("Rendered foundation pipeline diagrams:")
    for track in TRACKS:
        path = OUT_DIR / track["file"]
        print(f"- {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
