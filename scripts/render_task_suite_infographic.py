#!/usr/bin/env python3
"""
Render a polished Ropedia Xperience-10M 12-task infographic.

The task names, inputs, and metrics are read from
results/episode_task_suite/summary_report.json. The output is a deterministic
PNG rendered from HTML/CSS so the labels stay legible and reviewable.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/episode_task_suite/summary_report.json"
DEFAULT_BASE = ROOT / "docs/assets/task_suite_infographic_base.png"
DEFAULT_SAMPLE_DIR = ROOT.parent / "data/sample/xperience-10m-sample"
DEFAULT_OUTPUT = ROOT / "docs/assets/task_suite_infographic.png"
CANVAS_WIDTH = 1800
CANVAS_HEIGHT = 1650
THUMB_WIDTH = 420
THUMB_HEIGHT = 160


GROUPS = [
    {
        "name": "Label + State",
        "tone": "teal",
        "color": "#197d83",
        "soft": "#e8f4f3",
        "tasks": [
            ("timeline_action", "supervised"),
            ("timeline_subtask", "supervised"),
            ("next_action", "supervised"),
        ],
    },
    {
        "name": "Prediction + Reconstruction",
        "tone": "blue",
        "color": "#1f6c9f",
        "soft": "#e8f1fb",
        "tasks": [
            ("hand_trajectory_forecast", "forecast"),
            ("modality_reconstruction", "forecast"),
            ("contact_prediction", "supervised"),
        ],
    },
    {
        "name": "Grounding + Retrieval",
        "tone": "amber",
        "color": "#9b6516",
        "soft": "#fbf3df",
        "tasks": [
            ("caption_grounding", "retrieval"),
            ("cross_modal_retrieval", "retrieval"),
            ("object_relevance", "supervised"),
        ],
    },
    {
        "name": "Temporal Diagnostics",
        "tone": "red",
        "color": "#b0443e",
        "soft": "#fdeceb",
        "tasks": [
            ("transition_detection", "diagnostic"),
            ("temporal_order", "diagnostic"),
            ("misalignment_detection", "diagnostic"),
        ],
    },
]

MODALITIES = [
    ("video", "6 camera streams", "fisheye + stereo"),
    ("audio", "AAC stream in MP4", "documented, not featurized"),
    ("depth", "depth + confidence", "spatial geometry"),
    ("pose / SLAM", "camera trajectory", "position + orientation"),
    ("motion capture", "body + hand joints", "mocap features"),
    ("inertial", "accel + gyro", "wearable motion"),
    ("language", "objects + captions", "annotation text"),
]

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def image_data_uri(image, fmt: str = "PNG", quality: int = 92) -> str:
    buffer = io.BytesIO()
    save_kwargs = {"format": fmt}
    if fmt.upper() in {"JPEG", "JPG"}:
        save_kwargs.update({"quality": quality, "optimize": True})
    image.save(buffer, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "jpeg" if fmt.upper() in {"JPEG", "JPG"} else "png"
    return f"data:image/{mime};base64,{encoded}"


def make_canvas(size=(THUMB_WIDTH, THUMB_HEIGHT), color=(255, 254, 253)):
    from PIL import Image

    return Image.new("RGB", size, color)


def fit_image(image, size=(THUMB_WIDTH, THUMB_HEIGHT)):
    from PIL import ImageOps

    return ImageOps.fit(image.convert("RGB"), size, method=3, centering=(0.5, 0.5))


def read_video_frame(video_path: Path, frame_index: int = 2400):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total:
        frame_index = max(0, min(frame_index, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def draw_label(draw, xy, text, fill=(31, 36, 33), size=18):
    from PIL import ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=fill, font=font)


def video_thumb(sample_dir: Path) -> str:
    from PIL import Image, ImageDraw

    fish = fit_image(read_video_frame(sample_dir / "fisheye_cam0.mp4", 2450), (194, THUMB_HEIGHT))
    stereo_path = sample_dir / "stereo_left.mp4"
    stereo = fit_image(read_video_frame(stereo_path, 2450), (194, THUMB_HEIGHT)) if stereo_path.exists() else fish.copy()
    canvas = make_canvas()
    canvas.paste(fish, (0, 0))
    canvas.paste(stereo, (226, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((188, 0, 232, THUMB_HEIGHT), radius=0, fill=(251, 250, 247, 235))
    draw_label(draw, (194, 16), "fisheye", fill=(255, 255, 255), size=14)
    draw_label(draw, (240, 16), "stereo", fill=(255, 255, 255), size=14)
    return image_data_uri(canvas, "JPEG")


def colorize(values):
    import numpy as np

    stops = np.array([
        [26, 35, 126],
        [36, 123, 160],
        [68, 170, 122],
        [238, 190, 76],
        [197, 79, 51],
    ], dtype=np.float32)
    x = np.clip(values, 0, 1)
    scaled = x * (len(stops) - 1)
    lo = np.floor(scaled).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = scaled - lo
    rgb = stops[lo] * (1 - frac[..., None]) + stops[hi] * frac[..., None]
    return rgb.astype("uint8")


def depth_thumb(h5) -> str:
    import numpy as np
    from PIL import Image, ImageDraw

    frame = np.array(h5["depth/depth"][2450], dtype=np.float32)
    valid = np.isfinite(frame)
    lo, hi = np.percentile(frame[valid], [3, 97])
    norm = (frame - lo) / max(hi - lo, 1e-6)
    rgb = colorize(norm)
    depth = fit_image(Image.fromarray(rgb), (204, THUMB_HEIGHT))
    conf = np.array(h5["depth/confidence"][2450], dtype=np.uint8)
    conf_img = Image.fromarray(conf, mode="L").convert("RGB")
    conf_img = fit_image(conf_img, (204, THUMB_HEIGHT))
    canvas = make_canvas()
    canvas.paste(depth, (0, 0))
    canvas.paste(conf_img, (216, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((0, 0, 116, 28), radius=6, fill=(31, 36, 33, 150))
    draw.rounded_rectangle((216, 0, 350, 28), radius=6, fill=(31, 36, 33, 150))
    draw_label(draw, (10, 6), "depth", fill=(255, 255, 255), size=14)
    draw_label(draw, (226, 6), "confidence", fill=(255, 255, 255), size=14)
    return image_data_uri(canvas, "JPEG")


def audio_thumb(sample_dir: Path) -> str:
    import numpy as np
    from PIL import ImageDraw

    canvas = make_canvas()
    draw = ImageDraw.Draw(canvas, "RGBA")
    try:
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                "45",
                "-t",
                "6",
                "-i",
                str(sample_dir / "fisheye_cam0.mp4"),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "s16le",
                "pipe:1",
            ],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            raise RuntimeError("empty audio stream")
        samples = samples / max(float(np.max(np.abs(samples))), 1.0)
        bins = 180
        trimmed = samples[: bins * max(1, len(samples) // bins)]
        chunks = np.array_split(trimmed, bins)
        rms = np.array([np.sqrt(np.mean(chunk * chunk)) if len(chunk) else 0.0 for chunk in chunks])
        waveform = np.array([float(np.mean(chunk)) if len(chunk) else 0.0 for chunk in chunks])
        for i, value in enumerate(rms):
            x = 18 + i / max(bins - 1, 1) * (THUMB_WIDTH - 36)
            h = 8 + np.clip(value * 86, 0, 86)
            draw.line((x, 126, x, 126 - h), fill=(31, 108, 159, 150), width=2)
        points = []
        for i, value in enumerate(waveform):
            x = 18 + i / max(bins - 1, 1) * (THUMB_WIDTH - 36)
            y = 74 - np.clip(value, -1, 1) * 42
            points.append((x, y))
        draw.line(points, fill=(155, 101, 22, 210), width=2)
    except Exception:
        for i in range(48):
            x = 22 + i * 8
            h = 16 + (i % 7) * 7
            draw.rounded_rectangle((x, 128 - h, x + 4, 128), radius=2, fill=(31, 108, 159, 150))
    draw_label(draw, (16, 12), "AAC audio waveform", fill=(31, 36, 33), size=17)
    return image_data_uri(canvas, "PNG")


def normalize_points(points, width, height, pad=16):
    import numpy as np

    xy = points[:, :2].copy()
    lo = np.percentile(xy, 2, axis=0)
    hi = np.percentile(xy, 98, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    norm = (xy - lo) / span
    norm = np.clip(norm, 0, 1)
    norm[:, 1] = 1 - norm[:, 1]
    out = np.empty_like(norm)
    out[:, 0] = pad + norm[:, 0] * (width - pad * 2)
    out[:, 1] = pad + norm[:, 1] * (height - pad * 2)
    return out


def slam_thumb(h5) -> str:
    import numpy as np
    from PIL import ImageDraw

    canvas = make_canvas()
    draw = ImageDraw.Draw(canvas, "RGBA")
    points = np.array(h5["slam/point_cloud"], dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > 2600:
        points = points[np.linspace(0, len(points) - 1, 2600).astype(int)]
    xy = normalize_points(points[:, [0, 2, 1]], THUMB_WIDTH, THUMB_HEIGHT)
    z = points[:, 1]
    z_norm = (z - np.percentile(z, 2)) / max(np.percentile(z, 98) - np.percentile(z, 2), 1e-6)
    colors = colorize(z_norm)
    for (x, y), color in zip(xy, colors):
        draw.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=tuple(color.tolist()) + (165,))
    traj = np.array(h5["slam/trans_xyz"][:2450:36], dtype=np.float64)
    traj_xy = normalize_points(traj[:, [0, 2, 1]], THUMB_WIDTH, THUMB_HEIGHT)
    for a, b in zip(traj_xy[:-1], traj_xy[1:]):
        draw.line((a[0], a[1], b[0], b[1]), fill=(31, 108, 159, 190), width=2)
    draw_label(draw, (16, 14), "camera pose + SLAM map", fill=(31, 36, 33), size=17)
    return image_data_uri(canvas, "PNG")


def imu_thumb(h5) -> str:
    import numpy as np
    from PIL import ImageDraw

    canvas = make_canvas()
    draw = ImageDraw.Draw(canvas, "RGBA")
    key_idx = int(h5["imu/keyframe_indices"][2450])
    accel = np.array(h5["imu/accel_xyz"][max(0, key_idx - 220): key_idx + 220], dtype=np.float64)
    gyro = np.array(h5["imu/gyro_xyz"][max(0, key_idx - 220): key_idx + 220], dtype=np.float64)
    series = [accel[:, 0], accel[:, 1], accel[:, 2], gyro[:, 0], gyro[:, 1], gyro[:, 2]]
    colors = [(31, 108, 159), (52, 101, 56), (176, 68, 62), (155, 101, 22), (46, 119, 117), (96, 109, 128)]
    for row in range(4):
        y = 26 + row * 33
        draw.line((18, y, THUMB_WIDTH - 18, y), fill=(228, 222, 212, 180), width=1)
    for values, color in zip(series, colors):
        values = values[:420]
        if len(values) < 2:
            continue
        lo, hi = np.percentile(values, [3, 97])
        norm = (values - lo) / max(hi - lo, 1e-6)
        pts = []
        for i, v in enumerate(norm):
            x = 18 + i / max(len(values) - 1, 1) * (THUMB_WIDTH - 36)
            y = 138 - np.clip(v, 0, 1) * 112
            pts.append((x, y))
        draw.line(pts, fill=color + (200,), width=2)
    draw_label(draw, (16, 12), "inertial accel / gyro", fill=(31, 36, 33), size=17)
    return image_data_uri(canvas, "PNG")


def mocap_thumb(h5) -> str:
    import numpy as np
    from PIL import ImageDraw

    canvas = make_canvas()
    draw = ImageDraw.Draw(canvas, "RGBA")
    body = np.array(h5["full_body_mocap/keypoints"][2450], dtype=np.float32)
    left = np.array(h5["hand_mocap/left_joints_3d"][2450], dtype=np.float32)
    right = np.array(h5["hand_mocap/right_joints_3d"][2450], dtype=np.float32)
    all_points = np.concatenate([body, left, right], axis=0)
    lo = np.percentile(all_points[:, :2], 2, axis=0)
    hi = np.percentile(all_points[:, :2], 98, axis=0)
    span = np.maximum(hi - lo, 1e-6)

    def project(points, x_offset, width):
        xy = (points[:, :2] - lo) / span
        xy[:, 1] = 1 - xy[:, 1]
        xy[:, 0] = x_offset + xy[:, 0] * width
        xy[:, 1] = 26 + xy[:, 1] * 108
        return xy

    body_xy = project(body, 18, 165)
    for x, y in body_xy:
        draw.ellipse((x - 2.4, y - 2.4, x + 2.4, y + 2.4), fill=(52, 101, 56, 175))
    for a, b in zip(body_xy[:-1], body_xy[1:]):
        draw.line((a[0], a[1], b[0], b[1]), fill=(52, 101, 56, 70), width=1)

    for points, x_offset, color in [(left, 218, (31, 108, 159)), (right, 314, (155, 101, 22))]:
        xy = project(points, x_offset, 82)
        for a, b in HAND_EDGES:
            draw.line((xy[a][0], xy[a][1], xy[b][0], xy[b][1]), fill=color + (180,), width=2)
        for x, y in xy:
            draw.ellipse((x - 2.4, y - 2.4, x + 2.4, y + 2.4), fill=color + (220,))
    draw_label(draw, (16, 12), "body + hand mocap", fill=(31, 36, 33), size=17)
    return image_data_uri(canvas, "PNG")


def text_thumb(h5) -> str:
    from PIL import ImageDraw

    raw = h5["caption"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    data = json.loads(raw)
    segment = data["segments"][0]
    objects = sorted({item for values in segment.get("objects", {}).values() for item in values})[:5]
    actions = [a.get("label", "") for a in segment.get("Current Action", [])][:2]
    canvas = make_canvas()
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_label(draw, (16, 13), "language annotation", fill=(31, 36, 33), size=17)
    y = 46
    for label in objects:
        draw.rounded_rectangle((16, y, 16 + 20 + len(label) * 8, y + 24), radius=6, fill=(251, 243, 219, 230), outline=(226, 200, 144, 255))
        draw_label(draw, (26, y + 5), label, fill=(83, 74, 56), size=12)
        y += 30
    x = 184
    y = 48
    for action in actions:
        wrapped = action[:32] + ("..." if len(action) > 32 else "")
        draw.rounded_rectangle((x, y, THUMB_WIDTH - 16, y + 36), radius=7, fill=(232, 244, 243, 230), outline=(169, 204, 202, 255))
        draw_label(draw, (x + 10, y + 10), wrapped, fill=(31, 36, 33), size=12)
        y += 44
    return image_data_uri(canvas, "PNG")


def load_sample_thumbnails(sample_dir: Path | None) -> dict[str, str]:
    if sample_dir is None or not sample_dir.exists():
        return {}
    hdf5_path = sample_dir / "annotation.hdf5"
    required = [sample_dir / "fisheye_cam0.mp4", hdf5_path]
    if not all(path.exists() for path in required):
        return {}
    try:
        import h5py

        thumbnails = {"video": video_thumb(sample_dir), "audio": audio_thumb(sample_dir)}
        with h5py.File(hdf5_path, "r") as h5:
            thumbnails.update({
                "depth": depth_thumb(h5),
                "pose / SLAM": slam_thumb(h5),
                "motion capture": mocap_thumb(h5),
                "inertial": imu_thumb(h5),
                "language": text_thumb(h5),
            })
        return thumbnails
    except Exception as exc:
        print(f"Warning: could not build sample modality thumbnails: {exc}")
        return {}


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def metric_for(task_name: str, metrics: dict) -> tuple[str, str]:
    if task_name == "hand_trajectory_forecast":
        return "MPJPE", fmt(metrics["mpjpe"])
    if task_name == "cross_modal_retrieval":
        return "top-5", fmt(metrics["top5_accuracy"])
    if task_name == "caption_grounding":
        return "MRR", fmt(metrics["mrr"])
    if task_name == "object_relevance":
        return "micro-F1", fmt(metrics["micro_f1"])
    if task_name == "modality_reconstruction":
        return "R2", fmt(metrics["r2"])
    if task_name in {"temporal_order", "misalignment_detection"}:
        return "F1", fmt(metrics["f1"])
    if "macro_f1" in metrics:
        return "macro-F1", fmt(metrics["macro_f1"])
    if "accuracy" in metrics:
        return "accuracy", fmt(metrics["accuracy"])
    raise KeyError(f"No main metric configured for {task_name}")


def short_io(task_name: str, metrics: dict) -> str:
    custom = {
        "timeline_action": "all featurized modalities -> action label",
        "timeline_subtask": "all featurized modalities -> subtask label",
        "transition_detection": "all featurized modalities -> boundary vs steady",
        "next_action": "window at t -> action at t+20 frames",
        "hand_trajectory_forecast": "all featurized modalities -> future hand joints",
        "contact_prediction": "non-contact modalities -> contact state",
        "object_relevance": "non-caption feature blocks -> relevant objects",
        "caption_grounding": "text query -> matching sensor window",
        "cross_modal_retrieval": "motion / IMU / camera -> depth / video match",
        "modality_reconstruction": "motion / IMU / camera -> depth / video vector",
        "temporal_order": "two adjacent windows -> correct order",
        "misalignment_detection": "motion + visual pair -> aligned or shifted",
    }
    return custom.get(task_name, metrics.get("input", ""))


def task_card(task_name: str, kind: str, metrics: dict, group: dict, index: int, neural_metrics: dict | None = None) -> str:
    label, value = metric_for(task_name, metrics)
    neural_html = ""
    if neural_metrics and "error" not in neural_metrics:
        neural_label, neural_value = metric_for(task_name, neural_metrics)
        neural_html = f"""
        <div class="metric neural">
          <span>NN {html.escape(neural_label)}</span>
          <strong>{html.escape(neural_value)}</strong>
        </div>
        """
    io = short_io(task_name, metrics)
    return f"""
      <article class="task-card" style="--accent:{group['color']};--soft:{group['soft']};">
        <div class="task-meta">
          <span class="index">{index:02d}</span>
          <span class="kind">{html.escape(kind)}</span>
        </div>
        <h3>{html.escape(task_name)}</h3>
        <p>{html.escape(io)}</p>
        <div class="metric">
          <span>min {html.escape(label)}</span>
          <strong>{html.escape(value)}</strong>
        </div>
        {neural_html}
      </article>
    """


def modality_card(name: str, line_one: str, line_two: str, index: int, thumbnail: str | None) -> str:
    thumb_html = ""
    if thumbnail:
        thumb_html = f'<div class="modality-thumb"><img src="{thumbnail}" alt=""></div>'
    return f"""
      <article class="modality">
        {thumb_html}
        <div class="modality-index">{index:02d}</div>
        <h3>{html.escape(name)}</h3>
        <p>{html.escape(line_one)}</p>
        <span>{html.escape(line_two)}</span>
      </article>
    """


def build_html(summary: dict, base_image: Path | None, sample_dir: Path | None) -> str:
    suite = summary["tasks"]
    neural_suite = summary.get("neural_tasks", {})
    thumbnails = load_sample_thumbnails(sample_dir)
    base_layer = ""
    if base_image is not None and base_image.exists():
        base_layer = f'<div class="image-background" style="background-image:url(\'{base_image.resolve().as_uri()}\');"></div>'
    stats = [
        (f"{summary['num_frames']:,}", "frames"),
        (f"{summary['num_windows']:,}", "windows"),
        (f"{summary['feature_dim']:,}", "features"),
        (f"{len(suite)}+{len(neural_suite)}", "min + NN tasks"),
        ("70/30", "chronological split"),
    ]
    stats_html = "".join(
        f"<div class=\"stat\"><strong>{html.escape(value)}</strong><span>{html.escape(label)}</span></div>"
        for value, label in stats
    )
    modalities_html = "".join(
        modality_card(name, line_one, line_two, index, thumbnails.get(name))
        for index, (name, line_one, line_two) in enumerate(MODALITIES, start=1)
    )

    task_index = 1
    families = []
    for group in GROUPS:
        cards = []
        for task_name, kind in group["tasks"]:
            cards.append(task_card(task_name, kind, suite[task_name], group, task_index, neural_suite.get(task_name)))
            task_index += 1
        families.append(
            f"""
            <section class="family" style="--accent:{group['color']};--soft:{group['soft']};">
              <div class="family-head">
                <span>{html.escape(group['tone'])}</span>
                <h2>{html.escape(group['name'])}</h2>
              </div>
              <div class="family-cards">{''.join(cards)}</div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width={CANVAS_WIDTH}, initial-scale=1">
  <title>Xperience-10M 12-Task Episode Suite Infographic</title>
  <style>
    * {{ box-sizing: border-box; }}
    html,
    body {{
      margin: 0;
      width: {CANVAS_WIDTH}px;
      height: {CANVAS_HEIGHT}px;
      background: #fbfaf7;
    }}
    body {{
      font-family: "Avenir Next", "SF Pro Display", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2421;
      text-rendering: optimizeLegibility;
    }}
    .canvas {{
      position: relative;
      width: {CANVAS_WIDTH}px;
      height: {CANVAS_HEIGHT}px;
      overflow: hidden;
      padding: 54px 64px 44px;
      background:
        radial-gradient(circle at 9% 6%, rgba(31,108,159,0.13), transparent 20%),
        radial-gradient(circle at 90% 9%, rgba(155,101,22,0.10), transparent 22%),
        linear-gradient(90deg, rgba(68,55,38,0.035) 1px, transparent 1px),
        linear-gradient(0deg, rgba(68,55,38,0.027) 1px, transparent 1px),
        #fbfaf7;
      background-size: auto, auto, 54px 54px, 54px 54px, auto;
    }}
    .image-background {{
      position: absolute;
      inset: 0;
      background-position: center;
      background-repeat: no-repeat;
      background-size: cover;
      opacity: 0.30;
      filter: saturate(0.85) contrast(0.98);
    }}
    .content {{
      position: relative;
      z-index: 1;
    }}
    .header {{
      display: grid;
      grid-template-columns: 1.25fr 0.75fr;
      gap: 44px;
      align-items: end;
      padding-bottom: 30px;
      border-bottom: 1px solid #e4ded4;
    }}
    .kicker {{
      display: inline-flex;
      align-items: center;
      gap: 12px;
      color: #5f625d;
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .kicker::before {{
      content: "";
      width: 44px;
      height: 1px;
      background: #1f2421;
    }}
    h1 {{
      margin: 18px 0 0;
      max-width: 930px;
      font-size: 72px;
      line-height: 0.95;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 18px 0 0;
      max-width: 900px;
      color: #5f625d;
      font-size: 23px;
      line-height: 1.35;
      font-weight: 520;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }}
    .stat {{
      min-height: 78px;
      padding: 14px 15px;
      border: 1px solid #e4ded4;
      background: rgba(255,254,253,0.76);
      border-radius: 10px;
    }}
    .stat strong {{
      display: block;
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 25px;
      line-height: 1;
      font-variant-numeric: tabular-nums;
    }}
    .stat span {{
      display: block;
      margin-top: 8px;
      color: #6f716c;
      font-size: 13px;
      line-height: 1.15;
    }}
    .section-label {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin: 28px 0 14px;
      color: #5f625d;
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .section-label span:last-child {{
      color: #7e817b;
      text-transform: none;
      letter-spacing: 0;
      font-family: inherit;
    }}
    .modalities {{
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 12px;
    }}
    .modality {{
      min-height: 204px;
      padding: 11px 12px 14px;
      border: 1px solid #e4ded4;
      background: rgba(255,254,253,0.84);
      border-radius: 12px;
    }}
    .modality-thumb {{
      height: 86px;
      overflow: hidden;
      border: 1px solid #eee9e1;
      border-radius: 9px;
      background: #f5f1e9;
    }}
    .modality-thumb img {{
      display: block;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }}
    .modality-index,
    .index {{
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-variant-numeric: tabular-nums;
    }}
    .modality-index {{
      color: #8a8072;
      font-size: 12px;
      margin-top: 10px;
    }}
    .modality h3 {{
      margin: 8px 0 0;
      font-size: 19px;
      line-height: 1;
      text-transform: uppercase;
    }}
    .modality p {{
      margin: 9px 0 0;
      color: #4f565f;
      font-size: 14px;
      font-weight: 650;
    }}
    .modality span {{
      display: block;
      margin-top: 5px;
      color: #7a7d77;
      font-size: 13px;
    }}
    .shared-band {{
      display: grid;
      grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr;
      gap: 12px;
      align-items: center;
      margin-top: 20px;
      padding: 14px;
      border: 1px solid #e4ded4;
      background: rgba(245,241,233,0.82);
      border-radius: 12px;
    }}
    .step {{
      min-height: 62px;
      padding: 13px 15px;
      background: #fffefd;
      border: 1px solid #eee9e1;
      border-radius: 9px;
    }}
    .step strong {{
      display: block;
      font-size: 17px;
      line-height: 1.1;
    }}
    .step span {{
      display: block;
      margin-top: 5px;
      color: #6f716c;
      font-size: 13px;
    }}
    .arrow {{
      color: #938a7d;
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 22px;
    }}
    .families {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 20px;
      margin-top: 26px;
    }}
    .family {{
      padding: 17px;
      border: 1px solid color-mix(in srgb, var(--accent) 24%, #e4ded4);
      background: rgba(255,254,253,0.82);
      border-radius: 16px;
    }}
    .family-head {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      min-height: 78px;
      padding-bottom: 14px;
      border-bottom: 1px solid color-mix(in srgb, var(--accent) 18%, #eee9e1);
    }}
    .family-head span {{
      color: var(--accent);
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .family-head h2 {{
      margin: 0;
      color: var(--accent);
      font-size: 29px;
      line-height: 1.02;
      text-align: right;
    }}
    .family-cards {{
      display: grid;
      gap: 13px;
      margin-top: 15px;
    }}
    .task-card {{
      min-height: 168px;
      padding: 17px 18px;
      border: 1px solid color-mix(in srgb, var(--accent) 22%, #e4ded4);
      background: linear-gradient(180deg, #fffefd, color-mix(in srgb, var(--soft) 45%, #fffefd));
      border-radius: 13px;
    }}
    .task-meta {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .index {{
      color: #8a8072;
      font-size: 12px;
    }}
    .kind {{
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 9px;
      border-radius: 6px;
      border: 1px solid color-mix(in srgb, var(--accent) 30%, #ffffff);
      color: var(--accent);
      background: rgba(255,255,255,0.72);
      text-transform: uppercase;
      font-size: 11px;
      line-height: 1;
      font-weight: 830;
    }}
    .task-card h3 {{
      margin: 12px 0 0;
      color: #111827;
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 21px;
      line-height: 1.18;
      overflow-wrap: anywhere;
    }}
    .task-card p {{
      margin: 11px 0 0;
      min-height: 39px;
      color: #4f565f;
      font-size: 15px;
      line-height: 1.28;
      font-weight: 560;
    }}
    .metric {{
      display: inline-flex;
      align-items: baseline;
      gap: 10px;
      margin-top: 10px;
      min-height: 32px;
      padding: 7px 10px;
      border-radius: 8px;
      border: 1px solid color-mix(in srgb, var(--accent) 32%, #ffffff);
      background: rgba(255,255,255,0.82);
    }}
    .metric.neural {{
      margin-left: 8px;
      border-color: rgba(31,36,33,0.18);
      background: rgba(245,241,233,0.82);
    }}
    .metric span {{
      color: #64748b;
      font-size: 13px;
      font-weight: 760;
    }}
    .metric strong {{
      color: var(--accent);
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 20px;
      line-height: 1;
      font-weight: 860;
      font-variant-numeric: tabular-nums;
    }}
    .footer {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 32px;
      margin-top: 22px;
      padding-top: 20px;
      border-top: 1px solid #e4ded4;
      color: #5f625d;
      font-size: 18px;
      line-height: 1.35;
      font-weight: 620;
    }}
    .footer code {{
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      color: #1f2421;
      background: #f5f1e9;
      border: 1px solid #e4ded4;
      border-radius: 7px;
      padding: 6px 9px;
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <main class="canvas" aria-label="Ropedia Xperience-10M 12-task suite infographic">
    {base_layer}
    <div class="content">
    <header class="header">
      <div>
        <div class="kicker">verified single-episode task suite</div>
        <h1>Ropedia Xperience-10M 12-task suite</h1>
        <p class="subtitle">A clean map from synchronized multimodal windows to 12 auditable task heads, comparing minimal heads with neural MLP results. Next TODO: Qwen3-Omni fine-tuning plus sensor-bridge evaluation.</p>
      </div>
      <div class="stats">{stats_html}</div>
    </header>

    <div class="section-label">
      <span>Xperience-10M modalities</span>
      <span>audio is present in the sample MP4 stream; the current 8,378-d baseline manifest does not featurize it</span>
    </div>
    <section class="modalities">{modalities_html}</section>

    <section class="shared-band" aria-label="shared processing contract">
      <div class="step"><strong>raw public episode</strong><span>video, audio, depth, pose, mocap, IMU, language</span></div>
      <div class="arrow">-></div>
      <div class="step"><strong>20-frame windows</strong><span>stride 5, chronological order</span></div>
      <div class="arrow">-></div>
      <div class="step"><strong>8,378-d vector</strong><span>current manifest excludes audio features</span></div>
      <div class="arrow">-></div>
      <div class="step"><strong>12 minimal + NN heads</strong><span>softmax/ridge/logistic plus PyTorch MLP</span></div>
    </section>

    <section class="families">{''.join(families)}</section>

    <footer class="footer">
      <span>Single public sample episode: useful for pipeline validation and task design, not cross-episode generalization.</span>
      <code>results/episode_task_suite/summary_report.json</code>
    </footer>
    </div>
  </main>
</body>
</html>
"""


def render_html(html_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "npx",
            "--yes",
            "playwright",
            "screenshot",
            "--full-page",
            f"--viewport-size={CANVAS_WIDTH},{CANVAS_HEIGHT}",
            html_path.resolve().as_uri(),
            str(output_path),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-image", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--no-export", action="store_true", help="Only write the HTML used to render the image.")
    args = parser.parse_args()

    summary = load_summary()
    html_text = build_html(summary, args.base_image, args.sample_dir)
    if args.html is None:
        with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=False) as handle:
            handle.write(html_text)
            html_path = Path(handle.name)
    else:
        html_path = args.html
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_text, encoding="utf-8")

    if not args.no_export:
        render_html(html_path, args.output)
        print(f"Wrote image: {args.output}")
    print(f"Wrote render HTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
