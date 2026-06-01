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
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/episode_task_suite/summary_report.json"
DEFAULT_BASE = ROOT / "docs/assets/task_suite_infographic_base.png"
DEFAULT_SAMPLE_DIR = ROOT.parent / "data/sample/xperience-10m-sample"
DROPBOX_SAMPLE_DIR = Path.home() / "Library/CloudStorage/Dropbox/Ropedia/data/sample/xperience-10m-sample"
DEFAULT_OUTPUT = ROOT / "docs/assets/task_suite_infographic.png"
CANVAS_WIDTH = 1800
CANVAS_HEIGHT = 6600
THUMB_WIDTH = 880
THUMB_HEIGHT = 520


GROUPS = [
    {
        "name": "Label + State",
        "tone": "teal",
        "color": "#9bdfff",
        "soft": "#071d20",
        "tasks": [
            ("timeline_action", "supervised"),
            ("timeline_subtask", "supervised"),
            ("next_action", "supervised"),
        ],
    },
    {
        "name": "Prediction + Reconstruction",
        "tone": "blue",
        "color": "#a7f078",
        "soft": "#10210a",
        "tasks": [
            ("hand_trajectory_forecast", "forecast"),
            ("modality_reconstruction", "forecast"),
            ("contact_prediction", "supervised"),
        ],
    },
    {
        "name": "Grounding + Retrieval",
        "tone": "amber",
        "color": "#7ae5c3",
        "soft": "#092019",
        "tasks": [
            ("caption_grounding", "retrieval"),
            ("cross_modal_retrieval", "retrieval"),
            ("object_relevance", "supervised"),
        ],
    },
    {
        "name": "Temporal Diagnostics",
        "tone": "red",
        "color": "#d8f4a5",
        "soft": "#1b210d",
        "tasks": [
            ("transition_detection", "diagnostic"),
            ("temporal_order", "diagnostic"),
            ("misalignment_detection", "diagnostic"),
        ],
    },
]

MODALITIES = [
    ("video", "visual stream", "6 synchronized camera MP4 streams", "RGB/fisheye/stereo frame statistics"),
    ("audio", "acoustic stream", "AAC stream embedded in MP4", "documented, not featurized in 8,378-d vector"),
    ("depth", "geometry map", "depth map + confidence channel", "spatial geometry feature block"),
    ("pose / SLAM", "camera pose", "trajectory + sparse SLAM map", "position + orientation features"),
    ("motion capture", "human motion", "body + hand joint tracks", "3D mocap feature statistics"),
    ("inertial", "wearable sensor", "accelerometer + gyroscope", "wearable motion statistics"),
    ("language", "semantic annotation", "object tags + action captions", "task labels + semantic targets"),
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


def make_canvas(size=(THUMB_WIDTH, THUMB_HEIGHT), color=(2, 5, 2)):
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


def draw_label(draw, xy, text, fill=(244, 248, 239), size=18):
    from PIL import ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=fill, font=font)


def video_thumb(sample_dir: Path) -> str:
    from PIL import Image, ImageDraw

    gutter = 18
    panel_width = (THUMB_WIDTH - gutter) // 2
    fish = fit_image(read_video_frame(sample_dir / "fisheye_cam0.mp4", 2450), (panel_width, THUMB_HEIGHT))
    stereo_path = sample_dir / "stereo_left.mp4"
    stereo = fit_image(read_video_frame(stereo_path, 2450), (panel_width, THUMB_HEIGHT)) if stereo_path.exists() else fish.copy()
    canvas = make_canvas()
    canvas.paste(fish, (0, 0))
    canvas.paste(stereo, (panel_width + gutter, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((panel_width - 4, 0, panel_width + gutter + 4, THUMB_HEIGHT), radius=0, fill=(2, 5, 2, 220))
    draw_label(draw, (18, 20), "fisheye", fill=(255, 255, 255), size=22)
    draw_label(draw, (panel_width + gutter + 18, 20), "stereo", fill=(255, 255, 255), size=22)
    return image_data_uri(canvas, "JPEG")


def colorize(values):
    import numpy as np

    stops = np.array([
        [2, 5, 2],
        [58, 136, 102],
        [122, 229, 195],
        [167, 240, 120],
        [216, 244, 165],
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

    gutter = 18
    panel_width = (THUMB_WIDTH - gutter) // 2
    frame = np.array(h5["depth/depth"][2450], dtype=np.float32)
    valid = np.isfinite(frame)
    lo, hi = np.percentile(frame[valid], [3, 97])
    norm = (frame - lo) / max(hi - lo, 1e-6)
    rgb = colorize(norm)
    depth = fit_image(Image.fromarray(rgb), (panel_width, THUMB_HEIGHT))
    conf = np.array(h5["depth/confidence"][2450], dtype=np.uint8)
    conf_img = Image.fromarray(conf, mode="L").convert("RGB")
    conf_img = fit_image(conf_img, (panel_width, THUMB_HEIGHT))
    canvas = make_canvas()
    canvas.paste(depth, (0, 0))
    canvas.paste(conf_img, (panel_width + gutter, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((0, 0, 158, 44), radius=8, fill=(2, 5, 2, 178))
    draw.rounded_rectangle((panel_width + gutter, 0, panel_width + gutter + 220, 44), radius=8, fill=(2, 5, 2, 178))
    draw_label(draw, (14, 11), "depth", fill=(255, 255, 255), size=22)
    draw_label(draw, (panel_width + gutter + 14, 11), "confidence", fill=(255, 255, 255), size=22)
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
        bins = 220
        trimmed = samples[: bins * max(1, len(samples) // bins)]
        chunks = np.array_split(trimmed, bins)
        rms = np.array([np.sqrt(np.mean(chunk * chunk)) if len(chunk) else 0.0 for chunk in chunks])
        waveform = np.array([float(np.mean(chunk)) if len(chunk) else 0.0 for chunk in chunks])
        baseline = THUMB_HEIGHT - 72
        for i, value in enumerate(rms):
            x = 18 + i / max(bins - 1, 1) * (THUMB_WIDTH - 36)
            h = 14 + np.clip(value * 158, 0, 158)
            draw.line((x, baseline, x, baseline - h), fill=(167, 240, 120, 170), width=2)
        points = []
        for i, value in enumerate(waveform):
            x = 18 + i / max(bins - 1, 1) * (THUMB_WIDTH - 36)
            y = 126 - np.clip(value, -1, 1) * 82
            points.append((x, y))
        draw.line(points, fill=(122, 229, 195, 220), width=2)
    except Exception:
        for i in range(48):
            x = 22 + i * 8
            h = 16 + (i % 7) * 7
            draw.rounded_rectangle((x, THUMB_HEIGHT - 72 - h, x + 4, THUMB_HEIGHT - 72), radius=2, fill=(167, 240, 120, 170))
    draw_label(draw, (18, 18), "AAC audio waveform", fill=(244, 248, 239), size=22)
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
        draw.line((a[0], a[1], b[0], b[1]), fill=(167, 240, 120, 205), width=2)
    draw_label(draw, (18, 18), "camera pose + SLAM map", fill=(244, 248, 239), size=22)
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
    colors = [(167, 240, 120), (122, 229, 195), (155, 223, 255), (216, 244, 165), (244, 248, 239), (165, 175, 162)]
    for row in range(6):
        y = 68 + row * 44
        draw.line((18, y, THUMB_WIDTH - 18, y), fill=(167, 240, 120, 48), width=1)
    for values, color in zip(series, colors):
        values = values[:420]
        if len(values) < 2:
            continue
        lo, hi = np.percentile(values, [3, 97])
        norm = (values - lo) / max(hi - lo, 1e-6)
        pts = []
        for i, v in enumerate(norm):
            x = 18 + i / max(len(values) - 1, 1) * (THUMB_WIDTH - 36)
            y = THUMB_HEIGHT - 48 - np.clip(v, 0, 1) * (THUMB_HEIGHT - 116)
            pts.append((x, y))
        draw.line(pts, fill=color + (200,), width=2)
    draw_label(draw, (18, 18), "inertial accel / gyro", fill=(244, 248, 239), size=22)
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
        xy[:, 1] = 72 + xy[:, 1] * (THUMB_HEIGHT - 136)
        return xy

    body_xy = project(body, 28, 270)
    for x, y in body_xy:
        draw.ellipse((x - 2.4, y - 2.4, x + 2.4, y + 2.4), fill=(167, 240, 120, 185))
    for a, b in zip(body_xy[:-1], body_xy[1:]):
        draw.line((a[0], a[1], b[0], b[1]), fill=(167, 240, 120, 82), width=1)

    for points, x_offset, color in [(left, 392, (122, 229, 195)), (right, 562, (216, 244, 165))]:
        xy = project(points, x_offset, 126)
        for a, b in HAND_EDGES:
            draw.line((xy[a][0], xy[a][1], xy[b][0], xy[b][1]), fill=color + (180,), width=2)
        for x, y in xy:
            draw.ellipse((x - 2.4, y - 2.4, x + 2.4, y + 2.4), fill=color + (220,))
    draw_label(draw, (18, 18), "body + hand mocap", fill=(244, 248, 239), size=22)
    return image_data_uri(canvas, "PNG")


def text_thumb(h5) -> str:
    from PIL import ImageDraw

    width = 1500
    raw = h5["caption"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    data = json.loads(raw)
    segment = data["segments"][0]
    objects = sorted({item for values in segment.get("objects", {}).values() for item in values})[:5]
    actions = [a.get("label", "") for a in segment.get("Current Action", [])][:2]
    canvas = make_canvas((width, THUMB_HEIGHT))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_label(draw, (28, 24), "language annotation", fill=(244, 248, 239), size=28)
    y = 82
    for label in objects:
        chip_width = 52 + len(label) * 16
        draw.rounded_rectangle((28, y, 28 + chip_width, y + 38), radius=8, fill=(7, 18, 7, 235), outline=(167, 240, 120, 170), width=2)
        draw_label(draw, (44, y + 8), label, fill=(244, 248, 239), size=18)
        y += 47
    x = 560
    y = 92
    for action in actions:
        wrapped = action[:66] + ("..." if len(action) > 66 else "")
        draw.rounded_rectangle((x, y, width - 28, y + 54), radius=9, fill=(7, 18, 7, 235), outline=(122, 229, 195, 180), width=2)
        draw_label(draw, (x + 22, y + 15), wrapped, fill=(244, 248, 239), size=20)
        y += 68
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


def valid_sample_dir(sample_dir: Path | None) -> bool:
    if sample_dir is None:
        return False
    return (sample_dir / "annotation.hdf5").exists() and (sample_dir / "fisheye_cam0.mp4").exists()


def resolve_sample_dir(sample_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    env_sample_dir = os.environ.get("XPERIENCE10M_SAMPLE_DIR")
    if env_sample_dir:
        candidates.append(Path(env_sample_dir).expanduser())
    workspace = os.environ.get("WORKSPACE")
    if workspace:
        candidates.append(Path(workspace).expanduser() / "data/sample/xperience-10m-sample")
    if sample_dir is not None:
        candidates.append(sample_dir)
    candidates.extend([
        DEFAULT_SAMPLE_DIR,
        DROPBOX_SAMPLE_DIR,
    ])
    for candidate in candidates:
        if valid_sample_dir(candidate):
            return candidate
    return sample_dir


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


def modality_card(name: str, modality_type: str, sample_text: str, feature_text: str, index: int, thumbnail: str | None) -> str:
    thumb_html = ""
    if thumbnail:
        thumb_html = f'<div class="modality-thumb"><img src="{thumbnail}" alt=""></div>'
    return f"""
      <article class="modality">
        <div class="modality-heading">
          <div>
            <span class="modality-index">{index:02d}</span>
            <h3>{html.escape(name)}</h3>
          </div>
          <span class="modality-type">{html.escape(modality_type)}</span>
        </div>
        {thumb_html}
        <div class="modality-copy">
          <div class="modality-row">
            <span>Sample contains</span>
            <p>{html.escape(sample_text)}</p>
          </div>
          <div class="modality-row">
            <span>Current baseline use</span>
            <p>{html.escape(feature_text)}</p>
          </div>
        </div>
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
        modality_card(name, modality_type, sample_text, feature_text, index, thumbnails.get(name))
        for index, (name, modality_type, sample_text, feature_text) in enumerate(MODALITIES, start=1)
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
      background: #020502;
    }}
    body {{
      font-family: "Inter Tight", "Space Grotesk", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #f4f8ef;
      text-rendering: optimizeLegibility;
    }}
    .canvas {{
      position: relative;
      width: {CANVAS_WIDTH}px;
      height: {CANVAS_HEIGHT}px;
      overflow: hidden;
      padding: 54px 64px 44px;
      background:
        radial-gradient(circle at 72% 10%, rgba(167,240,120,0.18), transparent 24%),
        radial-gradient(circle at 20% 28%, rgba(255,255,255,0.10) 1px, transparent 2px),
        #020502;
      background-size: auto, 18px 18px, auto;
    }}
    .image-background {{
      position: absolute;
      inset: 0;
      background-position: center;
      background-repeat: no-repeat;
      background-size: cover;
      opacity: 0.36;
      filter: saturate(1.05) contrast(1.08) brightness(0.42);
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
      border-bottom: 1px solid rgba(167,240,120,0.20);
    }}
    .kicker {{
      display: inline-flex;
      align-items: center;
      gap: 12px;
      color: #a7f078;
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .kicker::before {{
      content: "";
      width: 44px;
      height: 1px;
      background: #a7f078;
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
      color: #dce8d7;
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
      border: 1px solid rgba(167,240,120,0.24);
      background: rgba(7,18,7,0.80);
      border-radius: 8px;
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
      color: #a5afa2;
      font-size: 13px;
      line-height: 1.15;
    }}
    .section-label {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      align-items: start;
      margin: 44px 0 24px;
      color: #a5afa2;
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 22px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .section-label span:last-child {{
      max-width: 1400px;
      color: #dce8d7;
      text-transform: none;
      letter-spacing: 0;
      font-family: inherit;
      font-size: 21px;
      line-height: 1.42;
      text-align: left;
    }}
    .modalities {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 34px;
    }}
    .modality {{
      min-height: 560px;
      padding: 34px;
      border: 1px solid rgba(167,240,120,0.22);
      background: rgba(7,18,7,0.84);
      border-radius: 8px;
      display: grid;
      grid-template-columns: 880px minmax(0, 1fr);
      grid-template-areas:
        "thumb heading"
        "thumb copy";
      column-gap: 46px;
      row-gap: 28px;
      align-items: start;
    }}
    .modality-thumb {{
      grid-area: thumb;
      height: 492px;
      overflow: hidden;
      border: 1px solid rgba(167,240,120,0.16);
      border-radius: 8px;
      background: #020502;
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
    .modality-heading {{
      grid-area: heading;
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 24px;
      padding-bottom: 26px;
      border-bottom: 1px solid rgba(167,240,120,0.16);
    }}
    .modality-index {{
      color: #a5afa2;
      font-size: 24px;
    }}
    .modality-type {{
      color: #a7f078;
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 16px;
      line-height: 1.15;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      text-align: right;
      max-width: 330px;
      padding-top: 8px;
    }}
    .modality h3 {{
      margin: 14px 0 0;
      font-size: 76px;
      line-height: 0.98;
      text-transform: uppercase;
    }}
    .modality-copy {{
      grid-area: copy;
      display: grid;
      grid-template-columns: 1fr;
      gap: 22px;
    }}
    .modality-row {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      align-items: baseline;
      padding: 22px 24px;
      border: 1px solid rgba(167,240,120,0.16);
      border-radius: 8px;
      background: rgba(2,5,2,0.40);
    }}
    .modality-row span {{
      display: block;
      color: #a5afa2;
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 16px;
      letter-spacing: 0.06em;
      line-height: 1.25;
      text-transform: uppercase;
    }}
    .modality-row p {{
      margin: 0;
      color: #dce8d7;
      font-size: 40px;
      font-weight: 650;
      line-height: 1.15;
    }}
    .shared-band {{
      display: grid;
      grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr;
      gap: 12px;
      align-items: center;
      margin-top: 30px;
      padding: 14px;
      border: 1px solid rgba(167,240,120,0.22);
      background: rgba(7,18,7,0.72);
      border-radius: 8px;
    }}
    .step {{
      min-height: 62px;
      padding: 13px 15px;
      background: rgba(7,18,7,0.92);
      border: 1px solid rgba(167,240,120,0.16);
      border-radius: 8px;
    }}
    .step strong {{
      display: block;
      font-size: 17px;
      line-height: 1.1;
    }}
    .step span {{
      display: block;
      margin-top: 5px;
      color: #a5afa2;
      font-size: 13px;
    }}
    .arrow {{
      color: #a7f078;
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 22px;
    }}
    .families {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 24px;
      margin-top: 30px;
    }}
    .family {{
      padding: 20px;
      border: 1px solid color-mix(in srgb, var(--accent) 28%, #020502);
      background: rgba(7,18,7,0.82);
      border-radius: 8px;
    }}
    .family-head {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      min-height: 66px;
      padding-bottom: 16px;
      border-bottom: 1px solid color-mix(in srgb, var(--accent) 24%, #020502);
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
      font-size: 32px;
      line-height: 1.02;
      text-align: right;
    }}
    .family-cards {{
      display: grid;
      gap: 16px;
      margin-top: 18px;
    }}
    .task-card {{
      min-height: 178px;
      padding: 18px 20px;
      border: 1px solid color-mix(in srgb, var(--accent) 28%, #020502);
      background: linear-gradient(180deg, rgba(10,24,10,0.96), color-mix(in srgb, var(--soft) 24%, #071207));
      border-radius: 8px;
    }}
    .task-meta {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .index {{
      color: #a5afa2;
      font-size: 12px;
    }}
    .kind {{
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 9px;
      border-radius: 6px;
      border: 1px solid color-mix(in srgb, var(--accent) 40%, #020502);
      color: var(--accent);
      background: rgba(2,5,2,0.48);
      text-transform: uppercase;
      font-size: 11px;
      line-height: 1;
      font-weight: 830;
    }}
    .task-card h3 {{
      margin: 12px 0 0;
      color: #f4f8ef;
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: 21px;
      line-height: 1.18;
      overflow-wrap: anywhere;
    }}
    .task-card p {{
      margin: 11px 0 0;
      min-height: 39px;
      color: #dce8d7;
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
      border: 1px solid color-mix(in srgb, var(--accent) 42%, #020502);
      background: rgba(2,5,2,0.42);
    }}
    .metric.neural {{
      margin-left: 8px;
      border-color: rgba(255,255,255,0.20);
      background: rgba(255,255,255,0.08);
    }}
    .metric span {{
      color: #a5afa2;
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
      border-top: 1px solid rgba(167,240,120,0.20);
      color: #a5afa2;
      font-size: 18px;
      line-height: 1.35;
      font-weight: 620;
    }}
    .footer code {{
      font-family: "SF Mono", "JetBrains Mono", ui-monospace, monospace;
      color: #020502;
      background: #a7f078;
      border: 1px solid #a7f078;
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
        <p class="subtitle">A clean map from synchronized multimodal windows to 12 auditable task heads, comparing minimal heads with neural MLP results. Next milestone: Qwen3-Omni fine-tuning with sensor-bridge evaluation.</p>
      </div>
      <div class="stats">{stats_html}</div>
    </header>

    <section class="shared-band" aria-label="shared processing contract">
      <div class="step"><strong>raw public episode</strong><span>video, audio, depth, pose, mocap, IMU, language</span></div>
      <div class="arrow">-></div>
      <div class="step"><strong>20-frame windows</strong><span>stride 5, chronological order</span></div>
      <div class="arrow">-></div>
      <div class="step"><strong>8,378-d vector</strong><span>current manifest excludes audio features</span></div>
      <div class="arrow">-></div>
      <div class="step"><strong>12 minimal + NN heads</strong><span>softmax/ridge/logistic plus PyTorch MLP</span></div>
    </section>

    <div class="section-label">
      <span>12 task families</span>
      <span>Every task below has a minimal baseline and a neural MLP head over the same aligned window contract, so the figure prioritizes task design before visual decoration.</span>
    </div>
    <section class="families">{''.join(families)}</section>

    <div class="section-label">
      <span>Xperience-10M modalities</span>
      <span>Public-sample thumbnails are enlarged here so each data stream is legible. Audio is present in the sample MP4 stream; the current 8,378-d baseline manifest does not featurize it.</span>
    </div>
    <section class="modalities">{modalities_html}</section>

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
    sample_dir = resolve_sample_dir(args.sample_dir)
    html_text = build_html(summary, args.base_image, sample_dir)
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
