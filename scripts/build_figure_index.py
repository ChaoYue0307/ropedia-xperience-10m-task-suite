#!/usr/bin/env python3
"""Build an index for public visual assets."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "docs/data/figure_index.json"
OUTPUT_MD = ROOT / "FIGURE_INDEX.md"

FIGURES = [
    {
        "id": "brand_logo_mark",
        "title": "Project logo mark",
        "path": "docs/assets/brand/xperience10m-logo-mark-512.png",
        "role": "Primary X-shaped multimodal camera mark used for the website header, README, HF cards, and brand identity.",
        "source_script": "scripts/build_brand_assets.py",
        "surface": "README, website, HF Space, artifact dataset, model card, favicon variants",
    },
    {
        "id": "brand_social_card",
        "title": "Project logo social card",
        "path": "docs/assets/brand/xperience10m-logo-social-card.png",
        "role": "Large preview image for README, Hugging Face cards, and Open Graph/Twitter social sharing.",
        "source_script": "scripts/build_brand_assets.py",
        "surface": "README, website metadata, HF Space, artifact dataset, model card",
    },
    {
        "id": "brand_favicon",
        "title": "Project favicon",
        "path": "docs/assets/brand/xperience10m-logo-favicon-64.png",
        "role": "Small dark-tile logo for browser tabs and compact navigation.",
        "source_script": "scripts/build_brand_assets.py",
        "surface": "website favicon and header",
    },
    {
        "id": "task_suite_infographic",
        "title": "12-task suite infographic",
        "path": "docs/assets/task_suite_infographic.png",
        "role": "Primary visual map of the task suite, verified metrics, and sample modalities.",
        "source_script": "scripts/render_task_suite_infographic.py",
        "surface": "README, website, HF Space, artifact dataset, model card",
    },
    {
        "id": "pipeline_diagram",
        "title": "Episode-to-task pipeline diagram",
        "path": "docs/assets/pipeline_diagram.png",
        "role": "End-to-end data processing and evaluation pipeline overview.",
        "source_script": "scripts/generate_visualizations.py",
        "surface": "README, website, HF artifact dataset",
    },
    {
        "id": "qwen3_omni_lora_pipeline",
        "title": "Qwen3-Omni LoRA training pipeline",
        "path": "docs/assets/qwen3_omni_lora_pipeline.png",
        "role": "Detailed raw-data-to-adapter flow for staged Xperience-10M Qwen3-Omni LoRA training.",
        "source_script": "docs/assets/qwen3_omni_lora_pipeline.prompt.md",
        "surface": "README, website, HF Space, artifact dataset, model card",
    },
    {
        "id": "task_architectures",
        "title": "Minimal and neural task architecture map",
        "path": "docs/assets/task_architectures.png",
        "role": "All 12 task heads and shared feature contracts.",
        "source_script": "scripts/render_overview_figures.py",
        "surface": "README, website, HF artifact dataset, model card",
    },
    {
        "id": "video_modality",
        "title": "Video modality thumbnail",
        "path": "docs/assets/modalities/video.jpg",
        "role": "Derived thumbnail for synchronized camera streams.",
        "source_script": "scripts/export_modality_atlas_assets.py",
        "surface": "website modality atlas, HF mirrors",
    },
    {
        "id": "audio_modality",
        "title": "Audio modality thumbnail",
        "path": "docs/assets/modalities/audio.png",
        "role": "Derived waveform thumbnail for the MP4 AAC stream.",
        "source_script": "scripts/export_modality_atlas_assets.py",
        "surface": "website modality atlas, HF mirrors",
    },
    {
        "id": "depth_modality",
        "title": "Depth modality thumbnail",
        "path": "docs/assets/modalities/depth.jpg",
        "role": "Derived depth and confidence thumbnail.",
        "source_script": "scripts/export_modality_atlas_assets.py",
        "surface": "website modality atlas, HF mirrors",
    },
    {
        "id": "pose_slam_modality",
        "title": "Pose / SLAM modality thumbnail",
        "path": "docs/assets/modalities/pose_slam.png",
        "role": "Derived camera trajectory and sparse map thumbnail.",
        "source_script": "scripts/export_modality_atlas_assets.py",
        "surface": "website modality atlas, HF mirrors",
    },
    {
        "id": "motion_capture_modality",
        "title": "Motion capture modality thumbnail",
        "path": "docs/assets/modalities/motion_capture.png",
        "role": "Derived body and hand motion-capture thumbnail.",
        "source_script": "scripts/export_modality_atlas_assets.py",
        "surface": "website modality atlas, HF mirrors",
    },
    {
        "id": "inertial_modality",
        "title": "Inertial modality thumbnail",
        "path": "docs/assets/modalities/inertial.png",
        "role": "Derived accelerometer and gyroscope trace thumbnail.",
        "source_script": "scripts/export_modality_atlas_assets.py",
        "surface": "website modality atlas, HF mirrors",
    },
    {
        "id": "language_modality",
        "title": "Language modality thumbnail",
        "path": "docs/assets/modalities/language.png",
        "role": "Derived object-tag and caption thumbnail.",
        "source_script": "scripts/export_modality_atlas_assets.py",
        "surface": "website modality atlas, HF mirrors",
    },
    {
        "id": "model_macro_f1_chart",
        "title": "Model macro-F1 comparison chart",
        "path": "docs/assets/charts/model_macro_f1.svg",
        "role": "Minimal-vs-neural classification score comparison.",
        "source_script": "scripts/generate_visualizations.py",
        "surface": "website diagnostics",
    },
    {
        "id": "neural_score_chart",
        "title": "Neural MLP task score chart",
        "path": "docs/assets/charts/episode_task_scores_neural_mlp.svg",
        "role": "Neural MLP metric snapshot across the task suite.",
        "source_script": "scripts/generate_visualizations.py",
        "surface": "website diagnostics",
    },
    {
        "id": "minimal_vs_neural_score_chart",
        "title": "Minimal-vs-neural task score chart",
        "path": "docs/assets/charts/episode_task_scores_minimal_vs_neural.svg",
        "role": "Side-by-side baseline comparison over the same window contracts.",
        "source_script": "scripts/generate_visualizations.py",
        "surface": "website diagnostics",
    },
    {
        "id": "research_direction_coverage_chart",
        "title": "Research direction coverage chart",
        "path": "docs/assets/charts/research_direction_coverage.svg",
        "role": "Four-track coverage map for Ropedia research directions.",
        "source_script": "scripts/generate_visualizations.py",
        "surface": "website directions",
    },
    {
        "id": "research_direction_extension_chart",
        "title": "Research direction extension chart",
        "path": "docs/assets/charts/research_direction_extension_tasks.svg",
        "role": "Four coded extension probes, one per Ropedia research direction.",
        "source_script": "scripts/generate_visualizations.py",
        "surface": "website directions",
    },
    {
        "id": "feature_blocks_chart",
        "title": "Feature block chart",
        "path": "docs/assets/charts/feature_blocks.svg",
        "role": "Feature allocation by modality block.",
        "source_script": "scripts/generate_visualizations.py",
        "surface": "website features",
    },
    {
        "id": "episode_task_scores_chart",
        "title": "Minimal task score chart",
        "path": "docs/assets/charts/episode_task_scores.svg",
        "role": "Minimal baseline metric snapshot across the task suite.",
        "source_script": "scripts/generate_visualizations.py",
        "surface": "website diagnostics",
    },
    {
        "id": "cross_modal_retrieval_chart",
        "title": "Cross-modal retrieval chart",
        "path": "docs/assets/charts/cross_modal_retrieval.svg",
        "role": "Retrieval behavior chart for the cross-modal task.",
        "source_script": "scripts/generate_visualizations.py",
        "surface": "website diagnostics",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


def svg_dimensions(path: Path) -> dict:
    root = ElementTree.fromstring(path.read_text(encoding="utf-8", errors="ignore"))
    width = parse_number(root.attrib.get("width"))
    height = parse_number(root.attrib.get("height"))
    view_box = root.attrib.get("viewBox")
    if (width is None or height is None) and view_box:
        parts = [float(item) for item in re.split(r"[\s,]+", view_box.strip()) if item]
        if len(parts) == 4:
            width = width if width is not None else parts[2]
            height = height if height is not None else parts[3]
    return {
        "format": "SVG",
        "width": int(round(width or 0)),
        "height": int(round(height or 0)),
        "view_box": view_box,
    }


def image_dimensions(path: Path) -> dict:
    if path.suffix.lower() == ".svg":
        return svg_dimensions(path)
    with Image.open(path) as image:
        return {
            "format": image.format,
            "width": int(image.width),
            "height": int(image.height),
        }


def figure_record(spec: dict) -> dict:
    path = ROOT / spec["path"]
    exists = path.exists()
    record = {
        **spec,
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "sha256": sha256(path) if exists else None,
        "dimensions": None,
        "source_script_exists": (ROOT / spec["source_script"]).exists(),
    }
    if exists:
        try:
            record["dimensions"] = image_dimensions(path)
        except Exception as exc:  # noqa: BLE001 - report the exact bad asset.
            record["dimension_error"] = str(exc)
    return record


def build_payload() -> dict:
    figures = [figure_record(item) for item in FIGURES]
    failures = []
    for figure in figures:
        if not figure["exists"]:
            failures.append({"figure": figure["id"], "kind": "missing_asset", "path": figure["path"]})
        if not figure["source_script_exists"]:
            failures.append({"figure": figure["id"], "kind": "missing_source_script", "path": figure["source_script"]})
        dimensions = figure.get("dimensions") or {}
        if dimensions.get("width", 0) <= 0 or dimensions.get("height", 0) <= 0:
            failures.append({"figure": figure["id"], "kind": "invalid_dimensions", "path": figure["path"]})
    return {
        "title": "Ropedia Xperience-10M Figure Index",
        "status": "pass" if not failures else "fail",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Public figures, diagrams, charts, and derived modality thumbnails. Raw Xperience-10M videos, annotations, RRD files, and Qwen weights are excluded.",
        "figure_count": len(figures),
        "figures": figures,
        "failures": failures,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Figure Index",
        "",
        "This file is generated by `scripts/build_figure_index.py`. It catalogs",
        "the public visual assets used by the repo, website, and Hugging Face mirrors.",
        "",
        f"Current status: **{payload['status']}**",
        "",
        payload["scope"],
        "",
        "## Figures",
        "",
        "| Figure | Path | Size | Source script | Role |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for figure in payload["figures"]:
        dimensions = figure.get("dimensions") or {}
        size = f"{dimensions.get('width', 0)} x {dimensions.get('height', 0)}"
        lines.append(
            f"| {figure['title']} | `{figure['path']}` | {size} | `{figure['source_script']}` | {figure['role']} |"
        )
    lines.extend([
        "",
        "## Use and Scope",
        "",
        "- These figures are derived presentation artifacts or small thumbnails.",
        "- The index records file hashes and dimensions for reproducibility checks.",
        "- Raw Xperience-10M MP4/HDF5/RRD files and full model weights are not redistributed.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"{payload['status'].upper()}: wrote {OUTPUT_JSON}")
    print(f"{payload['status'].upper()}: wrote {OUTPUT_MD}")
    if payload["status"] != "pass":
        for failure in payload["failures"]:
            print(f"- {failure}")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
