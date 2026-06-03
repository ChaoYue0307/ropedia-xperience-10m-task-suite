#!/usr/bin/env python3
"""Export standalone modality thumbnails and a website data manifest.

The large 12-task infographic embeds modality thumbnails as data URIs. This
script writes those same sample-derived thumbnails as first-class public assets
so the website can present a responsive, readable modality atlas on small
screens without redistributing raw videos or annotations.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from PIL import Image

from render_task_suite_infographic import MODALITIES, load_sample_thumbnails, resolve_sample_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs/assets/modalities"
DEFAULT_MANIFEST = ROOT / "docs/data/modality_atlas.json"

SLUGS = {
    "video": "video",
    "audio": "audio",
    "depth": "depth",
    "pose / SLAM": "pose_slam",
    "motion capture": "motion_capture",
    "inertial": "inertial",
    "language": "language",
}


def decode_data_uri(uri: str) -> tuple[str, bytes]:
    prefix, payload = uri.split(",", 1)
    mime = prefix.removeprefix("data:").split(";")[0]
    suffix = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(mime)
    if suffix is None:
        raise ValueError(f"Unsupported thumbnail MIME type: {mime}")
    return suffix, base64.b64decode(payload)


def image_size(path: Path) -> list[int]:
    with Image.open(path) as image:
        return [int(image.width), int(image.height)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    sample_dir = resolve_sample_dir(args.sample_dir)
    thumbnails = load_sample_thumbnails(sample_dir)
    if len(thumbnails) != len(MODALITIES):
        missing = [name for name, *_ in MODALITIES if name not in thumbnails]
        raise RuntimeError(f"Could not export all modality thumbnails. Missing: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (name, modality_type, sample_text, feature_text) in enumerate(MODALITIES, start=1):
        suffix, data = decode_data_uri(thumbnails[name])
        slug = SLUGS[name]
        output_path = args.output_dir / f"{slug}{suffix}"
        output_path.write_bytes(data)
        records.append({
            "index": index,
            "id": slug,
            "name": name,
            "type": modality_type,
            "sample_contains": sample_text,
            "current_baseline_use": feature_text,
            "image": f"assets/modalities/{output_path.name}",
            "image_size": image_size(output_path),
            "source": "Derived thumbnail from the public Xperience-10M sample episode.",
            "feature_status": "featurized_or_label_source",
        })

    manifest = {
        "title": "Xperience-10M Public Sample Modality Atlas",
        "source_sample_reference": "ropedia-ai/xperience-10m-sample public episode",
        "raw_data_redistributed": False,
        "notes": [
            "Images are lightweight derived thumbnails for review and website presentation.",
            "Raw MP4, HDF5, and RRD files remain excluded from the public repo and Hugging Face bundles.",
            "AAC audio is extracted from the sample MP4 stream and included in the current baseline feature vector.",
        ],
        "modalities": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(records)} modality thumbnails to {args.output_dir}")
    print(f"Wrote manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
