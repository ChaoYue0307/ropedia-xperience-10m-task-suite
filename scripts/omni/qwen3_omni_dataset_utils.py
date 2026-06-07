#!/usr/bin/env python3
"""Shared helpers for Ropedia -> Qwen3-Omni episode-understanding fine-tuning."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


VIDEO_NAMES = [
    "fisheye_cam0.mp4",
    "fisheye_cam1.mp4",
    "fisheye_cam2.mp4",
    "fisheye_cam3.mp4",
    "stereo_left.mp4",
    "stereo_right.mp4",
]


DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

JSON_FIELDS = [
    "action",
    "subtask",
    "objects",
    "contact",
    "transition",
    "next_action",
    "evidence_window",
]

STRICT_JSON_SCHEMA_TEXT = (
    '{"action":"<exact action option or unknown>",'
    '"subtask":"<exact subtask option or unknown>",'
    '"objects":["<0 to 8 short object names>"],'
    '"contact":"yes|no|unknown",'
    '"transition":"yes|no|unknown",'
    '"next_action":"<exact action option or unknown>",'
    '"evidence_window":{"start_frame":0,"end_frame":0}}'
)

SYSTEM_PROMPT = (
    "You are an embodied episode-understanding model for Ropedia/Xperience-10M. "
    "Return exactly one valid JSON object and no markdown, no prose, no code fences, "
    "and no repeated text. The JSON must use exactly these keys: action, subtask, "
    "objects, contact, transition, next_action, evidence_window. For action, "
    "subtask, and next_action, copy one label exactly from the provided option "
    "lists or use \"unknown\"; do not invent synonyms or paraphrases. Use "
    "\"unknown\" when evidence is missing instead of guessing. Keep objects to at "
    f"most 8 short names. Schema example: {STRICT_JSON_SCHEMA_TEXT}"
)


def add_repo_paths(workspace: Path) -> None:
    scripts = workspace / "scripts"
    toolkit = workspace / "HOMIE-toolkit"
    for path in (scripts, toolkit):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def episode_dirs_from_sources(episode_roots: list[Path] | None, manifest: Path | None, split: str = "all") -> list[Path]:
    episode_dirs: list[Path] = []
    if episode_roots:
        episode_dirs.extend(path.expanduser().resolve() for path in episode_roots)
    if manifest:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for ep in payload.get("episodes", []):
            if split != "all" and ep.get("split") != split:
                continue
            path = Path(ep["path"]).expanduser().resolve()
            if path not in episode_dirs:
                episode_dirs.append(path)
    return episode_dirs


def split_for_episode(episode_id: str, manifest: Path | None, episode_path: Path | None = None) -> str:
    if manifest is None:
        return "unspecified"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    resolved_episode_path = episode_path.expanduser().resolve() if episode_path is not None else None
    for ep in payload.get("episodes", []):
        manifest_path = Path(ep.get("path", "")).expanduser()
        if resolved_episode_path is not None and manifest_path.resolve() == resolved_episode_path:
            return str(ep.get("split", "unspecified"))
        if ep.get("episode_id") == episode_id or manifest_path.name == episode_id:
            return str(ep.get("split", "unspecified"))
    return "unspecified"


def existing_videos(episode_dir: Path) -> list[dict]:
    videos = []
    for name in VIDEO_NAMES:
        path = episode_dir / name
        if path.exists():
            videos.append({"name": name, "path": str(path)})
    return videos


def primary_video_path(videos: list[dict]) -> str | None:
    if not videos:
        return None
    preferred = ["fisheye_cam0.mp4", "stereo_left.mp4", "stereo_right.mp4"]
    by_name = {Path(item["path"]).name: item["path"] for item in videos}
    for name in preferred:
        if name in by_name:
            return by_name[name]
    return videos[0]["path"]


def label_options_text(label_options: list[str]) -> str:
    return "\n".join(f"- {label}" for label in label_options)


def canonical_answer(answer: dict) -> dict:
    window = answer.get("evidence_window") if isinstance(answer.get("evidence_window"), dict) else {}
    objects = answer.get("objects") if isinstance(answer.get("objects"), list) else []
    clean_objects = []
    seen = set()
    for obj in objects:
        value = normalize_label(obj)
        if not value or value.lower() in seen:
            continue
        seen.add(value.lower())
        clean_objects.append(value)
        if len(clean_objects) >= 8:
            break
    return {
        "action": normalize_label(answer.get("action") or "unknown") or "unknown",
        "subtask": normalize_label(answer.get("subtask") or "unknown") or "unknown",
        "objects": clean_objects,
        "contact": normalize_label(answer.get("contact") or "unknown").lower() or "unknown",
        "transition": normalize_label(answer.get("transition") or "unknown").lower() or "unknown",
        "next_action": normalize_label(answer.get("next_action") or "unknown") or "unknown",
        "evidence_window": {
            "start_frame": int(window.get("start_frame", 0) or 0),
            "end_frame": int(window.get("end_frame", 0) or 0),
        },
    }


def answer_json_text(sample: dict) -> str:
    answer = sample.get("answer_json")
    if answer is None:
        answer = {
            "action": sample.get("label", "unknown"),
            "subtask": sample.get("subtask", "unknown"),
            "objects": sample.get("objects", []),
            "contact": sample.get("contact", "unknown"),
            "transition": sample.get("transition", "unknown"),
            "next_action": sample.get("next_action", "unknown"),
            "evidence_window": sample.get("evidence_window", {}),
        }
    return json.dumps(canonical_answer(answer), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_user_prompt(sample: dict, label_options: list[str]) -> str:
    center_window = sample.get("center_window", {})
    start_frame = center_window.get("start_frame", sample.get("start_frame", "unknown"))
    end_frame = center_window.get("end_frame", sample.get("end_frame", "unknown"))
    action_options = sample.get("action_options") or label_options
    subtask_options = sample.get("subtask_options") or []
    prompt = [
        sample.get(
            "question",
            "Answer embodied episode-understanding questions for the current centered window.",
        ),
        f"Episode: {sample['episode_id']}",
        f"Label window frames: {start_frame}-{end_frame}",
        "Return exactly one compact JSON object only. Do not add markdown, prose, analysis, comments, or a second object.",
        f"Required schema: {STRICT_JSON_SCHEMA_TEXT}",
        "For action, subtask, and next_action, copy exactly one label from the option lists or use \"unknown\".",
        "Do not paraphrase labels, merge labels, or create new action/subtask strings.",
        "Use \"unknown\" for fields that cannot be determined.",
        "Keep objects as a short list with at most 8 entries.",
    ]
    if action_options:
        prompt.extend(["Known action labels:", label_options_text(action_options)])
    if subtask_options:
        prompt.extend(["Known subtask labels:", label_options_text(subtask_options)])
    if sample.get("sensor_bridge_summary"):
        prompt.extend(["Sensor adapter summary:", sample["sensor_bridge_summary"]])
    return "\n".join(prompt)


def build_messages(sample: dict, label_options: list[str], include_answer: bool) -> list[dict]:
    content = []
    media = sample.get("media", {})
    video_path = media.get("mosaic_video_path") or sample.get("primary_video_path")
    audio_path = media.get("audio_path")
    if video_path:
        content.append({"type": "video", "video": video_path})
    if audio_path:
        content.append({"type": "audio", "audio": audio_path})
    content.append({"type": "text", "text": build_user_prompt(sample, label_options)})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": answer_json_text(sample)})
    return messages


def sample_without_audio(sample: dict) -> dict:
    copied = dict(sample)
    media = dict(copied.get("media") or {})
    media["audio_path"] = None
    copied["media"] = media
    return copied


def sample_has_audio(sample: dict) -> bool:
    return bool((sample.get("media") or {}).get("audio_path"))


def audio_num_elements(audio) -> int:
    if audio is None:
        return 0
    if hasattr(audio, "numel"):
        try:
            return int(audio.numel())
        except TypeError:
            pass
    shape = getattr(audio, "shape", None)
    if shape is not None:
        total = 1
        for dim in shape:
            total *= int(dim)
        return total
    try:
        return len(audio)
    except TypeError:
        return 1


def has_empty_audio_items(audios) -> bool:
    if audios is None:
        return False
    items = audios if isinstance(audios, (list, tuple)) else [audios]
    return any(audio_num_elements(item) == 0 for item in items)


def is_empty_audio_exception(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "[1, 1, 0]" in text
        or "zero-size" in text
        or ("stft" in text and "expected 2d or 3d" in text)
    )


def parse_answer_json(text: str) -> dict:
    raw = str(text).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def json_validity_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    valid = sum(1 for text in texts if all(field in parse_answer_json(text) for field in JSON_FIELDS))
    return valid / len(texts)


def normalize_label(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).strip())
    text = text.strip("`'\". ")
    return text


def match_label(prediction: str, label_options: list[str]) -> str:
    normalized = normalize_label(prediction)
    if normalized in label_options:
        return normalized
    lowered = normalized.lower()
    by_lower = {label.lower(): label for label in label_options}
    if lowered in by_lower:
        return by_lower[lowered]
    for label in label_options:
        if label.lower() in lowered:
            return label
    return normalized


def class_metrics(y_true: list[str], y_pred: list[str], label_options: list[str]) -> tuple[dict, list[dict], list[list[int]]]:
    labels = list(label_options)
    for label in y_true + y_pred:
        if label not in labels:
            labels.append(label)
    index = {label: idx for idx, label in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]
    for true, pred in zip(y_true, y_pred):
        cm[index[true]][index[pred]] += 1

    per_class = []
    f1s = []
    correct = 0
    for idx, label in enumerate(labels):
        tp = cm[idx][idx]
        correct += tp
        fp = sum(row[idx] for row in cm) - tp
        fn = sum(cm[idx]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class.append({
            "class_name": label,
            "support": sum(cm[idx]),
            "predicted": sum(row[idx] for row in cm),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    metrics = {
        "num_samples": len(y_true),
        "accuracy": correct / len(y_true) if y_true else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "labels": labels,
    }
    return metrics, per_class, cm


def label_counts(samples: list[dict]) -> dict:
    counts = Counter(sample.get("label", sample.get("answer_json", {}).get("action", "unknown")) for sample in samples)
    return dict(counts.most_common())
