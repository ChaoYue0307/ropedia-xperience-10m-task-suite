#!/usr/bin/env python3
"""
Build a static interactive explorer for the single Xperience-10M sample episode.

The explorer is generated from committed/exported artifacts only. Raw MP4/HDF5
files are not embedded or redistributed.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from task_display import TASK_DISPLAY_NAMES, task_display_name


TASK_DISPLAY = {
    "timeline_action": task_display_name("timeline_action"),
    "timeline_subtask": task_display_name("timeline_subtask"),
    "transition_detection": task_display_name("transition_detection"),
    "next_action": task_display_name("next_action"),
    "contact_prediction": task_display_name("contact_prediction"),
    "object_relevance": task_display_name("object_relevance"),
}
TASK_DISPLAY_ALL = dict(TASK_DISPLAY_NAMES)


BLOCK_DISPLAY = {
    "hand_left_joints": "Left Hand",
    "hand_right_joints": "Right Hand",
    "body_joints": "Body Joints",
    "body_contacts": "Body Contacts",
    "camera_translation": "Camera Translation",
    "camera_rotation_matrix": "Camera Rotation",
    "imu_accel_gyro": "IMU Accel/Gyro",
    "depth_confidence": "Depth + Confidence",
    "audio_fisheye_cam0_aac": "Audio",
    "caption_objects_interaction_text": "Language Text",
    "slam_point_cloud": "SLAM Point Cloud",
    "calibration": "Calibration",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build static single-episode explorer page.")
    parser.add_argument("--workspace", type=Path, default=root)
    parser.add_argument("--suite-dir", type=Path, default=root / "results/episode_task_suite")
    parser.add_argument("--diagnostics-dir", type=Path, default=root / "results/single_episode_diagnostics")
    parser.add_argument("--docs-dir", type=Path, default=root / "docs")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def block_modality(name: str) -> str:
    if name.startswith("video_"):
        return "video"
    if name.startswith("hand_") or name.startswith("body_"):
        return "motion_capture"
    if name.startswith("camera_") or name in {"slam_point_cloud", "calibration"}:
        return "pose_slam"
    if name.startswith("depth_"):
        return "depth"
    if name.startswith("imu_"):
        return "inertial"
    if name.startswith("audio_"):
        return "audio"
    if name.startswith("caption_"):
        return "language"
    return "other"


def load_predictions(suite_dir: Path) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for task in TASK_DISPLAY:
        path = suite_dir / task / "predictions.csv"
        rows_by_window: dict[int, dict] = {}
        if not path.exists():
            out[task] = rows_by_window
            continue
        for row in read_csv(path):
            if "window_index" not in row:
                continue
            idx = int(row["window_index"])
            true_value = row.get("true_label") or row.get("true_objects") or row.get("true") or ""
            pred_value = row.get("predicted_label") or row.get("predicted_objects") or row.get("predicted") or ""
            if "correct" in row and row["correct"] != "":
                correct = int(float(row["correct"]))
            else:
                correct = int(str(true_value) == str(pred_value))
            rows_by_window[idx] = {
                "true": true_value,
                "predicted": pred_value,
                "correct": correct,
                "confidence": row.get("confidence", ""),
            }
        out[task] = rows_by_window
    return out


def build_action_segments(windows: list[dict]) -> list[dict]:
    segments = []
    if not windows:
        return segments
    current = windows[0]["action_label"]
    start = int(windows[0]["start_frame"])
    start_idx = int(windows[0]["window_index"])
    last = windows[0]
    for row in windows[1:]:
        if row["action_label"] != current:
            segments.append({
                "action": current,
                "start_frame": start,
                "end_frame": int(last["end_frame"]),
                "start_window": start_idx,
                "end_window": int(last["window_index"]),
            })
            current = row["action_label"]
            start = int(row["start_frame"])
            start_idx = int(row["window_index"])
        last = row
    segments.append({
        "action": current,
        "start_frame": start,
        "end_frame": int(last["end_frame"]),
        "start_window": start_idx,
        "end_window": int(last["window_index"]),
    })
    return segments


def build_data(args: argparse.Namespace) -> dict:
    suite_dir = args.suite_dir
    diagnostics_dir = args.diagnostics_dir
    windows = read_csv(suite_dir / "windows.csv")
    manifest = read_json(suite_dir / "feature_manifest.json")
    summary = read_json(suite_dir / "summary_report.json")
    provenance = read_json(diagnostics_dir / "provenance.json")
    object_rows = {int(r["window_index"]): r for r in read_csv(diagnostics_dir / "object_labels/window_object_labels.csv")}
    ablation_rows = read_csv(diagnostics_dir / "modality_ablation/ablation_metrics.csv")
    for row in ablation_rows:
        task = row.get("task")
        if task in TASK_DISPLAY_ALL:
            row["task_display_name"] = TASK_DISPLAY_ALL[task]
    alignment_rows = read_csv(diagnostics_dir / "alignment_stress/alignment_shift_metrics.csv")
    timeline_rows = read_csv(diagnostics_dir / "timeline_overlay/timeline_overlay.csv")
    predictions = load_predictions(suite_dir)
    X = np.load(suite_dir / "shared_windows.npz")["X"].astype(np.float32)

    block_stats = {}
    block_meta = []
    for block in manifest:
        name = block["name"]
        start, end = int(block["start"]), int(block["end"])
        values = X[:, start:end]
        l2 = np.linalg.norm(values, axis=1)
        mean_abs = np.mean(np.abs(values), axis=1)
        max_l2 = float(max(l2.max(), 1e-8))
        block_stats[name] = {
            "l2": l2,
            "mean_abs": mean_abs,
            "relative": l2 / max_l2,
        }
        block_meta.append({
            "name": name,
            "display": BLOCK_DISPLAY.get(name, name.replace("_", " ").title()),
            "modality": block_modality(name),
            "start": start,
            "end": end,
            "dim": int(block["dim"]),
        })

    explorer_windows = []
    for i, row in enumerate(windows):
        idx = int(row["window_index"])
        obj = object_rows.get(idx, {})
        feature_stats = []
        for block in block_meta:
            s = block_stats[block["name"]]
            feature_stats.append({
                "name": block["name"],
                "l2": round(float(s["l2"][i]), 6),
                "mean_abs": round(float(s["mean_abs"][i]), 6),
                "relative": round(float(s["relative"][i]), 6),
            })
        task_predictions = {}
        for task, rows_by_window in predictions.items():
            task_predictions[task] = rows_by_window.get(idx)
        explorer_windows.append({
            "window_index": idx,
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "center_frame": int(row["center_frame"]),
            "action": row["action_label"],
            "subtask": row["subtask_label"],
            "objects": [x for x in obj.get("objects", "").split("|") if x],
            "feature_stats": feature_stats,
            "predictions": task_predictions,
        })

    best_ablation = {}
    for task in sorted({r["task"] for r in ablation_rows}):
        computed = [r for r in ablation_rows if r["task"] == task and r["status"] == "computed" and r["score"]]
        if not computed:
            continue
        best = max(computed, key=lambda r: float(r["score"]))
        non_overlap = [r for r in computed if r.get("target_source_overlap") == "false"]
        best_non_overlap = max(non_overlap, key=lambda r: float(r["score"])) if non_overlap else None
        best_ablation[task] = {
            "task": task,
            "task_display_name": TASK_DISPLAY.get(task, task_display_name(task)),
            "best": {
                "modality_group": best["modality_group"],
                "modality_display": best["modality_display"],
                "score": float(best["score"]),
                "primary_metric": best["primary_metric"],
                "target_source_overlap": best["target_source_overlap"],
            },
            "best_non_overlap": None if best_non_overlap is None else {
                "modality_group": best_non_overlap["modality_group"],
                "modality_display": best_non_overlap["modality_display"],
                "score": float(best_non_overlap["score"]),
                "primary_metric": best_non_overlap["primary_metric"],
            },
        }

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_count": len(explorer_windows),
            "feature_dim": int(X.shape[1]),
            "object_label_rows": len(object_rows),
            "object_vocab_count": len(read_json(diagnostics_dir / "object_labels/object_vocab.json")["vocab"]),
            "timeline_prediction_rows": len(timeline_rows),
            "source_policy": "Window-level labels, features, predictions, and diagnostics only. Raw Xperience-10M MP4/HDF5/RRD files are not embedded.",
            "annotation_hash_recorded": any("annotation.hdf5" in key for key in provenance["input_file_hashes"]),
            "summary": {
                "num_windows": summary.get("num_windows"),
                "feature_dim": summary.get("feature_dim"),
                "window_frames": summary.get("window_frames"),
                "stride_frames": summary.get("stride_frames"),
            },
        },
        "tasks": TASK_DISPLAY,
        "task_display_names": TASK_DISPLAY_ALL,
        "feature_blocks": block_meta,
        "segments": build_action_segments(windows),
        "windows": explorer_windows,
        "ablation": {
            "best_by_task": best_ablation,
            "rows": ablation_rows,
        },
        "alignment": alignment_rows,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Single-Episode Explorer | Ropedia Xperience-10M</title>
  <meta name="description" content="Interactive window-level explorer for the Ropedia Xperience-10M single-episode diagnostics.">
  <meta name="theme-color" content="#020502">
  <link rel="icon" href="favicon.png" type="image/png" sizes="64x64">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root { color-scheme: dark; --page:#020502; --panel:#071207; --surface:#0b1709; --ink:#f4f8ef; --muted:#a7b5a3; --line:rgba(204,255,160,.24); --soft:rgba(204,255,160,.14); --green:#ccffa0; --cyan:#7ae5c3; --blue:#9bdfff; --red:#ff8f7a; --amber:#d8f4a5; --card:rgba(5,10,6,.84); --pill:rgba(255,255,255,.05); --font-ui:"Inter Tight",ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; --font-copy:"Inter Tight",ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; --font-btn:"Space Grotesk",ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; --max:1400px; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--page); color:var(--ink); font-family:var(--font-copy); line-height:1.5; font-synthesis-weight:none; }
    a { color:inherit; }
    .wrap { width:min(var(--max), calc(100% - 42px)); margin:0 auto; }
    header { position:sticky; top:0; z-index:10; background:rgba(2,5,2,.92); backdrop-filter:blur(16px); border-bottom:1px solid var(--soft); }
    .nav { height:64px; display:flex; align-items:center; justify-content:space-between; gap:18px; }
    .brand { display:flex; gap:11px; align-items:center; text-decoration:none; font-family:var(--font-ui); font-weight:700; }
    .brand img { width:38px; height:38px; border:1px solid rgba(204,255,160,.42); border-radius:8px; background:#061006; }
    .nav-links { display:flex; gap:10px; color:#c9d5c5; font-family:var(--font-btn); font-size:14px; }
    .nav-links a { min-height:36px; display:inline-flex; align-items:center; justify-content:center; padding:0 14px; border:1px solid transparent; border-radius:999px; background:var(--pill); text-decoration:none; }
    .nav-links a:hover { border-color:var(--green); color:var(--green); background:rgba(255,255,255,.08); }
    .hero { padding:78px 0 38px; border-bottom:1px solid var(--soft); background:linear-gradient(90deg, rgba(2,5,2,.95) 0%, rgba(2,5,2,.82) 55%, rgba(2,5,2,.70) 100%), linear-gradient(180deg, rgba(2,5,2,.08), rgba(5,6,11,.96)), url("assets/modalities/video.jpg") center right / cover no-repeat; }
    h1 { max-width:900px; margin:0; font-family:var(--font-ui); font-size:clamp(42px, 6vw, 76px); line-height:.98; letter-spacing:0; }
    .hero p { max-width:820px; margin:22px 0 0; color:#c7d1c3; font-size:18px; line-height:1.62; }
    .stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:28px; max-width:900px; }
    .stat { border:1px solid rgba(204,255,160,.18); border-radius:8px; background:var(--card); padding:13px 14px; }
    .stat strong { display:block; font-family:var(--font-ui); font-size:24px; line-height:1; }
    .stat span { display:block; margin-top:6px; color:var(--muted); font-size:12px; }
    main { padding:26px 0 70px; }
    .shell { display:grid; grid-template-columns:330px minmax(0,1fr); gap:18px; align-items:start; }
    .panel { border:1px solid rgba(204,255,160,.18); border-radius:8px; background:var(--card); box-shadow:0 18px 48px rgba(0,0,0,.32); }
    .side { position:sticky; top:84px; padding:18px; }
    label { display:block; color:var(--muted); font-size:12px; font-family:var(--font-btn); font-weight:700; margin:14px 0 7px; }
    input[type=range] { width:100%; accent-color:var(--green); }
    select, input[type=search] { width:100%; min-height:40px; border:1px solid var(--soft); border-radius:999px; background:#020802; color:var(--ink); padding:9px 12px; font:inherit; }
    .button-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; }
    button { border:1px solid rgba(255,255,255,.12); border-radius:999px; background:var(--pill); color:var(--ink); min-height:38px; font:700 13px var(--font-btn); cursor:pointer; }
    button:hover { border-color:var(--green); color:var(--green); background:rgba(255,255,255,.08); }
    .timeline { padding:18px; margin-bottom:18px; }
    .timeline-strip { position:relative; height:60px; border:1px solid var(--soft); border-radius:8px; overflow:hidden; background:#030803; }
    .segment { position:absolute; top:0; bottom:0; border-right:1px solid rgba(2,5,2,.45); opacity:.92; }
    .marker { position:absolute; top:0; bottom:0; width:3px; background:var(--ink); box-shadow:0 0 0 2px rgba(2,5,2,.9), 0 0 18px rgba(204, 255, 160,.6); }
    .timeline-meta { display:flex; justify-content:space-between; gap:12px; margin-top:10px; color:var(--muted); font-size:12px; }
    .content { display:grid; gap:18px; }
    .window-panel { padding:22px; }
    .window-head { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:18px; align-items:start; border-bottom:1px solid var(--soft); padding-bottom:18px; }
    h2 { margin:0; font-family:var(--font-ui); font-size:30px; line-height:1.1; }
    .frame-pill { border:1px solid var(--line); border-radius:999px; padding:8px 10px; color:var(--green); font-family:var(--font-btn); font-size:13px; font-weight:700; white-space:nowrap; background:var(--pill); }
    .subtle { color:var(--muted); }
    .chips { display:flex; flex-wrap:wrap; gap:7px; margin-top:12px; }
    .chip { border:1px solid var(--soft); background:var(--pill); color:#e7f2df; border-radius:999px; padding:5px 8px; font-size:12px; }
    .grid { display:grid; gap:12px; }
    .pred-grid { grid-template-columns:repeat(3,minmax(0,1fr)); margin-top:18px; }
    .pred { border:1px solid rgba(204,255,160,.18); border-radius:8px; background:rgba(5,10,6,.72); padding:13px; min-height:118px; }
    .pred h3 { margin:0 0 8px; font-family:var(--font-ui); font-size:15px; }
    .pred p { margin:4px 0; color:#cdd8c8; font-size:13px; overflow-wrap:anywhere; }
    .pred .ok { color:var(--green); font-weight:700; }
    .pred .bad { color:var(--red); font-weight:700; }
    .feature-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .feature { display:grid; grid-template-columns:130px 1fr 62px; gap:10px; align-items:center; border-bottom:1px solid var(--soft); padding:10px 0; }
    .feature:last-child { border-bottom:0; }
    .feature-name { font-family:var(--font-ui); font-size:13px; color:#edf6e8; }
    .bar { height:10px; border-radius:999px; background:rgba(204,255,160,.16); overflow:hidden; }
    .bar span { display:block; height:100%; width:calc(var(--w) * 1%); background:linear-gradient(90deg,var(--green),rgba(204,255,160,.45)); }
    .num { text-align:right; color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }
    .analysis-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    .analysis { padding:18px; }
    .analysis h3 { margin:0 0 12px; font-family:var(--font-ui); font-size:20px; }
    .rows { display:grid; gap:8px; }
    .row { display:grid; grid-template-columns:1fr auto; gap:12px; border-bottom:1px solid var(--soft); padding:8px 0; color:#d8e4d3; font-size:13px; }
    .row strong { color:var(--green); font-variant-numeric:tabular-nums; }
    .note { margin-top:12px; color:var(--muted); font-size:12px; line-height:1.55; }
    @media (max-width: 980px) { .shell,.analysis-grid { grid-template-columns:1fr; } .side { position:static; } .pred-grid,.feature-grid,.stats { grid-template-columns:1fr; } .window-head { grid-template-columns:1fr; } .nav-links { display:none; } }
  </style>
</head>
<body>
  <header>
    <div class="wrap nav">
      <a class="brand" href="index.html"><img src="assets/brand/xperience10m-logo-mark-192.png" alt=""><span>Ropedia Xperience-10M</span></a>
      <nav class="nav-links"><a href="index.html">Project</a><a href="single_episode_explorer.html">Explorer</a><a href="data/single_episode_explorer.json">Data JSON</a></nav>
    </div>
  </header>
  <section class="hero">
    <div class="wrap">
      <h1>Single-Episode Research Explorer</h1>
      <p>Inspect the exported Xperience-10M sample windows, real object labels, model predictions, feature-block statistics, and diagnostic scores from one aligned episode.</p>
      <div class="stats">
        <div class="stat"><strong id="statWindows">-</strong><span>windows</span></div>
        <div class="stat"><strong id="statDim">-</strong><span>feature dimensions</span></div>
        <div class="stat"><strong id="statObjects">-</strong><span>object labels</span></div>
        <div class="stat"><strong id="statPreds">-</strong><span>prediction rows</span></div>
      </div>
    </div>
  </section>
  <main>
    <div class="wrap shell">
      <aside class="panel side">
        <label for="windowRange">Window</label>
        <input id="windowRange" type="range" min="0" max="0" value="0">
        <div class="button-row"><button id="prevWindow" type="button">Previous</button><button id="nextWindow" type="button">Next</button></div>
        <label for="taskSelect">Task Focus</label>
        <select id="taskSelect"></select>
        <label for="searchBox">Search Action or Object</label>
        <input id="searchBox" type="search" placeholder="e.g. Pour coffee, kettle">
        <div class="button-row"><button id="firstMatch" type="button">First Match</button><button id="firstPred" type="button">First Predicted</button></div>
        <p class="note">The page uses window-level exported artifacts only. Raw video, raw HDF5, and RRD assets are not embedded.</p>
      </aside>
      <section class="content">
        <div class="panel timeline">
          <div class="timeline-strip" id="timelineStrip"></div>
          <div class="timeline-meta"><span id="timelineLeft"></span><span id="timelineRight"></span></div>
        </div>
        <section class="panel window-panel">
          <div class="window-head">
            <div>
              <h2 id="windowTitle">Window</h2>
              <p id="windowSubtitle" class="subtle"></p>
              <div class="chips" id="objectChips"></div>
            </div>
            <div class="frame-pill" id="framePill"></div>
          </div>
          <div class="grid pred-grid" id="predictionGrid"></div>
        </section>
        <section class="analysis-grid">
          <div class="panel analysis">
            <h3>Feature Blocks</h3>
            <div class="grid feature-grid" id="featureGrid"></div>
          </div>
          <div class="panel analysis">
            <h3>Diagnostics</h3>
            <div class="rows" id="diagnosticRows"></div>
            <p class="note" id="diagnosticNote"></p>
          </div>
        </section>
      </section>
    </div>
  </main>
  <script id="explorer-data" type="application/json">__DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("explorer-data").textContent);
    function hasPrediction(windowRecord, taskKey) {
      return taskKey === "all" ? Object.values(windowRecord.predictions).some(Boolean) : Boolean(windowRecord.predictions[taskKey]);
    }
    function defaultWindowIndex() {
      let best = 0;
      let bestCount = -1;
      DATA.windows.forEach((w) => {
        const count = Object.values(w.predictions).filter(Boolean).length;
        if (count > bestCount) { best = w.window_index; bestCount = count; }
      });
      return best;
    }
    const state = { index: defaultWindowIndex(), task: "all" };
    const range = document.getElementById("windowRange");
    const taskSelect = document.getElementById("taskSelect");
    const searchBox = document.getElementById("searchBox");
    const colors = ["#5ccf7d", "#7ae5c3", "#9bdfff", "#d8f4a5", "#f0a45e", "#cba8ff", "#ff8f7a"];
    document.getElementById("statWindows").textContent = DATA.meta.window_count;
    document.getElementById("statDim").textContent = DATA.meta.feature_dim;
    document.getElementById("statObjects").textContent = DATA.meta.object_vocab_count;
    document.getElementById("statPreds").textContent = DATA.meta.timeline_prediction_rows;
    range.max = DATA.windows.length - 1;
    for (const [key, label] of Object.entries(DATA.tasks)) {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = label;
      taskSelect.appendChild(option);
    }
    const allOption = document.createElement("option");
    allOption.value = "all";
    allOption.textContent = "All Prediction Cards";
    taskSelect.insertBefore(allOption, taskSelect.firstChild);
    taskSelect.value = state.task;
    function pct(value, min, max) { return ((value - min) / Math.max(1, max - min)) * 100; }
    function splitObjects(value) { return String(value || "").split("|").filter(Boolean); }
    function renderTimeline() {
      const strip = document.getElementById("timelineStrip");
      strip.innerHTML = "";
      const minFrame = DATA.windows[0].start_frame;
      const maxFrame = DATA.windows[DATA.windows.length - 1].end_frame;
      DATA.segments.forEach((seg, i) => {
        const el = document.createElement("div");
        el.className = "segment";
        el.style.left = pct(seg.start_frame, minFrame, maxFrame) + "%";
        el.style.width = Math.max(0.3, pct(seg.end_frame, minFrame, maxFrame) - pct(seg.start_frame, minFrame, maxFrame)) + "%";
        el.style.background = colors[i % colors.length];
        el.title = `${seg.action} (${seg.start_frame}-${seg.end_frame})`;
        el.addEventListener("click", () => { state.index = seg.start_window; render(); });
        strip.appendChild(el);
      });
      const marker = document.createElement("div");
      marker.className = "marker";
      marker.style.left = pct(DATA.windows[state.index].center_frame, minFrame, maxFrame) + "%";
      strip.appendChild(marker);
      document.getElementById("timelineLeft").textContent = `frame ${minFrame}`;
      document.getElementById("timelineRight").textContent = `frame ${maxFrame}`;
    }
    function renderPredictions(w) {
      const grid = document.getElementById("predictionGrid");
      grid.innerHTML = "";
      const taskEntries = Object.entries(DATA.tasks).filter(([key]) => state.task === "all" || key === state.task);
      for (const [key, label] of taskEntries) {
        const pred = w.predictions[key];
        const card = document.createElement("article");
        card.className = "pred";
        let body = "";
        if (!pred) {
          body = `<p class="subtle">No held-out prediction row for this window.</p>`;
        } else {
          const status = pred.correct ? `<span class="ok">correct</span>` : `<span class="bad">mismatch</span>`;
          body = `<p>${status}</p><p><strong>true</strong>: ${escapeHtml(pred.true || "")}</p><p><strong>pred</strong>: ${escapeHtml(pred.predicted || "")}</p>`;
          if (pred.confidence) body += `<p><strong>confidence</strong>: ${Number(pred.confidence).toFixed(3)}</p>`;
        }
        card.innerHTML = `<h3>${escapeHtml(label)}</h3>${body}`;
        grid.appendChild(card);
      }
    }
    function renderFeatures(w) {
      const grid = document.getElementById("featureGrid");
      grid.innerHTML = "";
      for (const stat of w.feature_stats) {
        const block = DATA.feature_blocks.find((b) => b.name === stat.name);
        const row = document.createElement("div");
        row.className = "feature";
        row.innerHTML = `<span class="feature-name">${escapeHtml(block.display)}</span><span class="bar"><span style="--w:${Math.round(stat.relative * 100)}"></span></span><span class="num">${stat.l2.toFixed(2)}</span>`;
        grid.appendChild(row);
      }
    }
    function renderDiagnostics() {
      const rows = document.getElementById("diagnosticRows");
      rows.innerHTML = "";
      const task = state.task === "all" ? "object_relevance" : state.task;
      const diag = DATA.ablation.best_by_task[task];
      if (diag) {
        rows.innerHTML += `<div class="row"><span>Best modality for ${escapeHtml(DATA.tasks[task] || task)}</span><strong>${escapeHtml(diag.best.modality_display)} ${diag.best.score.toFixed(3)}</strong></div>`;
        if (diag.best_non_overlap) rows.innerHTML += `<div class="row"><span>Best non-overlap modality</span><strong>${escapeHtml(diag.best_non_overlap.modality_display)} ${diag.best_non_overlap.score.toFixed(3)}</strong></div>`;
      }
      const zeroRows = DATA.alignment.filter((r) => Number(r.shift_windows) === 0);
      zeroRows.slice(0, 5).forEach((r) => {
        rows.innerHTML += `<div class="row"><span>${escapeHtml(r.query_display)} zero-shift retrieval MRR</span><strong>${Number(r.mrr).toFixed(3)}</strong></div>`;
      });
      document.getElementById("diagnosticNote").textContent = DATA.meta.source_policy;
    }
    function renderWindow() {
      const w = DATA.windows[state.index];
      range.value = state.index;
      document.getElementById("windowTitle").textContent = `Window ${w.window_index}: ${w.action || "unlabeled action"}`;
      document.getElementById("windowSubtitle").textContent = w.subtask || "No subtask label";
      document.getElementById("framePill").textContent = `frames ${w.start_frame}-${w.end_frame}`;
      const chips = document.getElementById("objectChips");
      chips.innerHTML = "";
      (w.objects.length ? w.objects : ["no object label"]).forEach((obj) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = obj;
        chips.appendChild(chip);
      });
      renderPredictions(w);
      renderFeatures(w);
      renderDiagnostics();
    }
    function render() { renderTimeline(); renderWindow(); }
    function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c])); }
    range.addEventListener("input", () => { state.index = Number(range.value); render(); });
    taskSelect.addEventListener("change", () => { state.task = taskSelect.value; render(); });
    document.getElementById("prevWindow").addEventListener("click", () => { state.index = Math.max(0, state.index - 1); render(); });
    document.getElementById("nextWindow").addEventListener("click", () => { state.index = Math.min(DATA.windows.length - 1, state.index + 1); render(); });
    document.getElementById("firstPred").addEventListener("click", () => {
      const found = DATA.windows.find((w) => hasPrediction(w, state.task));
      if (found) { state.index = found.window_index; render(); }
    });
    document.getElementById("firstMatch").addEventListener("click", () => {
      const q = searchBox.value.trim().toLowerCase();
      if (!q) return;
      const found = DATA.windows.find((w) => [w.action, w.subtask, ...w.objects].join(" ").toLowerCase().includes(q));
      if (found) { state.index = found.window_index; render(); }
    });
    render();
  </script>
</body>
</html>
"""


def write_html(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False).replace("</script", "<\\/script")
    path.write_text(HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = build_data(args)
    write_json(args.docs_dir / "data/single_episode_explorer.json", data)
    write_html(args.docs_dir / "single_episode_explorer.html", data)
    print(f"Wrote {args.docs_dir / 'data/single_episode_explorer.json'}")
    print(f"Wrote {args.docs_dir / 'single_episode_explorer.html'}")


if __name__ == "__main__":
    main()
