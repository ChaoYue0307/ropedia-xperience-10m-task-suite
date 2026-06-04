#!/usr/bin/env python3
"""Minimum real-data adapter setup check for a Ropedia -> Qwen3-Omni path.

This script does not pretend to fine-tune Qwen3-Omni itself. It validates the
part that Ropedia-specific sensor modalities need before they can be attached
to an omni backbone: windowing real episodes, turning sensor blocks into
adapter tokens, and training/evaluating a small task head on real labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DIRECT_QWEN3_INPUTS = [
    "rgb/fisheye video",
    "embedded mp4 audio",
    "language annotation prompt",
]

ADAPTER_INPUTS = [
    "depth/confidence",
    "pose/SLAM camera trajectory",
    "motion capture hand/body joints",
    "IMU accel/gyro",
    "contacts/object state features",
]


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run a real-data Ropedia sensor-adapter setup check.")
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--run-id", default="adapter_only")
    parser.add_argument(
        "--episode-root",
        type=Path,
        action="append",
        help="Episode folder containing annotation.hdf5. May be passed multiple times.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest produced by build_episode_manifest.py. Episodes from this file are appended.",
    )
    parser.add_argument("--target", choices=["action", "subtask"], default="action")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=workspace_default / "outputs/omni_exploration/feature_cache")
    parser.add_argument("--base-model-id", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--window-frames", type=int, default=20)
    parser.add_argument("--stride-frames", type=int, default=20)
    parser.add_argument("--min-label-fraction", type=float, default=0.6)
    parser.add_argument("--max-windows-per-episode", type=int, default=128)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--video-image-size", type=int, default=32)
    parser.add_argument("--video-grid-size", type=int, default=8)
    parser.add_argument("--video-hist-bins", type=int, default=8)
    parser.add_argument("--depth-grid-size", type=int, default=8)
    parser.add_argument("--text-hash-dim", type=int, default=128)
    parser.add_argument("--include-label-text", action="store_true")
    parser.add_argument(
        "--skip-video-features",
        action="store_true",
        help="Do not decode MP4s for handcrafted visual features. The direct Qwen3 path is still recorded.",
    )
    return parser.parse_args()


def add_repo_imports(workspace: Path) -> None:
    scripts = workspace / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))


def add_toolkit_to_path(workspace: Path) -> None:
    toolkit = workspace / "HOMIE-toolkit"
    if not toolkit.exists():
        raise FileNotFoundError(f"HOMIE-toolkit not found: {toolkit}")
    if str(toolkit) not in sys.path:
        sys.path.insert(0, str(toolkit))


def episode_dirs_from_args(args: argparse.Namespace) -> list[Path]:
    episode_dirs: list[Path] = []
    if args.episode_root:
        episode_dirs.extend(path.expanduser().resolve() for path in args.episode_root)
    if args.manifest:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        for ep in payload.get("episodes", []):
            path = Path(ep["path"]).expanduser().resolve()
            if path not in episode_dirs:
                episode_dirs.append(path)
    if not episode_dirs:
        default = args.workspace / "data/sample/xperience-10m-sample"
        episode_dirs.append(default.resolve())
    return episode_dirs


@dataclass
class EpisodeDataset:
    episode_id: str
    X: np.ndarray
    labels: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    feature_manifest: list[dict]
    available_modalities: list[dict]


def load_episode(args: argparse.Namespace, episode_dir: Path) -> EpisodeDataset:
    from data_loader import load_from_annotation_hdf5
    from train_all_modalities_model import build_feature_dataset, prepare_modalities, VIDEO_FILES

    annotation = episode_dir / "annotation.hdf5"
    if not annotation.exists():
        raise FileNotFoundError(f"Missing annotation.hdf5: {annotation}")

    ann = load_from_annotation_hdf5(annotation, 0, None, load_slam_point_cloud=True)
    local_args = argparse.Namespace(**vars(args))
    local_args.annotation = annotation
    local_args.cache_dir = args.cache_dir / f"{episode_dir.parent.name}__{episode_dir.name}"

    if args.skip_video_features:
        original_video_files = VIDEO_FILES.copy()
        VIDEO_FILES.clear()
        try:
            extras, available_modalities = prepare_modalities(local_args, ann)
        finally:
            VIDEO_FILES.clear()
            VIDEO_FILES.update(original_video_files)
    else:
        extras, available_modalities = prepare_modalities(local_args, ann)

    X, labels, starts, ends, _label_fracs, feature_manifest = build_feature_dataset(
        ann,
        extras,
        target=args.target,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        min_label_fraction=args.min_label_fraction,
    )
    if args.max_windows_per_episode > 0 and len(labels) > args.max_windows_per_episode:
        keep = np.linspace(0, len(labels) - 1, args.max_windows_per_episode, dtype=np.int64)
        X = X[keep]
        labels = labels[keep]
        starts = starts[keep]
        ends = ends[keep]
    return EpisodeDataset(
        episode_id=episode_dir.name,
        X=X.astype(np.float32),
        labels=labels,
        starts=starts,
        ends=ends,
        feature_manifest=feature_manifest,
        available_modalities=available_modalities,
    )


def align_feature_dims(episodes: list[EpisodeDataset]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    max_dim = max(ep.X.shape[1] for ep in episodes)
    Xs, labels, episode_ids, window_ids = [], [], [], []
    for ep in episodes:
        X = ep.X
        if X.shape[1] < max_dim:
            padded = np.zeros((X.shape[0], max_dim), dtype=np.float32)
            padded[:, : X.shape[1]] = X
            X = padded
        Xs.append(X)
        labels.extend([str(x) for x in ep.labels])
        episode_ids.extend([ep.episode_id] * len(ep.labels))
        window_ids.extend(range(len(ep.labels)))
    manifest = max(episodes, key=lambda ep: ep.X.shape[1]).feature_manifest
    return (
        np.concatenate(Xs, axis=0).astype(np.float32),
        np.asarray(labels, dtype=object),
        np.asarray(episode_ids, dtype=object),
        np.asarray(window_ids, dtype=np.int64),
        manifest,
    )


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    seen = OrderedDict()
    for label in labels:
        if label not in seen:
            seen[label] = len(seen)
    return np.asarray([seen[label] for label in labels], dtype=np.int64), list(seen.keys())


def split_indices(episode_ids: np.ndarray, labels: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    unique_episodes = np.unique(episode_ids)
    if len(unique_episodes) >= 2:
        n_test = max(1, int(round(len(unique_episodes) * test_fraction)))
        heldout = set(unique_episodes[-n_test:].tolist())
        test = np.asarray([i for i, ep in enumerate(episode_ids) if ep in heldout], dtype=np.int64)
        train = np.asarray([i for i, ep in enumerate(episode_ids) if ep not in heldout], dtype=np.int64)
        return train, test, "held_out_episode"

    # Single-episode smoke: use a chronological split, not shuffled windows.
    n = len(labels)
    cut = max(1, min(n - 1, int(round(n * (1.0 - test_fraction)))))
    return np.arange(cut, dtype=np.int64), np.arange(cut, n, dtype=np.int64), "single_episode_chronological"


def block_slices(feature_manifest: list[dict], input_dim: int) -> list[tuple[str, int, int]]:
    slices = []
    for block in feature_manifest:
        start = int(block["start"])
        end = min(int(block["end"]), input_dim)
        if start < end:
            slices.append((str(block["name"]), start, end))
    if not slices:
        slices.append(("all_features", 0, input_dim))
    return slices


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, list[dict], np.ndarray]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    rows = []
    f1s = []
    for idx in range(n_classes):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        rows.append({
            "class_id": idx,
            "support": int(cm[idx, :].sum()),
            "predicted": int(cm[:, idx].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return float(np.mean(f1s)) if f1s else 0.0, rows, cm


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_adapter_model(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    blocks: list[tuple[str, int, int]],
    args: argparse.Namespace,
) -> tuple[dict, np.ndarray, list[dict]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SensorAdapterClassifier(nn.Module):
        def __init__(self, block_specs: list[tuple[str, int, int]], hidden_dim: int, n_classes: int, n_layers: int):
            super().__init__()
            self.block_specs = block_specs
            self.adapters = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(end - start),
                    nn.Linear(end - start, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _name, start, end in block_specs
            ])
            self.type_embedding = nn.Parameter(torch.randn(len(block_specs), hidden_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, min(8, hidden_dim // 32)),
                dim_feedforward=hidden_dim * 4,
                dropout=0.10,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.fusion = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_classes))

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            tokens = []
            for adapter, (_name, start, end) in zip(self.adapters, self.block_specs):
                tokens.append(adapter(features[:, start:end]))
            x = torch.stack(tokens, dim=1) + self.type_embedding.unsqueeze(0)
            x = self.fusion(x)
            pooled = x.mean(dim=1)
            return self.head(pooled)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    X_mean = X[train_idx].mean(axis=0, keepdims=True)
    X_std = X[train_idx].std(axis=0, keepdims=True)
    X_std[X_std < 1e-6] = 1.0
    Xs = (X - X_mean) / X_std

    x_tensor = torch.from_numpy(Xs.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.int64))
    train_tensor = torch.from_numpy(train_idx.astype(np.int64))
    test_tensor = torch.from_numpy(test_idx.astype(np.int64))

    n_classes = int(y.max()) + 1
    model = SensorAdapterClassifier(blocks, args.hidden_dim, n_classes, args.transformer_layers).to(device)
    counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        perm = train_tensor[torch.randperm(len(train_tensor), generator=gen)]
        total_loss = 0.0
        correct = 0
        seen = 0
        model.train()
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            xb = x_tensor[idx].to(device)
            yb = y_tensor[idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().detach().cpu())
            seen += len(idx)
        history.append({"epoch": epoch, "loss": total_loss / max(seen, 1), "train_accuracy": correct / max(seen, 1)})

    model.eval()
    with torch.no_grad():
        logits = []
        for start in range(0, len(test_tensor), args.batch_size):
            idx = test_tensor[start : start + args.batch_size]
            logits.append(model(x_tensor[idx].to(device)).detach().cpu())
        test_logits = torch.cat(logits, dim=0)
        probs = torch.softmax(test_logits, dim=1).numpy()
        pred = probs.argmax(axis=1).astype(np.int64)

    artifact = {
        "model_state": model.state_dict(),
        "feature_mean": X_mean.astype(np.float32),
        "feature_std": X_std.astype(np.float32),
        "blocks": blocks,
        "history": history,
    }
    return artifact, pred, history


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.workspace / "results" / "omni_finetune" / args.run_id / "adapter_only"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    add_repo_imports(args.workspace)
    add_toolkit_to_path(args.workspace)

    episode_dirs = episode_dirs_from_args(args)
    episodes = [load_episode(args, path) for path in episode_dirs]
    X, labels, episode_ids, window_ids, feature_manifest = align_feature_dims(episodes)
    y, class_names = encode_labels(labels)
    train_idx, test_idx, split_name = split_indices(episode_ids, labels, args.test_fraction, args.seed)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Need non-empty train and test splits.")
    blocks = block_slices(feature_manifest, X.shape[1])

    artifact, pred, history = train_adapter_model(X, y, train_idx, test_idx, blocks, args)
    y_test = y[test_idx]
    accuracy = float(np.mean(pred == y_test))
    macro, per_class, cm = macro_f1(y_test, pred, len(class_names))
    for row in per_class:
        row["class_name"] = class_names[int(row["class_id"])]

    import torch

    model_path = args.output_dir / "sensor_adapter_model.pt"
    torch.save(artifact, model_path)

    prediction_rows = []
    for local_pos, pred_id in enumerate(pred):
        absolute_idx = int(test_idx[local_pos])
        true_id = int(y[absolute_idx])
        prediction_rows.append({
            "sample_index": absolute_idx,
            "episode_id": str(episode_ids[absolute_idx]),
            "window_id": int(window_ids[absolute_idx]),
            "true_label": class_names[true_id],
            "predicted_label": class_names[int(pred_id)],
            "correct": int(true_id == int(pred_id)),
        })

    metrics = {
        "task": f"qwen3_omni_sensor_adapter_smoke_{args.target}",
        "base_model_target": args.base_model_id,
        "qwen3_loaded": False,
        "qwen3_note": "This run validates Ropedia sensor-adapter tokens and task heads before loading or LoRA-tuning Qwen3-Omni.",
        "split": split_name,
        "num_episodes": len(episodes),
        "num_windows": int(len(labels)),
        "num_train_windows": int(len(train_idx)),
        "num_test_windows": int(len(test_idx)),
        "num_classes": int(len(class_names)),
        "feature_dim": int(X.shape[1]),
        "num_adapter_tokens": int(len(blocks)),
        "accuracy": accuracy,
        "macro_f1": macro,
        "train_final_loss": float(history[-1]["loss"]),
        "train_final_accuracy": float(history[-1]["train_accuracy"]),
        "direct_qwen3_inputs": DIRECT_QWEN3_INPUTS,
        "adapter_required_inputs": ADAPTER_INPUTS,
    }

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    (args.output_dir / "adapter_blocks.json").write_text(
        json.dumps([{"name": name, "start": start, "end": end, "dim": end - start} for name, start, end in blocks], indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "available_modalities.json").write_text(
        json.dumps([{"episode_id": ep.episode_id, "modalities": ep.available_modalities} for ep in episodes], indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "predictions.csv", prediction_rows, ["sample_index", "episode_id", "window_id", "true_label", "predicted_label", "correct"])
    write_csv(
        args.output_dir / "per_class_metrics.csv",
        per_class,
        ["class_id", "class_name", "support", "predicted", "precision", "recall", "f1"],
    )
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true\\pred"] + class_names)
        for i, name in enumerate(class_names):
            writer.writerow([name] + [int(x) for x in cm[i]])

    report = [
        "# Qwen3-Omni Adapter Setup Check",
        "",
        f"- Base model target: `{args.base_model_id}`",
        "- Qwen3-Omni weights loaded: `false`",
        f"- Episodes: `{len(episodes)}`",
        f"- Windows: `{len(labels)}` total, `{len(train_idx)}` train, `{len(test_idx)}` test",
        f"- Split: `{split_name}`",
        f"- Feature dimension: `{X.shape[1]}`",
        f"- Adapter soft-token blocks: `{len(blocks)}`",
        f"- Accuracy: `{accuracy:.4f}`",
        f"- Macro-F1: `{macro:.4f}`",
        "",
        "## Why this is the minimum real test",
        "",
        "This run uses real Ropedia annotation/video-derived feature blocks. It tests the sensor-adapter side that depth, pose, mocap, contacts, and IMU need before those tokens are attached to Qwen3-Omni. It deliberately avoids downloading the 30B Qwen3-Omni weights until the data path, labels, splits, and storage plan are confirmed.",
    ]
    (args.output_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
