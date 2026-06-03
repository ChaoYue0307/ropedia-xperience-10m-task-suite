#!/usr/bin/env python3
"""Download selected Xperience-10M episodes in relay-sized batches.

Intended host: an HF-reachable relay machine with limited disk. The script
downloads one batch, optionally rsyncs it to a training host, writes progress
records, and can delete the local batch after transfer.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


REQUIRED_FILES = [
    "annotation.hdf5",
    "fisheye_cam0.mp4",
    "fisheye_cam1.mp4",
    "fisheye_cam2.mp4",
    "fisheye_cam3.mp4",
    "stereo_left.mp4",
    "stereo_right.mp4",
]


@dataclass
class Batch:
    index: int
    episodes: list[dict[str, Any]]

    @property
    def bytes(self) -> int:
        return sum(int(ep["training_bytes_excluding_visualization_rrd"]) for ep in self.episodes)

    @property
    def file_count(self) -> int:
        return sum(len(ep["download_files"]) for ep in self.episodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ropedia-ai/xperience-10m")
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--relay-root", type=Path, required=True)
    parser.add_argument("--batch-max-gib", type=float, default=24.0)
    parser.add_argument("--batch-max-episodes", type=int, default=8)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all remaining batches.")
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future use; downloads are sequential for disk safety.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", "").strip())
    parser.add_argument("--progress-jsonl", type=Path, default=Path("relay_progress.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("relay_summary.json"))
    parser.add_argument("--transfer-host", default="", help="Remote destination, e.g. user@training-host")
    parser.add_argument("--transfer-root", default="", help="Remote directory that receives session/episode folders.")
    parser.add_argument("--transfer-mode", choices=["rsync", "chunked"], default="rsync")
    parser.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh" / "xperience10m_relay_ed25519")
    parser.add_argument("--ssh-extra", default="-o BatchMode=yes -o StrictHostKeyChecking=accept-new")
    parser.add_argument("--chunk-transfer-script", type=Path, default=None)
    parser.add_argument("--chunk-parallel", type=int, default=8)
    parser.add_argument("--chunk-size-mib", type=int, default=8)
    parser.add_argument("--chunk-threshold-mib", type=int, default=8)
    parser.add_argument("--delete-after-transfer", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Do not download; validate selected files already under relay-root.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human_bytes(num: float | int) -> str:
    value = float(num)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_selection(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("selected_episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"No selected_episodes found in {path}")
    for ep in episodes:
        missing = [key for key in ("episode_path", "download_files", "training_bytes_excluding_visualization_rrd") if key not in ep]
        if missing:
            raise ValueError(f"Selection episode missing {missing}: {ep}")
    return episodes


def make_batches(episodes: list[dict[str, Any]], max_bytes: int, max_episodes: int) -> list[Batch]:
    batches: list[Batch] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for ep in episodes:
        ep_bytes = int(ep["training_bytes_excluding_visualization_rrd"])
        would_exceed_bytes = current and current_bytes + ep_bytes > max_bytes
        would_exceed_count = current and len(current) >= max_episodes
        if would_exceed_bytes or would_exceed_count:
            batches.append(Batch(index=len(batches), episodes=current))
            current = []
            current_bytes = 0
        current.append(ep)
        current_bytes += ep_bytes
    if current:
        batches.append(Batch(index=len(batches), episodes=current))
    return batches


def local_file(root: Path, filename: str) -> Path:
    return root / filename


def validate_batch(root: Path, batch: Batch) -> dict[str, Any]:
    missing = []
    size_mismatches = []
    total_bytes = 0
    for ep in batch.episodes:
        expected_by_name = {
            "annotation.hdf5": int(ep["annotation_bytes"]),
        }
        for filename in ep["download_files"]:
            path = local_file(root, filename)
            if not path.exists():
                missing.append(filename)
                continue
            actual = path.stat().st_size
            total_bytes += actual
            expected = expected_by_name.get(Path(filename).name)
            if expected is not None and actual != expected:
                size_mismatches.append({"path": filename, "expected": expected, "actual": actual})
    return {
        "ok": not missing and not size_mismatches,
        "missing": missing,
        "size_mismatches": size_mismatches,
        "local_bytes": total_bytes,
        "local_human": human_bytes(total_bytes),
    }


def download_batch(repo_id: str, token: str, root: Path, batch: Batch, progress_path: Path) -> None:
    for ep in batch.episodes:
        for filename in ep["download_files"]:
            start = time.time()
            append_jsonl(
                progress_path,
                {
                    "time": utc_now(),
                    "event": "download_start",
                    "batch": batch.index,
                    "episode_path": ep["episode_path"],
                    "path": filename,
                },
            )
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(root),
                token=token,
            )
            local = local_file(root, filename)
            append_jsonl(
                progress_path,
                {
                    "time": utc_now(),
                    "event": "download_done",
                    "batch": batch.index,
                    "episode_path": ep["episode_path"],
                    "path": filename,
                    "bytes": local.stat().st_size if local.exists() else 0,
                    "seconds": round(time.time() - start, 3),
                },
            )


def run_command(cmd: list[str], progress_path: Path, event_prefix: str, batch_index: int, dry_run: bool) -> None:
    append_jsonl(progress_path, {"time": utc_now(), "event": f"{event_prefix}_start", "batch": batch_index, "cmd": cmd})
    if dry_run:
        append_jsonl(progress_path, {"time": utc_now(), "event": f"{event_prefix}_dry_run", "batch": batch_index})
        return
    subprocess.run(cmd, check=True)
    append_jsonl(progress_path, {"time": utc_now(), "event": f"{event_prefix}_done", "batch": batch_index})


def transfer_batch(args: argparse.Namespace, batch_root: Path, batch: Batch) -> None:
    if not args.transfer_host or not args.transfer_root:
        return
    mkdir_cmd = [
        "ssh",
        "-i",
        str(args.ssh_key),
        *shlex.split(args.ssh_extra),
        args.transfer_host,
        f"mkdir -p {shlex.quote(args.transfer_root)}",
    ]
    run_command(mkdir_cmd, args.progress_jsonl, "remote_mkdir", batch.index, args.dry_run)
    if args.transfer_mode == "chunked":
        script = args.chunk_transfer_script
        if script is None:
            script = Path(__file__).with_name("parallel_chunk_transfer.py")
        script = script.expanduser().resolve()
        chunk_progress = args.progress_jsonl.parent / f"chunk_transfer_batch_{batch.index:04d}.jsonl"
        chunk_cmd = [
            "python3",
            str(script),
            "--src-root",
            str(batch_root),
            "--dst-host",
            args.transfer_host,
            "--dst-root",
            args.transfer_root,
            "--ssh-key",
            str(args.ssh_key),
            "--ssh-extra",
            args.ssh_extra,
            "--chunk-size-mib",
            str(args.chunk_size_mib),
            "--parallel",
            str(args.chunk_parallel),
            "--split-threshold-mib",
            str(args.chunk_threshold_mib),
            "--progress-jsonl",
            str(chunk_progress),
        ]
        run_command(chunk_cmd, args.progress_jsonl, "chunk_transfer", batch.index, args.dry_run)
        return

    ssh_cmd = f"ssh -i {shlex.quote(str(args.ssh_key))} {args.ssh_extra}"
    rsync_cmd = [
        "rsync",
        "-avP",
        "--partial",
        "--append-verify",
        "--exclude",
        "visualization.rrd",
        "-e",
        ssh_cmd,
        f"{batch_root}/",
        f"{args.transfer_host}:{args.transfer_root}/",
    ]
    run_command(rsync_cmd, args.progress_jsonl, "rsync", batch.index, args.dry_run)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.workers != 1:
        print("NOTE: --workers is reserved; using sequential downloads for relay disk safety.")

    token = args.token
    if not args.dry_run and not args.validate_only and not token:
        token = getpass.getpass("HF token: ").strip()
    if not args.dry_run and not args.validate_only and not token:
        raise SystemExit("HF token is required unless --dry-run or --validate-only is set.")

    args.relay_root = args.relay_root.expanduser().resolve()
    args.progress_jsonl = (args.relay_root / args.progress_jsonl).resolve() if not args.progress_jsonl.is_absolute() else args.progress_jsonl
    args.summary_json = (args.relay_root / args.summary_json).resolve() if not args.summary_json.is_absolute() else args.summary_json
    args.relay_root.mkdir(parents=True, exist_ok=True)

    episodes = load_selection(args.selection_json)
    batches = make_batches(episodes, int(args.batch_max_gib * 1024**3), args.batch_max_episodes)
    selected_batches = batches[args.start_batch :]
    if args.max_batches > 0:
        selected_batches = selected_batches[: args.max_batches]

    summary = {
        "status": "running" if selected_batches else "nothing_to_do",
        "generated_at_utc": utc_now(),
        "repo_id": args.repo_id,
        "selection_json": str(args.selection_json),
        "relay_root": str(args.relay_root),
        "batch_max_gib": args.batch_max_gib,
        "batch_max_episodes": args.batch_max_episodes,
        "total_batches": len(batches),
        "scheduled_batches": [batch.index for batch in selected_batches],
        "scheduled_episode_count": sum(len(batch.episodes) for batch in selected_batches),
        "scheduled_bytes": sum(batch.bytes for batch in selected_batches),
        "scheduled_human": human_bytes(sum(batch.bytes for batch in selected_batches)),
        "transfer_host": args.transfer_host,
        "transfer_root": args.transfer_root,
        "delete_after_transfer": args.delete_after_transfer,
        "dry_run": args.dry_run,
        "validate_only": args.validate_only,
    }
    write_summary(args.summary_json, summary)

    for batch in selected_batches:
        batch_root = args.relay_root / f"batch_{batch.index:04d}"
        batch_root.mkdir(parents=True, exist_ok=True)
        append_jsonl(
            args.progress_jsonl,
            {
                "time": utc_now(),
                "event": "batch_start",
                "batch": batch.index,
                "episode_count": len(batch.episodes),
                "expected_bytes": batch.bytes,
                "expected_human": human_bytes(batch.bytes),
                "batch_root": str(batch_root),
            },
        )
        if args.dry_run:
            append_jsonl(
                args.progress_jsonl,
                {
                    "time": utc_now(),
                    "event": "batch_planned",
                    "batch": batch.index,
                    "episode_paths": [ep["episode_path"] for ep in batch.episodes],
                    "files": [filename for ep in batch.episodes for filename in ep["download_files"]],
                    "validation": "skipped_for_dry_run",
                },
            )
            transfer_batch(args, batch_root, batch)
            append_jsonl(args.progress_jsonl, {"time": utc_now(), "event": "batch_done", "batch": batch.index})
            continue
        if not args.dry_run and not args.validate_only:
            download_batch(args.repo_id, token, batch_root, batch, args.progress_jsonl)
        validation = validate_batch(batch_root, batch)
        append_jsonl(args.progress_jsonl, {"time": utc_now(), "event": "batch_validated", "batch": batch.index, **validation})
        if not validation["ok"]:
            raise SystemExit(f"Batch {batch.index} validation failed: {validation}")
        transfer_batch(args, batch_root, batch)
        if args.delete_after_transfer and args.transfer_host and args.transfer_root and not args.dry_run:
            shutil.rmtree(batch_root)
            append_jsonl(args.progress_jsonl, {"time": utc_now(), "event": "batch_deleted", "batch": batch.index, "batch_root": str(batch_root)})
        append_jsonl(args.progress_jsonl, {"time": utc_now(), "event": "batch_done", "batch": batch.index})

    summary["status"] = "complete"
    summary["completed_at_utc"] = utc_now()
    write_summary(args.summary_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
