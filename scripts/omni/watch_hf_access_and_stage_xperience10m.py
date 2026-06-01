#!/usr/bin/env python3
"""Poll Hugging Face gated access, then stage and optionally transfer Xperience-10M.

This is intended for a staging host that can reach Hugging Face. It does a
cheap HEAD request against one gated file. When access is approved, it starts
the selective 32-episode staging script and can optionally launch a generic
host-to-host transfer script.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_url
from huggingface_hub.file_download import get_hf_file_metadata


DEFAULT_PROBE_FILE = "003dcaf0-edba-4787-ada0-187d2748f684/ep1/fisheye_cam0.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ropedia-ai/xperience-10m")
    parser.add_argument("--probe-file", default=DEFAULT_PROBE_FILE)
    parser.add_argument("--local-dir", type=Path, default=Path(os.environ.get("XPERIENCE10M_STAGE_DIR", "xperience10m_hf_staging")))
    parser.add_argument("--stage-script", type=Path, default=Path(os.environ.get("XPERIENCE10M_STAGE_SCRIPT", "scripts/omni/stage_xperience10m_from_hf.py")))
    parser.add_argument("--transfer-script", type=Path, default=Path(os.environ.get("XPERIENCE10M_TRANSFER_SCRIPT", "scripts/omni/transfer_xperience10m_between_hosts.sh")))
    parser.add_argument("--log-dir", type=Path, default=Path(os.environ.get("XPERIENCE10M_LOG_DIR", "xperience10m_logs")))
    parser.add_argument("--target-episodes", type=int, default=32)
    parser.add_argument("--max-top-level", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reserve-gb", type=float, default=250)
    parser.add_argument("--min-episode-gb", type=float, default=0.25)
    parser.add_argument("--selection-strategy", default="stratified", choices=["stratified", "first"])
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means run until approved.")
    parser.add_argument("--run-transfer", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token

    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    token_path = hf_home / "token"
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()
    return ""


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def check_access(repo_id: str, probe_file: str, token: str) -> tuple[bool, dict]:
    if not token:
        return False, {"status": "missing_token"}

    url = hf_hub_url(repo_id=repo_id, filename=probe_file, repo_type="dataset")
    try:
        metadata = get_hf_file_metadata(url, token=token, timeout=30)
        return True, {
            "status": "approved",
            "etag": metadata.etag,
            "size": metadata.size,
        }
    except Exception as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return False, {
            "status": "not_approved" if status_code in (401, 403) else "check_error",
            "http_status": status_code,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_logged(cmd: list[str], log_file: Path, env: dict[str, str]) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] RUN {' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
        handle.write(f"[{utc_now()}] EXIT {proc.returncode}\n")
        return int(proc.returncode)


def main() -> int:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.log_dir / "hf_access_watch.jsonl"
    token = read_token()

    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
    env.setdefault("HF_HOME", str(args.log_dir.parent / "hf_home"))
    env.setdefault("HF_HUB_CACHE", str(args.log_dir.parent / "hf_cache"))

    attempt = 0
    while True:
        attempt += 1
        approved, detail = check_access(args.repo_id, args.probe_file, token)
        record = {"time": utc_now(), "attempt": attempt, "approved": approved, **detail}
        append_jsonl(status_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

        if approved:
            break
        if args.max_attempts and attempt >= args.max_attempts:
            return 2
        time.sleep(max(60, args.poll_seconds))

    stage_cmd = [
        "python3",
        str(args.stage_script),
        "--local-dir",
        str(args.local_dir),
        "--target-episodes",
        str(args.target_episodes),
        "--max-top-level",
        str(args.max_top_level),
        "--workers",
        str(args.workers),
        "--reserve-gb",
        str(args.reserve_gb),
        "--min-episode-gb",
        str(args.min_episode_gb),
        "--selection-strategy",
        args.selection_strategy,
    ]
    stage_rc = run_logged(stage_cmd, args.log_dir / "stage_32ep.log", env)
    append_jsonl(status_path, {"time": utc_now(), "stage_exit": stage_rc})
    if stage_rc != 0:
        return stage_rc

    if args.run_transfer:
        transfer_rc = run_logged([str(args.transfer_script)], args.log_dir / "transfer_to_h20.log", env)
        append_jsonl(status_path, {"time": utc_now(), "transfer_exit": transfer_rc})
        return transfer_rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
