#!/usr/bin/env python3
"""Transfer large dataset files over several SSH/SCP streams.

This utility is intended for relay situations where a single rsync/ssh stream is
slow. Files above a configurable threshold are split locally, copied as chunks in
parallel, reassembled on the destination host, and size-validated before the
final file is moved into place.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-host", required=True, help="Destination SSH host, e.g. cy@host")
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--ssh-extra", default="-o BatchMode=yes -o StrictHostKeyChecking=accept-new")
    parser.add_argument("--chunk-size-mib", type=int, default=128)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--split-threshold-mib", type=int, default=256)
    parser.add_argument("--local-temp-root", type=Path, default=None)
    parser.add_argument("--remote-temp-root", default="/tmp/xperience10m_chunk_transfer")
    parser.add_argument("--exclude", action="append", default=["visualization.rrd", ".cache/**", "**/.cache/**", ".chunk_transfer_tmp/**"])
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files.")
    parser.add_argument("--progress-jsonl", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human_bytes(num: int | float) -> str:
    value = float(num)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def log_event(path: Path | None, payload: dict) -> None:
    payload = {"time": utc_now(), **payload}
    line = json.dumps(payload, sort_keys=True)
    print(line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def run(cmd: list[str], dry_run: bool = False) -> None:
    if dry_run:
        print("DRY-RUN", " ".join(shlex.quote(part) for part in cmd), flush=True)
        return
    subprocess.run(cmd, check=True)


def ssh_base(args: argparse.Namespace) -> list[str]:
    return ["ssh", "-i", str(args.ssh_key), *shlex.split(args.ssh_extra), args.dst_host]


def scp_base(args: argparse.Namespace) -> list[str]:
    return ["scp", "-q", "-i", str(args.ssh_key), *shlex.split(args.ssh_extra)]


def excluded(rel_path: str, patterns: list[str]) -> bool:
    name = Path(rel_path).name
    if rel_path.startswith(".cache/") or "/.cache/" in rel_path:
        return True
    if ".chunk_transfer_tmp/" in rel_path or rel_path.startswith(".chunk_transfer_tmp/"):
        return True
    return any(fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(name, pattern) for pattern in patterns)


def iter_files(root: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if excluded(rel, patterns):
            continue
        files.append(path)
    return sorted(files)


def remote_mkdir(args: argparse.Namespace, path: str) -> None:
    run([*ssh_base(args), "mkdir", "-p", path], args.dry_run)


def remote_size(args: argparse.Namespace, path: str) -> int | None:
    code = "import os,sys; p=sys.argv[1]; print(os.path.getsize(p) if os.path.exists(p) else -1)"
    remote_cmd = f"python3 -c {shlex.quote(code)} {shlex.quote(path)}"
    cmd = [*ssh_base(args), remote_cmd]
    if args.dry_run:
        return None
    out = subprocess.check_output(cmd, text=True).strip()
    size = int(out)
    return None if size < 0 else size


def copy_one(args: argparse.Namespace, src: Path, remote_path: str) -> None:
    remote_mkdir(args, shlex.quote(str(Path(remote_path).parent)))
    run([*scp_base(args), str(src), f"{args.dst_host}:{remote_path}"], args.dry_run)


def split_file(src: Path, chunk_dir: Path, chunk_size: int) -> list[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    with src.open("rb") as handle:
        index = 0
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            chunk = chunk_dir / f"part_{index:05d}"
            chunk.write_bytes(data)
            chunks.append(chunk)
            index += 1
    return chunks


def transfer_chunk(args: argparse.Namespace, chunk: Path, remote_chunk_dir: str) -> None:
    run([*scp_base(args), str(chunk), f"{args.dst_host}:{remote_chunk_dir}/{chunk.name}"], args.dry_run)


def assemble_remote(args: argparse.Namespace, remote_chunk_dir: str, chunk_names: list[str], remote_path: str, expected_size: int) -> None:
    code = (
        "import os, shutil, sys\n"
        "chunk_dir, dest, expected, *names = sys.argv[1:]\n"
        "expected = int(expected)\n"
        "os.makedirs(os.path.dirname(dest), exist_ok=True)\n"
        "partial = dest + '.partial_chunked'\n"
        "with open(partial, 'wb') as out:\n"
        "    for name in names:\n"
        "        with open(os.path.join(chunk_dir, name), 'rb') as inp:\n"
        "            shutil.copyfileobj(inp, out, 1024 * 1024)\n"
        "actual = os.path.getsize(partial)\n"
        "if actual != expected:\n"
        "    raise SystemExit(f'size mismatch for {dest}: expected {expected}, got {actual}')\n"
        "os.replace(partial, dest)\n"
    )
    argv = [remote_chunk_dir, remote_path, str(expected_size), *chunk_names]
    remote_cmd = " ".join(["python3", "-c", shlex.quote(code), *(shlex.quote(value) for value in argv)])
    run([*ssh_base(args), remote_cmd], args.dry_run)


def transfer_split(args: argparse.Namespace, src: Path, remote_path: str, local_temp: Path, remote_temp: str) -> None:
    size = src.stat().st_size
    if remote_size(args, remote_path) == size:
        log_event(args.progress_jsonl, {"event": "skip_existing", "path": str(src), "bytes": size})
        return

    if local_temp.exists():
        shutil.rmtree(local_temp)
    chunks = split_file(src, local_temp, args.chunk_size_mib * 1024 * 1024)
    remote_mkdir(args, remote_temp)
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(transfer_chunk, args, chunk, remote_temp) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    assemble_remote(args, remote_temp, [chunk.name for chunk in chunks], remote_path, size)
    elapsed = time.time() - start
    rate = size / elapsed if elapsed else 0
    log_event(
        args.progress_jsonl,
        {
            "event": "split_transfer_done",
            "path": str(src),
            "bytes": size,
            "human": human_bytes(size),
            "chunks": len(chunks),
            "seconds": round(elapsed, 3),
            "rate_human_per_s": f"{human_bytes(rate)}/s",
        },
    )


def main() -> int:
    args = parse_args()
    args.src_root = args.src_root.expanduser().resolve()
    args.ssh_key = args.ssh_key.expanduser().resolve()
    if args.local_temp_root is None:
        args.local_temp_root = args.src_root.parent / f".{args.src_root.name}_chunk_transfer_tmp"
    else:
        args.local_temp_root = args.local_temp_root.expanduser().resolve()

    files = iter_files(args.src_root, args.exclude)
    if args.max_files:
        files = files[: args.max_files]
    threshold = args.split_threshold_mib * 1024 * 1024
    log_event(
        args.progress_jsonl,
        {
            "event": "start",
            "src_root": str(args.src_root),
            "dst_host": args.dst_host,
            "dst_root": args.dst_root,
            "files": len(files),
            "parallel": args.parallel,
            "chunk_size_mib": args.chunk_size_mib,
        },
    )

    for index, src in enumerate(files):
        rel = src.relative_to(args.src_root).as_posix()
        remote_path = f"{args.dst_root.rstrip('/')}/{rel}"
        size = src.stat().st_size
        log_event(args.progress_jsonl, {"event": "file_start", "index": index, "path": rel, "bytes": size, "human": human_bytes(size)})
        if size >= threshold:
            safe_name = rel.replace("/", "__")
            transfer_split(
                args,
                src,
                remote_path,
                args.local_temp_root / safe_name,
                f"{args.remote_temp_root.rstrip('/')}/{safe_name}",
            )
        else:
            start = time.time()
            copy_one(args, src, remote_path)
            elapsed = time.time() - start
            log_event(
                args.progress_jsonl,
                {
                    "event": "direct_transfer_done",
                    "path": rel,
                    "bytes": size,
                    "seconds": round(elapsed, 3),
                    "rate_human_per_s": f"{human_bytes(size / elapsed if elapsed else 0)}/s",
                },
            )

    log_event(args.progress_jsonl, {"event": "done"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
