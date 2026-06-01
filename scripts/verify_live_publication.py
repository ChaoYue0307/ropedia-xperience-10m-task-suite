#!/usr/bin/env python3
"""Verify the already-published GitHub Pages and Hugging Face mirrors.

This is the post-publish companion to the local publication gates. It fetches
public URLs and compares them with the local release artifacts so a reviewer can
see that the live surfaces match the repo/HF bundles that were prepared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/data/live_publication_status.json"
TIMEOUT_SECONDS = 30
USER_AGENT = "ropedia-xperience-10m-live-verifier/1.0"


HASH_GROUPS = [
    {
        "id": "task_suite_infographic",
        "title": "Task-suite infographic",
        "local_path": "docs/assets/task_suite_infographic.png",
        "urls": {
            "github_pages": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/assets/task_suite_infographic.png",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/resolve/main/assets/task_suite_infographic.png",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/resolve/main/docs/assets/task_suite_infographic.png",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/resolve/main/assets/task_suite_infographic.png",
        },
    },
    {
        "id": "quality_gates_json",
        "title": "Quality-gate JSON",
        "local_path": "docs/data/quality_gates.json",
        "urls": {
            "github_pages": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/data/quality_gates.json",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/data/quality_gates.json",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/resolve/main/docs/data/quality_gates.json",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/resolve/main/metrics/quality_gates.json",
        },
    },
    {
        "id": "quality_gates_markdown",
        "title": "Quality-gate Markdown",
        "local_path": "QUALITY_GATES.md",
        "urls": {
            "github_raw": "https://raw.githubusercontent.com/ChaoYue0307/ropedia-xperience-10m-task-suite/main/QUALITY_GATES.md",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/QUALITY_GATES.md",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/raw/main/QUALITY_GATES.md",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/raw/main/QUALITY_GATES.md",
        },
    },
]


MARKER_CHECKS = [
    {
        "id": "github_pages_index_current",
        "title": "GitHub Pages index contains current publication markers",
        "url": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/",
        "required": [
            "Release gates are explicit",
            "quality_gates.json",
            "xperience10m-taskfirst-v12-modality-xl",
        ],
        "forbidden": [
            "xperience10m-" + "taskfirst-v10",
            "xperience10m-" + "modalities-v9-large-atlas",
        ],
    },
    {
        "id": "hf_space_index_current",
        "title": "HF Space index contains current publication markers",
        "url": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/index.html",
        "required": [
            "Release gates are explicit",
            "quality_gates.json",
            "xperience10m-taskfirst-v12-modality-xl",
        ],
        "forbidden": [
            "xperience10m-" + "taskfirst-v10",
            "xperience10m-" + "modalities-v9-large-atlas",
        ],
    },
    {
        "id": "hf_artifacts_card_current",
        "title": "HF artifact card links quality gates",
        "url": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/raw/main/README.md",
        "required": ["QUALITY_GATES.md", "docs/data/quality_gates.json"],
        "forbidden": ["xperience10m-" + "taskfirst-v10"],
    },
    {
        "id": "hf_model_card_current",
        "title": "HF model card links quality gates",
        "url": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/raw/main/README.md",
        "required": ["QUALITY_GATES.md", "metrics/quality_gates.json"],
        "forbidden": ["xperience10m-" + "taskfirst-v10"],
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def fetch(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read()
            return {
                "ok": True,
                "status_code": int(getattr(response, "status", 200)),
                "bytes": len(body),
                "sha256": sha256_bytes(body),
                "body": body,
                "final_url": sanitize_url(response.geturl()),
            }
    except HTTPError as exc:
        return {
            "ok": False,
            "status_code": exc.code,
            "bytes": 0,
            "sha256": None,
            "error": str(exc),
            "final_url": url,
        }
    except URLError as exc:
        return {
            "ok": False,
            "status_code": None,
            "bytes": 0,
            "sha256": None,
            "error": str(exc.reason),
            "final_url": url,
        }


def hash_group_record(group: dict) -> dict:
    local_path = ROOT / group["local_path"]
    local = {
        "path": group["local_path"],
        "exists": local_path.exists(),
        "bytes": local_path.stat().st_size if local_path.exists() else 0,
        "sha256": sha256_file(local_path) if local_path.exists() else None,
    }
    mirrors = {}
    failures = []
    if not local["exists"]:
        failures.append({"surface": "local", "kind": "missing", "path": group["local_path"]})
    for surface, url in group["urls"].items():
        result = fetch(url)
        record = {key: value for key, value in result.items() if key != "body"}
        record["url"] = url
        mirrors[surface] = record
        if not result["ok"]:
            failures.append({"surface": surface, "kind": "fetch_failed", "url": url, "error": result.get("error")})
            continue
        if local["exists"] and result["sha256"] != local["sha256"]:
            failures.append(
                {
                    "surface": surface,
                    "kind": "hash_mismatch",
                    "url": url,
                    "expected_sha256": local["sha256"],
                    "actual_sha256": result["sha256"],
                }
            )
    return {
        "id": group["id"],
        "title": group["title"],
        "status": "pass" if not failures else "fail",
        "local": local,
        "mirrors": mirrors,
        "failures": failures,
    }


def marker_record(check: dict) -> dict:
    result = fetch(check["url"])
    failures = []
    missing = []
    forbidden_hits = []
    if not result["ok"]:
        failures.append({"kind": "fetch_failed", "url": check["url"], "error": result.get("error")})
        text = ""
    else:
        text = result["body"].decode("utf-8", errors="ignore")
        missing = [marker for marker in check["required"] if marker not in text]
        forbidden_hits = [marker for marker in check["forbidden"] if marker in text]
        if missing:
            failures.append({"kind": "missing_required_markers", "markers": missing})
        if forbidden_hits:
            failures.append({"kind": "forbidden_markers_present", "markers": forbidden_hits})
    return {
        "id": check["id"],
        "title": check["title"],
        "url": check["url"],
        "status": "pass" if not failures else "fail",
        "fetch": {key: value for key, value in result.items() if key != "body"},
        "required_marker_count": len(check["required"]),
        "missing_markers": missing,
        "forbidden_markers_present": forbidden_hits,
        "failures": failures,
    }


def build_report() -> dict:
    hash_records = [hash_group_record(group) for group in HASH_GROUPS]
    marker_records = [marker_record(check) for check in MARKER_CHECKS]
    failures = [
        {"check": record["id"], **failure}
        for record in [*hash_records, *marker_records]
        for failure in record["failures"]
    ]
    return {
        "title": "Ropedia Xperience-10M Live Publication Status",
        "status": "pass" if not failures else "fail",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Live GitHub Pages, GitHub raw, Hugging Face Space, artifact dataset, and model card mirrors.",
        "hash_groups": hash_records,
        "marker_checks": marker_records,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true", help="Verify and print status without updating the report file.")
    args = parser.parse_args()

    report = build_report()
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{report['status'].upper()}: wrote {args.output}")
    else:
        print(f"{report['status'].upper()}: live publication verification")
    if report["status"] != "pass":
        for failure in report["failures"][:20]:
            print(f"- {failure}")
        if len(report["failures"]) > 20:
            print(f"- ... {len(report['failures']) - 20} more failures")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
