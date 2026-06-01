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
import subprocess
import tempfile
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
        "id": "xperience10m_dataset_card_alignment_json",
        "title": "Official Xperience-10M dataset-card alignment JSON",
        "local_path": "docs/data/xperience10m_dataset_card_alignment.json",
        "urls": {
            "github_pages": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/data/xperience10m_dataset_card_alignment.json",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/data/xperience10m_dataset_card_alignment.json",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/resolve/main/docs/data/xperience10m_dataset_card_alignment.json",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/resolve/main/metrics/xperience10m_dataset_card_alignment.json",
        },
    },
    {
        "id": "source_alignment_audit_json",
        "title": "Source-alignment audit JSON",
        "local_path": "docs/data/source_alignment_audit.json",
        "urls": {
            "github_pages": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/data/source_alignment_audit.json",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/data/source_alignment_audit.json",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/resolve/main/docs/data/source_alignment_audit.json",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/resolve/main/metrics/source_alignment_audit.json",
        },
    },
    {
        "id": "reviewer_scorecard_json",
        "title": "Reviewer scorecard JSON",
        "local_path": "docs/data/reviewer_scorecard.json",
        "urls": {
            "github_pages": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/data/reviewer_scorecard.json",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/data/reviewer_scorecard.json",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/resolve/main/docs/data/reviewer_scorecard.json",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/resolve/main/metrics/reviewer_scorecard.json",
        },
    },
    {
        "id": "evaluation_protocol_json",
        "title": "Evaluation protocol JSON",
        "local_path": "docs/data/evaluation_protocol.json",
        "urls": {
            "github_pages": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/data/evaluation_protocol.json",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/data/evaluation_protocol.json",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/resolve/main/docs/data/evaluation_protocol.json",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/resolve/main/metrics/evaluation_protocol.json",
        },
    },
    {
        "id": "figure_index_json",
        "title": "Figure index JSON",
        "local_path": "docs/data/figure_index.json",
        "urls": {
            "github_pages": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/data/figure_index.json",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/data/figure_index.json",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/resolve/main/docs/data/figure_index.json",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/resolve/main/metrics/figure_index.json",
        },
    },
    {
        "id": "task_walkthroughs_json",
        "title": "Task walkthrough JSON",
        "local_path": "docs/data/task_walkthroughs.json",
        "urls": {
            "github_pages": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/data/task_walkthroughs.json",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/data/task_walkthroughs.json",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/resolve/main/docs/data/task_walkthroughs.json",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/resolve/main/metrics/task_walkthroughs.json",
        },
    },
    {
        "id": "task_surface_integrity_json",
        "title": "Task-surface integrity JSON",
        "local_path": "docs/data/task_surface_integrity.json",
        "urls": {
            "github_pages": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/data/task_surface_integrity.json",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/data/task_surface_integrity.json",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/resolve/main/docs/data/task_surface_integrity.json",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/resolve/main/metrics/task_surface_integrity.json",
        },
    },
    {
        "id": "brand_assets_json",
        "title": "Brand assets JSON",
        "local_path": "docs/data/brand_assets.json",
        "urls": {
            "github_pages": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/data/brand_assets.json",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/data/brand_assets.json",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/resolve/main/docs/data/brand_assets.json",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/resolve/main/metrics/brand_assets.json",
        },
    },
    {
        "id": "brand_logo_social_card",
        "title": "Brand logo social card",
        "local_path": "docs/assets/brand/xperience10m-logo-social-card.png",
        "urls": {
            "github_pages": "https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/assets/brand/xperience10m-logo-social-card.png",
            "hf_space": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/resolve/main/assets/brand/xperience10m-logo-social-card.png",
            "hf_artifacts": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/resolve/main/docs/assets/brand/xperience10m-logo-social-card.png",
            "hf_model": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/resolve/main/assets/brand/xperience10m-logo-social-card.png",
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
            "xperience10m_dataset_card_alignment.json",
            "source_alignment_audit.json",
            "evaluation_protocol.json",
            "figure_index.json",
            "brand_assets.json",
            "xperience10m-logo-social-card.png",
            "reviewer_scorecard.json",
            "cc-by-nc-4.0",
            "12,103 episode folders",
            "xperience10m-taskfirst-v12-modality-xl",
            "Interactive task walkthrough.",
            "taskPlayer",
            "Action Recognition",
            "data/task_walkthroughs.json",
            "task_surface_integrity.json",
        ],
        "forbidden": [
            "xperience10m-" + "taskfirst-v10",
            "xperience10m-" + "modalities-v9-large-atlas",
            "artifact-id",
        ],
    },
    {
        "id": "hf_space_index_current",
        "title": "HF Space index contains current publication markers",
        "url": "https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite/raw/main/index.html",
        "required": [
            "Release gates are explicit",
            "quality_gates.json",
            "xperience10m_dataset_card_alignment.json",
            "source_alignment_audit.json",
            "evaluation_protocol.json",
            "figure_index.json",
            "brand_assets.json",
            "xperience10m-logo-social-card.png",
            "reviewer_scorecard.json",
            "cc-by-nc-4.0",
            "12,103 episode folders",
            "xperience10m-taskfirst-v12-modality-xl",
            "Interactive task walkthrough.",
            "taskPlayer",
            "Action Recognition",
            "data/task_walkthroughs.json",
            "task_surface_integrity.json",
        ],
        "forbidden": [
            "xperience10m-" + "taskfirst-v10",
            "xperience10m-" + "modalities-v9-large-atlas",
            "artifact-id",
        ],
    },
    {
        "id": "hf_artifacts_card_current",
        "title": "HF artifact card links quality gates",
        "url": "https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/raw/main/README.md",
        "required": [
            "QUALITY_GATES.md",
            "docs/data/quality_gates.json",
            "xperience10m_dataset_card_alignment.json",
            "source_alignment_audit.json",
            "evaluation_protocol.json",
            "figure_index.json",
            "brand_assets.json",
            "xperience10m-logo-social-card.png",
            "reviewer_scorecard.json",
            "cc-by-nc-4.0",
            "12,103 episode folders",
            "interactive scrub/play walkthrough storyboard",
            "task_surface_integrity.json",
        ],
        "forbidden": ["xperience10m-" + "taskfirst-v10"],
    },
    {
        "id": "hf_model_card_current",
        "title": "HF model card links quality gates",
        "url": "https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines/raw/main/README.md",
        "required": [
            "QUALITY_GATES.md",
            "metrics/quality_gates.json",
            "xperience10m_dataset_card_alignment.json",
            "source_alignment_audit.json",
            "evaluation_protocol.json",
            "figure_index.json",
            "brand_assets.json",
            "xperience10m-logo-social-card.png",
            "reviewer_scorecard.json",
            "cc-by-nc-4.0",
            "12,103 episode folders",
            "interactive scrub/play storyboard",
            "task_surface_integrity.json",
        ],
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
    except (TimeoutError, OSError) as exc:
        fallback = fetch_with_curl(url)
        if fallback["ok"]:
            return fallback
        return {
            "ok": False,
            "status_code": None,
            "bytes": 0,
            "sha256": None,
            "error": str(exc),
            "final_url": url,
        }
    except URLError as exc:
        fallback = fetch_with_curl(url)
        if fallback["ok"]:
            return fallback
        return {
            "ok": False,
            "status_code": None,
            "bytes": 0,
            "sha256": None,
            "error": str(exc.reason),
            "final_url": url,
        }


def fetch_with_curl(url: str) -> dict:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [
                "curl",
                "-L",
                "-sS",
                "--max-time",
                str(TIMEOUT_SECONDS),
                "-A",
                USER_AGENT,
                "-o",
                str(tmp_path),
                "-w",
                "%{http_code}\n%{url_effective}",
                url,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        status_text, _, final_url = result.stdout.partition("\n")
        status_code = int(status_text.strip() or "0")
        body = tmp_path.read_bytes() if tmp_path.exists() else b""
        if result.returncode != 0:
            return {
                "ok": False,
                "status_code": status_code or None,
                "bytes": 0,
                "sha256": None,
                "error": result.stderr.strip() or f"curl exited {result.returncode}",
                "final_url": sanitize_url(final_url.strip() or url),
            }
        return {
            "ok": 200 <= status_code < 400,
            "status_code": status_code,
            "bytes": len(body),
            "sha256": sha256_bytes(body),
            "body": body,
            "final_url": sanitize_url(final_url.strip() or url),
        }
    finally:
        tmp_path.unlink(missing_ok=True)


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
