#!/usr/bin/env python3
"""Validate Qwen3-Omni scale-up status against the actual Xperience-10M artifacts.

This check exists because several setup/provenance files retain historical
`32ep` run identifiers in their paths. Those identifiers are useful provenance,
but public project surfaces should present them as setup artifacts until the
held-out 32-episode pilot is actually completed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/data/scope_claims_audit.json"

PUBLIC_PRESENTATION_FILES = [
    "README.md",
    "ARTIFACT_GUIDE.md",
    "EVIDENCE_CONTRACT.md",
    "REPRODUCIBILITY.md",
    "docs/index.html",
    "docs/data/artifact_index.json",
    "docs/data/evidence_contract.json",
    "docs/data/project_manifest.json",
    "docs/data/mirror_parity.json",
    "docs/data/reproducibility_matrix.json",
    "docs/data/project_packet.json",
    "docs/data/summary_metrics.json",
]

RESULT_TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}
HISTORICAL_PATTERNS = [
    "qwen3_omni_32ep",
    "xperience10m_qwen3_omni_32ep",
    "ropedia-episode-task-suite",
]
MISLEADING_PHRASES = [
    re.compile(r"\breal\s+32-episode\s+(?:result|metric|fine-?tune)\b", re.IGNORECASE),
    re.compile(r"\b32-episode\s+(?:result|metric|fine-?tune)\s+is\s+claimed\b", re.IGNORECASE),
    re.compile(r"\bfull\s+32-episode\s+(?:result|metric|fine-?tune)\b", re.IGNORECASE),
]
NEGATION_HINTS = {
    "not",
    "no",
    "never",
    "blocked",
    "pending",
    "gated",
    "until",
    "after",
    "requires",
    "must not",
    "not yet",
    "no real",
}


def read_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def check(name: str, passed: bool, detail: str, evidence: list[str]) -> dict:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "detail": detail,
        "evidence": evidence,
    }


def sentence_windows(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?\n])\s+", text) if part.strip()]


def has_negation(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(hint in lowered for hint in NEGATION_HINTS)


def scan_public_docs() -> tuple[list[dict], list[dict]]:
    failures: list[dict] = []
    observations: list[dict] = []
    for relative_path in PUBLIC_PRESENTATION_FILES:
        path = ROOT / relative_path
        if not path.exists():
            failures.append({"kind": "missing_public_file", "path": relative_path})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in HISTORICAL_PATTERNS:
            if pattern in text:
                failures.append(
                    {
                        "kind": "historical_identifier_in_public_presentation",
                        "path": relative_path,
                        "pattern": pattern,
                    }
                )
        for sentence in sentence_windows(text):
            for phrase in MISLEADING_PHRASES:
                if phrase.search(sentence) and not has_negation(sentence):
                    failures.append(
                        {
                            "kind": "misleading_32_episode_phrase",
                            "path": relative_path,
                            "phrase": phrase.pattern,
                            "sentence": sentence[:260],
                        }
                    )
        if "32-episode" in text:
            observations.append({"path": relative_path, "contains_32_episode_status_text": True})
    return failures, observations


def scan_historical_result_identifiers() -> list[dict]:
    results_root = ROOT / "results/omni_finetune"
    records: list[dict] = []
    if not results_root.exists():
        return records
    try:
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "results/omni_finetune"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.splitlines()
        paths = [ROOT / item for item in tracked if item]
    except (OSError, subprocess.CalledProcessError):
        paths = [item for item in results_root.rglob("*") if item.is_file()]
    for path in sorted(item for item in paths if item.is_file()):
        if path.suffix.lower() not in RESULT_TEXT_SUFFIXES:
            continue
        relative_path = path.relative_to(ROOT).as_posix()
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, line in enumerate(handle, start=1):
                matched = [pattern for pattern in HISTORICAL_PATTERNS if pattern in line]
                if not matched:
                    continue
                records.append(
                    {
                        "classification": "historical_identifier_in_readiness_artifact",
                        "path": relative_path,
                        "line": line_number,
                        "patterns": matched,
                        "example": line.strip()[:260],
                    }
                )
    return records


def build_report() -> dict:
    checks: list[dict] = []
    failures: list[dict] = []

    project_manifest = read_json("docs/data/project_manifest.json")
    project_packet = read_json("docs/data/project_packet.json")
    summary_metrics = read_json("docs/data/summary_metrics.json")
    dataset_manifest = read_json("results/omni_finetune/dataset_manifest.json")
    training_metadata = read_json("results/omni_finetune/training_metadata.json")
    source_discovery = read_json("results/omni_finetune/source_discovery.json")

    project_qwen_claim = project_manifest["scope_boundary"].get("qwen3_omni_32_episode_claim")
    checks.append(
        check(
            "project_manifest_records_pending_32_episode_qwen_result",
            project_qwen_claim is False,
            f"project_manifest scope_boundary.qwen3_omni_32_episode_claim={project_qwen_claim!r}",
            ["docs/data/project_manifest.json"],
        )
    )

    project_qwen_claim = project_packet["scope_status"].get("qwen3_omni_32_episode_claim")
    checks.append(
        check(
            "project_packet_records_pending_32_episode_qwen_result",
            project_qwen_claim is False,
            f"project_packet scope_status.qwen3_omni_32_episode_claim={project_qwen_claim!r}",
            ["docs/data/project_packet.json"],
        )
    )
    reading_notes = " ".join(project_packet.get("current_reading_notes", []))
    checks.append(
        check(
            "project_packet_describes_32_episode_setup_status",
            "32-episode" in reading_notes and ("setup" in reading_notes or "gated data" in reading_notes),
            "project packet describes the setup-stage Qwen3-Omni run separately from the planned 32-episode fine-tune",
            ["docs/data/project_packet.json"],
        )
    )

    current_scope = summary_metrics.get("omni_relay", {}).get("current_scope", "")
    checks.append(
        check(
            "summary_metrics_preserves_omni_scale_up_status",
            "32-episode Qwen3-Omni fine-tune requires gated data staging" in current_scope,
            current_scope,
            ["docs/data/summary_metrics.json"],
        )
    )

    split_counts = dataset_manifest.get("split_counts", {})
    checks.append(
        check(
            "omni_dataset_manifest_is_setup_stage",
            dataset_manifest.get("num_episodes") == 1
            and dataset_manifest.get("num_samples") == 128
            and split_counts == {"train": 128},
            (
                f"episodes={dataset_manifest.get('num_episodes')}, "
                f"samples={dataset_manifest.get('num_samples')}, split_counts={split_counts}"
            ),
            ["results/omni_finetune/dataset_manifest.json"],
        )
    )

    checks.append(
        check(
            "omni_training_metadata_is_setup_stage",
            training_metadata.get("num_train_samples") == 128
            and training_metadata.get("num_val_samples") == 0,
            (
                f"train={training_metadata.get('num_train_samples')}, "
                f"val={training_metadata.get('num_val_samples')}, "
                f"processes={training_metadata.get('num_processes')}"
            ),
            ["results/omni_finetune/training_metadata.json"],
        )
    )

    checks.append(
        check(
            "source_discovery_gate_is_closed",
            source_discovery.get("ready_for_32_episode_pilot") is False
            and source_discovery.get("local", {}).get("num_degraded_valid_episodes") == 1,
            (
                f"ready_for_32_episode_pilot={source_discovery.get('ready_for_32_episode_pilot')}, "
                f"local_valid={source_discovery.get('local', {}).get('num_degraded_valid_episodes')}"
            ),
            ["results/omni_finetune/source_discovery.json"],
        )
    )

    doc_failures, public_observations = scan_public_docs()
    failures.extend(doc_failures)
    checks.append(
        check(
            "public_presentation_has_no_historical_32ep_identifiers",
            not doc_failures,
            f"public presentation scan failures={len(doc_failures)}",
            PUBLIC_PRESENTATION_FILES,
        )
    )

    historical_identifiers = scan_historical_result_identifiers()
    checks.append(
        check(
            "historical_32ep_identifiers_are_confined_to_readiness_artifacts",
            bool(historical_identifiers),
            f"historical identifiers found in result provenance files={len(historical_identifiers)}",
            ["results/omni_finetune/"],
        )
    )

    failures.extend(
        {
            "kind": "failed_check",
            "name": item["name"],
            "detail": item["detail"],
            "evidence": item["evidence"],
        }
        for item in checks
        if item["status"] != "pass"
    )

    status = "pass" if not failures else "fail"
    return {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            "qwen3_omni_32_episode_claim": False,
            "dataset_manifest_num_episodes": dataset_manifest.get("num_episodes"),
            "dataset_manifest_num_samples": dataset_manifest.get("num_samples"),
            "training_metadata_num_train_samples": training_metadata.get("num_train_samples"),
            "source_discovery_ready_for_32_episode_pilot": source_discovery.get("ready_for_32_episode_pilot"),
            "historical_identifier_count": len(historical_identifiers),
            "public_32_episode_status_file_count": len(public_observations),
            "failure_count": len(failures),
        },
        "checks": checks,
        "public_status_observations": public_observations,
        "historical_identifiers": historical_identifiers[:30],
        "historical_identifier_total_count": len(historical_identifiers),
        "failures": failures,
    }


def main() -> int:
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: wrote {OUTPUT}")
    if report["status"] != "pass":
        for failure in report["failures"][:30]:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
