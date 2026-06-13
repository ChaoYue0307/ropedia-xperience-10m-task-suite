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

    project_packet = read_json("docs/data/project_packet.json")
    summary_metrics = read_json("docs/data/summary_metrics.json")
    verified_result = read_json("docs/data/omni_finetune_verified_result.json")
    package_path = verified_result["public_package"]["path"]
    package_audit = read_json(f"{package_path}/package_audit.json")
    dataset_manifest = read_json(f"{package_path}/dataset/dataset_manifest.json")
    training_metadata = read_json(f"{package_path}/training/training_metadata.json")
    eval_metrics = read_json(f"{package_path}/eval/metrics.json")
    verified_evaluation = verified_result.get("evaluation", {})
    expected_json_validity = float(verified_evaluation.get("json_validity_rate", 0.0))

    reading_notes = " ".join(project_packet.get("current_reading_notes", []))
    has_verified_qwen_note = (
        "diagnostic pilot is verified" in reading_notes
        or "diagnostic branch is verified" in reading_notes
        or "diagnostic result is verified" in reading_notes
    )
    checks.append(
        check(
            "project_packet_records_verified_diagnostic_status",
            has_verified_qwen_note and "strong model quality is not yet shown" in reading_notes,
            "project packet describes the verified diagnostic pilot and quality boundary",
            ["docs/data/project_packet.json"],
        )
    )

    current_scope = summary_metrics.get("omni_relay", {}).get("current_scope", "")
    has_verified_scope = (
        "diagnostic pilot is verified" in current_scope
        or "diagnostic branch is verified" in current_scope
        or "diagnostic result is verified" in current_scope
    )
    checks.append(
        check(
            "summary_metrics_preserves_verified_diagnostic_status",
            has_verified_scope and "98% target" in current_scope,
            current_scope,
            ["docs/data/summary_metrics.json"],
        )
    )

    split_counts = dataset_manifest.get("split_counts", {})
    expected_split_counts = verified_result.get("split_policy", {}).get("exported_window_counts", {})
    expected_dataset_samples = sum(expected_split_counts.values()) if expected_split_counts else None
    checks.append(
        check(
            "verified_package_dataset_has_expected_windows",
            dataset_manifest.get("num_episodes") == 119
            and dataset_manifest.get("num_samples") == expected_dataset_samples
            and split_counts == expected_split_counts,
            (
                f"episodes={dataset_manifest.get('num_episodes')}, "
                f"samples={dataset_manifest.get('num_samples')}, split_counts={split_counts}, "
                f"expected_samples={expected_dataset_samples}, expected_split_counts={expected_split_counts}"
            ),
            [f"{package_path}/dataset/dataset_manifest.json"],
        )
    )

    expected_train = verified_result.get("training", {}).get("num_train_samples")
    expected_val = verified_result.get("training", {}).get("num_val_samples")
    expected_processes = verified_result.get("training", {}).get("num_processes")
    checks.append(
        check(
            "verified_package_training_records_8_processes",
            training_metadata.get("num_train_samples") == expected_train
            and training_metadata.get("num_val_samples") == expected_val
            and training_metadata.get("num_processes") == expected_processes,
            (
                f"train={training_metadata.get('num_train_samples')}, "
                f"val={training_metadata.get('num_val_samples')}, "
                f"processes={training_metadata.get('num_processes')}, "
                f"expected_train={expected_train}, expected_val={expected_val}, "
                f"expected_processes={expected_processes}"
            ),
            [f"{package_path}/training/training_metadata.json"],
        )
    )

    expected_eval_samples = verified_evaluation.get("num_samples")
    expected_eval_episodes = verified_evaluation.get("held_out_episode_count")
    checks.append(
        check(
            "verified_package_eval_records_real_held_out_metrics",
            eval_metrics.get("num_samples") == expected_eval_samples
            and eval_metrics.get("eval_split") == "test"
            and eval_metrics.get("held_out_episode_count", eval_metrics.get("num_eval_episodes")) == expected_eval_episodes
            and abs(float(eval_metrics.get("json_validity_rate", 0.0)) - expected_json_validity) < 1e-12,
            (
                f"samples={eval_metrics.get('num_samples')}, "
                f"split={eval_metrics.get('eval_split')}, "
                f"held_out={eval_metrics.get('held_out_episode_count', eval_metrics.get('num_eval_episodes'))}, "
                f"json_validity={eval_metrics.get('json_validity_rate')}, "
                f"expected_samples={expected_eval_samples}, expected_held_out={expected_eval_episodes}"
            ),
            [f"{package_path}/eval/metrics.json"],
        )
    )

    checks.append(
        check(
            "verified_package_audit_passes",
            package_audit.get("status") == "pass" and not package_audit.get("issues"),
            f"audit_status={package_audit.get('status')}, issues={len(package_audit.get('issues', []))}",
            [f"{package_path}/package_audit.json"],
        )
    )

    doc_failures, public_observations = scan_public_docs()
    failures.extend(doc_failures)
    checks.append(
        check(
            "public_presentation_has_no_misleading_32ep_identifiers",
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
            "qwen3_omni_verified_diagnostic_pilot": True,
            "dataset_manifest_num_episodes": dataset_manifest.get("num_episodes"),
            "dataset_manifest_num_samples": dataset_manifest.get("num_samples"),
            "training_metadata_num_train_samples": training_metadata.get("num_train_samples"),
            "eval_num_samples": eval_metrics.get("num_samples"),
            "eval_json_validity_rate": eval_metrics.get("json_validity_rate"),
            "quality_target_met": verified_result.get("evaluation", {}).get("quality_target", {}).get("status") == "met",
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
