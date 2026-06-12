#!/usr/bin/env python3
"""Summarize Qwen3-Omni full-parameter feasibility gates.

These runs are evidence that full-parameter FSDP can load, prepare, step, and
run short guarded pilots on an 8-GPU remote worker. They are not promoted model results and they do
not publish checkpoints or weights.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results" / "omni_finetune"
OUTPUT_JSON = ROOT / "docs/data/qwen3_full_parameter_gates.json"
OUTPUT_MD = RESULT_ROOT / "QWEN3_FULL_PARAMETER_GATES_20260609.md"

RUNS = [
    {
        "id": "fullparam_smoke_1step",
        "title": "Full-Parameter 1-Step Feasibility Smoke",
        "summary": RESULT_ROOT
        / "xperience10m_qwen3_omni_128ep_fullparam_smoke_preemptible_8gpu_20260609"
        / "fullparam_feasibility_summary.json",
        "scope": "1 optimizer step over 8 train samples",
    },
    {
        "id": "fullparam_shorttrain8",
        "title": "Full-Parameter 8-Step Short Train",
        "summary": RESULT_ROOT
        / "xperience10m_qwen3_omni_128ep_fullparam_shorttrain8_preemptible_8gpu_20260609"
        / "fullparam_shorttrain8_summary.json",
        "scope": "8 optimizer steps over 64 train samples",
    },
    {
        "id": "fullparam_pilot32",
        "title": "Full-Parameter 32-Step Pilot",
        "summary": RESULT_ROOT
        / "xperience10m_qwen3_omni_128ep_fullparam_pilot32_preemptible_8gpu_20260609"
        / "fullparam_pilot32_summary.json",
        "scope": "32 optimizer steps over 256 train samples",
    },
    {
        "id": "fullparam_pilot64",
        "title": "Full-Parameter 64-Step Pilot",
        "summary": RESULT_ROOT
        / "xperience10m_qwen3_omni_128ep_fullparam_pilot64_preemptible_8gpu_20260609"
        / "fullparam_pilot64_summary.json",
        "scope": "64 optimizer steps over 512 train samples",
    },
    {
        "id": "fullparam_pilot128_preempted",
        "title": "Full-Parameter 128-Step Opportunistic Pilot",
        "summary": RESULT_ROOT
        / "xperience10m_qwen3_omni_128ep_fullparam_pilot128_preemptible_8gpu_20260609"
        / "fullparam_pilot128_summary.json",
        "scope": "planned 128 optimizer steps over 1024 train samples; preempted for Qwen v5 handoff",
    },
    {
        "id": "fullparam_pilot128_after_qwen_v5",
        "title": "Full-Parameter 128-Step Post-Qwen-v5 Pilot",
        "metadata": RESULT_ROOT
        / "xperience10m_qwen3_omni_128ep_fullparam_pilot128_after_qwen_v5_preemptible_8gpu_20260609"
        / "training_metadata.json",
        "progress": RESULT_ROOT
        / "xperience10m_qwen3_omni_128ep_fullparam_pilot128_after_qwen_v5_preemptible_8gpu_20260609"
        / "progress.jsonl",
        "scope": "128 optimizer steps over 1024 train samples after verified Qwen v5 handoff",
    },
    {
        "id": "fullparam_pilot256_after_qwen_v6",
        "title": "Full-Parameter 256-Step Post-Qwen-v6 Pilot",
        "metadata": RESULT_ROOT
        / "xperience10m_qwen3_omni_128ep_fullparam_pilot256_after_qwen_v6_preemptible_8gpu_20260611"
        / "training_metadata.json",
        "progress": RESULT_ROOT
        / "xperience10m_qwen3_omni_128ep_fullparam_pilot256_after_qwen_v6_preemptible_8gpu_20260611"
        / "progress.jsonl",
        "scope": "256 optimizer steps over 2048 train samples after verified Qwen v6 handoff",
    },
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) else None


def progress_summary(path: Path, expected_steps: int | None) -> dict[str, Any]:
    if not path.exists():
        return {}
    train_losses: list[float] = []
    observed_steps = 0
    saw_complete = False
    saw_save_skipped = False
    saw_max_steps = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_name = event.get("event")
        if event_name == "train_step":
            observed_steps = max(observed_steps, int(event.get("global_step") or 0))
            loss = number(event.get("rank0_batch_loss"))
            if loss is not None:
                train_losses.append(float(loss))
        elif event_name == "train_loop_stopped_max_steps":
            saw_max_steps = True
            observed_steps = max(observed_steps, int(event.get("global_step") or 0))
        elif event_name == "save_skipped":
            saw_save_skipped = True
        elif event_name == "complete":
            saw_complete = True
    status = "passed" if saw_complete and saw_save_skipped and observed_steps == expected_steps else "review"
    return {
        "status": status,
        "observed_train_steps": observed_steps or None,
        "first_step_loss": train_losses[0] if train_losses else None,
        "final_step_loss": train_losses[-1] if train_losses else None,
        "min_step_loss": min(train_losses) if train_losses else None,
        "max_step_loss": max(train_losses) if train_losses else None,
        "saw_max_steps": saw_max_steps,
        "saw_save_skipped": saw_save_skipped,
        "saw_complete": saw_complete,
    }


def row_from_metadata(config: dict[str, Any]) -> dict[str, Any]:
    metadata_path = Path(config["metadata"])
    progress_path = Path(config["progress"])
    payload = read_json(metadata_path)
    history = payload.get("history", []) if isinstance(payload.get("history"), list) else []
    last_history = history[-1] if history else {}
    max_train_steps = payload.get("max_train_steps")
    progress = progress_summary(progress_path, max_train_steps if isinstance(max_train_steps, int) else None)
    status = progress.get("status", "missing") if payload else "missing"
    save_mode = payload.get("save_mode")
    return {
        "id": config["id"],
        "title": config["title"],
        "status": status,
        "scope": config["scope"],
        "summary_path": rel(metadata_path),
        "progress_path": rel(progress_path),
        "run_id": payload.get("run_id"),
        "purpose": "post_verified_qwen_v5_full_parameter_feasibility_pilot",
        "tuning_mode": payload.get("tuning_mode"),
        "training_objective": payload.get("backbone", {}).get("training_objective")
        if isinstance(payload.get("backbone"), dict)
        else None,
        "num_processes": payload.get("num_processes"),
        "num_train_samples": payload.get("num_train_samples"),
        "configured_max_train_steps": max_train_steps,
        "observed_train_steps": progress.get("observed_train_steps") or last_history.get("global_step"),
        "first_step_loss": number(progress.get("first_step_loss")),
        "final_step_loss": number(progress.get("final_step_loss")),
        "epoch_train_loss": number(last_history.get("train_loss")),
        "min_step_loss": number(progress.get("min_step_loss")),
        "max_step_loss": number(progress.get("max_step_loss")),
        "model_load_seconds": None,
        "accelerator_prepare_seconds": None,
        "train_loop_seconds": None,
        "save_mode": save_mode,
        "checkpoint_saved": False,
        "checkpoint_policy": "no full-parameter checkpoint or public weights; save_mode=none",
        "preempt_event": None,
        "parent_resume_event": None,
        "progress_events": {
            "max_steps_reached": progress.get("saw_max_steps"),
            "save_skipped": progress.get("saw_save_skipped"),
            "complete": progress.get("saw_complete"),
        },
    }


def run_row(config: dict[str, Any]) -> dict[str, Any]:
    if "metadata" in config:
        return row_from_metadata(config)
    path = Path(config["summary"])
    payload = read_json(path)
    status = payload.get("status", "missing") if payload else "missing"
    max_train_steps = (
        payload.get("max_train_steps")
        or payload.get("configured_max_train_steps")
        or payload.get("global_step")
    )
    final_step_loss = payload.get("final_step_loss", payload.get("rank0_batch_loss"))
    epoch_train_loss = payload.get("epoch_train_loss", payload.get("train_loss"))
    return {
        "id": config["id"],
        "title": config["title"],
        "status": status,
        "scope": config["scope"],
        "summary_path": rel(path),
        "run_id": payload.get("run_id"),
        "purpose": payload.get("purpose"),
        "tuning_mode": payload.get("tuning_mode"),
        "training_objective": payload.get("training_objective"),
        "num_processes": payload.get("num_processes"),
        "num_train_samples": payload.get("num_train_samples")
        or payload.get("configured_max_train_samples"),
        "configured_max_train_steps": max_train_steps,
        "observed_train_steps": payload.get("observed_train_steps")
        if payload.get("observed_train_steps") is not None
        else payload.get("global_step"),
        "first_step_loss": number(payload.get("first_step_loss")),
        "final_step_loss": number(final_step_loss),
        "epoch_train_loss": number(epoch_train_loss),
        "min_step_loss": number(payload.get("min_step_loss")),
        "max_step_loss": number(payload.get("max_step_loss")),
        "model_load_seconds": number(payload.get("model_load_seconds")),
        "accelerator_prepare_seconds": number(payload.get("accelerator_prepare_seconds")),
        "train_loop_seconds": number(payload.get("train_loop_seconds")),
        "save_mode": payload.get("save_mode"),
        "checkpoint_saved": bool(payload.get("checkpoint_saved", False)),
        "checkpoint_policy": "no full-parameter checkpoint or public weights; save_mode=none",
        "preempt_event": payload.get("preempt_event"),
        "parent_resume_event": payload.get("parent_resume_event"),
    }


def build_payload() -> dict[str, Any]:
    runs = [run_row(config) for config in RUNS]
    passed = [run for run in runs if run["status"] == "passed"]
    preempted = [run for run in runs if str(run["status"]).startswith("preempted")]
    missing_or_review = [
        run
        for run in runs
        if run["status"] not in {"passed"} and not str(run["status"]).startswith("preempted")
    ]
    completed_steps = sum(int(run.get("observed_train_steps") or 0) for run in passed)
    longest_passed = max(passed, key=lambda run: int(run.get("observed_train_steps") or 0), default=None)
    return {
        "title": "Qwen3-Omni Full-Parameter Feasibility Gates",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass" if len(passed) >= 5 and len(preempted) <= 1 and not missing_or_review else "review",
        "decision": "full_parameter_feasible_for_guarded_short_runs_not_promoted",
        "interpretation": (
            "The 2026-06-09 gates prove that Qwen3-Omni full-parameter FSDP can load, "
            "prepare, run backward/optimizer steps, and complete guarded pilots up to "
            "128 optimizer steps on an 8-GPU remote worker. They do not prove a production full-parameter fine-tune, and "
            "they intentionally save no full checkpoints or public weights."
        ),
        "aggregate": {
            "run_count": len(runs),
            "passed_run_count": len(passed),
            "preempted_run_count": len(preempted),
            "review_or_missing_run_count": len(missing_or_review),
            "completed_full_parameter_train_steps": completed_steps,
            "longest_passed_run_id": longest_passed.get("run_id") if longest_passed else None,
            "longest_passed_steps": longest_passed.get("observed_train_steps") if longest_passed else None,
            "num_processes": sorted({run.get("num_processes") for run in runs if run.get("num_processes")}),
            "checkpoint_saved": any(run.get("checkpoint_saved") for run in runs),
        },
        "runs": runs,
        "publication_policy": {
            "public_summary_allowed": True,
            "publish_full_parameter_weights": False,
            "publish_full_checkpoints": False,
            "reason": "All completed 2026-06-09 full-parameter runs used save_mode=none; the preempted pilot saved nothing. These are feasibility evidence only.",
        },
        "next_steps": [
            "Keep the verified Qwen3-Omni LoRA adapter as the published production result for the 128-episode suite.",
            "For a production full-parameter run, add a sharded checkpoint/resume plan before any long training launch.",
            "Run a separate checkpointed full-parameter pilot only when GPUs are not needed by verified LoRA evaluation/publication work.",
        ],
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-Omni Full-Parameter Feasibility Gates",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        payload["interpretation"],
        "",
        "## Summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Passed runs: `{payload['aggregate']['passed_run_count']}`",
        f"- Preempted runs: `{payload['aggregate']['preempted_run_count']}`",
        f"- Review/missing runs: `{payload['aggregate']['review_or_missing_run_count']}`",
        f"- Completed full-parameter optimizer steps: `{payload['aggregate']['completed_full_parameter_train_steps']}`",
        f"- Longest passed run: `{payload['aggregate']['longest_passed_run_id']}` ({payload['aggregate']['longest_passed_steps']} steps)",
        f"- Checkpoint saved: `{payload['aggregate']['checkpoint_saved']}`",
        "",
        "## Runs",
        "",
        "| run | status | steps | samples | final loss | epoch/train loss | policy | source |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run in payload["runs"]:
        lines.append(
            "| {title} | {status} | {steps} | {samples} | {final_loss} | {epoch_loss} | {policy} | `{source}` |".format(
                title=run["title"],
                status=run["status"],
                steps=fmt(run.get("observed_train_steps")),
                samples=fmt(run.get("num_train_samples")),
                final_loss=fmt(run.get("final_step_loss")),
                epoch_loss=fmt(run.get("epoch_train_loss")),
                policy="no weights/checkpoints",
                source=run["summary_path"],
            )
        )
    lines.extend(
        [
            "",
            "## Publication Policy",
            "",
            "- Public summary allowed: `true`",
            "- Publish full-parameter weights: `false`",
            "- Publish full checkpoints: `false`",
            f"- Reason: {payload['publication_policy']['reason']}",
            "",
            "## Next Steps",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["next_steps"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(markdown(payload), encoding="utf-8")
    print(f"PASS: wrote {rel(OUTPUT_JSON)}")
    print(f"PASS: wrote {rel(OUTPUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
