#!/usr/bin/env python3
"""Build the public Qwen3-Omni v1-v6 run-lineage summary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIED = ROOT / "results/omni_finetune/verified_public"
OUTPUT_JSON = ROOT / "docs/data/qwen3_omni_run_lineage.json"
OUTPUT_MD = ROOT / "QWEN3_OMNI_RUN_LINEAGE.md"

RUNS = [
    {
        "version": "v1",
        "title": "Selected-128 validation-aware LoRA baseline",
        "package": "xperience10m_qwen3_omni_128ep_96train_16val_16test_valmon_20260605_eval",
        "purpose": "Prove that the selected-128 split, LoRA training, held-out eval, validation, and public packaging loop works end to end.",
        "change_from_previous": "First verified Qwen3-Omni selected-128 LoRA run.",
        "role": "First verified 96/16/16 selected-episode Qwen3-Omni LoRA package; establishes dataset, training, eval, and packaging plumbing.",
        "reader_use": "Use only as lineage evidence for the first working pipeline.",
        "public_matrix_role": "superseded lineage evidence, not the current 20-task Qwen row",
    },
    {
        "version": "v2",
        "title": "Structured-JSON reuse full-8-GPU LoRA",
        "package": "xperience10m_qwen3_omni_128ep_structured_json_v2_reuse_full8gpu_lora_eval_test_full",
        "purpose": "Make the answer format schema-checked and reduce invalid JSON before expanding scale.",
        "change_from_previous": "Reused the selected-128 split with a stricter structured-JSON answer contract and full 8-GPU LoRA training.",
        "role": "Reuses the selected-128 split with a stricter structured JSON answer contract and full 8-GPU LoRA training.",
        "reader_use": "Use as evidence that schema-constrained evaluation improved validity and contact accuracy over v1.",
        "public_matrix_role": "superseded lineage evidence, not the current 20-task Qwen row",
    },
    {
        "version": "v3",
        "title": "Strict-label prompt evaluation",
        "package": "xperience10m_qwen3_omni_128ep_structured_json_v3_strict_label_prompt_reuse_lora_eval_test_full",
        "purpose": "Separate prompt/eval formatting effects from adapter-training effects.",
        "change_from_previous": "Evaluated the v2 adapter with stricter labels and prompts; no new adapter training.",
        "role": "Strict-label prompt/eval pass over the v2 adapter; improves JSON validity without introducing a new adapter training run.",
        "reader_use": "Use as prompt/eval ablation evidence, not as a separate trained model.",
        "public_matrix_role": "superseded prompt/eval lineage evidence",
    },
    {
        "version": "v4",
        "title": "Four-epoch structured-JSON LoRA",
        "package": "xperience10m_qwen3_omni_128ep_structured_json_v4_4epoch_full8gpu_lora_eval_test_full",
        "purpose": "Test whether longer structured-JSON LoRA training improves the same selected split.",
        "change_from_previous": "Trained a new four-epoch full-8-GPU LoRA adapter on the structured-JSON setup.",
        "role": "Four-epoch full-8-GPU LoRA run on the same selected split; useful for overfit/metric tradeoff analysis.",
        "reader_use": "Use as overfit and metric-tradeoff evidence before the multiscale export.",
        "public_matrix_role": "superseded lineage evidence, not the current 20-task Qwen row",
    },
    {
        "version": "v5",
        "title": "Multiscale cap96 LoRA",
        "package": "xperience10m_qwen3_omni_128ep_multiscale_cap96_v5_full8gpu_lora_eval_test_full",
        "purpose": "Move from the 448-sample compact eval to a denser multiscale 4,032-sample held-out eval.",
        "change_from_previous": "Introduced the multiscale cap96 export and larger held-out evaluation surface.",
        "role": "Dense/multiscale selected-128 run with 4,032 held-out predictions; kept as the pinned prior release because several metrics remain stronger than v6.",
        "reader_use": "Use as the pinned prior release; it remains stronger on JSON validity, subtask, next-action, object, and transition metrics.",
        "public_matrix_role": "pinned prior release row and comparison baseline",
    },
    {
        "version": "v6",
        "title": "Rank64 lr5e-5 multiscale LoRA",
        "package": "xperience10m_qwen3_omni_128ep_multiscale_cap96_v6_rank64_lr5e5_full8gpu_lora_eval_test_full",
        "purpose": "Promote the current public Qwen3-Omni 20-task row with multiscale LoRA plus task-specific probes.",
        "change_from_previous": "Kept the multiscale setup, changed LoRA rank/lr to rank64/lr5e-5, and added verified task-specific probes for full 20-task coverage.",
        "role": "Current verified Qwen3-Omni row: rank64/lr5e-5 multiscale LoRA plus task-specific probe artifacts used for the 20/20 Qwen matrix coverage.",
        "reader_use": "Use as the current public 20-task Qwen row; it improves action macro-F1 and contact accuracy while v5 remains the prior comparator.",
        "public_matrix_role": "current public 20-task Qwen3-Omni v6 LoRA row",
    },
]

METRIC_KEYS = [
    "json_validity_rate",
    "action_macro_f1",
    "subtask_accuracy",
    "transition_accuracy",
    "next_action_accuracy",
    "contact_accuracy",
    "object_micro_f1",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict, key: str):
    return metrics.get(key)


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in row) + " |")
    return "\n".join(out)


def build_payload() -> dict:
    rows = []
    for spec in RUNS:
        package_dir = VERIFIED / spec["package"]
        summary = read_json(package_dir / "verified_result_summary.json")
        metrics = read_json(package_dir / "eval/metrics.json")
        row = {
            **spec,
            "status": summary.get("status", "verified"),
            "package_path": str(package_dir.relative_to(ROOT)),
            "dataset_run_id": summary.get("dataset_run_id"),
            "train_run_id": summary.get("train_run_id"),
            "eval_run_id": summary.get("eval_run_id"),
            "dataset_contract": summary.get("dataset_contract"),
            "eval_samples": metrics.get("num_samples") or metrics.get("eval_samples"),
            "metrics": {key: metric_value(metrics, key) for key in METRIC_KEYS},
        }
        rows.append(row)
    return {
        "title": "Qwen3-Omni v1-v6 Run Lineage",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Verified public-safe Qwen3-Omni LoRA/eval packages over the selected Xperience-10M 128-episode surface.",
        "interpretation_rule": (
            "Do not confuse the Qwen run versions with the project evidence lines. "
            "The project evidence lines are one public sample episode and selected 128-episode "
            "artifacts. Qwen v1-v6 are only the Qwen3-Omni run lineage inside the selected-128 line. "
            "The 20-task matrix uses Qwen3-Omni v6 LoRA; v5 remains the pinned prior release; "
            "v1-v4 are lineage and ablation evidence."
        ),
        "current_public_matrix_row": "qwen3_omni_v6_lora",
        "pinned_prior_release": "v5",
        "runs": rows,
        "related_engineering_artifacts": [
            {
                "name": "Full-parameter gates",
                "path": "results/omni_finetune/QWEN3_FULL_PARAMETER_GATES_20260609.md",
                "role": "Feasibility and short-train gates; not a public 20-task matrix method row.",
            },
            {
                "name": "Alternate fullsplit v6 package",
                "path": "results/omni_finetune/verified_public/xperience10m_qwen3_omni_128ep_fullsplit_fast8gpu_lora_fsdp_full_train_noval_tail_logits_fullstatesave_v6_eval_test_full",
                "role": "Verified alternate no-validation/fullsplit artifact retained for audit, not the current matrix row.",
            },
        ],
    }


def write_outputs(payload: dict) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for run in payload["runs"]:
        m = run["metrics"]
        rows.append(
            [
                run["version"],
                run["title"],
                run["purpose"],
                run["change_from_previous"],
                run["eval_samples"],
                fmt(m["json_validity_rate"]),
                fmt(m["action_macro_f1"]),
                fmt(m["contact_accuracy"]),
                run["reader_use"],
            ]
        )
    detail_rows = [
        [
            run["version"],
            run["train_run_id"],
            run["eval_run_id"],
            run["role"],
            run["package_path"],
        ]
        for run in payload["runs"]
    ]
    text = f"""# Qwen3-Omni v1-v6 Run Lineage

Generated: `{payload['generated_at_utc']}`.

Scope: {payload['scope']}

Interpretation rule: {payload['interpretation_rule']}

Read the versions as an engineering audit trail, not as six separate benchmark
rows. v1-v4 explain how the Qwen3-Omni pipeline was hardened, v5 is the pinned
prior multiscale release, and v6 is the current 20-task Qwen3-Omni row.

## Compact Lineage

{markdown_table(['Version', 'Run', 'Purpose', 'Change from previous', 'Eval samples', 'JSON validity', 'Action macro-F1', 'Contact acc.', 'Use now'], rows)}

## Run IDs And Packages

{markdown_table(['Version', 'Train run', 'Eval run', 'Role', 'Package'], detail_rows)}

## Related Engineering Artifacts

{markdown_table(['Artifact', 'Path', 'Role'], [[row['name'], row['path'], row['role']] for row in payload['related_engineering_artifacts']])}
"""
    OUTPUT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    write_outputs(build_payload())
    print(f"Wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
