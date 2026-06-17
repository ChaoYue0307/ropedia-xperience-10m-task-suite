#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPU_HOST_SUFFIX="${GPU_HOST_SUFFIX:-$(printf 'A%s-80Gx4' 100)}"
REMOTE_HOST="${REMOTE_HOST:-ANGEL-${GPU_HOST_SUFFIX}}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/kgc/chaoyue/ropedia-h20-side/ropedia-episode-task-suite}"
RUN_ID="${RUN_ID:-xperience10m_qwen3_omni_v6_future_task_probes_a100_20260616T143608Z}"
RESULT_ROOT="${RESULT_ROOT:-results/omni_finetune}"

REMOTE_RUN_DIR="${REMOTE_ROOT}/${RESULT_ROOT}/${RUN_ID}"
LOCAL_RUN_DIR="${PROJECT_ROOT}/${RESULT_ROOT}/${RUN_ID}"
LOCAL_LAUNCHER_DIR="${PROJECT_ROOT}/${RESULT_ROOT}/deferred_launchers"
REMOTE_LAUNCHER_LOG="${REMOTE_ROOT}/${RESULT_ROOT}/deferred_launchers/${RUN_ID}.launcher.log"
TASKS_CSV="${TASKS_CSV:-long_horizon_next_action,next_subtask_forecast,object_set_forecast}"

IFS=',' read -r -a TASKS <<< "$TASKS_CSV"

echo "checking remote run ${REMOTE_HOST}:${REMOTE_RUN_DIR}"
ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && test -s '${RESULT_ROOT}/${RUN_ID}/summary.json'"
for task_id in "${TASKS[@]}"; do
  ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && test -s '${RESULT_ROOT}/${RUN_ID}/${task_id}/metrics.json'"
done

mkdir -p "$LOCAL_RUN_DIR" "$LOCAL_LAUNCHER_DIR"
rsync -av "${REMOTE_HOST}:${REMOTE_RUN_DIR}/" "$LOCAL_RUN_DIR/"
ssh "$REMOTE_HOST" "test -s '$REMOTE_LAUNCHER_LOG'" >/dev/null 2>&1 \
  && rsync -av "${REMOTE_HOST}:${REMOTE_LAUNCHER_LOG}" "$LOCAL_LAUNCHER_DIR/" \
  || true

python3 - "$PROJECT_ROOT" "$RUN_ID" "$TASKS_CSV" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_id = sys.argv[2]
task_ids = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
run_dir = root / "results/omni_finetune" / run_id
metric_key_by_task = {
    "caption_grounding": "caption_grounding_mrr",
    "temporal_order": "temporal_order_f1",
    "misalignment_detection": "misalignment_detection_f1",
    "long_horizon_next_action": "long_horizon_next_action_macro_f1",
    "next_subtask_forecast": "next_subtask_forecast_macro_f1",
    "object_set_forecast": "object_set_forecast_micro_f1",
    "time_to_transition": "time_to_transition_mae",
}
expected = {task_id: metric_key_by_task[task_id] for task_id in task_ids}

summary_path = run_dir / "summary.json"
if not summary_path.exists():
    raise SystemExit(f"missing summary: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("status") != "pass":
    raise SystemExit(f"run summary is not pass: {summary.get('status')}")

records = []
for task_id, metric_key in expected.items():
    metrics_path = run_dir / task_id / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"missing metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    score = metrics.get(metric_key)
    if metrics.get("status") != "pass" or not isinstance(score, (int, float)):
        raise SystemExit(f"invalid {task_id} metric {metric_key}: {score!r}")
    records.append(
        {
            "task_id": task_id,
            "metric_key": metric_key,
            "primary_score": score,
            "num_samples": metrics.get("num_samples"),
            "source": str(metrics_path.relative_to(root)),
        }
    )

validation = {
    "title": "Qwen3 Future Task Probe Collection Validation",
    "status": "pass",
    "run_id": run_id,
    "summary": str(summary_path.relative_to(root)),
    "validated_task_count": len(records),
    "records": records,
}
(run_dir / "collection_validation.json").write_text(
    json.dumps(validation, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(validation, indent=2, sort_keys=True))
PY

echo "collected and validated ${LOCAL_RUN_DIR}"
