#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPU_HOST_SUFFIX="${GPU_HOST_SUFFIX:-$(printf 'A%s-80Gx4' 100)}"
REMOTE_HOST="${REMOTE_HOST:-ANGEL-${GPU_HOST_SUFFIX}}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/kgc/chaoyue/ropedia-h20-side/ropedia-episode-task-suite}"
RUN_ID="${RUN_ID:-xperience10m_qwen3_omni_v6_sensor_target_probes_a100_20260619T000000Z}"
RESULT_ROOT="${RESULT_ROOT:-results/omni_finetune}"
TASKS_CSV="${TASKS_CSV:-hand_trajectory_forecast,modality_reconstruction,imu_to_hand_pose}"

REMOTE_RUN_DIR="${REMOTE_ROOT}/${RESULT_ROOT}/${RUN_ID}"
LOCAL_RUN_DIR="${PROJECT_ROOT}/${RESULT_ROOT}/${RUN_ID}"
LOCAL_LAUNCHER_DIR="${PROJECT_ROOT}/${RESULT_ROOT}/deferred_launchers"
REMOTE_LAUNCHER_LOGS=(
  "${REMOTE_ROOT}/${RESULT_ROOT}/${RUN_ID}.launch.log"
  "${REMOTE_ROOT}/${RESULT_ROOT}/${RUN_ID}.resume_when_free.log"
  "${REMOTE_ROOT}/${RESULT_ROOT}/${RUN_ID}.resume_when_free.launch.log"
  "${REMOTE_ROOT}/${RESULT_ROOT}/${RUN_ID}.shared_vram_resume.log"
  "${REMOTE_ROOT}/${RESULT_ROOT}/${RUN_ID}.shared_vram_autoresume_guard.log"
  "${REMOTE_ROOT}/${RESULT_ROOT}/${RUN_ID}.autoresume_guard.launch.log"
  "${REMOTE_ROOT}/${RESULT_ROOT}/deferred_launchers/${RUN_ID}.launch.log"
  "${REMOTE_ROOT}/${RESULT_ROOT}/deferred_launchers/${RUN_ID}.launcher.log"
)

IFS=',' read -r -a TASKS <<< "$TASKS_CSV"

echo "checking remote run ${REMOTE_HOST}:${REMOTE_RUN_DIR}"
ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && test -s '${RESULT_ROOT}/${RUN_ID}/summary.json'"
for task_id in "${TASKS[@]}"; do
  ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && test -s '${RESULT_ROOT}/${RUN_ID}/${task_id}/metrics.json'"
done

mkdir -p "$LOCAL_RUN_DIR" "$LOCAL_LAUNCHER_DIR"
rsync -av "${REMOTE_HOST}:${REMOTE_RUN_DIR}/" "$LOCAL_RUN_DIR/"
for remote_launcher_log in "${REMOTE_LAUNCHER_LOGS[@]}"; do
  ssh "$REMOTE_HOST" "test -s '$remote_launcher_log'" >/dev/null 2>&1 \
    && rsync -av "${REMOTE_HOST}:${remote_launcher_log}" "$LOCAL_LAUNCHER_DIR/" \
    || true
done

python3 - "$PROJECT_ROOT" "$RUN_ID" "$TASKS_CSV" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_id = sys.argv[2]
task_ids = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
run_dir = root / "results/omni_finetune" / run_id
metric_key_by_task = {
    "hand_trajectory_forecast": "hand_trajectory_forecast_mrr",
    "caption_grounding": "caption_grounding_mrr",
    "cross_modal_retrieval": "cross_modal_retrieval_mrr",
    "modality_reconstruction": "modality_reconstruction_mrr",
    "imu_to_hand_pose": "imu_to_hand_pose_mrr",
    "camera_view_sync_retrieval": "camera_view_sync_retrieval_mrr",
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
    "title": "Qwen3 Retrieval Task Probe Collection Validation",
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
