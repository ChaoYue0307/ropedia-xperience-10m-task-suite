#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPU_HOST_SUFFIX="${GPU_HOST_SUFFIX:-$(printf 'A%s-80Gx4' 100)}"
REMOTE_HOST="${REMOTE_HOST:-ANGEL-${GPU_HOST_SUFFIX}}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/kgc/chaoyue/ropedia-h20-side/ropedia-episode-task-suite}"
RUN_ID="${RUN_ID:-xperience10m_qwen3_omni_v6_interaction_text_task15_a100_20260620T010305Z}"
RESULT_ROOT="${RESULT_ROOT:-results/omni_finetune}"
TASK_ID="${TASK_ID:-interaction_text_prediction}"
METRIC_KEY="${METRIC_KEY:-macro_f1}"

REMOTE_RUN_DIR="${REMOTE_ROOT}/${RESULT_ROOT}/${RUN_ID}"
LOCAL_RUN_DIR="${PROJECT_ROOT}/${RESULT_ROOT}/${RUN_ID}"
LOCAL_LAUNCHER_DIR="${PROJECT_ROOT}/${RESULT_ROOT}/deferred_launchers"
REMOTE_LAUNCHER_LOG="${REMOTE_ROOT}/${RESULT_ROOT}/${RUN_ID}.launch.log"
REMOTE_DEFERRED_LAUNCHER_LOG="${REMOTE_ROOT}/${RESULT_ROOT}/deferred_launchers/${RUN_ID}.launcher.log"

echo "checking remote run ${REMOTE_HOST}:${REMOTE_RUN_DIR}"
ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && test -s '${RESULT_ROOT}/${RUN_ID}/summary.json'"
ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && test -s '${RESULT_ROOT}/${RUN_ID}/${TASK_ID}/metrics.json'"
ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && test -s '${RESULT_ROOT}/${RUN_ID}/${TASK_ID}/predictions.jsonl'"

mkdir -p "$LOCAL_RUN_DIR" "$LOCAL_LAUNCHER_DIR"
rsync -av --exclude 'tail_helper_supervisor*' "${REMOTE_HOST}:${REMOTE_RUN_DIR}/" "$LOCAL_RUN_DIR/"
for remote_launcher_log in "$REMOTE_LAUNCHER_LOG" "$REMOTE_DEFERRED_LAUNCHER_LOG"; do
  ssh "$REMOTE_HOST" "test -s '$remote_launcher_log'" >/dev/null 2>&1 \
    && rsync -av "${REMOTE_HOST}:${remote_launcher_log}" "$LOCAL_LAUNCHER_DIR/" \
    || true
done

python3 - "$PROJECT_ROOT" "$RUN_ID" "$TASK_ID" "$METRIC_KEY" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_id = sys.argv[2]
task_id = sys.argv[3]
metric_key = sys.argv[4]
run_dir = root / "results/omni_finetune" / run_id
summary_path = run_dir / "summary.json"
metrics_path = run_dir / task_id / "metrics.json"
predictions_path = run_dir / task_id / "predictions.jsonl"

if not summary_path.exists():
    raise SystemExit(f"missing summary: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("status") != "pass":
    raise SystemExit(f"run summary is not pass: {summary.get('status')}")

if not metrics_path.exists():
    raise SystemExit(f"missing metrics: {metrics_path}")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
score = metrics.get(metric_key)
if metrics.get("status") != "pass" or not isinstance(score, (int, float)):
    raise SystemExit(f"invalid {task_id} metric {metric_key}: {score!r}")
if metrics.get("caption_manifest_status") != "pass":
    raise SystemExit(f"caption manifest status is not pass: {metrics.get('caption_manifest_status')}")

prediction_rows = 0
with predictions_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            prediction_rows += 1
if prediction_rows <= 0:
    raise SystemExit(f"no prediction rows in {predictions_path}")

declared_rows = metrics.get("num_samples") or metrics.get("scored_rows")
if isinstance(declared_rows, int) and declared_rows != prediction_rows:
    raise SystemExit(
        f"prediction row mismatch: metrics={declared_rows} file={prediction_rows}"
    )

validation = {
    "title": "Qwen3 Interaction Text Task-15 Collection Validation",
    "status": "pass",
    "run_id": run_id,
    "summary": str(summary_path.relative_to(root)),
    "validated_task_count": 1,
    "records": [
        {
            "task_id": task_id,
            "metric_key": metric_key,
            "primary_score": score,
            "prediction_rows": prediction_rows,
            "caption_manifest_status": metrics.get("caption_manifest_status"),
            "requested_annotation_file_count": metrics.get("requested_annotation_file_count"),
            "processed_annotation_file_count": metrics.get("processed_annotation_file_count"),
            "source": str(metrics_path.relative_to(root)),
        }
    ],
}
(run_dir / "collection_validation.json").write_text(
    json.dumps(validation, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(validation, indent=2, sort_keys=True))
PY

echo "collected and validated ${LOCAL_RUN_DIR}"
