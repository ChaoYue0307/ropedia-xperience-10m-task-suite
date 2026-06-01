#!/usr/bin/env bash
set -euo pipefail

STAGING_DIR="${STAGING_DIR:?Set STAGING_DIR to the prepared Xperience-10M staging directory.}"
TRAINING_HOST="${TRAINING_HOST:?Set TRAINING_HOST to the remote training host, for example user@hostname.}"
TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:?Set TRAINING_DATA_ROOT to the remote Xperience-10M data directory.}"
REMOTE_WORKSPACE="${REMOTE_WORKSPACE:?Set REMOTE_WORKSPACE to the remote repo checkout.}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/xperience10m_transfer}"

rsync -avP --partial --append-verify \
  --exclude "visualization.rrd" \
  -e "ssh -i ${SSH_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
  "${STAGING_DIR}" \
  "${TRAINING_HOST}:${TRAINING_DATA_ROOT}"

ssh -i "${SSH_KEY}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${TRAINING_HOST}" \
  "cd ${REMOTE_WORKSPACE} && python3 scripts/omni/discover_xperience10m_sources.py --workspace ${REMOTE_WORKSPACE} --data-root ${TRAINING_DATA_ROOT} --output results/omni_finetune/source_discovery.json --report-output results/omni_finetune/DATA_BLOCKER_REPORT.md"
