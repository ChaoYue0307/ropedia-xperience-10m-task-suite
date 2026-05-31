#!/usr/bin/env bash
set -euo pipefail

A100_STAGE_DIR="${A100_STAGE_DIR:-/mnt/kgc/chaoyue/xperience10m_hf_staging/}"
H20_HOST="${H20_HOST:-cy@47.100.122.133}"
H20_DATA_ROOT="${H20_DATA_ROOT:-/home/cy/Ropedia/modelscope_data/}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/xperience10m_h20_transfer}"

rsync -avP --partial --append-verify \
  --exclude "visualization.rrd" \
  -e "ssh -i ${SSH_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
  "${A100_STAGE_DIR}" \
  "${H20_HOST}:${H20_DATA_ROOT}"

ssh -i "${SSH_KEY}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${H20_HOST}" \
  "cd /home/cy/Ropedia/ropedia-episode-task-suite && python3 scripts/omni/discover_xperience10m_sources.py --workspace /home/cy/Ropedia/ropedia-episode-task-suite --data-root /home/cy/Ropedia/modelscope_data --output results/omni_finetune/source_discovery.json --report-output results/omni_finetune/DATA_BLOCKER_REPORT.md"
