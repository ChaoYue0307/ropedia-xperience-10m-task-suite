# Xperience-10M 128-Episode Data Preparation and Fine-Tune Plan

This is the executable plan for moving from metadata selection to real
multi-episode training. It does not claim model-quality results until data is
downloaded, staged, audited, trained, and evaluated on held-out sessions.

## Current Preflight

| Host | Role | Status |
| --- | --- | --- |
| HF-reachable download host | Dataset download and transfer | Needs Hugging Face access and enough scratch storage for one batch |
| Training host | Persistent data + training | Needs enough storage for the staged selection and the training/eval stack |

Conclusion: use a Hugging Face reachable machine for dataset downloads and the
training machine as the persistent data store. The selected episodes are
prepared in bounded batches, transferred, validated, and then used for the
held-out Qwen3-Omni LoRA pilot.

Current execution status:

- a 128-episode data-preparation job has been launched on an HF-reachable host,
- staged-file transfer is active,
- later batches are scheduled after storage checks,
- no multi-episode model-quality training result is claimed yet.

## Selected Data

- Selection file: `results/omni_finetune/xperience10m_128_episode_selection.json`
- Download list: `results/omni_finetune/xperience10m_128_episode_download_files.txt`
- Episodes: 128
- Sessions: 128 unique sessions
- Split: 96 train / 16 val / 16 test
- Files: 896 training files
- Excluded: `visualization.rrd`
- Estimated training-host storage: 277.71 GiB excluding RRD

## Transfer Setup

Define host-specific paths outside the public artifact:

```bash
export RELAY_WORKDIR=/path/to/ropedia-episode-task-suite
export RELAY_ROOT=/path/to/xperience10m_relay
export TRAINING_HOST=<training-user>@<training-host>
export TRAINING_REPO=/path/to/ropedia-episode-task-suite
export TRAINING_DATA_ROOT=/path/to/xperience10m_128
```

Create a dedicated download-to-training SSH key:

```bash
ssh <relay-host> 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && test -f ~/.ssh/xperience10m_relay_ed25519 || ssh-keygen -t ed25519 -N "" -f ~/.ssh/xperience10m_relay_ed25519 -C xperience10m-relay-to-training'
ssh <relay-host> 'cat ~/.ssh/xperience10m_relay_ed25519.pub'
```

Append that public key to the training host `~/.ssh/authorized_keys`, then verify from the download host:

```bash
ssh <relay-host> 'ssh -i ~/.ssh/xperience10m_relay_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=accept-new <training-user>@<training-host> hostname'
```

## Copy Minimal Repo Files to Download Host

```bash
ssh <relay-host> 'mkdir -p "$RELAY_WORKDIR"'
rsync -av \
  scripts/omni/relay_xperience10m_selection.py \
  scripts/omni/parallel_chunk_transfer.py \
  results/omni_finetune/xperience10m_128_episode_selection.json \
  <relay-host>:"$RELAY_WORKDIR"/
```

## Transfer Dry Run

```bash
ssh <relay-host> '
cd "$RELAY_WORKDIR" &&
python3 relay_xperience10m_selection.py \
  --selection-json xperience10m_128_episode_selection.json \
  --relay-root "$RELAY_ROOT" \
  --batch-max-gib 40 \
  --batch-max-episodes 16 \
  --transfer-host "$TRAINING_HOST" \
  --transfer-root "$TRAINING_DATA_ROOT" \
  --ssh-key ~/.ssh/xperience10m_relay_ed25519 \
  --transfer-mode chunked \
  --chunk-parallel 8 \
  --chunk-size-mib 8 \
  --chunk-threshold-mib 8 \
  --delete-after-transfer \
  --dry-run
'
```

## Start Data Preparation

Run in a persistent terminal or `tmux` session on the download host:

```bash
export HF_TOKEN=...
cd "$RELAY_WORKDIR"
python3 relay_xperience10m_selection.py \
  --selection-json xperience10m_128_episode_selection.json \
  --relay-root "$RELAY_ROOT" \
  --batch-max-gib 40 \
  --batch-max-episodes 16 \
  --transfer-host "$TRAINING_HOST" \
  --transfer-root "$TRAINING_DATA_ROOT" \
  --ssh-key ~/.ssh/xperience10m_relay_ed25519 \
  --transfer-mode chunked \
  --chunk-parallel 8 \
  --chunk-size-mib 8 \
  --chunk-threshold-mib 8 \
  --delete-after-transfer
```

Batch sizing is intentionally conservative. A 40 GiB batch size keeps restarts
and partial-transfer cleanup cheaper than treating the full 277.71 GiB selection
as one unit. Later batches should start after disk headroom is checked.

## Training-Host Data Validation

After transfer completes:

```bash
cd "$TRAINING_REPO"
python3 scripts/omni/discover_xperience10m_sources.py \
  --workspace "$TRAINING_REPO" \
  --data-root "$TRAINING_DATA_ROOT" \
  --output results/omni_finetune/source_discovery_128.json \
  --report-output results/omni_finetune/DATA_BLOCKER_REPORT_128.md \
  --target-episodes 128 \
  --skip-modelscope \
  --skip-huggingface
```

Then build the episode manifest:

```bash
python3 scripts/omni/build_episode_manifest.py \
  --workspace "$TRAINING_REPO" \
  --data-root "$TRAINING_DATA_ROOT" \
  --max-episodes 128 \
  --train-fraction 0.75 \
  --val-fraction 0.125 \
  --test-fraction 0.125 \
  --split-seed 7 \
  --output results/omni_finetune/episode_manifest_128.json
```

## Content Rebalance Gate

Parse staged annotations before training:

```bash
python3 scripts/omni/audit_staged_xperience10m_content.py \
  --data-root "$TRAINING_DATA_ROOT" \
  --selection-json results/omni_finetune/xperience10m_128_episode_selection.json \
  --output-json results/omni_finetune/staged_content_audit_128.json \
  --output-csv results/omni_finetune/staged_content_audit_128.csv \
  --report-output results/omni_finetune/STAGED_CONTENT_AUDIT_128.md
```

If a category dominates train, val, or test, swap episodes before training.

## Training Order

### 1. Qwen3-Omni LoRA Baseline

Use this as the first real multi-episode SFT run because the repo already has
working Qwen3-Omni training/eval scripts.

Expected dataset:

- 128 episodes
- 32,768 max windows at 256 windows per episode
- held-out sessions in val/test

### 2. Cosmos3-Nano Compatibility

Cosmos3-Nano should be treated as a second branch:

- first run inference compatibility on a few staged clips,
- then adapt data format for Cosmos video/action tasks,
- then run post-training only after Qwen3-Omni and content audit pass.

Good Cosmos tasks:

- video + text -> physical reasoning text,
- video + text -> future state/action label,
- video + action/text -> future video,
- video + text -> action trajectory proxy.

Do not start with Cosmos3-Super. Cosmos3-Nano is the practical first target;
Super is for a later run after data format, metrics, and compute are stable.

## Acceptance Gates

- 128 selected episodes staged on the training host.
- No `visualization.rrd` in training data.
- 128 unique sessions preserved.
- Train/val/test session leakage is zero.
- Content audit reviewed before training.
- Qwen3-Omni eval runs on held-out sessions.
- Cosmos3-Nano branch starts with compatibility, not immediate full fine-tune.
