# A100 Hugging Face Relay Status

Current blocker: Hugging Face access to `ropedia-ai/xperience-10m` is still
pending approval from the dataset authors.

Verified:

- A100 SSH alias: `ANGEL-A100-80Gx4`
- H20 SSH alias: `ANGEL-H20-96GX8`
- A100 can reach `huggingface.co`
- A100 staging path: `/mnt/kgc/chaoyue/xperience10m_hf_staging`
- A100 HF cache path: `/mnt/kgc/chaoyue/hf_cache`
- A100 HF token path: `/mnt/kgc/chaoyue/hf_home/token`
- A100 has enough free space for the 32-episode stratified pilot subset
- Direct A100 -> H20 SSH/rsync works with `~/.ssh/xperience10m_h20_transfer`

Dry-run selection:

- Dataset: `ropedia-ai/xperience-10m`
- Target: 32 complete leaf episodes
- Strategy: stratified round-robin across top-level session UUIDs
- Candidate scan: first 64 top-level session UUIDs
- Valid candidates: `680`
- Selected sessions: `32`
- Minimum episode size: `0.25 GB`
- Estimated bytes: `72,031,620,552`
- Excludes: `visualization.rrd`

Background watcher:

```bash
ps -p $(cat /mnt/kgc/chaoyue/xperience10m_logs/hf_access_watch.pid) -o pid,etime,cmd
tail -f /mnt/kgc/chaoyue/xperience10m_logs/hf_access_watch.out
tail -f /mnt/kgc/chaoyue/xperience10m_logs/hf_access_watch.jsonl
```

Watcher behavior:

1. Polls one gated HF file every 15 minutes.
2. When access changes from 403 pending to approved, downloads 32 complete episodes from 32 different session UUIDs.
3. Validates the staged files.
4. Transfers staged data to `/home/cy/Ropedia/modelscope_data` on H20.
5. Runs the H20 readiness gate.

Manual restart on A100:

```bash
HF_HOME=/mnt/kgc/chaoyue/hf_home \
HF_HUB_CACHE=/mnt/kgc/chaoyue/hf_cache \
nohup python3 /mnt/kgc/chaoyue/xperience10m_tools/watch_hf_access_and_stage_xperience10m.py \
  --poll-seconds 900 \
  --target-episodes 32 \
  --max-top-level 64 \
  --workers 8 \
  --reserve-gb 250 \
  --selection-strategy stratified \
  --min-episode-gb 0.25 \
  --run-transfer \
  > /mnt/kgc/chaoyue/xperience10m_logs/hf_access_watch.out 2>&1 &
```

Stop watcher:

```bash
kill $(cat /mnt/kgc/chaoyue/xperience10m_logs/hf_access_watch.pid)
```
