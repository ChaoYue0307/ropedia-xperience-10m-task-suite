# Multi-Episode Access Status

Current blocker: access to the gated full `ropedia-ai/xperience-10m` dataset is
still pending approval from the dataset authors.

This file records only public-facing readiness facts. It intentionally excludes
machine aliases, private paths, SSH hosts, token locations, and local server
details.

## Selection Plan

| Item | Value |
| --- | ---: |
| Dataset | `ropedia-ai/xperience-10m` |
| Target | 32 complete leaf episodes |
| Strategy | stratified round-robin across top-level session UUIDs |
| Candidate scan | first 64 top-level session UUIDs |
| Valid candidates | 680 |
| Selected sessions | 32 |
| Minimum episode size | 0.25 GB |
| Estimated bytes | 72,031,620,552 |
| Excluded file | `visualization.rrd` |

## Boundary

The current Qwen3-Omni artifacts are readiness artifacts from the locally
available sample data. They are not 32-episode held-out model-quality results.

A real 32-episode pilot can be claimed only after:

- at least 32 valid episodes are available locally,
- the manifest builder confirms complete held-out episode splits,
- training finishes with recorded metadata and progress logs,
- evaluation runs on held-out test episodes,
- predictions, metrics, confusion matrices, and a run report are committed.

The source-of-truth blocker report remains:

`results/omni_finetune/DATA_BLOCKER_REPORT.md`
