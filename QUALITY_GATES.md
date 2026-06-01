# Publication Quality Gates

This file is the reviewer-facing release checklist for the Ropedia Xperience-10M Task Suite.

Current gate status: **pass**

Do not present a release as current unless every automated gate passes, then verify live GitHub/HF mirrors after publishing.

These gates validate public packaging, claim boundaries, mirror parity, and website integrity. They do not prove cross-episode model quality; the 32-episode Qwen3-Omni pilot remains gated on data access.

## Automated Gates

| Gate | Command | Report | Current report status | Blocks publication if |
| --- | --- | --- | --- | --- |
| Scope claims guard | `python scripts/validate_scope_claims.py` | `docs/data/scope_claims_audit.json` | `pass` | Historical 32ep readiness/provenance strings are presented as real 32-episode metrics. |
| Source alignment audit | `python scripts/validate_source_alignment.py` | `docs/data/source_alignment_audit.json` | `pass` | Official full-dataset facts, sample-card facts, API-listing caveats, or public-card boundary markers are missing or inconsistent. |
| Website integrity | `python scripts/validate_website_integrity.py` | `docs/data/website_integrity.json` | `pass` | Local links, anchors, JSON bundles, or referenced image assets are missing or invalid. |
| Evaluation protocol | `python scripts/build_evaluation_protocol.py` | `docs/data/evaluation_protocol.json` | `pass` | Windowing, split policy, leakage controls, task metrics, or unsupported interpretations are not explicit. |
| Figure index | `python scripts/build_figure_index.py` | `docs/data/figure_index.json` | `pass` | Public figures, charts, or modality thumbnails are missing, unreadable, or lack source-script provenance. |
| Brand assets | `python scripts/build_brand_assets.py` | `docs/data/brand_assets.json` | `pass` | The ChatGPT-image-generated logo, favicon, social card, or app icons are missing or not reproducibly packaged. |
| Quality-gate manifest | `python scripts/build_quality_gates.py` | `docs/data/quality_gates.json` | `pass` | A public reviewer cannot see the current packaging gates in one place. |
| Artifact index | `python scripts/build_artifact_index.py` | `docs/data/artifact_index.json` | `pass` | Reviewer-critical evidence files are missing from the indexed proof layer. |
| Publication hygiene | `python scripts/validate_publication_package.py` | `docs/data/publication_audit.json` | `pass` | Raw data, caches, heavy archives, token strings, missing required assets, or stale public-card figure references enter public bundles. |
| Prepared mirror parity | `python scripts/validate_mirror_parity.py` | `docs/data/mirror_parity.json` | `pass` | Prepared HF Space, artifact dataset, or model bundle diverges from the repo for critical files. |

## Post-Publish Checks

| Check | Evidence | Required result |
| --- | --- | --- |
| Live publication verifier | `python scripts/verify_live_publication.py` | live GitHub Pages, GitHub raw, HF Space, artifact dataset, and model mirrors match the current release assets |
| GitHub Pages deployment | `gh run list --repo ChaoYue0307/ropedia-xperience-10m-task-suite --limit 5` | latest pages-build-deployment run succeeds |
| Rendered browser check | `Browser/Playwright page identity, nonblank render, console health, and one local interaction` | no relevant console warnings/errors and target links work |

## Rerun Order

```bash
python scripts/validate_scope_claims.py
python scripts/validate_source_alignment.py
python scripts/build_evaluation_protocol.py
python scripts/build_brand_assets.py
python scripts/build_figure_index.py
python scripts/validate_website_integrity.py
python scripts/build_quality_gates.py
python scripts/build_artifact_index.py
python scripts/validate_publication_package.py
python scripts/validate_mirror_parity.py
```

After Hugging Face bundle sync, rerun `validate_publication_package.py` and `validate_mirror_parity.py` once more before upload.
