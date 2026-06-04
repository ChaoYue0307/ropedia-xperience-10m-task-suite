# Release Checks

This file is the release map for the Ropedia Xperience-10M Task Suite.

Current gate status: **pass**

A release is current when the automated reports pass and the live GitHub/Hugging Face mirrors are verified after publishing.

These checks cover public packaging, project status wording, mirror parity, and website integrity. Cross-episode model quality is measured by later held-out evaluation reports.

## Automated Checks

| Check | Command | Report | Current status | Needs attention when |
| --- | --- | --- | --- | --- |
| Multi-episode pilot status | `python scripts/validate_scope_claims.py` | `docs/data/scope_claims_audit.json` | `pass` | Setup identifiers are presented as completed held-out metrics. |
| Source alignment | `python scripts/validate_source_alignment.py` | `docs/data/source_alignment_audit.json` | `pass` | Official full-dataset facts, sample-card facts, API-listing notes, or project coverage are missing or inconsistent. |
| Website integrity | `python scripts/validate_website_integrity.py` | `docs/data/website_integrity.json` | `pass` | Local links, anchors, JSON bundles, or referenced image assets are missing or invalid. |
| Rendered website check | `python scripts/build_rendered_site_check.py --input /tmp/xperience_rendered_site_observations.json` | `docs/data/rendered_site_check.json` | `pass` | The local rendered site cannot load, switch tabs, deep-link to the walkthrough, update player controls, or stay console-clean. |
| Task surface integrity | `python scripts/validate_task_surface.py` | `docs/data/task_surface_integrity.json` | `pass` | Task cards expose raw artifact ids, human-readable task names drift, modality thumbnails are missing, or the interactive task player is not wired to the generated JSON. |
| Evaluation protocol | `python scripts/build_evaluation_protocol.py` | `docs/data/evaluation_protocol.json` | `pass` | Windowing, split policy, leakage controls, task metrics, or current limitations are not explicit. |
| Figure index | `python scripts/build_figure_index.py` | `docs/data/figure_index.json` | `pass` | Public figures, charts, or modality thumbnails are missing, unreadable, or lack source-script provenance. |
| Brand assets | `python scripts/build_brand_assets.py` | `docs/data/brand_assets.json` | `pass` | The generated logo system, favicon, social card, or app icons are missing or not reproducibly packaged. |
| Release-check manifest | `python scripts/build_quality_gates.py` | `docs/data/quality_gates.json` | `pass` | A public reader cannot see the current release state in one place. |
| Artifact index | `python scripts/build_artifact_index.py` | `docs/data/artifact_index.json` | `pass` | Project-critical evidence files are missing from the indexed artifact layer. |
| Public bundle contents | `python scripts/validate_publication_package.py` | `docs/data/publication_audit.json` | `pass` | Raw data, caches, heavy archives, credential text, missing required assets, or outdated public-card assets enter public bundles. |
| Public project surface | `python scripts/build_public_surface_qa.py` | `docs/data/public_surface_qa.json` | `pass` | Repo, website, or Hugging Face presentation loses SEO/social metadata, accessible tab semantics, source links, project-check links, or clear project language. |
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
python scripts/build_rendered_site_check.py --input /tmp/xperience_rendered_site_observations.json
python scripts/validate_task_surface.py
python scripts/build_quality_gates.py
python scripts/build_artifact_index.py
python scripts/validate_publication_package.py
python scripts/build_public_surface_qa.py
python scripts/validate_mirror_parity.py
```

After Hugging Face bundle sync, rerun `validate_publication_package.py` and `validate_mirror_parity.py` once more before upload.
