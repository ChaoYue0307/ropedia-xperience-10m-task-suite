# Public Presentation Check

This generated report checks whether the public repo, website, and Hugging Face cards read like one polished research project.

Current status: **pass**

This report validates public presentation quality and package contents. Multi-episode model metrics are tracked by the training and evaluation reports.

## Checks

| Check | Status | What it guards |
| --- | --- | --- |
| public_presentation_files_exist | `pass` | Repo README, website HTML, and three Hugging Face cards should all be present in the publication workspace. |
| core_status_reports_pass | `pass` | The presentation check depends on the existing project validators already reporting pass. |
| website_has_research_seo_metadata | `pass` | The website should expose search/social metadata and structured project metadata. |
| website_tabs_are_accessible_and_keyboardable | `pass` | The long research dashboard should be navigable as real tabs, including keyboard support. |
| responsive_navigation_guard_present | `pass` | Tablet/mobile navigation should not overflow and deep links should land below sticky navigation. |
| public_naming_consistent | `pass` | Public copy should consistently present the project as Ropedia Xperience-10M, with the Qwen3-Omni scale-up status. |
| public_links_cover_repo_hf_dataset_and_ropedia | `pass` | Public cards should link the repo, Space, artifacts, model baselines, upstream dataset, and Ropedia dataset page. |
| public_artifact_qa_files_are_exposed | `pass` | Readers should be able to find integrity, publication, mirror, and presentation-check files from public copy. |
| public_copy_uses_reader_facing_language | `pass` | Public copy should use reader-facing project language and avoid private tooling, hardware labels, assessment framing, or design-process notes. |

## Scope

| Surface | File |
| --- | --- |
| github_readme | `README.md` |
| website_html | `docs/index.html` |
| hf_space_card | `hf_publish/space/README.md` |
| hf_artifact_card | `hf_publish/artifacts/README.md` |
| hf_model_card | `hf_publish/model/README.md` |

## Regenerate

```bash
python scripts/build_public_surface_qa.py
```
