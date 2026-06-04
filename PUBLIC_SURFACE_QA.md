# Public Project Surface

This generated report checks whether the public repo, website, and Hugging Face cards read like one cohesive research project.

Current status: **pass**

This report covers the public repo, website, Hugging Face cards, and package contents. Multi-episode model metrics are tracked by the training and evaluation reports.

## Checks

| Area | Status | What it covers |
| --- | --- | --- |
| Public files | `pass` | Repo README, website HTML, and three Hugging Face cards should all be present in the publication workspace. |
| Project reports | `pass` | The public project surface depends on the existing project reports already passing. |
| Website metadata | `pass` | The website should expose search/social metadata and structured project metadata. |
| Keyboard navigation | `pass` | The long research dashboard should be navigable as real tabs, including keyboard support. |
| Responsive navigation | `pass` | Tablet/mobile navigation should not overflow and deep links should land below sticky navigation. |
| Project naming | `pass` | Public copy should consistently present the project as Ropedia Xperience-10M, with the Qwen3-Omni scale-up status. |
| Public links | `pass` | Public cards should link the repo, Space, artifacts, model baselines, upstream dataset, and Ropedia dataset page. |
| Artifact links | `pass` | Readers should be able to find website reference, release package, mirror, and public presentation files from public copy. |
| Project language | `pass` | Project language is clear and avoids hardware details or irrelevant implementation details. |

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
