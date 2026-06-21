# Foundation Pipeline Slide Diagrams

These three public images are high-resolution foundation-direction slide
diagrams. They are used for the pipeline tracks documented in
`THREE_FOUNDATION_PIPELINES.md` and
`docs/data/three_foundation_pipelines.json`.

They replace the earlier concept-art images and keep the public visuals tied to
the original direction slides. Spatial intelligence and human-video world
modeling use the clean slide PNGs supplied for publication and are exported as
2560-pixel public assets. VLA now uses the clean VLA slide PNG supplied
afterward and is exported through the same 2560-pixel public path.
They are still **pipeline communication assets**, not evidence of completed
foundation-model quality. Exact technical claims live in the surrounding
Markdown, JSON, and website labels.

| Track | Enhanced asset | Source |
| --- | --- | --- |
| Spatial intelligence models | `spatial-intelligence-pipeline.png` | `source-slides/spatial-intelligence-slide.png` |
| Human-video world models | `human-video-world-model-pipeline.png` | `source-slides/human-video-world-model-slide.png` |
| Vision-language-action models | `vision-language-action-pipeline.png` | `source-slides/vision-language-action-slide.png` |

The website places each figure beside a one-sample training I/O recipe:

| Track | One-sample training pair |
| --- | --- |
| Spatial intelligence models | Current 20-frame multiview/depth/pose/object window -> spatial relation, retrieval, reconstruction-proxy, or QA target. |
| Human-video world models | Current observed 20-frame window at time `t` -> shifted future action, subtask, object-set, contact, transition-time, or future-feature target. |
| Vision-language-action models | Egocentric video + caption/object/motion/contact context -> action-token, object-action, contact, interaction-text, subtask, or hand-trajectory proxy target. |

The deterministic restoration script is
`scripts/render_foundation_pipeline_diagrams.py`; restoration notes and source
mapping are in `prompts.md`.
