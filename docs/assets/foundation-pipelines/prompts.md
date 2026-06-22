# Foundation Pipeline Slide Source Notes

The current public assets are not generated concept art. They are restored
high-resolution PNGs rebuilt from original direction-slide sources supplied by
the project owner. The filename is kept as `prompts.md` because older public
manifests and mirrors already link here as the provenance note.

Update on 2026-06-19: the latest supplied clean Spatial intelligence and
Human-video world model PNGs are byte-identical to the committed source-slide
cache and are published as 2560-pixel public images. A clean
Vision-language-action PNG was then supplied and is now committed as the VLA
source slide, replacing the temporary redraw fallback.

| Track | Source | Enhanced public PNG |
| --- | --- | --- |
| Spatial intelligence models | `source-slides/spatial-intelligence-slide.png` | `spatial-intelligence-pipeline.png` |
| Human-video world models | `source-slides/human-video-world-model-slide.png` | `human-video-world-model-pipeline.png` |
| Vision-language-action models | `source-slides/vision-language-action-slide.png` | `vision-language-action-pipeline.png` |

Restoration is deterministic and local:

- Clean slide PNGs are used directly where available.
- EXIF orientation normalization.
- Autocontrast and moderate brightness/color/contrast correction.
- Lanczos resize to a 2560-pixel public width.
- Gentle sharpening and unsharp masking.
The restoration script deliberately avoids hallucinated model results or
non-source concept art. Technical task/training/evaluation scope is maintained in
`THREE_FOUNDATION_PIPELINES.md` and
`docs/data/three_foundation_pipelines.json`.
