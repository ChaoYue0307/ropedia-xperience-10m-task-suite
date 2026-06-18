# Foundation Pipeline Slide Source Notes

The current public assets are not generated concept art. They are restored
high-resolution PNGs rebuilt from original direction-slide sources supplied by
the project owner. The filename is kept as `prompts.md` because older public
manifests and mirrors already link here as the provenance note.

Update on 2026-06-19: the latest supplied clean Spatial intelligence and
Human-video world model PNGs are byte-identical to the committed source-slide
cache and are published as 2560-pixel public images. The third uploaded image
duplicates the Spatial intelligence PNG, so the Vision-language-action card is
published as a clean deterministic slide redraw from the original VLA
presentation-photo content.

| Track | Source | Enhanced public PNG |
| --- | --- | --- |
| Spatial intelligence models | `source-slides/spatial-intelligence-slide.png` | `spatial-intelligence-pipeline.png` |
| Human-video world models | `source-slides/human-video-world-model-slide.png` | `human-video-world-model-pipeline.png` |
| Vision-language-action models | `source-photos/vision-language-action-source.jpg` | `vision-language-action-pipeline.png` |

Restoration is deterministic and local:

- Clean slide PNGs are used directly where available.
- EXIF orientation normalization.
- Autocontrast and moderate brightness/color/contrast correction.
- Lanczos resize to a 2560-pixel public width.
- Gentle sharpening and unsharp masking.
- For VLA only, a deterministic clean slide redraw preserves the visible
  presentation content from the source photo while matching the clean black and
  lime public-slide style.

The restoration script deliberately avoids hallucinated model claims or
non-source concept art. Technical task/training/evaluation claims are maintained in
`THREE_FOUNDATION_PIPELINES.md` and
`docs/data/three_foundation_pipelines.json`.
