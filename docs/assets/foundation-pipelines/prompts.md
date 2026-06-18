# Foundation Pipeline Photo Restoration Notes

The current public assets are not generated concept art. They are restored
high-resolution PNGs rebuilt from original presentation photos supplied by the
project owner. The filename is kept as `prompts.md` because older public
manifests and mirrors already link here as the provenance note.

| Track | Source photo | Enhanced public PNG |
| --- | --- | --- |
| Spatial intelligence models | `source-photos/spatial-intelligence-source.jpg` | `spatial-intelligence-pipeline.png` |
| Human-video world models | `source-photos/human-video-world-model-source.jpg` | `human-video-world-model-pipeline.png` |
| Vision-language-action models | `source-photos/vision-language-action-source.jpg` | `vision-language-action-pipeline.png` |

Restoration is deterministic and local:

- EXIF orientation normalization.
- Autocontrast and moderate brightness/color/contrast correction.
- Lanczos resize to a 2560-pixel public width.
- Gentle sharpening and unsharp masking.

The restoration script deliberately does not synthesize, redraw, or hallucinate
slide text. Technical task/training/evaluation claims are maintained in
`THREE_FOUNDATION_PIPELINES.md` and
`docs/data/three_foundation_pipelines.json`.
