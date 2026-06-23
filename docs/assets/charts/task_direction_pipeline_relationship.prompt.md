# Task-Direction-Pipeline Relationship Map

Generated as a public infographic asset for the website relationship block.

## Public Asset

- `docs/assets/charts/task_direction_pipeline_relationship.png`

## Generation Prompt

Use case: infographic-diagram.

Asset type: high-resolution website figure for Ropedia Xperience-10M.

Primary request: generate a polished, readable, dark neon Ropedia-style
infographic that clearly explains the current expandable structure:
20 task contracts -> 4 research directions -> 3 foundation training pipelines.

Critical text accuracy and layout:

- Make the numbers 20, 4, and 3 unmistakable. Use large section headings:
  `20 TASKS`, `4 RESEARCH DIRECTIONS`, `3 FOUNDATION PIPELINES`.
- Add the subtitle: `Current public release, open to expand`.
- Left side: exactly twenty task tiles in a 5 x 4 grid. Use large numbers
  `01` through `20` plus simple icons. Keep task labels short and readable:
  Action, Step, Boundary, Next Action, Hand Path, Contact, Object, Grounding,
  Retrieval, Reconstruct, Order, Sync, Horizon, Subtask, Text, Relation,
  Object Set, IMU Pose, View Sync, Time.
- Middle: exactly four large direction panels labeled A, B, C, D. Use these
  direction names:
  - A. Human Modeling & Motion Understanding
  - B. 3D/4D Reconstruction & Neural Rendering
  - C. Egocentric Vision & Interaction
  - D. Scene Reconstruction & World Modeling
- Under each direction, include one concise focus line:
  - A: pose, hand/body motion, human-object interaction
  - B: multi-view 3D/4D reconstruction, NeRF/GS, novel views
  - C: egocentric actions, intention, objects, language
  - D: scene graphs, spatial reasoning, future world state
- Right side: exactly three foundation pipeline cards:
  - 1 Spatial Intelligence
  - 2 Human-Video World Model
  - 3 Vision-Language-Action
- Each pipeline card shows inputs -> targets:
  - Spatial Intelligence: RGB + depth + pose -> 3D geometry + spatial reasoning
  - Human-Video World Model: video + action + motion -> future frame + future action
  - Vision-Language-Action: video + language + motion -> robot action chunk
- Draw clean arrows from the 20-task grid into the 4 direction panels, then
  arrows into the 3 pipeline cards.
- Add a footer line: `Tasks, directions, and pipelines can expand as new evidence is added.`

Visual style:

- Deep black background, neon lime as primary, cyan/blue/violet accents,
  consistent with the Ropedia visual style.
- Clean scientific dashboard infographic, high contrast, sharp typography,
  generous spacing.
- No photorealistic presenter, no white page background, no watermark.
- Avoid tiny decorative clutter; prioritize readability at website size.

## Post-Processing

The generated image output was used as the visual base. The top title band was
locally corrected with an exact text overlay to avoid generated-text spelling
drift while preserving the generated diagram content.

## Verification

The committed asset was visually checked after post-processing. The page also
includes an HTML key that lists the exact 20 task names, four canonical research
directions, and three foundation-pipeline tracks, so reader-facing counts do
not depend only on the image.
