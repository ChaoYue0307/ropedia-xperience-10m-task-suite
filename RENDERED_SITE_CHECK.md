# Rendered Website Check

This report records the latest browser-level check of the local static website.

Current status: **pass**

## Browser Flow

- URL: `http://127.0.0.1:8765/#walkthroughs`
- Title: `Ropedia Xperience-10M Task Suite`
- Viewport: `481 x 716`
- Flow: load current docs server -> open #walkthroughs deep link -> click Next -> click Process story chapter
- Screenshot: `/tmp/xperience_site_walkthrough_fresh.png`

## Checks

| Check | Status | What it covers |
| --- | --- | --- |
| Page Identity | `pass` | The rendered page should load the expected local site title and URL. |
| First Meaningful Content | `pass` | The first meaningful heading should identify the research task lab. |
| Responsive Viewport Recorded | `pass` | The browser run should record a narrow responsive viewport large enough to expose the mobile/tablet layout. |
| Tabbed Research Navigation | `pass` | The rendered top-level tab system should switch to the Data tab for the walkthrough deep link. |
| Task And Modality Cards Render | `pass` | The rendered task and modality sections should expose all 12 task cards and seven modality cards. |
| Walkthrough Deep Link | `pass` | The walkthrough deep link should reveal the walkthrough player, all task selectors, and four chapter controls. |
| Walkthrough Interaction | `pass` | Clicking Next and the Process chapter should update the active task, chapter, counter, and frame label. |
| Rendered Check Resource Link | `pass` | The rendered page should expose the rendered website check JSON from the resource section. |
| Console Health | `pass` | The rendered flow should complete without browser console warnings or errors. |

## Regenerate

Run the local static website, exercise the walkthrough in a browser, save the observation JSON, then rebuild this report:

```bash
python scripts/build_rendered_site_check.py --input /tmp/xperience_rendered_site_observations.json
```
