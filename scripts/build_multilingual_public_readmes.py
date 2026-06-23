#!/usr/bin/env python3
"""Build public README entry points for the project mirrors."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPDATED = "2026-06-23"
LANGUAGES = [
    ("en", "English", "README.md"),
    ("zh", "中文", "README.zh.md"),
    ("es", "Español", "README.es.md"),
    ("fr", "Français", "README.fr.md"),
    ("de", "Deutsch", "README.de.md"),
    ("ja", "日本語", "README.ja.md"),
    ("ko", "한국어", "README.ko.md"),
    ("pt", "Português", "README.pt.md"),
]


def lang_bar(active: str) -> str:
    parts = []
    for code, label, filename in LANGUAGES:
        text = f"<b>{label}</b>" if code == active else label
        parts.append(f'  <a href="{filename}">{text}</a>')
    return "<!-- LANG-BAR:START -->\n<p align=\"center\">\n" + " ·\n".join(parts) + "\n</p>\n<!-- LANG-BAR:END -->"


def badges() -> str:
    return """<p align="center">
  <a href="https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/"><img alt="GitHub Pages" src="https://img.shields.io/badge/site-GitHub%20Pages-1f63e9"></a>
  <a href="https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite"><img alt="HF Space" src="https://img.shields.io/badge/Hugging%20Face-Space-ffb000"></a>
  <a href="https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts"><img alt="artifact dataset" src="https://img.shields.io/badge/HF-artifacts-008b9a"></a>
  <a href="https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines"><img alt="baseline model repo" src="https://img.shields.io/badge/HF-baselines-7ae5c3"></a>
  <a href="https://huggingface.co/datasets/ropedia-ai/xperience-10m"><img alt="Xperience-10M" src="https://img.shields.io/badge/dataset-Xperience--10M-344054"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-code%20MIT%20%2B%20data%20terms-ccffa0"></a>
</p>"""


def hero(title: str, tagline: str, active: str) -> str:
    return f"""<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-social-card.png" alt="Ropedia Xperience-10M Task Suite cover" width="100%">
</p>

<h1 align="center">{title}</h1>

<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-mark-192.png" alt="Ropedia Xperience-10M Task Suite logo" width="112">
</p>

<p align="center">
  <strong>{tagline}</strong>
</p>

{lang_bar(active)}

{badges()}
"""


ENGLISH_TOP = f"""{hero(
    "Ropedia Xperience-10M Task Suite",
    "Public task and evaluation layer for Xperience-10M: sample data, 20 embodied-AI tasks, baselines, Qwen3-Omni and Cosmos3 diagnostics, and foundation-model training directions.",
    "en",
)}

This project builds on the Xperience-10M dataset released by Ropedia to provide public, reproducible embodied-AI evaluation materials. It is organized into two evidence lines. **Line 1** turns one public sample episode into inspectable tasks, targets, and baseline runs. **Line 2** uses selected 128-episode public-safe artifacts for aligned metadata/raw baselines, Qwen3-Omni v6 LoRA, Cosmos3-Super Reasoner, and Cosmos3-Nano Future Window. Every score links back to its source artifact, and direct scores remain clearly separated from compact-proxy estimates.

**Updated:** {UPDATED}.

**Scope:** Line 1 uses one public sample episode. Line 2 uses selected 128-episode public-safe artifacts linked back to official gated episode paths. Raw Xperience-10M MP4/HDF5/RRD files, Qwen3 base weights, Cosmos3 base weights, and gated data are not redistributed here.

## Contents

- [Project Entry Points](#project-entry-points)
- [At A Glance](#at-a-glance)
- [Data Explorer Analysis](#data-explorer-analysis)
- [Two Evidence Lines](#two-evidence-lines)
- [Fast Project Map](#fast-project-map)
- [Why This Project Exists](#why-this-project-exists)
- [Start Here](#start-here)
- [Glossary](#glossary)
- [Current Research Scope](#current-research-scope)
- [Evaluation Protocol](#evaluation-protocol)
- [Dataset Context](#dataset-context)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

## Project Entry Points

Use the two evidence lines first, then choose the artifact that answers your question. The dashboard is the best visual overview; the GitHub repo is the source of truth for scripts and generated JSON; Hugging Face mirrors contain public-safe cards, metrics, figures, and model artifacts.

Quick rule: use **Line 1** for “can I inspect and reproduce the task?” Use **Line 2** for “how do aligned baselines and model diagnostics compare on the selected 128 episodes?”

The committed task contracts, result matrices, validation JSON, public-safe result packages, GitHub sources, and Hugging Face mirrors are the canonical technical evidence.

## At A Glance

<table>
  <thead>
    <tr>
      <th width="24%">Signal</th>
      <th>Current public state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Project identity</strong><br><img src="docs/assets/brand/xperience10m-logo-mark-192.png" alt="Ropedia Xperience-10M Task Suite logo" width="56"></td>
      <td>The same project logo mark is used across the GitHub README, GitHub Pages dashboard, Hugging Face Space, artifact dataset, model mirrors, favicon, and social preview. Ropedia is credited as the Xperience-10M data provider and releaser; this repository is the task-suite and evaluation layer built on that dataset. Reusable assets: <a href="docs/assets/brand/xperience10m-logo-mark-512.png">logo mark</a> and <a href="docs/assets/brand/xperience10m-logo-social-card.png">social card</a>.</td>
    </tr>
    <tr>
      <td><strong>Two-line contract</strong></td>
      <td><strong>Line 1: 1 sample episode</strong> for task construction and reproducibility. <strong>Line 2: 128 selected episodes</strong> for same-split metadata/raw baselines, Qwen3-Omni v6, and Cosmos3 diagnostics.</td>
    </tr>
    <tr>
      <td><strong>Data explorer analysis</strong></td>
      <td>A generated analysis layer separates the public sample, selected-128 feature exports, and authenticated Hugging Face gated full-dataset metadata with scope stats, split counts, modality breakdowns, and chart assets.</td>
    </tr>
    <tr>
      <td><strong>180 method-task records</strong></td>
      <td>9 methods x 20 tasks = 180/180 scored records. The ledger separates 174 direct scores from 6 compact-proxy scores.</td>
    </tr>
    <tr>
      <td><strong>20 task contracts</strong></td>
      <td>Action, procedure, transition, trajectory, contact, objects, language, retrieval, reconstruction, order, sync, long-horizon forecasting, interaction text, action-object binding, sensor bridging, camera sync, and transition timing.</td>
    </tr>
    <tr>
      <td><strong>4 research directions</strong></td>
      <td>Human Modeling & Motion Understanding; 3D/4D Reconstruction & Neural Rendering; Egocentric Vision & Interaction; Scene Reconstruction & World Modeling. These are analysis groups over the same 20 tasks, not separate benchmark tiers.</td>
    </tr>
    <tr>
      <td><strong>Line 1 methods</strong></td>
      <td>Minimal and Neural MLP baselines cover all 20 tasks on the one public sample episode: 40/40 direct scores.</td>
    </tr>
    <tr>
      <td><strong>Line 2 methods</strong></td>
      <td>Metadata simple/NN, raw-feature simple/NN, Qwen3-Omni v6 LoRA, Cosmos3-Super Reasoner, and Cosmos3-Nano Future Window cover all 20 selected-128 task axes: 140/140 scores.</td>
    </tr>
    <tr>
      <td><strong>3 foundation pipelines</strong></td>
      <td>Spatial intelligence, human-video world modeling, and vision-language-action pipelines are documented as training recipes with task mappings, input-output contracts, and model-evidence requirements.</td>
    </tr>
    <tr>
      <td><strong>1 unified target</strong></td>
      <td>The long-term embodied foundation-model target connects perception, 3D memory, language-grounded reasoning, action, and planning without adding a new score axis.</td>
    </tr>
    <tr>
      <td><strong>Public mirrors</strong></td>
      <td>GitHub, GitHub Pages, HF Space, HF artifact dataset, HF baseline model repo, Qwen3-Omni and Cosmos3 model repos, and HF collection.</td>
    </tr>
  </tbody>
</table>

## Public Structure: 20 Tasks / 4 Directions / 3 Pipelines / 1 Unified Target

The project has four connected layers. The **20 tasks** are the scored benchmark contracts. The **4 directions** are research groupings over those same tasks. The **3 foundation pipelines** are training recipes that reuse the same modalities, windows, and task targets. The **1 unified embodied model target** is the long-term integration goal after those pipelines mature.

Layer rule: if it has a metric, it is a **task**; if it explains what the evidence studies, it is a **direction**; if it describes model inputs and training targets, it is a **pipeline**; if it combines perception, 3D memory, language, action, and planning, it is the **unified target** rather than an extra score axis.

<p align="center">
  <img src="docs/assets/charts/task_direction_pipeline_relationship.png" alt="Relationship map showing 20 task contracts, 4 research directions, 3 foundation-model pipeline tracks, and 1 unified embodied model target" width="100%">
</p>

| Layer | Count | Role | Exact public labels |
| --- | ---: | --- | --- |
| Task contracts | 20 | Score axes used by the matrix, radars, task cards, and method rows. | Action Recognition; Procedure Step Recognition; Action Boundary Detection; Next-Action Prediction; Hand Trajectory Forecasting; Contact State Prediction; Object Relevance Prediction; Language Grounding; Cross-Modal Retrieval; Cross-Modal Reconstruction; Temporal Order Verification; Multimodal Synchronization Detection; Long-Horizon Next-Action Forecasting; Long-Horizon Next-Subtask Forecasting; Interaction Text Prediction; Action-Object Relation Prediction; Future Object-Set Forecasting; IMU-to-Hand Pose Reconstruction; Camera-View Synchronization Retrieval; Time-to-Next-Transition Regression. |
| Research directions | 4 | Ways to interpret what the 20 tasks study; not separate benchmark tiers. | Human Modeling & Motion Understanding; 3D/4D Reconstruction & Neural Rendering; Egocentric Vision & Interaction; Scene Reconstruction & World Modeling. |
| Foundation pipelines | 3 | Larger-model training tracks with separate input-output recipes and result gates. | Spatial intelligence models; Human-video world models; Vision-language-action models. |
| Unified embodied model target | 1 | Long-term integration target, not a task/method row in the 180-result matrix. | Perception; 3D memory; language-grounded reasoning; action; planning. |

## Data Explorer Analysis

The data explorer is now a three-scope analysis layer, not only a raw-file browser. It compares the public sample episode, selected 128-episode feature exports, and the Hugging Face-hosted gated full-dataset metadata without mixing their evidence boundaries.

| Scope | Question | Current public analysis |
| --- | --- | --- |
| Public sample | What files and signals are directly inspectable? | 1 episode, 5,821 frames, 1,161 aligned 20-frame windows, 8,546 feature dimensions, raw-file browser, modality breakdowns, action-window distribution. |
| Selected 128 | What selected-episode surface supports model comparison? | 96/16/16 split, 34,269 Qwen3-Omni v6 multiscale rows, 106,095 dense compact rows, selected episode links, public-safe matrices. |
| Full HF dataset | How large is the official upstream dataset? | Authenticated Hub file metadata: 804 sessions, 12,103 episode-like folders, 85,257 files, 24.63 TiB training-byte view, without redistributing raw gated data. |

Entry points: [website analysis section](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/#data-analysis), [analysis report](DATA_EXPLORER_ANALYSIS.md), and [structured analysis record](docs/data/data_explorer_analysis.json).

## Two Evidence Lines

The public suite is organized around two evidence lines. Keep them separate when comparing metrics.

<p align="center">
  <img src="docs/assets/charts/two_evidence_line_map.svg" alt="Two evidence-line map: 1 sample episode and 128 selected episodes combine into 180 scored method-task records" width="100%">
</p>

<table>
  <thead>
    <tr>
      <th width="20%">Line</th>
      <th width="24%">Data unit</th>
      <th width="22%">Score statement</th>
      <th width="20%">Best use</th>
      <th>Read separately from</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1 sample episode</strong></td>
      <td>One public Xperience-10M sample episode: 5,821 frames, 1,161 aligned 20-frame windows, 8,546 feature dimensions.</td>
      <td>40/40 direct scores from Minimal and Neural MLP heads.</td>
      <td>Inspect the raw sample, understand file organization, reproduce the 20 task targets, and compare Minimal vs Neural MLP behavior inside one episode.</td>
      <td>The selected-128 comparison rows and any broader held-out model behavior.</td>
    </tr>
    <tr>
      <td><strong>128 selected episodes</strong></td>
      <td>Selected held-out 96/16/16 split: 34,269 exported windows with public-safe processed features linked to official gated episode paths. The Hugging Face artifact dataset exposes these rows separately as <a href="https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts/viewer/selected_128_windows/selected_128"><code>selected_128_windows/selected_128</code></a>; it is not mixed with the one-sample <code>episode_sample/public_sample</code> viewer.</td>
      <td>140/140 selected-128 scores: 134 direct + 6 compact-proxy.</td>
      <td>Compare same-split metadata/raw baselines, Qwen3-Omni v6, Cosmos3-Super, and Cosmos3-Nano while keeping the 6 compact-proxy cells visible.</td>
      <td>Direct raw-target measurements for the proxy-marked cells.</td>
    </tr>
  </tbody>
</table>

### Result Ledger

<table>
  <thead>
    <tr>
      <th width="20%">Line</th>
      <th width="14%">Methods</th>
      <th width="14%">Tasks</th>
      <th width="18%">Scored records</th>
      <th width="16%">Direct scores</th>
      <th>Proxy scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1 sample episode</strong></td>
      <td>2</td>
      <td>20</td>
      <td>40/40</td>
      <td>40</td>
      <td>0</td>
    </tr>
    <tr>
      <td><strong>128 selected episodes</strong></td>
      <td>7</td>
      <td>20</td>
      <td>140/140</td>
      <td>134</td>
      <td>6 compact-proxy scores, each source-linked and reasoned.</td>
    </tr>
    <tr>
      <td><strong>Total public matrix</strong></td>
      <td>9</td>
      <td>20</td>
      <td>180/180</td>
      <td>174</td>
      <td>6</td>
    </tr>
  </tbody>
</table>

### Method Blocks

<table>
  <thead>
    <tr>
      <th width="20%">Evidence line</th>
      <th width="20%">Method block</th>
      <th width="24%">Methods</th>
      <th width="18%">Score statement</th>
      <th>Read as</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1 sample episode</strong></td>
      <td>Task-head baselines</td>
      <td>Minimal; Neural MLP</td>
      <td>40/40 direct scores.</td>
      <td>Task-lab reproducibility and simple-vs-neural behavior.</td>
    </tr>
    <tr>
      <td><strong>128 selected episodes</strong></td>
      <td>Aligned baseline heads</td>
      <td>Metadata simple/NN; raw-feature simple/NN</td>
      <td>80/80 scores: 74 direct + 6 compact-proxy.</td>
      <td>Same-split metadata/raw-feature baseline comparison.</td>
    </tr>
    <tr>
      <td><strong>128 selected episodes</strong></td>
      <td>Qwen3-Omni series</td>
      <td>Qwen3-Omni v6 LoRA</td>
      <td>20/20 direct scores from verified selected-128 Qwen3-Omni LoRA and task-specific probes.</td>
      <td>Trainable Qwen3-Omni diagnostic baseline on the selected-128 surface.</td>
    </tr>
    <tr>
      <td><strong>128 selected episodes</strong></td>
      <td>Cosmos3 series</td>
      <td>Cosmos3-Super Reasoner; Cosmos3-Nano Future Window</td>
      <td>40/40 direct scores from verified public-safe reasoner and future-window artifacts.</td>
      <td>Cosmos3 reasoner and future-window diagnostics on the selected-128 surface.</td>
    </tr>
  </tbody>
</table>

Cosmos3-Super Forward-Dynamics LoRA is published as a separate fine-tuned adapter artifact with weights/results; it is not counted as a 20-task matrix method row.

### Qwen3-Omni Run Versions

These are Qwen3-Omni run versions inside **Line 2: selected 128 episodes**. They are not the project evidence lines. The 20-task matrix uses **Qwen3-Omni v6 LoRA**; **v5** remains the pinned prior multiscale release; **v1-v4** are lineage and ablation evidence.

<table>
  <thead>
    <tr>
      <th width="8%">Run</th>
      <th width="26%">Purpose</th>
      <th width="28%">Main change</th>
      <th width="16%">Eval signal</th>
      <th>Use now</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>v1</strong></td><td>Prove the selected-128 LoRA/eval/package loop.</td><td>First verified 96/16/16 selected-episode Qwen3-Omni LoRA run.</td><td>448 eval; JSON 0.8750; contact 0.6451.</td><td>Lineage only.</td></tr>
    <tr><td><strong>v2</strong></td><td>Make answers schema-checked.</td><td>Structured-JSON contract with full-8-GPU LoRA on the same split.</td><td>448 eval; JSON 0.9978; contact 0.7188.</td><td>Structured-output ablation.</td></tr>
    <tr><td><strong>v3</strong></td><td>Separate prompt/eval effects from training.</td><td>Strict-label prompt/eval over the v2 adapter; no new adapter training.</td><td>448 eval; JSON 1.0000; contact 0.7210.</td><td>Prompt/eval ablation.</td></tr>
    <tr><td><strong>v4</strong></td><td>Test longer structured-JSON LoRA training.</td><td>New four-epoch full-8-GPU adapter on the same selected split.</td><td>448 eval; JSON 1.0000; contact 0.7299.</td><td>Overfit/metric-tradeoff evidence.</td></tr>
    <tr><td><strong>v5</strong></td><td>Move to denser multiscale evaluation.</td><td>Multiscale cap96 export with 4,032 held-out predictions.</td><td>4,032 eval; JSON 1.0000; contact 0.7865.</td><td>Pinned prior release; stronger on several non-contact metrics.</td></tr>
    <tr><td><strong>v6</strong></td><td>Publish the current Qwen 20-task row.</td><td>Rank64/lr5e-5 multiscale LoRA plus verified task-specific probes.</td><td>4,032 eval; JSON 0.9990; contact 0.8177.</td><td>Current public 20-task Qwen3-Omni row.</td></tr>
  </tbody>
</table>

Detailed lineage:
[`QWEN3_OMNI_RUN_LINEAGE.md`](QWEN3_OMNI_RUN_LINEAGE.md) and
[`qwen3_omni_run_lineage.json`](docs/data/qwen3_omni_run_lineage.json).

Result entry points:
[`TWO_EVIDENCE_LINES.md`](TWO_EVIDENCE_LINES.md),
[`two_evidence_lines.json`](docs/data/two_evidence_lines.json),
[`TWO_EVIDENCE_LINE_RESULT_SUMMARY.md`](TWO_EVIDENCE_LINE_RESULT_SUMMARY.md),
[`two_evidence_line_result_summary.json`](docs/data/two_evidence_line_result_summary.json),
[`QWEN3_OMNI_RUN_LINEAGE.md`](QWEN3_OMNI_RUN_LINEAGE.md),
[`qwen3_omni_run_lineage.json`](docs/data/qwen3_omni_run_lineage.json),
[`single_episode_task_model_radar.json`](docs/data/single_episode_task_model_radar.json),
[`episode128_task_model_radar.json`](docs/data/episode128_task_model_radar.json),
[`task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json), and
[`xperience10m_128_episode_feature_index.json`](docs/data/xperience10m_128_episode_feature_index.json).

## Fast Project Map

<table>
  <thead>
    <tr>
      <th width="26%">Goal</th>
      <th width="32%">Start here</th>
      <th>Then inspect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Understand quickly</strong></td>
      <td><a href="PROJECT_BRIEF.md">Project brief</a><br><a href="PROJECT_STATUS.md">Project status</a></td>
      <td><a href="https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/">Dashboard</a></td>
    </tr>
    <tr>
      <td><strong>Choose the published mirror</strong></td>
      <td><a href="PUBLIC_READER_MAP.md">Public evidence map</a></td>
      <td><a href="docs/data/public_reader_map.json">evidence-map data</a></td>
    </tr>
    <tr>
      <td><strong>Decode project terms</strong></td>
      <td><a href="GLOSSARY.md">Glossary</a></td>
      <td><a href="docs/data/glossary.json">glossary data</a></td>
    </tr>
    <tr>
      <td><strong>Inspect the 20 tasks</strong></td>
      <td><a href="TASK_SUITE_20.md">20-task guide</a></td>
      <td><a href="docs/data/task_suite_20.json">task contract data</a><br><a href="results/episode_task_suite/task_walkthroughs/">task walkthroughs</a></td>
    </tr>
    <tr>
      <td><strong>Explore data scales</strong></td>
      <td><a href="DATA_EXPLORER_ANALYSIS.md">Data explorer analysis</a></td>
      <td><a href="https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/#data-analysis">website analysis section</a><br><a href="docs/data/data_explorer_analysis.json">structured analysis record</a></td>
    </tr>
    <tr>
      <td><strong>Compare results</strong></td>
      <td><a href="RESEARCH_TAKEAWAYS.md">Research takeaways</a></td>
      <td><a href="docs/data/two_evidence_line_result_summary.json">two-line result summary</a><br><a href="docs/data/task_method_20_result_matrix.json">180-record result table</a><br><a href="docs/data/unified_task_model_radar.json">radar data</a><br><a href="docs/data/task_method_20_gap_audit.json">score/proxy audit</a></td>
    </tr>
    <tr>
      <td><strong>Understand one sample</strong></td>
      <td><a href="https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html">Single-episode explorer</a></td>
      <td><a href="docs/data/raw_sample_files.json">sample-file map</a><br><a href="results/episode_task_suite/feature_manifest.json">feature manifest</a></td>
    </tr>
    <tr>
      <td><strong>Read foundation directions</strong></td>
      <td><a href="THREE_FOUNDATION_PIPELINES.md">Three foundation pipelines</a></td>
      <td><a href="docs/data/three_foundation_pipelines.json">pipeline contract data</a><br><a href="FOUNDATION_MODEL_PLAN.md">foundation model plan</a></td>
    </tr>
    <tr>
      <td><strong>Reproduce or audit</strong></td>
      <td><a href="REPRODUCIBILITY.md">Reproducibility</a><br><a href="EVIDENCE_CONTRACT.md">Evidence contract</a></td>
      <td><a href="docs/data/quality_gates.json">quality gates</a><br><a href="docs/data/publication_audit.json">publication audit</a><br><a href="docs/data/mirror_parity.json">mirror parity</a></td>
    </tr>
  </tbody>
</table>
"""


LANGUAGE_GUIDES = {
    "zh": {
        "title": "Xperience-10M 任务套件",
        "tagline": "面向 Xperience-10M 的公开任务与评估层：样本数据、20 个具身智能任务、基线、Qwen3-Omni 与 Cosmos3 诊断结果，以及基础模型训练方向。",
        "body": f"""## 项目概览

这个仓库把 Ropedia 公开的 Xperience-10M sample episode 变成一个可检查的具身智能任务实验室。请先看仪表盘和项目状态，再进入 20 个任务、结果矩阵和 Hugging Face 镜像。

**更新时间：** {UPDATED}。

**范围：** 完整可复现的任务套件来自一个公开样本 episode；128-episode 结果只发布 public-safe 的指标、报告、预测摘要和模型卡。原始 MP4/HDF5/RRD、完整 Qwen 权重和 gated 数据不在本仓库重新分发。

## 两条证据线

| 线 | 数据单元 | 方法与结果 | 用途 |
| --- | --- | --- | --- |
| 1 sample episode | 5,821 帧；1,161 个 20-frame 对齐窗口；8,546 维特征。 | Minimal + Neural MLP；20 个任务全覆盖；40/40 scored records；全部为 direct scores。 | 检查原始 sample 文件、任务定义、可复现基线和每个任务是否成立。 |
| 128 selected episodes | 96/16/16 split；34,269 个导出窗口；public-safe 特征链接到官方 gated episode path。 | Metadata simple/NN、raw-feature simple/NN、Qwen3-Omni、Cosmos3-Super、Cosmos3-Nano；140/140 scored records；134 direct + 6 compact proxy。 | 比较同一 split 上的基线和模型分支；proxy target 会显式标注。 |

公式：2 个单 episode 方法 x 20 个任务 = 40；7 个 128-episode 方法 x 20 个任务 = 140；公开矩阵总计 180/180 scored records。

方法块：Line 1 是 task-head baselines（Minimal、Neural MLP）。Line 2 分成 aligned baseline heads（metadata simple/NN、raw-feature simple/NN）、Qwen3-Omni series（Qwen3-Omni v6 LoRA）和 Cosmos3 series（Cosmos3-Super Reasoner、Cosmos3-Nano Future Window）。Qwen3 run v1-v6 是 Line 2 内部的 LoRA/评估演进线，不是项目的 evidence lines；20-task matrix 使用 v6，v5 是 pinned prior release。Cosmos3-Super Forward-Dynamics LoRA 是单独发布的 adapter 权重/结果，不计入 20-task matrix method row。

入口：[双证据线说明](TWO_EVIDENCE_LINES.md)、[双证据线数据](docs/data/two_evidence_lines.json)、[180 条结果表](docs/data/task_method_20_result_matrix.json)、[双线结果摘要](docs/data/two_evidence_line_result_summary.json)。

## 快速入口

| 目标 | 入口 |
| --- | --- |
| 快速理解项目 | [项目简报](PROJECT_BRIEF.md), [项目状态](PROJECT_STATUS.md) |
| 选择 GitHub / 网页 / HF 入口 | [公开证据地图](PUBLIC_READER_MAP.md) |
| 查看 20 个任务定义 | [20 任务指南](TASK_SUITE_20.md), [任务契约数据](docs/data/task_suite_20.json) |
| 比较结果 | [研究结论](RESEARCH_TAKEAWAYS.md), [180 条结果表](docs/data/task_method_20_result_matrix.json) |
| 查看一个 sample 的全部文件关系 | [单 episode 浏览器](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [sample 文件地图](docs/data/raw_sample_files.json) |
| 对比 sample / 128 / 全量数据规模 | [数据探索分析](DATA_EXPLORER_ANALYSIS.md), [网页分析区](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/#data-analysis) |
| 阅读三个基础模型方向 | [三条基础模型 pipeline](THREE_FOUNDATION_PIPELINES.md), [pipeline 契约数据](docs/data/three_foundation_pipelines.json) |
| 复现与审计 | [复现指南](REPRODUCIBILITY.md), [证据契约](EVIDENCE_CONTRACT.md) |

## 核心结构

- 识别规则：有 metric 的是 20 个任务层；解释这些 evidence 研究什么的是 4 个 research directions；描述模型 input/output 和训练目标的是 3 条 foundation pipelines；把感知、3D 记忆、语言推理、action 和 planning 合并起来的是 unified embodied model target，不是新的评分轴。
- 数据层：公开 sample episode 被切成 20-frame 窗口，并连接视频、音频、深度、pose/SLAM、mocap、IMU、calibration 和语言标注。
- 数据探索：新增分析层把 1 sample、selected 128 feature exports 和官方 gated full-dataset metadata 分开呈现，避免把 raw inspection、derived features 和 full-corpus metadata 混在一起。
- 任务层：20 个统一任务覆盖识别、预测、检索、重建、同步、长时预测、action-object 关系和 sensor bridge。
- 结果层：单 episode minimal/NN 覆盖 20/20；128-episode metadata/raw、Qwen3-Omni v6 LoRA、Cosmos3-Super Reasoner、Cosmos3-Nano Future Window 分开标注；当前公开矩阵为 180/180 scored records，其中 174 direct、6 compact proxy，proxy target 显式保留。
- 训练方向：spatial intelligence、human-video world model、vision-language-action 三条 pipeline 已经有任务映射和需要的证据清单；长期目标是一个 unified embodied foundation model。

## 公开边界

本项目只发布小型 derived artifacts、指标、图表、README、模型卡和 public-safe 预测摘要。原始 Xperience-10M 数据使用仍以 Ropedia 官方 Hugging Face 数据卡和访问条款为准。
""",
    },
    "es": {
        "title": "Ropedia Xperience-10M Task Suite",
        "tagline": "Capa pública de tareas y evaluación para Xperience-10M: datos de muestra, 20 tareas embodied-AI, baselines, diagnósticos Qwen3-Omni y Cosmos3, y direcciones de entrenamiento.",
        "body": f"""## Descripción Del Proyecto

Este repositorio convierte el episodio público de muestra de Xperience-10M en un laboratorio verificable de tareas para embodied AI. Empieza por el panel visual y el estado del proyecto; después entra en las tareas, matrices de resultados y espejos de Hugging Face.

**Actualizado:** {UPDATED}.

**Alcance:** la suite reproducible usa un episodio público; los resultados de 128 episodios publican solo métricas, reportes, predicciones seguras y tarjetas de modelo. No se redistribuyen MP4/HDF5/RRD originales, pesos completos de Qwen ni datos gated.

## Dos Líneas de Evidencia

| Línea | Unidad de datos | Métodos y resultados | Uso |
| --- | --- | --- | --- |
| 1 episodio de muestra | 5,821 frames; 1,161 ventanas alineadas de 20 frames; 8,546 dimensiones. | Minimal + Neural MLP en 20 tareas; 40/40 registros con score; todos son direct scores. | Inspeccionar archivos de muestra, definiciones de tarea, baselines reproducibles y validez de tareas. |
| 128 episodios seleccionados | Split 96/16/16; 34,269 ventanas exportadas; features public-safe ligadas a episode paths oficiales gated. | Metadata simple/NN, raw-feature simple/NN, Qwen3-Omni, Cosmos3-Super y Cosmos3-Nano; 140/140 registros con score; 134 direct + 6 compact proxy. | Comparar baselines y ramas de modelo en el mismo split; los proxy targets permanecen visibles. |

Fórmula: 2 métodos de un episodio x 20 tareas = 40; 7 métodos de 128 episodios x 20 tareas = 140; matriz pública total = 180/180 registros con score.

Bloques de métodos: la línea 1 contiene task-head baselines (Minimal, Neural MLP). La línea 2 separa aligned baseline heads (metadata simple/NN, raw-feature simple/NN), la serie Qwen3-Omni (Qwen3-Omni v6 LoRA) y la serie Cosmos3 (Cosmos3-Super Reasoner, Cosmos3-Nano Future Window). Qwen3 v1-v6 es una línea interna de evolución LoRA/evaluación dentro de la línea 2, no las evidence lines del proyecto; la matriz de 20 tareas usa v6 y v5 queda como pinned prior release. Cosmos3-Super Forward-Dynamics LoRA se publica como adapter/pesos/resultados aparte y no cuenta como fila de método en la matriz de 20 tareas.

Entradas: [guia de dos lineas de evidencia](TWO_EVIDENCE_LINES.md), [datos de dos lineas](docs/data/two_evidence_lines.json), [tabla de 180 resultados](docs/data/task_method_20_result_matrix.json), [resumen de resultados de dos lineas](docs/data/two_evidence_line_result_summary.json).

## Ruta Rápida

| Objetivo | Entrada |
| --- | --- |
| Entender el proyecto | [resumen del proyecto](PROJECT_BRIEF.md), [estado del proyecto](PROJECT_STATUS.md) |
| Elegir entrada GitHub / web / HF | [mapa publico de evidencia](PUBLIC_READER_MAP.md) |
| Ver las 20 tareas | [guia de 20 tareas](TASK_SUITE_20.md), [datos de contratos de tarea](docs/data/task_suite_20.json) |
| Comparar resultados | [conclusiones de investigacion](RESEARCH_TAKEAWAYS.md), [tabla de 180 resultados](docs/data/task_method_20_result_matrix.json) |
| Inspeccionar una muestra | [explorador de un episodio](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [mapa de archivos de muestra](docs/data/raw_sample_files.json) |
| Comparar sample / 128 / dataset completo | [analisis del data explorer](DATA_EXPLORER_ANALYSIS.md), [seccion web de analisis](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/#data-analysis) |
| Leer las tres direcciones foundation | [tres pipelines foundation](THREE_FOUNDATION_PIPELINES.md), [datos de contratos de pipeline](docs/data/three_foundation_pipelines.json) |
| Reproducir o auditar | [guia de reproducibilidad](REPRODUCIBILITY.md), [contrato de evidencia](EVIDENCE_CONTRACT.md) |

## Estructura

- Regla de lectura: si tiene una métrica, es una tarea de las 20; si explica qué estudia la evidencia, es una de las 4 research directions; si define inputs/outputs de entrenamiento, es una de las 3 foundation pipelines; si combina percepción, memoria 3D, lenguaje, acción y planificación, es el unified embodied model target, no otro eje de score.
- Datos: ventanas de 20 frames con video, audio, profundidad, pose/SLAM, mocap, IMU, calibración y lenguaje.
- Exploracion de datos: la nueva capa separa el sample publico, los exports selected-128 y la metadata del dataset gated completo.
- Tareas: 20 contratos para reconocimiento, predicción, recuperación, reconstrucción, sincronización, horizonte largo, relación acción-objeto y puentes de sensores.
- Resultados: minimal/NN de un episodio cubren 20/20; las ramas de 128 episodios separan metadata, raw features, Qwen3 y Cosmos; la matriz pública está en 180/180 registros con score: 174 direct y 6 compact proxy, con proxy targets visibles.
- Direcciones: spatial intelligence, human-video world model y vision-language-action tienen mapeo de tareas y requisitos de evidencia; el objetivo largo plazo es un unified embodied foundation model.

## Límite Público

El proyecto publica solo artifacts derivados, métricas, figuras, tarjetas y resúmenes public-safe. El uso de Xperience-10M sigue las condiciones oficiales de Ropedia en Hugging Face.
""",
    },
    "fr": {
        "title": "Ropedia Xperience-10M Task Suite",
        "tagline": "Couche publique de tâches et d'évaluation pour Xperience-10M : échantillon, 20 tâches embodied-AI, baselines, diagnostics Qwen3-Omni et Cosmos3, et pistes d'entraînement.",
        "body": f"""## Présentation Du Projet

Ce dépôt transforme l'épisode public d'exemple Xperience-10M en laboratoire de tâches vérifiable pour l'IA incarnée. Commencez par le tableau de bord et le statut du projet, puis ouvrez les contrats de tâches, les matrices de résultats et les miroirs Hugging Face.

**Mise à jour :** {UPDATED}.

**Portée :** la suite entièrement reproductible utilise un épisode public; les résultats 128 épisodes ne publient que des métriques, rapports, prédictions sûres et cartes de modèles. Les MP4/HDF5/RRD bruts, les poids Qwen complets et les données gated ne sont pas redistribués.

## Deux Lignes de Preuve

| Ligne | Unité de données | Méthodes et résultats | Usage |
| --- | --- | --- | --- |
| 1 épisode d'exemple | 5,821 frames; 1,161 fenêtres alignées de 20 frames; 8,546 dimensions. | Minimal + Neural MLP sur 20 tâches; 40/40 enregistrements scorés; tous sont des direct scores. | Inspecter les fichiers sample, les définitions de tâches, les baselines reproductibles et la validité des tâches. |
| 128 épisodes sélectionnés | Split 96/16/16; 34,269 fenêtres exportées; features public-safe liées aux chemins gated officiels. | Metadata simple/NN, raw-feature simple/NN, Qwen3-Omni v6, Cosmos3-Super et Cosmos3-Nano; 140/140 enregistrements scorés; 134 direct + 6 compact proxy. | Comparer les baselines, Qwen3-Omni diagnostics et Cosmos3 diagnostics sur le même split; les proxy targets restent visibles. |

Formule : 2 méthodes sur 1 épisode x 20 tâches = 40; 7 méthodes sur 128 épisodes x 20 tâches = 140; matrice publique totale = 180/180 enregistrements scorés.

Blocs de méthodes : la ligne 1 contient les task-head baselines (Minimal, Neural MLP). La ligne 2 sépare les aligned baseline heads (metadata simple/NN, raw-feature simple/NN), la série Qwen3-Omni (Qwen3-Omni v6 LoRA) et la série Cosmos3 (Cosmos3-Super Reasoner, Cosmos3-Nano Future Window). Qwen3 v1-v6 est une lignée LoRA/évaluation interne à la ligne 2, pas les evidence lines du projet; la matrice 20 tâches utilise v6 et v5 reste le pinned prior release. Cosmos3-Super Forward-Dynamics LoRA est publié comme adapter/poids/résultats séparé et ne compte pas comme ligne de méthode dans la matrice 20 tâches.

Entrées : [guide des deux lignes de preuve](TWO_EVIDENCE_LINES.md), [données des deux lignes](docs/data/two_evidence_lines.json), [table des 180 résultats](docs/data/task_method_20_result_matrix.json), [résumé des résultats par ligne](docs/data/two_evidence_line_result_summary.json).

## Parcours Rapide

| Objectif | Point d'entrée |
| --- | --- |
| Comprendre le projet | [résumé du projet](PROJECT_BRIEF.md), [état du projet](PROJECT_STATUS.md) |
| Choisir l’entrée GitHub / web / HF | [carte publique des preuves](PUBLIC_READER_MAP.md) |
| Lire les 20 tâches | [guide des 20 tâches](TASK_SUITE_20.md), [données des contrats de tâche](docs/data/task_suite_20.json) |
| Comparer les résultats | [conclusions de recherche](RESEARCH_TAKEAWAYS.md), [table des 180 résultats](docs/data/task_method_20_result_matrix.json) |
| Inspecter un sample | [explorateur d'un épisode](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [carte des fichiers sample](docs/data/raw_sample_files.json) |
| Comparer sample / 128 / dataset complet | [analyse data explorer](DATA_EXPLORER_ANALYSIS.md), [section web d'analyse](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/#data-analysis) |
| Lire les trois pipelines foundation | [trois pipelines foundation](THREE_FOUNDATION_PIPELINES.md), [données des contrats de pipeline](docs/data/three_foundation_pipelines.json) |
| Reproduire et auditer | [guide de reproductibilité](REPRODUCIBILITY.md), [contrat de preuve](EVIDENCE_CONTRACT.md) |

## Structure

- Règle de lecture : avec une métrique, c'est l'une des 20 tâches; si cela explique ce que les preuves étudient, c'est l'une des 4 research directions; si cela définit des inputs/outputs d'entraînement, c'est l'une des 3 foundation pipelines; si cela combine perception, mémoire 3D, langage, action et planification, c'est la cible unified embodied model, pas un nouvel axe de score.
- Données : fenêtres de 20 frames reliant vidéo, audio, profondeur, pose/SLAM, mocap, IMU, calibration et annotations de langage.
- Exploration des donnees : la nouvelle couche separe le sample public, les exports selected-128 et les metadonnees du dataset gated complet.
- Tâches : 20 contrats couvrant reconnaissance, prévision, retrieval, reconstruction, ordre, synchronisation, horizon long, relations action-objet et sensor bridge.
- Résultats : minimal/NN sur l'épisode public couvrent 20/20; la ligne 128 épisodes sépare metadata, raw features, Qwen3-Omni et Cosmos3; la matrice publique atteint 180/180 enregistrements scorés: 174 direct et 6 compact proxy, avec proxy targets visibles.
- Directions : spatial intelligence, human-video world model et vision-language-action sont documentés avec tâches et preuves nécessaires; l'objectif à long terme est un unified embodied foundation model.

## Frontière Publique

Le projet publie des artifacts dérivés, métriques, figures et cartes public-safe. L'accès aux données Xperience-10M reste régi par la carte officielle Ropedia sur Hugging Face.
""",
    },
    "de": {
        "title": "Ropedia Xperience-10M Task Suite",
        "tagline": "Öffentliche Aufgaben- und Evaluationsschicht für Xperience-10M: Sample-Daten, 20 Embodied-AI-Aufgaben, Baselines, Qwen3-Omni- und Cosmos3-Diagnostik und Trainingsrichtungen.",
        "body": f"""## Projektüberblick

Dieses Repository macht aus dem öffentlichen Xperience-10M-Sample eine prüfbare Aufgabenoberfläche für Embodied AI. Beginnen Sie mit Dashboard und Projektstatus, danach mit Aufgabenverträgen, Ergebnismatrizen und Hugging-Face-Spiegeln.

**Aktualisiert:** {UPDATED}.

**Umfang:** die vollständig reproduzierbare Suite nutzt ein öffentliches Sample-Episode; 128-Episode-Ergebnisse veröffentlichen nur public-safe Metriken, Berichte, Vorhersagen und Modellkarten. Rohdaten wie MP4/HDF5/RRD, vollständige Qwen-Gewichte und gated Daten werden nicht weitergegeben.

## Zwei Evidenzlinien

| Linie | Dateneinheit | Methoden und Ergebnisse | Zweck |
| --- | --- | --- | --- |
| 1 Sample-Episode | 5,821 Frames; 1,161 ausgerichtete 20-Frame-Fenster; 8,546 Dimensionen. | Minimal + Neural MLP auf 20 Aufgaben; 40/40 gescorte Einträge; alle sind direct scores. | Sample-Dateien, Aufgaben, reproduzierbare Baselines und Aufgabenqualität prüfen. |
| 128 ausgewählte Episoden | 96/16/16 Split; 34,269 exportierte Fenster; public-safe Features mit offiziellen gated Episode-Pfaden. | Metadata simple/NN, raw-feature simple/NN, Qwen3-Omni, Cosmos3-Super und Cosmos3-Nano; 140/140 gescorte Einträge; 134 direct + 6 compact proxy. | Baselines und Modellzweige auf demselben Split vergleichen; Proxy-Targets bleiben sichtbar. |

Formel: 2 Single-Episode-Methoden x 20 Aufgaben = 40; 7 128-Episode-Methoden x 20 Aufgaben = 140; öffentliche Gesamtmatrix = 180/180 gescorte Einträge.

Methodenblöcke: Linie 1 enthält task-head baselines (Minimal, Neural MLP). Linie 2 trennt aligned baseline heads (metadata simple/NN, raw-feature simple/NN), die Qwen3-Omni series (Qwen3-Omni v6 LoRA) und die Cosmos3 series (Cosmos3-Super Reasoner, Cosmos3-Nano Future Window). Qwen3 v1-v6 ist eine LoRA-/Evaluationslinie innerhalb von Linie 2, nicht die evidence lines des Projekts; die 20-Task-Matrix nutzt v6 und v5 bleibt der pinned prior release. Cosmos3-Super Forward-Dynamics LoRA ist ein separat veröffentlichter Adapter/Gewichts-/Ergebnis-Artefakt und zählt nicht als Methodenreihe der 20-Task-Matrix.

Einstieg: [Leitfaden zu zwei Evidenzlinien](TWO_EVIDENCE_LINES.md), [Daten der zwei Evidenzlinien](docs/data/two_evidence_lines.json), [Tabelle mit 180 Ergebnissen](docs/data/task_method_20_result_matrix.json), [Ergebniszusammenfassung der zwei Linien](docs/data/two_evidence_line_result_summary.json).

## Schneller Einstieg

| Ziel | Einstieg |
| --- | --- |
| Projekt verstehen | [Projektbrief](PROJECT_BRIEF.md), [Projektstatus](PROJECT_STATUS.md) |
| GitHub-/Web-/HF-Einstieg wählen | [öffentliche Evidenzkarte](PUBLIC_READER_MAP.md) |
| 20 Aufgaben prüfen | [20-Aufgaben-Leitfaden](TASK_SUITE_20.md), [Aufgabenvertragsdaten](docs/data/task_suite_20.json) |
| Ergebnisse vergleichen | [Forschungsergebnisse](RESEARCH_TAKEAWAYS.md), [Tabelle mit 180 Ergebnissen](docs/data/task_method_20_result_matrix.json) |
| Ein Sample untersuchen | [Ein-Episode-Explorer](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [Sample-Dateikarte](docs/data/raw_sample_files.json) |
| Sample / 128 / Gesamtdatensatz vergleichen | [Data-Explorer-Analyse](DATA_EXPLORER_ANALYSIS.md), [Web-Analysebereich](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/#data-analysis) |
| Drei Foundation-Pipelines lesen | [drei Foundation-Pipelines](THREE_FOUNDATION_PIPELINES.md), [Pipeline-Vertragsdaten](docs/data/three_foundation_pipelines.json) |
| Reproduzieren oder auditieren | [Reproduzierbarkeitsleitfaden](REPRODUCIBILITY.md), [Evidenzvertrag](EVIDENCE_CONTRACT.md) |

## Struktur

- Leseregel: Hat es eine Metrik, gehört es zu den 20 Aufgaben; erklärt es, was die Evidenz untersucht, gehört es zu den 4 research directions; beschreibt es Trainings-Inputs und Targets, gehört es zu den 3 foundation pipelines; verbindet es Wahrnehmung, 3D-Gedächtnis, Sprache, Aktion und Planung, ist es das unified embodied model target, keine zusätzliche Score-Achse.
- Daten: 20-Frame-Fenster über Video, Audio, Tiefe, Pose/SLAM, Mocap, IMU, Kalibrierung und Sprachannotation.
- Datenexploration: Die neue Analyse trennt Public Sample, Selected-128-Exports und Full-Dataset-Metadaten des gated Upstreams.
- Aufgaben: 20 Verträge für Erkennung, Vorhersage, Retrieval, Rekonstruktion, Ordnung, Synchronisierung, Langhorizont-Prognose, Aktion-Objekt-Bindung und Sensor-Brücken.
- Ergebnisse: Single-Episode minimal/NN decken 20/20 ab; 128-Episode-Zweige trennen Metadata, Raw Features, Qwen3 und Cosmos; die öffentliche Matrix steht bei 180/180 gescorten Einträgen: 174 direct und 6 compact proxy, mit sichtbaren Proxy-Targets.
- Richtungen: spatial intelligence, human-video world model und vision-language-action sind mit Aufgaben und Evidenzanforderungen dokumentiert; das langfristige Ziel ist ein unified embodied foundation model.

## Öffentliche Grenze

Dieses Projekt veröffentlicht nur abgeleitete Artefakte, Metriken, Figuren, Karten und public-safe Zusammenfassungen. Xperience-10M bleibt unter den offiziellen Ropedia/Hugging-Face-Bedingungen.
""",
    },
    "ja": {
        "title": "Ropedia Xperience-10M Task Suite",
        "tagline": "Xperience-10M の公開タスク・評価レイヤー: サンプルデータ、20 個の embodied-AI タスク、ベースライン、Qwen3-Omni と Cosmos3 診断、基盤モデル訓練方向。",
        "body": f"""## プロジェクト概要

このリポジトリは、公開 Xperience-10M サンプル episode を、検証可能な embodied AI タスク実験面に変換します。まずダッシュボードとプロジェクト状態を見て、その後 20 タスク、結果行列、Hugging Face ミラーを確認してください。

**更新日:** {UPDATED}。

**範囲:** 完全に再現可能なタスク suite は 1 つの公開サンプル episode に基づきます。128-episode の結果は public-safe な指標、レポート、予測要約、モデルカードのみを公開します。元の MP4/HDF5/RRD、完全な Qwen 重み、gated データは再配布しません。

## 2 つの証拠ライン

| ライン | データ単位 | 手法と結果 | 用途 |
| --- | --- | --- | --- |
| 1 sample episode | 5,821 frames、1,161 aligned 20-frame windows、8,546 dimensions。 | Minimal + Neural MLP が 20 tasks を覆盖; 40/40 scored records; すべて direct scores。 | Raw sample files、task definitions、reproducible baselines、task validity を確認。 |
| 128 selected episodes | 96/16/16 split、34,269 exported windows、public-safe features が official gated episode paths に対応。 | Metadata simple/NN、raw-feature simple/NN、Qwen3-Omni v6、Cosmos3-Super、Cosmos3-Nano; 140/140 scored records; 134 direct + 6 compact proxy。 | 同一 split の metadata/raw baselines、Qwen3-Omni diagnostics、Cosmos3 diagnostics を比較; proxy targets は明示。 |

式: 1-episode methods 2 個 x 20 tasks = 40、128-episode methods 7 個 x 20 tasks = 140、公開 matrix 合計は 180/180 scored records。

Method blocks: Line 1 は task-head baselines（Minimal、Neural MLP）。Line 2 は aligned baseline heads（metadata simple/NN、raw-feature simple/NN）、Qwen3-Omni series（Qwen3-Omni v6 LoRA）、Cosmos3 series（Cosmos3-Super Reasoner、Cosmos3-Nano Future Window）に分かれます。Qwen3 v1-v6 は Line 2 内の LoRA/eval lineage で、project evidence lines とは別です。20-task matrix は v6 を使い、v5 は pinned prior release です。Cosmos3-Super Forward-Dynamics LoRA は別の adapter/weights/results artifact として公開され、20-task matrix の method row には含めません。

入口: [2 つの証拠ラインのガイド](TWO_EVIDENCE_LINES.md)、[2 ラインのデータ](docs/data/two_evidence_lines.json)、[180 件の結果表](docs/data/task_method_20_result_matrix.json)、[2 ライン結果サマリ](docs/data/two_evidence_line_result_summary.json)。

## クイックルート

| 目的 | 入口 |
| --- | --- |
| プロジェクトを素早く理解 | [プロジェクト概要](PROJECT_BRIEF.md), [プロジェクト状態](PROJECT_STATUS.md) |
| GitHub / Web / HF の入口を選ぶ | [公開エビデンスマップ](PUBLIC_READER_MAP.md) |
| 20 タスクを見る | [20 タスクガイド](TASK_SUITE_20.md), [タスク契約データ](docs/data/task_suite_20.json) |
| 結果を比較 | [研究結果まとめ](RESEARCH_TAKEAWAYS.md), [180 件の結果表](docs/data/task_method_20_result_matrix.json) |
| 1 サンプルを調べる | [1 episode ブラウザ](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [sample ファイルマップ](docs/data/raw_sample_files.json) |
| sample / 128 / HF full dataset を比較 | [data explorer analysis](DATA_EXPLORER_ANALYSIS.md), [web analysis section](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/#data-analysis) |
| 3 つの foundation pipeline を読む | [3 つの foundation pipeline](THREE_FOUNDATION_PIPELINES.md), [pipeline 契約データ](docs/data/three_foundation_pipelines.json) |
| 再現・監査 | [再現ガイド](REPRODUCIBILITY.md), [証拠契約](EVIDENCE_CONTRACT.md) |

## 構造

- 読み方のルール: metric があるものは 20 tasks、evidence が何を調べるかを説明するものは 4 research directions、training input/output を定義するものは 3 foundation pipelines です。perception、3D memory、language reasoning、action、planning を統合するものは unified embodied model target であり、新しい score axis ではありません。
- データ: 20-frame window が video、audio、depth、pose/SLAM、mocap、IMU、calibration、language annotation を結びます。
- データ探索: 新しい analysis layer は public sample、selected-128 exports、Hugging Face gated full dataset metadata を分けて提示します。
- タスク: 認識、予測、retrieval、reconstruction、order、sync、long-horizon、action-object、sensor bridge など 20 契約。
- 結果: single-episode minimal/NN は 20/20。128-episode 側は metadata、raw feature、Qwen3、Cosmos を証拠タイプ別に分けます。公開 matrix は 180/180 scored records で、174 direct と 6 compact proxy を分離し、proxy targets は明示します。
- 方向: spatial intelligence、human-video world model、vision-language-action に対して、タスク対応と必要証拠を記録しています。長期目標は unified embodied foundation model です。

## 公開境界

本プロジェクトは派生 artifacts、指標、図、カード、public-safe 要約のみを公開します。Xperience-10M の利用は Ropedia 公式 Hugging Face データカードとアクセス条件に従います。
""",
    },
    "ko": {
        "title": "Ropedia Xperience-10M Task Suite",
        "tagline": "Xperience-10M을 위한 공개 과제 및 평가 레이어: 샘플 데이터, 20개 embodied-AI 과제, 베이스라인, Qwen3-Omni 및 Cosmos3 진단, foundation 모델 학습 방향.",
        "body": f"""## 프로젝트 개요

이 저장소는 공개 Xperience-10M sample episode를 검증 가능한 embodied AI 과제 실험 표면으로 정리합니다. 먼저 대시보드와 프로젝트 상태를 보고, 이후 20개 과제, 결과 행렬, Hugging Face 미러를 확인하세요.

**업데이트:** {UPDATED}.

**범위:** 완전히 재현 가능한 task suite는 공개 sample episode 하나를 사용합니다. 128-episode 결과는 public-safe 지표, 리포트, 예측 요약, 모델 카드만 공개합니다. 원본 MP4/HDF5/RRD, 전체 Qwen 가중치, gated 데이터는 재배포하지 않습니다.

## 두 증거 라인

| 라인 | 데이터 단위 | 방법과 결과 | 용도 |
| --- | --- | --- | --- |
| 1 sample episode | 5,821 frames, 1,161 aligned 20-frame windows, 8,546 dimensions. | Minimal + Neural MLP가 20 tasks 전체를 평가; 40/40 scored records; 모두 direct scores. | Raw sample files, task definitions, reproducible baselines, task validity 확인. |
| 128 selected episodes | 96/16/16 split, 34,269 exported windows, public-safe features가 official gated episode paths에 연결됨. | Metadata simple/NN, raw-feature simple/NN, Qwen3-Omni v6, Cosmos3-Super, Cosmos3-Nano; 140/140 scored records; 134 direct + 6 compact proxy. | 같은 split에서 metadata/raw baselines, Qwen3-Omni diagnostics, Cosmos3 diagnostics 비교; proxy targets는 명시 유지. |

공식: single-episode 방법 2개 x 20 tasks = 40; 128-episode 방법 7개 x 20 tasks = 140; 전체 공개 matrix = 180/180 scored records.

방법 블록: Line 1은 task-head baselines(Minimal, Neural MLP)입니다. Line 2는 aligned baseline heads(metadata simple/NN, raw-feature simple/NN), Qwen3-Omni series(Qwen3-Omni v6 LoRA), Cosmos3 series(Cosmos3-Super Reasoner, Cosmos3-Nano Future Window)로 분리됩니다. Qwen3 v1-v6은 Line 2 내부의 LoRA/eval lineage이며 project evidence lines와 다릅니다. 20-task matrix는 v6을 사용하고 v5는 pinned prior release입니다. Cosmos3-Super Forward-Dynamics LoRA는 별도의 adapter/weights/results artifact로 공개되며 20-task matrix method row에는 포함되지 않습니다.

입구: [두 증거 라인 가이드](TWO_EVIDENCE_LINES.md), [두 라인 데이터](docs/data/two_evidence_lines.json), [180개 결과 표](docs/data/task_method_20_result_matrix.json), [두 라인 결과 요약](docs/data/two_evidence_line_result_summary.json).

## 빠른 경로

| 목표 | 시작점 |
| --- | --- |
| 프로젝트 빠르게 이해 | [프로젝트 요약](PROJECT_BRIEF.md), [프로젝트 상태](PROJECT_STATUS.md) |
| GitHub / 웹 / HF 입구 선택 | [공개 증거 맵](PUBLIC_READER_MAP.md) |
| 20개 과제 확인 | [20개 과제 가이드](TASK_SUITE_20.md), [과제 계약 데이터](docs/data/task_suite_20.json) |
| 결과 비교 | [연구 결과 요약](RESEARCH_TAKEAWAYS.md), [180개 결과 표](docs/data/task_method_20_result_matrix.json) |
| 샘플 하나 검사 | [단일 episode 브라우저](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [sample 파일 지도](docs/data/raw_sample_files.json) |
| sample / 128 / HF full dataset 비교 | [data explorer analysis](DATA_EXPLORER_ANALYSIS.md), [web analysis section](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/#data-analysis) |
| 세 foundation pipeline 읽기 | [세 foundation pipeline](THREE_FOUNDATION_PIPELINES.md), [pipeline 계약 데이터](docs/data/three_foundation_pipelines.json) |
| 재현 및 감사 | [재현 가이드](REPRODUCIBILITY.md), [증거 계약](EVIDENCE_CONTRACT.md) |

## 구조

- 읽기 규칙: metric이 있으면 20개 task layer이고, evidence가 무엇을 연구하는지 설명하면 4개 research direction layer이며, model input/output과 training target을 설명하면 3개 foundation pipeline layer입니다. perception, 3D memory, language reasoning, action, planning을 합치는 것은 unified embodied model target이며 새 score axis가 아닙니다.
- 데이터: 20-frame window가 video, audio, depth, pose/SLAM, mocap, IMU, calibration, language annotation을 연결합니다.
- 데이터 탐색: 새 분석 레이어는 public sample, selected-128 exports, Hugging Face gated full dataset metadata를 분리해 보여줍니다.
- 과제: 인식, 예측, retrieval, reconstruction, order, sync, long-horizon, action-object binding, sensor bridge 등 20개 계약.
- 결과: single-episode minimal/NN은 20/20; 128-episode 레이어는 metadata, raw feature, Qwen3, Cosmos를 증거 유형별로 분리합니다. 공개 matrix는 180/180 scored records이며 174 direct와 6 compact proxy를 분리하고 proxy targets를 명시합니다.
- 방향: spatial intelligence, human-video world model, vision-language-action에 대해 과제 매핑과 필요한 증거를 기록합니다. 장기 목표는 unified embodied foundation model입니다.

## 공개 경계

이 프로젝트는 파생 artifacts, 지표, 그림, 카드, public-safe 요약만 공개합니다. Xperience-10M 사용은 Ropedia 공식 Hugging Face 데이터 카드와 접근 조건을 따릅니다.
""",
    },
    "pt": {
        "title": "Ropedia Xperience-10M Task Suite",
        "tagline": "Camada pública de tarefas e avaliação para Xperience-10M: dados de amostra, 20 tarefas embodied-AI, baselines, diagnósticos Qwen3-Omni e Cosmos3 e direções de treino.",
        "body": f"""## Visão Geral Do Projeto

Este repositório transforma o episódio público de amostra do Xperience-10M em um laboratório verificável de tarefas para embodied AI. Comece pelo painel visual e pelo status do projeto; depois abra os contratos de tarefas, matrizes de resultados e espelhos no Hugging Face.

**Atualizado:** {UPDATED}.

**Escopo:** a suíte totalmente reproduzível usa um episódio público; os resultados de 128 episódios publicam apenas métricas, relatórios, predições seguras e model cards. MP4/HDF5/RRD originais, pesos completos do Qwen e dados gated não são redistribuídos.

## Duas Linhas de Evidência

| Linha | Unidade de dados | Métodos e resultados | Uso |
| --- | --- | --- | --- |
| 1 episódio de amostra | 5,821 frames; 1,161 janelas alinhadas de 20 frames; 8,546 dimensões. | Minimal + Neural MLP em 20 tarefas; 40/40 registros com score; todos são direct scores. | Inspecionar arquivos da amostra, definições de tarefas, baselines reproduzíveis e validade das tarefas. |
| 128 episódios selecionados | Split 96/16/16; 34,269 janelas exportadas; features public-safe ligadas aos caminhos oficiais gated. | Metadata simple/NN, raw-feature simple/NN, Qwen3-Omni, Cosmos3-Super e Cosmos3-Nano; 140/140 registros com score; 134 direct + 6 compact proxy. | Comparar baselines e ramos de modelo no mesmo split; proxy targets permanecem visíveis. |

Fórmula: 2 métodos de um episódio x 20 tarefas = 40; 7 métodos de 128 episódios x 20 tarefas = 140; matriz pública total = 180/180 registros com score.

Blocos de métodos: a linha 1 contém task-head baselines (Minimal, Neural MLP). A linha 2 separa aligned baseline heads (metadata simple/NN, raw-feature simple/NN), a série Qwen3-Omni (Qwen3-Omni v6 LoRA) e a série Cosmos3 (Cosmos3-Super Reasoner, Cosmos3-Nano Future Window). Qwen3 v1-v6 é uma linhagem LoRA/eval interna à linha 2, não as evidence lines do projeto; a matriz de 20 tarefas usa v6 e v5 fica como pinned prior release. Cosmos3-Super Forward-Dynamics LoRA é publicado como adapter/pesos/resultados separado e não conta como linha de método na matriz de 20 tarefas.

Entradas: [guia de duas linhas de evidencia](TWO_EVIDENCE_LINES.md), [dados das duas linhas](docs/data/two_evidence_lines.json), [tabela de 180 resultados](docs/data/task_method_20_result_matrix.json), [resumo de resultados das duas linhas](docs/data/two_evidence_line_result_summary.json).

## Rota Rápida

| Objetivo | Entrada |
| --- | --- |
| Entender o projeto | [resumo do projeto](PROJECT_BRIEF.md), [status do projeto](PROJECT_STATUS.md) |
| Escolher entrada GitHub / web / HF | [mapa publico de evidencias](PUBLIC_READER_MAP.md) |
| Ver as 20 tarefas | [guia das 20 tarefas](TASK_SUITE_20.md), [dados dos contratos de tarefa](docs/data/task_suite_20.json) |
| Comparar resultados | [conclusões de pesquisa](RESEARCH_TAKEAWAYS.md), [tabela de 180 resultados](docs/data/task_method_20_result_matrix.json) |
| Inspecionar uma amostra | [explorador de um episódio](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [mapa dos arquivos de amostra](docs/data/raw_sample_files.json) |
| Comparar sample / 128 / dataset completo | [analise do data explorer](DATA_EXPLORER_ANALYSIS.md), [secao web de analise](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/#data-analysis) |
| Ler as três pipelines foundation | [três pipelines foundation](THREE_FOUNDATION_PIPELINES.md), [dados dos contratos de pipeline](docs/data/three_foundation_pipelines.json) |
| Reproduzir ou auditar | [guia de reprodutibilidade](REPRODUCIBILITY.md), [contrato de evidencia](EVIDENCE_CONTRACT.md) |

## Estrutura

- Regra de leitura: se tem uma métrica, pertence às 20 tarefas; se explica o que a evidência estuda, pertence às 4 research directions; se define inputs/outputs de treino, pertence às 3 foundation pipelines; se combina percepção, memória 3D, linguagem, ação e planejamento, pertence ao unified embodied model target, não a um novo eixo de score.
- Dados: janelas de 20 frames ligam vídeo, áudio, profundidade, pose/SLAM, mocap, IMU, calibração e anotações de linguagem.
- Exploracao de dados: a nova camada separa o sample publico, os exports selected-128 e a metadata do dataset gated completo.
- Tarefas: 20 contratos cobrem reconhecimento, previsão, retrieval, reconstrução, ordem, sincronização, horizonte longo, relação ação-objeto e pontes de sensores.
- Resultados: minimal/NN de um episódio cobrem 20/20; a camada de 128 episódios separa metadata, raw features, Qwen3 e Cosmos; a matriz pública está em 180/180 registros com score: 174 direct e 6 compact proxy, com proxy targets visíveis.
- Direções: spatial intelligence, human-video world model e vision-language-action têm mapeamento de tarefas e requisitos de evidência; o objetivo de longo prazo é um unified embodied foundation model.

## Fronteira Pública

O projeto publica apenas artifacts derivados, métricas, figuras, cards e resumos public-safe. O uso do Xperience-10M segue o dataset card oficial da Ropedia no Hugging Face.
""",
    },
}


COMMON_FOOTER = """## Published Mirrors

| Surface | Link |
| --- | --- |
| GitHub | https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite |
| Website | https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/ |
| HF Space | https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite |
| HF artifacts | https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts |
| HF baselines | https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines |
| HF weights/results | https://huggingface.co/cy0307/ropedia-xperience-10m-weights-results |
| HF collection | https://huggingface.co/collections/cy0307/ropedia-xperience-10m-task-suite |

## Glossary

Use `GLOSSARY.md` and `docs/data/glossary.json` for project terminology:
evidence line, 20-frame window, compact-proxy score, Qwen v1-v6,
Cosmos3-Super, LoRA adapter, HF artifact dataset, and related terms.

## Citation

Use `CITATION.cff` and cite the upstream Ropedia Xperience-10M dataset according to its official card.
"""


def write_language_json() -> None:
    rows = [
        {
            "code": code,
            "label": label,
            "readme": filename,
            "github_url": f"https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite/blob/main/{filename}",
        }
        for code, label, filename in LANGUAGES
    ]
    payload = {
        "title": "Ropedia Xperience-10M Task Suite Language Versions",
        "status": "pass",
        "updated": UPDATED,
        "language_count": len(rows),
        "languages": rows,
    }
    path = ROOT / "docs/data/language_versions.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def current_body_suffix() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    marker = "\n## Why This Project Exists\n"
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit("Could not find README marker: ## Why This Project Exists")
    return text[idx + 1 :].rstrip() + "\n"


def main() -> int:
    suffix = current_body_suffix()
    (ROOT / "README.md").write_text(ENGLISH_TOP.rstrip() + "\n\n" + suffix, encoding="utf-8")
    for code, payload in LANGUAGE_GUIDES.items():
        filename = next(item[2] for item in LANGUAGES if item[0] == code)
        content = (
            hero(payload["title"], payload["tagline"], code).rstrip()
            + "\n\n"
            + payload["body"].rstrip()
            + "\n\n"
            + COMMON_FOOTER
        )
        (ROOT / filename).write_text(content, encoding="utf-8")
    write_language_json()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
