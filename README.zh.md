<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-social-card.png" alt="Ropedia Xperience-10M Task Suite cover" width="100%">
</p>

<h1 align="center">Ropedia Xperience-10M 任务套件</h1>

<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-mark-192.png" alt="Ropedia Xperience-10M logo" width="112">
</p>

<p align="center">
  <strong>面向 Xperience-10M 的多语言公开研究入口：样本数据、20 个具身智能任务、基线、Qwen3/Cosmos 诊断结果，以及基础模型训练方向。</strong>
</p>

<!-- LANG-BAR:START -->
<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md"><b>中文</b></a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.pt.md">Português</a>
</p>
<!-- LANG-BAR:END -->

<p align="center">
  <a href="https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/"><img alt="GitHub Pages" src="https://img.shields.io/badge/site-GitHub%20Pages-1f63e9"></a>
  <a href="https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite"><img alt="HF Space" src="https://img.shields.io/badge/Hugging%20Face-Space-ffb000"></a>
  <a href="https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts"><img alt="artifact dataset" src="https://img.shields.io/badge/HF-artifacts-008b9a"></a>
  <a href="https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines"><img alt="baseline model repo" src="https://img.shields.io/badge/HF-baselines-7ae5c3"></a>
  <a href="https://huggingface.co/datasets/ropedia-ai/xperience-10m"><img alt="Xperience-10M" src="https://img.shields.io/badge/dataset-Xperience--10M-344054"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-code%20MIT%20%2B%20data%20terms-ccffa0"></a>
</p>

## 如何阅读这个项目

这个仓库把 Ropedia 公开的 Xperience-10M sample episode 变成一个可检查的具身智能任务实验室。请先看仪表盘和项目状态，再进入 20 个任务、结果矩阵和 Hugging Face 镜像。

**更新时间：** 2026-06-21。

**范围：** 完整可复现的任务套件来自一个公开样本 episode；128-episode 结果只发布 public-safe 的指标、报告、预测摘要和模型卡。原始 MP4/HDF5/RRD、完整 Qwen 权重和 gated 数据不在本仓库重新分发。

## 两条证据线

| 线 | 数据单元 | 方法与结果 | 用途 |
| --- | --- | --- | --- |
| 1 sample episode | 5,821 帧；1,161 个 20-frame 对齐窗口；8,546 维特征。 | Minimal + Neural MLP；20 个任务全覆盖；40/40 scored records。 | 检查原始 sample 文件、任务定义、可复现基线和每个任务是否成立。 |
| 128 selected episodes | 96/16/16 split；34,269 个导出窗口；public-safe 特征链接到官方 gated episode path。 | Metadata simple/NN、raw-feature simple/NN、Qwen3-Omni、Cosmos3-Super、Cosmos3-Nano；140/140 scored records。 | 比较同一 split 上的基线和模型分支；proxy target 会显式标注。 |

入口：[`TWO_EVIDENCE_LINES.md`](TWO_EVIDENCE_LINES.md)、[`two_evidence_lines.json`](docs/data/two_evidence_lines.json)、[`task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json)。

## 快速入口

| 目标 | 入口 |
| --- | --- |
| 快速理解项目 | [PROJECT_BRIEF.md](PROJECT_BRIEF.md), [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| 选择 GitHub / 网页 / HF 的正确入口 | [PUBLIC_READER_MAP.md](PUBLIC_READER_MAP.md) |
| 查看 20 个任务定义 | [TASK_SUITE_20.md](TASK_SUITE_20.md), [task_suite_20.json](docs/data/task_suite_20.json) |
| 比较结果 | [RESEARCH_TAKEAWAYS.md](RESEARCH_TAKEAWAYS.md), [task_method_20_result_matrix.json](docs/data/task_method_20_result_matrix.json) |
| 查看一个 sample 的全部文件关系 | [single_episode_explorer.html](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [raw_sample_files.json](docs/data/raw_sample_files.json) |
| 阅读三个基础模型方向 | [THREE_FOUNDATION_PIPELINES.md](THREE_FOUNDATION_PIPELINES.md), [three_foundation_pipelines.json](docs/data/three_foundation_pipelines.json) |
| 复现与审计 | [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [EVIDENCE_CONTRACT.md](EVIDENCE_CONTRACT.md) |

## 核心结构

- 数据层：公开 sample episode 被切成 20-frame 窗口，并连接视频、音频、深度、pose/SLAM、mocap、IMU、calibration 和语言标注。
- 任务层：20 个统一任务覆盖识别、预测、检索、重建、同步、长时预测、action-object 关系和 sensor bridge。
- 结果层：单 episode minimal/NN 覆盖 20/20；128-episode metadata/raw/Qwen3/Cosmos 分开标注；当前公开矩阵为 180/180 scored records，proxy target 显式保留。
- 训练方向：spatial intelligence、human-video world model、vision-language-action 三条 pipeline 已经有任务映射和需要的证据清单。

## 公开边界

本项目只发布小型 derived artifacts、指标、图表、README、模型卡和 public-safe 预测摘要。原始 Xperience-10M 数据使用仍以 Ropedia 官方 Hugging Face 数据卡和访问条款为准。

## Public Surfaces

| Surface | Link |
| --- | --- |
| GitHub | https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite |
| Website | https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/ |
| HF Space | https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite |
| HF artifacts | https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts |
| HF baselines | https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines |
| HF weights/results | https://huggingface.co/cy0307/ropedia-xperience-10m-weights-results |
| HF collection | https://huggingface.co/collections/cy0307/ropedia-xperience-10m-task-suite |

## Citation

Use `CITATION.cff` and cite the upstream Ropedia Xperience-10M dataset according to its official card.
