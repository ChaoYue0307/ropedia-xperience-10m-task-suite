<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-social-card.png" alt="Ropedia Xperience-10M Task Suite cover" width="100%">
</p>

<h1 align="center">Ropedia Xperience-10M Task Suite</h1>

<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-mark-192.png" alt="Ropedia Xperience-10M logo" width="112">
</p>

<p align="center">
  <strong>Xperience-10M の多言語公開研究面: サンプルデータ、20 個の embodied-AI タスク、ベースライン、Qwen3-Omni と Cosmos3 診断、基盤モデル訓練方向。</strong>
</p>

<!-- LANG-BAR:START -->
<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.ja.md"><b>日本語</b></a> ·
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

## このプロジェクトの読み方

このリポジトリは、公開 Xperience-10M サンプル episode を、検証可能な embodied AI タスク実験面に変換します。まずダッシュボードとプロジェクト状態を見て、その後 20 タスク、結果行列、Hugging Face ミラーを確認してください。

**更新日:** 2026-06-21。

**範囲:** 完全に再現可能なタスク suite は 1 つの公開サンプル episode に基づきます。128-episode の結果は public-safe な指標、レポート、予測要約、モデルカードのみを公開します。元の MP4/HDF5/RRD、完全な Qwen 重み、gated データは再配布しません。

## 2 つの証拠ライン

| ライン | データ単位 | 手法と結果 | 用途 |
| --- | --- | --- | --- |
| 1 sample episode | 5,821 frames、1,161 aligned 20-frame windows、8,546 dimensions。 | Minimal + Neural MLP が 20 tasks を覆盖; 40/40 scored records; すべて direct scores。 | Raw sample files、task definitions、reproducible baselines、task validity を確認。 |
| 128 selected episodes | 96/16/16 split、34,269 exported windows、public-safe features が official gated episode paths に対応。 | Metadata simple/NN、raw-feature simple/NN、Qwen3-Omni v6、Cosmos3-Super、Cosmos3-Nano; 140/140 scored records; 134 direct + 6 compact proxy。 | 同一 split の metadata/raw baselines、Qwen3-Omni diagnostics、Cosmos3 diagnostics を比較; proxy targets は明示。 |

式: 1-episode methods 2 個 x 20 tasks = 40、128-episode methods 7 個 x 20 tasks = 140、公開 matrix 合計は 180/180 scored records。

Method blocks: Line 1 は task-head baselines（Minimal、Neural MLP）。Line 2 は aligned baseline heads（metadata simple/NN、raw-feature simple/NN）、Qwen3-Omni series（Qwen3-Omni v6 LoRA）、Cosmos3 series（Cosmos3-Super Reasoner、Cosmos3-Nano Future Window）に分かれます。Qwen3 v1-v6 は Line 2 内の LoRA/eval lineage で、project evidence lines とは別です。20-task matrix は v6 を使い、v5 は pinned prior release です。Cosmos3-Super Forward-Dynamics LoRA は別の adapter/weights/results artifact として公開され、20-task matrix の method row には含めません。

入口: [`TWO_EVIDENCE_LINES.md`](TWO_EVIDENCE_LINES.md)、[`two_evidence_lines.json`](docs/data/two_evidence_lines.json)、[`task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json)、[`two_evidence_line_result_summary.json`](docs/data/two_evidence_line_result_summary.json)。

## クイックルート

| 目的 | 入口 |
| --- | --- |
| プロジェクトを素早く理解 | [PROJECT_BRIEF.md](PROJECT_BRIEF.md), [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| 公開面を選ぶ | [PUBLIC_READER_MAP.md](PUBLIC_READER_MAP.md) |
| 20 タスクを見る | [TASK_SUITE_20.md](TASK_SUITE_20.md), [task_suite_20.json](docs/data/task_suite_20.json) |
| 結果を比較 | [RESEARCH_TAKEAWAYS.md](RESEARCH_TAKEAWAYS.md), [task_method_20_result_matrix.json](docs/data/task_method_20_result_matrix.json) |
| 1 サンプルを調べる | [single_episode_explorer.html](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [raw_sample_files.json](docs/data/raw_sample_files.json) |
| 3 つの foundation pipeline を読む | [THREE_FOUNDATION_PIPELINES.md](THREE_FOUNDATION_PIPELINES.md), [three_foundation_pipelines.json](docs/data/three_foundation_pipelines.json) |
| 再現・監査 | [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [EVIDENCE_CONTRACT.md](EVIDENCE_CONTRACT.md) |

## 構造

- データ: 20-frame window が video、audio、depth、pose/SLAM、mocap、IMU、calibration、language annotation を結びます。
- タスク: 認識、予測、retrieval、reconstruction、order、sync、long-horizon、action-object、sensor bridge など 20 契約。
- 構造: 20 tasks / 4 research directions / 3 foundation pipelines。20 tasks は scoring axes、4 directions は同じタスクを読むための研究グループ、3 pipelines は大規模モデルの training tracks であり、新しい task tier ではありません。
- 結果: single-episode minimal/NN は 20/20。128-episode 側は metadata、raw feature、Qwen3、Cosmos を証拠タイプ別に分けます。公開 matrix は 180/180 scored records で、174 direct と 6 compact proxy を分離し、proxy targets は明示します。
- パイプライン: spatial intelligence models、human-video world models、vision-language-action models に対して、タスク対応と必要証拠を記録しています。

## 公開境界

本プロジェクトは派生 artifacts、指標、図、カード、public-safe 要約のみを公開します。Xperience-10M の利用は Ropedia 公式 Hugging Face データカードとアクセス条件に従います。

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

## Glossary

Use `GLOSSARY.md` and `docs/data/glossary.json` for project terminology:
evidence line, 20-frame window, compact-proxy score, Qwen v1-v6,
Cosmos3-Super, LoRA adapter, HF artifact dataset, and related terms.

## Citation

Use `CITATION.cff` and cite the upstream Ropedia Xperience-10M dataset according to its official card.
