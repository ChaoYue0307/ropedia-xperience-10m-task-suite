<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-social-card.png" alt="Ropedia Xperience-10M Task Suite cover" width="100%">
</p>

<h1 align="center">Ropedia Xperience-10M Task Suite</h1>

<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-mark-192.png" alt="Ropedia Xperience-10M logo" width="112">
</p>

<p align="center">
  <strong>Xperience-10M을 위한 다국어 공개 연구 표면: 샘플 데이터, 20개 embodied-AI 과제, 베이스라인, Qwen3-Omni 및 Cosmos3 진단, foundation 모델 학습 방향.</strong>
</p>

<!-- LANG-BAR:START -->
<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md"><b>한국어</b></a> ·
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

## 이 프로젝트를 읽는 방법

이 저장소는 공개 Xperience-10M sample episode를 검증 가능한 embodied AI 과제 실험 표면으로 정리합니다. 먼저 대시보드와 프로젝트 상태를 보고, 이후 20개 과제, 결과 행렬, Hugging Face 미러를 확인하세요.

**업데이트:** 2026-06-23.

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
| 공개 표면 선택 | [공개 리더 맵](PUBLIC_READER_MAP.md) |
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
