<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-social-card.png" alt="Ropedia Xperience-10M Task Suite cover" width="100%">
</p>

<h1 align="center">Ropedia Xperience-10M Task Suite</h1>

<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-mark-192.png" alt="Ropedia Xperience-10M logo" width="112">
</p>

<p align="center">
  <strong>Superfície pública multilíngue para Xperience-10M: dados de amostra, 20 tarefas embodied-AI, baselines, diagnósticos Qwen3/Cosmos e direções de treino.</strong>
</p>

<!-- LANG-BAR:START -->
<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.pt.md"><b>Português</b></a>
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

## Como Ler Este Projeto

Este repositório transforma o episódio público de amostra do Xperience-10M em um laboratório verificável de tarefas para embodied AI. Comece pelo painel visual e pelo status do projeto; depois abra os contratos de tarefas, matrizes de resultados e espelhos no Hugging Face.

**Atualizado:** 2026-06-21.

**Escopo:** a suíte totalmente reproduzível usa um episódio público; os resultados de 128 episódios publicam apenas métricas, relatórios, predições seguras e model cards. MP4/HDF5/RRD originais, pesos completos do Qwen e dados gated não são redistribuídos.

## Duas Linhas de Evidência

| Linha | Unidade de dados | Métodos e resultados | Uso |
| --- | --- | --- | --- |
| 1 episódio de amostra | 5,821 frames; 1,161 janelas alinhadas de 20 frames; 8,546 dimensões. | Minimal + Neural MLP em 20 tarefas; 40/40 registros com score; todos são direct scores. | Inspecionar arquivos da amostra, definições de tarefas, baselines reproduzíveis e validade das tarefas. |
| 128 episódios selecionados | Split 96/16/16; 34,269 janelas exportadas; features public-safe ligadas aos caminhos oficiais gated. | Metadata simple/NN, raw-feature simple/NN, Qwen3-Omni, Cosmos3-Super e Cosmos3-Nano; 140/140 registros com score; 134 direct + 6 compact proxy. | Comparar baselines e ramos de modelo no mesmo split; proxy targets permanecem visíveis. |

Fórmula: 2 métodos de um episódio x 20 tarefas = 40; 7 métodos de 128 episódios x 20 tarefas = 140; matriz pública total = 180/180 registros com score.

Entradas: [`TWO_EVIDENCE_LINES.md`](TWO_EVIDENCE_LINES.md), [`two_evidence_lines.json`](docs/data/two_evidence_lines.json), [`task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json), [`two_evidence_line_result_summary.json`](docs/data/two_evidence_line_result_summary.json).

## Rota Rápida

| Objetivo | Entrada |
| --- | --- |
| Entender o projeto | [PROJECT_BRIEF.md](PROJECT_BRIEF.md), [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| Escolher a superfície pública correta | [PUBLIC_READER_MAP.md](PUBLIC_READER_MAP.md) |
| Ver as 20 tarefas | [TASK_SUITE_20.md](TASK_SUITE_20.md), [task_suite_20.json](docs/data/task_suite_20.json) |
| Comparar resultados | [RESEARCH_TAKEAWAYS.md](RESEARCH_TAKEAWAYS.md), [task_method_20_result_matrix.json](docs/data/task_method_20_result_matrix.json) |
| Inspecionar uma amostra | [single_episode_explorer.html](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [raw_sample_files.json](docs/data/raw_sample_files.json) |
| Ler as três pipelines foundation | [THREE_FOUNDATION_PIPELINES.md](THREE_FOUNDATION_PIPELINES.md), [three_foundation_pipelines.json](docs/data/three_foundation_pipelines.json) |
| Reproduzir ou auditar | [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [EVIDENCE_CONTRACT.md](EVIDENCE_CONTRACT.md) |

## Estrutura

- Dados: janelas de 20 frames ligam vídeo, áudio, profundidade, pose/SLAM, mocap, IMU, calibração e anotações de linguagem.
- Tarefas: 20 contratos cobrem reconhecimento, previsão, retrieval, reconstrução, ordem, sincronização, horizonte longo, relação ação-objeto e pontes de sensores.
- Resultados: minimal/NN de um episódio cobrem 20/20; a camada de 128 episódios separa metadata, raw features, Qwen3 e Cosmos; a matriz pública está em 180/180 registros com score: 174 direct e 6 compact proxy, com proxy targets visíveis.
- Direções: spatial intelligence, human-video world model e vision-language-action têm mapeamento de tarefas e requisitos de evidência.

## Fronteira Pública

O projeto publica apenas artifacts derivados, métricas, figuras, cards e resumos public-safe. O uso do Xperience-10M segue o dataset card oficial da Ropedia no Hugging Face.

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
