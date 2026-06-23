<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-social-card.png" alt="Ropedia Xperience-10M Task Suite cover" width="100%">
</p>

<h1 align="center">Ropedia Xperience-10M Task Suite</h1>

<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-mark-192.png" alt="Ropedia Xperience-10M Task Suite logo" width="112">
</p>

<p align="center">
  <strong>Camada pública de tarefas e avaliação para Xperience-10M: dados de amostra, 20 tarefas embodied-AI, baselines, diagnósticos Qwen3-Omni e Cosmos3 e direções de treino.</strong>
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

## Visão Geral Do Projeto

Este repositório transforma o episódio público de amostra do Xperience-10M em um laboratório verificável de tarefas para embodied AI. Comece pelo painel visual e pelo status do projeto; depois abra os contratos de tarefas, matrizes de resultados e espelhos no Hugging Face.

**Atualizado:** 2026-06-23.

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

## Published Mirrors

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

If you use this project, cite the Ropedia Xperience-10M Task Suite by Chaoyue He and also cite the upstream Ropedia Xperience-10M dataset according to its official card.

```bibtex
@misc{he2026ropedia_xperience10m_task_suite,
  author       = {Chaoyue He},
  title        = {Ropedia Xperience-10M Task Suite},
  year         = {2026},
  url          = {https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite},
  note         = {Public task and evaluation layer for Ropedia Xperience-10M: two evidence lines, 20 embodied-AI tasks, 180 scored method-task records, and selected-128 model diagnostics}
}
```
