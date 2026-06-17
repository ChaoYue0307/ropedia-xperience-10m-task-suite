<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-social-card.png" alt="Ropedia Xperience-10M Task Suite cover" width="100%">
</p>

<h1 align="center">Ropedia Xperience-10M Task Suite</h1>

<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-mark-192.png" alt="Ropedia Xperience-10M logo" width="112">
</p>

<p align="center">
  <strong>Superficie pública multilingüe para Xperience-10M: datos de muestra, 20 tareas embodied-AI, baselines, diagnósticos Qwen3/Cosmos y direcciones de entrenamiento.</strong>
</p>

<!-- LANG-BAR:START -->
<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.es.md"><b>Español</b></a> ·
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

## Cómo Leer Este Proyecto

Este repositorio convierte el episodio público de muestra de Xperience-10M en un laboratorio verificable de tareas para embodied AI. Empieza por el panel visual y el estado del proyecto; después entra en las tareas, matrices de resultados y espejos de Hugging Face.

**Actualizado:** 2026-06-18.

**Alcance:** la suite reproducible usa un episodio público; los resultados de 128 episodios publican solo métricas, reportes, predicciones seguras y tarjetas de modelo. No se redistribuyen MP4/HDF5/RRD originales, pesos completos de Qwen ni datos gated.

## Ruta Rápida

| Objetivo | Entrada |
| --- | --- |
| Entender el proyecto | [PROJECT_BRIEF.md](PROJECT_BRIEF.md), [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| Elegir la superficie correcta | [PUBLIC_READER_MAP.md](PUBLIC_READER_MAP.md) |
| Ver las 20 tareas | [TASK_SUITE_20.md](TASK_SUITE_20.md), [task_suite_20.json](docs/data/task_suite_20.json) |
| Comparar resultados | [RESEARCH_TAKEAWAYS.md](RESEARCH_TAKEAWAYS.md), [task_method_20_result_matrix.json](docs/data/task_method_20_result_matrix.json) |
| Inspeccionar una muestra | [single_episode_explorer.html](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [raw_sample_files.json](docs/data/raw_sample_files.json) |
| Leer las tres direcciones foundation | [THREE_FOUNDATION_PIPELINES.md](THREE_FOUNDATION_PIPELINES.md), [three_foundation_pipelines.json](docs/data/three_foundation_pipelines.json) |
| Reproducir o auditar | [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [EVIDENCE_CONTRACT.md](EVIDENCE_CONTRACT.md) |

## Estructura

- Datos: ventanas de 20 frames con video, audio, profundidad, pose/SLAM, mocap, IMU, calibración y lenguaje.
- Tareas: 20 contratos para reconocimiento, predicción, recuperación, reconstrucción, sincronización, horizonte largo, relación acción-objeto y puentes de sensores.
- Resultados: minimal/NN de un episodio cubren 20/20; las ramas de 128 episodios separan metadata, raw features, Qwen3 y Cosmos con gaps explícitos.
- Direcciones: spatial intelligence, human-video world model y vision-language-action tienen mapeo de tareas y requisitos de evidencia.

## Límite Público

El proyecto publica solo artifacts derivados, métricas, figuras, tarjetas y resúmenes public-safe. El uso de Xperience-10M sigue las condiciones oficiales de Ropedia en Hugging Face.

## Public Surfaces

| Surface | Link |
| --- | --- |
| GitHub | https://github.com/ChaoYue0307/ropedia-xperience-10m-task-suite |
| Website | https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/ |
| HF Space | https://huggingface.co/spaces/cy0307/ropedia-xperience-10m-task-suite |
| HF artifacts | https://huggingface.co/datasets/cy0307/ropedia-xperience-10m-task-suite-artifacts |
| HF baselines | https://huggingface.co/cy0307/ropedia-xperience-10m-task-baselines |
| HF collection | https://huggingface.co/collections/cy0307/ropedia-xperience-10m-task-suite |

## Citation

Use `CITATION.cff` and cite the upstream Ropedia Xperience-10M dataset according to its official card.
