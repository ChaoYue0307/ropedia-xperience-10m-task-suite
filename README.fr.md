<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-social-card.png" alt="Ropedia Xperience-10M Task Suite cover" width="100%">
</p>

<h1 align="center">Ropedia Xperience-10M Task Suite</h1>

<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-mark-192.png" alt="Ropedia Xperience-10M logo" width="112">
</p>

<p align="center">
  <strong>Surface publique multilingue pour Xperience-10M : échantillon, 20 tâches embodied-AI, baselines, diagnostics Qwen3/Cosmos et pistes d'entraînement.</strong>
</p>

<!-- LANG-BAR:START -->
<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md"><b>Français</b></a> ·
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

## Comment Lire Ce Projet

Ce dépôt transforme l'épisode public d'exemple Xperience-10M en laboratoire de tâches vérifiable pour l'IA incarnée. Commencez par le tableau de bord et le statut du projet, puis ouvrez les contrats de tâches, les matrices de résultats et les miroirs Hugging Face.

**Mise à jour :** 2026-06-21.

**Portée :** la suite entièrement reproductible utilise un épisode public; les résultats 128 épisodes ne publient que des métriques, rapports, prédictions sûres et cartes de modèles. Les MP4/HDF5/RRD bruts, les poids Qwen complets et les données gated ne sont pas redistribués.

## Deux Lignes de Preuve

| Ligne | Unité de données | Méthodes et résultats | Usage |
| --- | --- | --- | --- |
| 1 épisode d'exemple | 5,821 frames; 1,161 fenêtres alignées de 20 frames; 8,546 dimensions. | Minimal + Neural MLP sur 20 tâches; 40/40 enregistrements scorés. | Inspecter les fichiers sample, les définitions de tâches, les baselines reproductibles et la validité des tâches. |
| 128 épisodes sélectionnés | Split 96/16/16; 34,269 fenêtres exportées; features public-safe liées aux chemins gated officiels. | Metadata simple/NN, raw-feature simple/NN, Qwen3-Omni, Cosmos3-Super et Cosmos3-Nano; 140/140 enregistrements scorés. | Comparer les baselines et branches de modèles sur le même split; les proxy targets restent visibles. |

Entrées : [`TWO_EVIDENCE_LINES.md`](TWO_EVIDENCE_LINES.md), [`two_evidence_lines.json`](docs/data/two_evidence_lines.json), [`task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json).

## Parcours Rapide

| Objectif | Point d'entrée |
| --- | --- |
| Comprendre le projet | [PROJECT_BRIEF.md](PROJECT_BRIEF.md), [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| Choisir la bonne surface publique | [PUBLIC_READER_MAP.md](PUBLIC_READER_MAP.md) |
| Lire les 20 tâches | [TASK_SUITE_20.md](TASK_SUITE_20.md), [task_suite_20.json](docs/data/task_suite_20.json) |
| Comparer les résultats | [RESEARCH_TAKEAWAYS.md](RESEARCH_TAKEAWAYS.md), [task_method_20_result_matrix.json](docs/data/task_method_20_result_matrix.json) |
| Inspecter un sample | [single_episode_explorer.html](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [raw_sample_files.json](docs/data/raw_sample_files.json) |
| Lire les trois pipelines foundation | [THREE_FOUNDATION_PIPELINES.md](THREE_FOUNDATION_PIPELINES.md), [three_foundation_pipelines.json](docs/data/three_foundation_pipelines.json) |
| Reproduire et auditer | [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [EVIDENCE_CONTRACT.md](EVIDENCE_CONTRACT.md) |

## Structure

- Données : fenêtres de 20 frames reliant vidéo, audio, profondeur, pose/SLAM, mocap, IMU, calibration et annotations de langage.
- Tâches : 20 contrats couvrant reconnaissance, prévision, retrieval, reconstruction, ordre, synchronisation, horizon long, relations action-objet et sensor bridge.
- Résultats : minimal/NN sur l'épisode public couvrent 20/20; les branches 128 épisodes séparent metadata, raw features, Qwen3 et Cosmos; la matrice publique atteint 180/180 enregistrements scorés avec proxy targets visibles.
- Directions : spatial intelligence, human-video world model et vision-language-action sont documentés avec tâches et preuves nécessaires.

## Frontière Publique

Le projet publie des artifacts dérivés, métriques, figures et cartes public-safe. L'accès aux données Xperience-10M reste régi par la carte officielle Ropedia sur Hugging Face.

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
