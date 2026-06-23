<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-social-card.png" alt="Ropedia Xperience-10M Task Suite cover" width="100%">
</p>

<h1 align="center">Ropedia Xperience-10M Task Suite</h1>

<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-mark-192.png" alt="Ropedia Xperience-10M Task Suite logo" width="112">
</p>

<p align="center">
  <strong>Couche publique de tâches et d'évaluation pour Xperience-10M : échantillon, 20 tâches embodied-AI, baselines, diagnostics Qwen3-Omni et Cosmos3, et pistes d'entraînement.</strong>
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

## Présentation Du Projet

Ce dépôt transforme l'épisode public d'exemple Xperience-10M en laboratoire de tâches vérifiable pour l'IA incarnée. Commencez par le tableau de bord et le statut du projet, puis ouvrez les contrats de tâches, les matrices de résultats et les miroirs Hugging Face.

**Mise à jour :** 2026-06-23.

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

Use `CITATION.cff` and cite the upstream Ropedia Xperience-10M dataset according to its official card.
