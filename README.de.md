<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-social-card.png" alt="Ropedia Xperience-10M Task Suite cover" width="100%">
</p>

<h1 align="center">Ropedia Xperience-10M Task Suite</h1>

<p align="center">
  <img src="docs/assets/brand/xperience10m-logo-mark-192.png" alt="Ropedia Xperience-10M logo" width="112">
</p>

<p align="center">
  <strong>Mehrsprachige öffentliche Forschungsoberfläche für Xperience-10M: Sample-Daten, 20 Embodied-AI-Aufgaben, Baselines, Qwen3-Omni- und Cosmos3-Diagnostik und Trainingsrichtungen.</strong>
</p>

<!-- LANG-BAR:START -->
<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md"><b>Deutsch</b></a> ·
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

## So Liest Man Dieses Projekt

Dieses Repository macht aus dem öffentlichen Xperience-10M-Sample eine prüfbare Aufgabenoberfläche für Embodied AI. Beginnen Sie mit Dashboard und Projektstatus, danach mit Aufgabenverträgen, Ergebnismatrizen und Hugging-Face-Spiegeln.

**Aktualisiert:** 2026-06-21.

**Umfang:** die vollständig reproduzierbare Suite nutzt ein öffentliches Sample-Episode; 128-Episode-Ergebnisse veröffentlichen nur public-safe Metriken, Berichte, Vorhersagen und Modellkarten. Rohdaten wie MP4/HDF5/RRD, vollständige Qwen-Gewichte und gated Daten werden nicht weitergegeben.

## Zwei Evidenzlinien

| Linie | Dateneinheit | Methoden und Ergebnisse | Zweck |
| --- | --- | --- | --- |
| 1 Sample-Episode | 5,821 Frames; 1,161 ausgerichtete 20-Frame-Fenster; 8,546 Dimensionen. | Minimal + Neural MLP auf 20 Aufgaben; 40/40 gescorte Einträge; alle sind direct scores. | Sample-Dateien, Aufgaben, reproduzierbare Baselines und Aufgabenqualität prüfen. |
| 128 ausgewählte Episoden | 96/16/16 Split; 34,269 exportierte Fenster; public-safe Features mit offiziellen gated Episode-Pfaden. | Metadata simple/NN, raw-feature simple/NN, Qwen3-Omni, Cosmos3-Super und Cosmos3-Nano; 140/140 gescorte Einträge; 134 direct + 6 compact proxy. | Baselines und Modellzweige auf demselben Split vergleichen; Proxy-Targets bleiben sichtbar. |

Formel: 2 Single-Episode-Methoden x 20 Aufgaben = 40; 7 128-Episode-Methoden x 20 Aufgaben = 140; öffentliche Gesamtmatrix = 180/180 gescorte Einträge.

Methodenblöcke: Linie 1 enthält task-head baselines (Minimal, Neural MLP). Linie 2 trennt aligned baseline heads (metadata simple/NN, raw-feature simple/NN), die Qwen3-Omni series (Qwen3-Omni v6 LoRA) und die Cosmos3 series (Cosmos3-Super Reasoner, Cosmos3-Nano Future Window). Qwen3 v1-v6 ist eine LoRA-/Evaluationslinie innerhalb von Linie 2, nicht die evidence lines des Projekts; die 20-Task-Matrix nutzt v6 und v5 bleibt der pinned prior release. Cosmos3-Super Forward-Dynamics LoRA ist ein separat veröffentlichter Adapter/Gewichts-/Ergebnis-Artefakt und zählt nicht als Methodenreihe der 20-Task-Matrix.

Einstieg: [`TWO_EVIDENCE_LINES.md`](TWO_EVIDENCE_LINES.md), [`two_evidence_lines.json`](docs/data/two_evidence_lines.json), [`task_method_20_result_matrix.json`](docs/data/task_method_20_result_matrix.json), [`two_evidence_line_result_summary.json`](docs/data/two_evidence_line_result_summary.json).

## Schneller Einstieg

| Ziel | Einstieg |
| --- | --- |
| Projekt verstehen | [PROJECT_BRIEF.md](PROJECT_BRIEF.md), [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| Richtige öffentliche Oberfläche wählen | [PUBLIC_READER_MAP.md](PUBLIC_READER_MAP.md) |
| 20 Aufgaben prüfen | [TASK_SUITE_20.md](TASK_SUITE_20.md), [task_suite_20.json](docs/data/task_suite_20.json) |
| Ergebnisse vergleichen | [RESEARCH_TAKEAWAYS.md](RESEARCH_TAKEAWAYS.md), [task_method_20_result_matrix.json](docs/data/task_method_20_result_matrix.json) |
| Ein Sample untersuchen | [single_episode_explorer.html](https://chaoyue0307.github.io/ropedia-xperience-10m-task-suite/single_episode_explorer.html), [raw_sample_files.json](docs/data/raw_sample_files.json) |
| Drei Foundation-Pipelines lesen | [THREE_FOUNDATION_PIPELINES.md](THREE_FOUNDATION_PIPELINES.md), [three_foundation_pipelines.json](docs/data/three_foundation_pipelines.json) |
| Reproduzieren oder auditieren | [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [EVIDENCE_CONTRACT.md](EVIDENCE_CONTRACT.md) |

## Struktur

- Daten: 20-Frame-Fenster über Video, Audio, Tiefe, Pose/SLAM, Mocap, IMU, Kalibrierung und Sprachannotation.
- Aufgaben: 20 Verträge für Erkennung, Vorhersage, Retrieval, Rekonstruktion, Ordnung, Synchronisierung, Langhorizont-Prognose, Aktion-Objekt-Bindung und Sensor-Brücken.
- Struktur: Die 20 Aufgaben sind die Score-Achsen; die 4 research directions sind Lese- und Forschungsgruppen; die 3 foundation pipelines sind Trainingspfade für größere Modelle.
- Ergebnisse: Single-Episode minimal/NN decken 20/20 ab; 128-Episode-Zweige trennen Metadata, Raw Features, Qwen3 und Cosmos; die öffentliche Matrix steht bei 180/180 gescorten Einträgen: 174 direct und 6 compact proxy, mit sichtbaren Proxy-Targets.
- Pipelines: spatial intelligence models, human-video world models und vision-language-action models sind mit Aufgaben und Evidenzanforderungen dokumentiert.

## Öffentliche Grenze

Dieses Projekt veröffentlicht nur abgeleitete Artefakte, Metriken, Figuren, Karten und public-safe Zusammenfassungen. Xperience-10M bleibt unter den offiziellen Ropedia/Hugging-Face-Bedingungen.

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
