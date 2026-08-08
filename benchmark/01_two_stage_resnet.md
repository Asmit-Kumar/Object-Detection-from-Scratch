# Benchmark Report 01 · Two-Stage ResNet Detection Pipeline

This document presents the complete 10,000-image evaluation report for the **Stage 4 Two-Stage Detection Pipeline**, combining a Stage 1 Detector (`ObjectDetectorResNet` v1) with a Stage 2 Character Classifier (`CharacterClassifierResNet`).

---

## Architecture Overview

```
                   Input Image (224x224 grayscale)
                                 │
                                 ▼
                     ResNet Backbone (Detector)
                                 │
                                 ▼
                  Prediction Head (24 Output Slots)
                                 │
                      Predicted Boxes + Scores
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       │                                                   │
 ──────── Training ────────                         ──────── Inference ────────
    Hungarian Matching                                 Confidence Threshold (≥ 0.70)
            │                                               │
      Detection Loss                                       NMS (IoU ≥ 0.35)
 (Huber + BCE w/ pos_weight)                                │
                                                       Final Boxes
                                                            │
                                                   Crop, Pad & Resize
                                                            │
                                                  Character Classifier
                                                      (47 classes)
```

Inference in the two-stage model requires extracting image crops for each detected bounding box, padding them to square 28×28 patches, and executing a secondary forward pass through the Stage 2 classifier.

---

## Model Parameter Audit

| Stage | Model | Size | Model Params | Combined Pipeline Params (Det + Cls) | Weights |
|:---|:---|:---:|:---:|:---:|:---|
| Stage 1 | `ObjectDetectorResNet` — Nano | `n` | **0.57M** (565,784) | **0.94M** (936,135) | `weights/detector_n_new_best.pth` |
| Stage 1 | `ObjectDetectorResNet` — Small | `s` | **2.23M** (2,227,128) | **2.60M** (2,597,479) | `weights/detector_s_new_best.pth` |
| Stage 1 | `ObjectDetectorResNet` — Medium | `m` | **8.84M** (8,836,856) | **9.21M** (9,207,207) | `weights/detector_m_new_best.pth` |
| Stage 2 | `CharacterClassifierResNet` — ByMerge | — | **0.37M** (370,351) | — | `weights/classifier_resent_bymerge_s_best.pth` |

---

## Evaluation Results (10,000 Test Images across 4 Layouts)

### 🔬 Nano — 0.57M Detector / 0.94M Total

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | 0.9242 | 0.8566 | 78.03% | 0.6938 | 661 FPS |
| 📐 Grid   | 0.8891 | 0.8158 | 77.38% | 0.6584 | 650 FPS |
| 📝 Words  | 0.9612 | 0.8584 | 71.58% | 0.6491 | 665 FPS |
| 📏 Line   | 0.9580 | 0.8512 | 71.59% | 0.6454 | 673 FPS |

### ⚡ Small — 2.23M Detector / 2.60M Total _(recommended)_

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | 0.9773 | 0.9520 | 78.30% | **0.7552** | 611 FPS |
| 📐 Grid   | 0.9548 | 0.9237 | 78.02% | **0.7326** | 605 FPS |
| 📝 Words  | 0.9870 | 0.9319 | 72.19% | **0.6920** | 618 FPS |
| 📏 Line   | 0.9800 | 0.9292 | 72.08% | **0.6876** | 615 FPS |

### 🎯 Medium — 8.84M Detector / 9.21M Total _(best accuracy)_

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | 0.9900 | 0.9538 | 78.84% | **0.7660** | 393 FPS |
| 📐 Grid   | 0.9809 | 0.9497 | 78.66% | **0.7591** | 387 FPS |
| 📝 Words  | 0.9936 | 0.9489 | 72.99% | **0.7085** | 378 FPS |
| 📏 Line   | 0.9934 | 0.9504 | 72.83% | **0.7075** | 386 FPS |

---

## Two-Stage Model Summary

| Variant | Detector Params | Classifier Params | Total Pipeline Params | Avg Det Precision | Avg Det Recall | Avg E2E F1 | Avg FPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nano (`n`)** | **565,784** (0.57M) | **370,351** (0.37M) | **936,135** (0.94M) | 0.933 | 0.846 | 0.662 | **662 FPS** |
| **Small (`s`)** | **2,227,128** (2.23M) | **370,351** (0.37M) | **2,597,479** (2.60M) | 0.975 | 0.934 | 0.717 | **612 FPS** |
| **Medium (`m`)** | **8,836,856** (8.84M) | **370,351** (0.37M) | **9,207,207** (9.21M) | **0.990** | **0.951** | **0.735** | **386 FPS** |

---

## 📈 Density-Stratified Recall Sweep (Recall vs. GT Object Count $n_{gt}$)

| Density Bucket ($n_{gt}$) | Test Images | Two-Stage Nano (`n`) | Two-Stage Small (`s`) | Two-Stage Medium (`m`) |
|:---:|:---:|:---:|:---:|:---:|
| **1 – 4 objects** | 236 | 93.70% | **96.05%** | 93.58% |
| **5 – 8 objects** | 1,244 | 89.25% | **96.57%** | 95.86% |
| **9 – 12 objects** | 850 | 84.26% | 95.35% | **95.84%** |
| **13 – 16 objects** | 170 | 76.62% | 89.16% | **92.88%** |


---

## Sample Detections

Each image shows a 224×224 canvas with detected bounding boxes. Labels display predicted character + joint confidence score.

### 🔬 Nano (0.57M Det / 0.94M Total)

| random | random | grid | grid |
|:---:|:---:|:---:|:---:|
| ![](../result/benchmark/n/random_1.png) | ![](../result/benchmark/n/random_2.png) | ![](../result/benchmark/n/grid_1.png) | ![](../result/benchmark/n/grid_2.png) |
| **words** | **words** | **line** | **line** |
| ![](../result/benchmark/n/words_1.png) | ![](../result/benchmark/n/words_2.png) | ![](../result/benchmark/n/line_1.png) | ![](../result/benchmark/n/line_2.png) |

### ⚡ Small (2.23M Det / 2.60M Total) — Recommended

| random | random | grid | grid |
|:---:|:---:|:---:|:---:|
| ![](../result/benchmark/s/random_1.png) | ![](../result/benchmark/s/random_2.png) | ![](../result/benchmark/s/grid_1.png) | ![](../result/benchmark/s/grid_2.png) |
| **words** | **words** | **line** | **line** |
| ![](../result/benchmark/s/words_1.png) | ![](../result/benchmark/s/words_2.png) | ![](../result/benchmark/s/line_1.png) | ![](../result/benchmark/s/line_2.png) |

### 🎯 Medium (8.84M Det / 9.21M Total) — Best Accuracy

| random | random | grid | grid |
|:---:|:---:|:---:|:---:|
| ![](../result/benchmark/m/random_1.png) | ![](../result/benchmark/m/random_2.png) | ![](../result/benchmark/m/grid_1.png) | ![](../result/benchmark/m/grid_2.png) |
| **words** | **words** | **line** | **line** |
| ![](../result/benchmark/m/words_1.png) | ![](../result/benchmark/m/words_2.png) | ![](../result/benchmark/m/line_1.png) | ![](../result/benchmark/m/line_2.png) |

