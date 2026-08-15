# Benchmark Report 03 · Grid-Based Spatial ResNet Detector

This document presents the complete 10,000-image evaluation report for the **Stage 6 Grid-Based Spatial Detector Architecture** (`ObjectDetectorResNet` v3 with `grid_head`).

---

## Architecture Overview

```
                   Input Image (224×224 grayscale)
                                 │
                                 ▼
                      ResNet Backbone (Stem + L1..L4)
                                 │
                      Spatial Feature Map (14×14)
                                 │
                      grid_head (Conv2d 1×1)
                                 │
                      Spatial Grid Tensor (B, 14, 14, 52)
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       │                                                   │
 ──────── Training ────────                         ──────── Inference ────────
    Direct Spatial Target Mapping                       Optimal Conf Threshold (0.90)
    (targets[gy, gx] = box, obj, cls)                               │
            │                                               NMS (IoU ≥ 0.35)
    Direct GPU Tensor Loss                                          │
 (Huber + BCE + Aligned-IoU + CE)                              Final Detections
                                                   (Boxes + Scores + Classes)
```

In contrast to the Stage 5 FC-based tri-head model, the Stage 6 Grid Detector eliminates global average pooling (`AdaptiveAvgPool2d((2,2))`) and fully-connected linear projection heads. Instead, it applies a spatial $1 \times 1$ convolutional head directly on top of the $14 \times 14$ feature map. Bounding box coordinates $[x, y, w, h]$, objectness confidence logits $[conf]$, and 47 character class logits $[cls_0 \dots cls_{46}]$ are bound directly to local $16 \times 16$ pixel receptive fields.

---

## Model Parameter Audit

| Size Preset | Model Name | Total Parameters | Focal Loss Checkpoint | Original BCE Checkpoint | Output Tensor Shape |
|:---|:---|:---:|:---|:---|:---:|
| **Nano (`n`)** | `ObjectDetectorResNet` (v3 Nano) | **392,044** (0.39M) | `weights/1_grid_detector_n_best.pth` | `weights/grid_detector_n_best.pth` | `(B, 14, 14, 52)` |
| **Small (`s`)** | `ObjectDetectorResNet` (v3 Small) | **1,552,652** (1.55M) | `weights/1_grid_detector_s_best.pth` | `weights/grid_detector_s_best.pth` | `(B, 14, 14, 52)` |
| **Medium (`m`)** | `ObjectDetectorResNet` (v3 Medium) | **6,183,948** (6.18M) | `weights/1_grid_detector_m_best.pth` | `weights/grid_detector_m_best.pth` | `(B, 14, 14, 52)` |

**Configuration**: `nms_iou=0.35`, `device=cuda`, `S=14` spatial resolution.

---

## Evaluation Results: Focal Loss ($K=1$, `1_grid_detector_*`)

### 🔬 Focal Nano — 0.39M Params (`conf = 0.50`)
> [!NOTE]
> When trained with pure Focal Loss without pos_weight boosting, the 0.39M parameter Nano model lacked the capacity for simultaneous box regression and classification across dense layouts. Small (1.55M) and Medium (6.18M) scaled robustly.

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | 0.0984 | 0.0945 | 14.98% | 0.0144 | 397.1 img/s |
| 📐 Grid   | 0.0566 | 0.0624 | 10.62% | 0.0063 | 413.1 img/s |
| 📝 Words  | 0.0684 | 0.0774 | 27.21% | 0.0198 | 408.9 img/s |
| 📏 Line   | 0.0711 | 0.0806 | 27.07% | 0.0205 | 401.3 img/s |

### ⚡ Focal Small — 1.55M Params (`conf = 0.50`)

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | **0.9983** | **0.9978** | **89.27%** | **0.8909** | 396.4 img/s |
| 📐 Grid   | 0.4968 | 0.4820 | 85.92% | 0.4204 | 400.1 img/s |
| 📝 Words  | 0.4169 | 0.4078 | 87.61% | 0.3613 | 404.4 img/s |
| 📏 Line   | 0.4089 | 0.4018 | 85.43% | 0.3463 | 396.1 img/s |

### 🎯 Focal Medium — 6.18M Params (`conf = 0.40`)

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | **0.9972** | **0.9974** | **89.63%** | **0.8939** | 367.1 img/s |
| 📐 Grid   | 0.4932 | 0.5010 | 86.05% | 0.4277 | 380.0 img/s |
| 📝 Words  | 0.4309 | 0.4354 | 85.06% | 0.3684 | 380.2 img/s |
| 📏 Line   | 0.4191 | 0.4237 | 82.90% | 0.3493 | 380.6 img/s |

---

## Evaluation Results: Original BCE ($K=1$, `grid_detector_*`)

### 🔬 BCE Nano — 0.39M Params (`conf = 0.95`)

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | 0.9972 | 0.9975 | 88.30% | 0.8806 | 410.4 img/s |
| 📐 Grid   | 0.4893 | 0.4843 | 85.37% | 0.4156 | 408.4 img/s |
| 📝 Words  | 0.4102 | 0.4076 | 87.06% | 0.3559 | 419.0 img/s |
| 📏 Line   | 0.3978 | 0.3955 | 85.63% | 0.3397 | 414.7 img/s |

### ⚡ BCE Small — 1.55M Params (`conf = 0.95`)

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | 0.9976 | 0.9982 | 89.62% | 0.8943 | 401.9 img/s |
| 📐 Grid   | 0.5004 | 0.5005 | 86.48% | 0.4328 | 402.2 img/s |
| 📝 Words  | 0.4201 | 0.4202 | 87.39% | 0.3671 | 399.4 img/s |
| 📏 Line   | 0.4179 | 0.4180 | 85.20% | 0.3561 | 399.0 img/s |

### 🎯 BCE Medium — 6.18M Params (`conf = 0.95`)

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | 0.9973 | 0.9979 | 89.77% | 0.8955 | 383.6 img/s |
| 📐 Grid   | 0.5011 | 0.5002 | 85.99% | 0.4305 | 378.7 img/s |
| 📝 Words  | 0.4346 | 0.4344 | 84.31% | 0.3663 | 381.7 img/s |
| 📏 Line   | 0.4241 | 0.4239 | 82.43% | 0.3496 | 375.2 img/s |

---

## Grid-Based Spatial ($K=1$) Summary Comparison

| Model Variant | Loss Formulation | Total Params | Optimal `conf` | Avg Det Precision | Avg Det Recall | Avg Classifier Acc | Avg End-to-End F1 | Avg Throughput (Image FPS) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nano (`n`)** | Focal Loss | 0.39M | `0.50` | 0.0736 ⚠️ | 0.0787 ⚠️ | 19.97% | 0.0153 | 405.1 img/s |
| **Small (`s`)** | Focal Loss | 1.55M | `0.50` | 0.5802 | 0.5724 | 87.06% | 0.5047 | 399.3 img/s |
| **Medium (`m`)** | Focal Loss | 6.18M | `0.40` | 0.5851 | 0.5894 | 85.91% | 0.5098 | 377.0 img/s |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **Nano (`n`)** | Original BCE | 0.39M | `0.95` | 0.5736 | 0.5712 | 86.59% | 0.4980 | **413.1 img/s** |
| **Small (`s`)** | Original BCE | 1.55M | `0.95` | 0.5840 | 0.5842 | **87.17%** | **0.5126** | **400.6 img/s** |
| **Medium (`m`)** | Original BCE | 6.18M | `0.95` | **0.5893** | **0.5891** | 85.63% | **0.5105** | **379.8 img/s** |

---

## 📈 Density-Stratified Recall Sweep (Recall vs. GT Object Count $n_{gt}$)

Evaluates detection recall performance across object density buckets on the `random` layout test set:

| Density Bucket ($n_{gt}$) | Test Images | Grid Nano (`n`) | Grid Small (`s`) | Grid Medium (`m`) |
|:---:|:---:|:---:|:---:|:---:|
| **1 – 4 objects** | 236 | 99.38% | 99.01% | 98.89% |
| **5 – 8 objects** | 1,244 | 99.72% | 99.80% | 99.76% |
| **9 – 12 objects** | 850 | 99.83% | 99.87% | **99.90%** |
| **13 – 16 objects** | 170 | 99.75% | **99.92%** | 99.83% |

> **Key Observation**: In contrast to Stage 5 Single-Stage FC models (which suffered severe density degradation dropping to ~75.6% recall at 13-16 objects on Nano), the Stage 6 Grid Spatial Detector maintains flawless **>99.7% recall even at 13–16 objects**. This demonstrates that binding predictions to spatial $14 \times 14$ grid cells completely resolves slot competition on non-overlapping scenes.
>
> > [!WARNING]
> > **Scope Caveat**: The Density Sweep evaluation is performed strictly on the `random` layout test set, where characters are sparsely distributed across the full image. While density collapse is solved for scattered objects, performance still drops sharply on dense structured layouts (see Root Cause below).

---

## ⚡ Technical Throughput & Spatial Grounding Dynamics

### 1. Classification Breakthrough via Spatial Grounding
* **Stage 5 Deficit**: Global average pooling (`AdaptiveAvgPool2d((2,2))`) collapsed spatial coordinate maps into abstract 1D vectors before FC projection, leading to an abysmal **36.34% classification accuracy on Nano**.
* **Stage 6 Breakthrough**: Replacing global pooling with a dense $14 \times 14$ spatial Conv head maintains local receptive field grounding, surging Nano classification accuracy to **88.30% (+51.96% gain)** and Small/Medium to **~89.6%**.

### 2. Parameter Efficiency
* Removing the global FC projection heads reduced parameter footprints significantly:
  - **Nano**: Reduced from **0.71M to 0.39M** (45% parameter reduction).
  - **Small**: Reduced from **2.52M to 1.55M** (38% parameter reduction).
  - **Medium**: Reduced from **9.42M to 6.18M** (34% parameter reduction).

### 3. The Layout Collapse Root Cause Explained
* **Sparse / Random Layouts**: On `random` placement, Stage 6 models achieve near-perfect performance: **~99.7% Precision and Recall**, and **~0.88–0.89 E2E F1**, far surpassing Stage 4 and Stage 5 models.
* **Structured / Dense Layouts (`grid`, `words`, `line`)**: Detection Precision/Recall drops sharply from ~99.7% down to **~40–50%**.
* **Root Cause (Visual Crowding)**: Detailed instrumentation of the `collate_fn` dataset mapping confirmed **exactly 0 grid-cell collisions** across all layouts (target centers never overwrite one another within the same $16 \times 16$ grid cell). Instead, the model's collapse on structured text is caused by **receptive field interference (visual crowding)**. In layouts like `grid`, `words`, and `line`, characters are packed densely next to one another. The ResNet's local receptive field blends these dense visual features together, causing the bounding box and classification heads to fail to disentangle adjacent targets, even though they technically occupy distinct logical grid cells.

---

## 🖼️ Sample Detections

Visual sample detection grid outputs generated from the test benchmark:

### 🔬 Nano (0.39M) — `conf = 0.90`

| random | random | grid | grid |
|:---:|:---:|:---:|:---:|
| ![](../result/benchmark/grid_stage/n/random_1.png) | ![](../result/benchmark/grid_stage/n/random_2.png) | ![](../result/benchmark/grid_stage/n/grid_1.png) | ![](../result/benchmark/grid_stage/n/grid_2.png) |
| **words** | **words** | **line** | **line** |
| ![](../result/benchmark/grid_stage/n/words_1.png) | ![](../result/benchmark/grid_stage/n/words_2.png) | ![](../result/benchmark/grid_stage/n/line_1.png) | ![](../result/benchmark/grid_stage/n/line_2.png) |

### ⚡ Small (1.55M) — `conf = 0.90`

| random | random | grid | grid |
|:---:|:---:|:---:|:---:|
| ![](../result/benchmark/grid_stage/s/random_1.png) | ![](../result/benchmark/grid_stage/s/random_2.png) | ![](../result/benchmark/grid_stage/s/grid_1.png) | ![](../result/benchmark/grid_stage/s/grid_2.png) |
| **words** | **words** | **line** | **line** |
| ![](../result/benchmark/grid_stage/s/words_1.png) | ![](../result/benchmark/grid_stage/s/words_2.png) | ![](../result/benchmark/grid_stage/s/line_1.png) | ![](../result/benchmark/grid_stage/s/line_2.png) |

### 🎯 Medium (6.18M) — `conf = 0.90`

| random | random | grid | grid |
|:---:|:---:|:---:|:---:|
| ![](../result/benchmark/grid_stage/m/random_1.png) | ![](../result/benchmark/grid_stage/m/random_2.png) | ![](../result/benchmark/grid_stage/m/grid_1.png) | ![](../result/benchmark/grid_stage/m/grid_2.png) |
| **words** | **words** | **line** | **line** |
| ![](../result/benchmark/grid_stage/m/words_1.png) | ![](../result/benchmark/grid_stage/m/words_2.png) | ![](../result/benchmark/grid_stage/m/line_1.png) | ![](../result/benchmark/grid_stage/m/line_2.png) |
