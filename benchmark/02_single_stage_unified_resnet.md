# Benchmark Report 02 · Single-Stage Unified ResNet Detector

This document presents the complete 10,000-image evaluation report for the **Stage 5 Single-Stage Unified Detector Architecture** (`ObjectDetectorResNet` v2 with `box_head`, `obj_head`, `class_head`).

---

## Architecture Overview

```
                   Input Image (224×224 grayscale)
                                 │
                                 ▼
                      ResNet Backbone (Shared)
                                 │
                 Tri-Head (Box + Obj + Class Logits)
                                 │
                      Predicted Tensor (B, 24, 52)
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       │                                                   │
 ──────── Training ────────                         ──────── Inference ────────
    Hungarian Matching                                 Optimal Conf Threshold (0.60 - 0.70)
            │                                               │
   Unified Tri-Head Loss                               NMS (IoU ≥ 0.35)
(Box + Obj + Class on Matched)                              │
                                                       Final Detections
                                                   (Boxes + Scores + Classes)
```

In contrast to the two-stage pipeline, the Single-Stage Unified Detector predicts bounding box coordinates $[x, y, w, h]$, objectness confidence logits $[conf]$, and 47 character class logits $[cls_0 \dots cls_{46}]$ **simultaneously in a single forward pass**. This completely eliminates image patch cropping, padding, resizing, and crop-tensor GPU memory transfers.

---

## Model Parameter Audit

| Size Preset | Model Name | Total Parameters | Optimal `conf` | Checkpoint Weight File | Output Tensor Shape |
|:---|:---|:---:|:---:|:---|:---:|
| **Nano (`n`)** | `ObjectDetectorResNet` (v2 Nano) | **711,296** (0.71M) | `0.70` | `checkpoint/s_detector_n_best.pth` | `(B, 24, 52)` |
| **Small (`s`)** | `ObjectDetectorResNet` (v2 Small) | **2,517,024** (2.52M) | `0.65` ⭐ | `checkpoint/s_detector_s_best.pth` | `(B, 24, 52)` |
| **Medium (`m`)** | `ObjectDetectorResNet` (v2 Medium) | **9,415,520** (9.42M) | `0.65` ⭐ | `checkpoint/s_detector_m_best.pth` | `(B, 24, 52)` |
| **Large (`l`)** | `ObjectDetectorResNet` (v2 Large) | **19,989,088** (19.99M) | `0.60` ⭐ | `checkpoint/s_detector_l_best.pth` | `(B, 24, 52)` |

**Configuration**: `iou_match_thresh=0.50` (Hungarian matching), `nms_iou=0.35`, `device=cuda`

---

## Evaluation Results (10,000 Test Images across 4 Layouts)

### 🔬 Nano — 0.71M Params (`conf = 0.70`)

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | 0.8961 | 0.8480 | 32.97% | 0.2873 | 577 img/s |
| 📐 Grid   | 0.8455 | 0.8026 | 33.16% | 0.2731 | 652 img/s |
| 📝 Words  | 0.8775 | 0.8390 | 40.27% | 0.3455 | 646 img/s |
| 📏 Line   | 0.9149 | 0.8649 | 38.94% | 0.3463 | 661 img/s |

### ⚡ Small — 2.52M Params (`conf = 0.65`)

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | 0.9339 | 0.9257 | 70.09% | 0.6517 | 604 img/s |
| 📐 Grid   | 0.9103 | 0.8948 | 69.07% | 0.6234 | 610 img/s |
| 📝 Words  | 0.9045 | 0.9001 | 74.05% | 0.6681 | 610 img/s |
| 📏 Line   | 0.9280 | 0.9207 | 72.57% | 0.6708 | 606 img/s |

### 🎯 Medium — 9.42M Params (`conf = 0.65`)

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | 0.9579 | 0.9577 | 82.43% | 0.7895 | 589 img/s |
| 📐 Grid   | 0.9437 | 0.9396 | 81.02% | 0.7629 | 587 img/s |
| 📝 Words  | 0.9517 | 0.9469 | 86.00% | 0.8164 | 593 img/s |
| 📏 Line   | 0.9520 | 0.9518 | 83.42% | 0.7941 | 585 img/s |

### 🏆 Large — 19.99M Params (`conf = 0.60`) — Best Overall Accuracy

| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🎲 Random | 0.9647 | 0.9654 | 86.34% | **0.8332** | 520 img/s |
| 📐 Grid   | 0.9562 | 0.9625 | 85.55% | **0.8207** | 532 img/s |
| 📝 Words  | 0.9292 | 0.9524 | 89.23% | **0.8394** | 536 img/s |
| 📏 Line   | 0.9331 | 0.9518 | 88.59% | **0.8349** | 541 img/s |

---

## Single-Stage Model Comparison Summary (Optimal `conf` per Size)

| Model Variant | Total Parameters | Optimal `conf` | Avg Det Precision | Avg Det Recall | Avg Classifier Acc | Avg End-to-End F1 | Avg Throughput (Image FPS) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nano (`n`)** | **0.71M** (711,296) | `0.70` | 0.8835 | 0.8386 | 36.34% | 0.3131 | **634 img/s** |
| **Small (`s`)** | **2.52M** (2,517,024) | `0.65` ⭐ | 0.9192 | 0.9101 | 71.44% | 0.6535 | **608 img/s** |
| **Medium (`m`)** | **9.42M** (9,415,520) | `0.65` ⭐ | 0.9513 | 0.9490 | 83.22% | 0.7907 | **589 img/s** |
| **Large (`l`)** | **19.99M** (19,989,088) | `0.60` ⭐ | 0.9458 | **0.9580** | **87.34%** | **0.8320** | **533 img/s** |

---

## ⚡ Technical Throughput & Architectural Scaling Dynamics

### 🔍 Why Two-Stage Nano/Small are Slightly Faster (662 vs. 634 img/s), but Single-Stage Crushes Medium (589 vs. 156 img/s)?

1. **Sparse vs. Dense Classification Computation**:
   - In **Two-Stage**, the secondary classifier is extremely tiny (~45k–150k parameters). It **only runs on detected bounding box crops** ($K \approx 8 - 10$ crops per image).
   - In **Single-Stage**, the unified tri-head evaluates character class logits **densely across all 24 spatial grid slots** ($24 \times 47 = 1,128$ class logits per image) during every forward pass.
   - On tiny backbones (Nano/Small) where neural network compute takes $< 0.5\text{ ms}$, evaluating 1,128 dense logits adds a tiny amount of GPU tensor operations. Because Two-Stage Nano/Small only classifies 8–10 crops with a lightweight classifier, it runs slightly faster by ~28 img/s.

2. **Secondary Classifier Scaling Bottleneck in Two-Stage Medium**:
   - As model capacity grows, the Two-Stage secondary classifier scales up (~1.2M parameters). Running $K$ crops through a heavy secondary network per image introduces continuous **CPU $\leftrightarrow$ GPU stream synchronizations**, crop slicing overhead, and sequential model passes. This causes Two-Stage throughput to drop dramatically from **612 img/s down to 156 img/s** (4.1ms added latency per image!).
   - In **Single-Stage**, the tri-head is baked directly into the backbone output feature map. Scaling the backbone from Nano (0.71M) to Large (19.99M) adds **zero extra forward passes and zero crop extraction overhead**. Latency remains nearly flat (~1.58 ms for Nano $\rightarrow$ 1.70 ms for Medium $\rightarrow$ 1.88 ms for Large), resulting in a **3.8× speedup on Medium (589 img/s vs. 156 img/s)**.


---

## 📈 Density-Stratified Recall Sweep (Recall vs. GT Object Count $n_{gt}$)

| Density Bucket ($n_{gt}$) | Test Images | Single-Stage Nano | Single-Stage Small | Single-Stage Medium | Single-Stage Large |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 – 4 objects** | 236 | 89.14% | **96.79%** | 94.94% | 89.01% |
| **5 – 8 objects** | 1,244 | 87.73% | 94.43% | **96.89%** | 95.06% |
| **9 – 12 objects** | 850 | 83.91% | 90.37% | 94.85% | **95.56%** |
| **13 – 16 objects** | 170 | 75.60% | 84.41% | **88.73%** | 88.10% |

---

## 🖼️ Sample Detections

Visual sample detection grid outputs generated from the test benchmark:

### 🔬 Nano (0.71M) — `conf = 0.70`

| random | random | grid | grid |
|:---:|:---:|:---:|:---:|
| ![](../result/benchmark/single_stage/n/random_1.png) | ![](../result/benchmark/single_stage/n/random_2.png) | ![](../result/benchmark/single_stage/n/grid_1.png) | ![](../result/benchmark/single_stage/n/grid_2.png) |
| **words** | **words** | **line** | **line** |
| ![](../result/benchmark/single_stage/n/words_1.png) | ![](../result/benchmark/single_stage/n/words_2.png) | ![](../result/benchmark/single_stage/n/line_1.png) | ![](../result/benchmark/single_stage/n/line_2.png) |

### ⚡ Small (2.52M) — `conf = 0.65`

| random | random | grid | grid |
|:---:|:---:|:---:|:---:|
| ![](../result/benchmark/single_stage/s/random_1.png) | ![](../result/benchmark/single_stage/s/random_2.png) | ![](../result/benchmark/single_stage/s/grid_1.png) | ![](../result/benchmark/single_stage/s/grid_2.png) |
| **words** | **words** | **line** | **line** |
| ![](../result/benchmark/single_stage/s/words_1.png) | ![](../result/benchmark/single_stage/s/words_2.png) | ![](../result/benchmark/single_stage/s/line_1.png) | ![](../result/benchmark/single_stage/s/line_2.png) |

### 🎯 Medium (9.42M) — `conf = 0.65`

| random | random | grid | grid |
|:---:|:---:|:---:|:---:|
| ![](../result/benchmark/single_stage/m/random_1.png) | ![](../result/benchmark/single_stage/m/random_2.png) | ![](../result/benchmark/single_stage/m/grid_1.png) | ![](../result/benchmark/single_stage/m/grid_2.png) |
| **words** | **words** | **line** | **line** |
| ![](../result/benchmark/single_stage/m/words_1.png) | ![](../result/benchmark/single_stage/m/words_2.png) | ![](../result/benchmark/single_stage/m/line_1.png) | ![](../result/benchmark/single_stage/m/line_2.png) |

### 🏆 Large (19.99M) — `conf = 0.60` (Best Overall Accuracy)

| random | random | grid | grid |
|:---:|:---:|:---:|:---:|
| ![](../result/benchmark/single_stage/l/random_1.png) | ![](../result/benchmark/single_stage/l/random_2.png) | ![](../result/benchmark/single_stage/l/grid_1.png) | ![](../result/benchmark/single_stage/l/grid_2.png) |
| **words** | **words** | **line** | **line** |
| ![](../result/benchmark/single_stage/l/words_1.png) | ![](../result/benchmark/single_stage/l/words_2.png) | ![](../result/benchmark/single_stage/l/line_1.png) | ![](../result/benchmark/single_stage/l/line_2.png) |
