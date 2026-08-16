# Master Benchmark Index & Architecture Comparison

Welcome to the central **Benchmark Hub** for the Object Detection from Scratch project.

This document maintains the master comparison index across 10,000 placement-stratified test images (`data/OD_benchmark/` across `random`, `grid`, `words`, `line`).

---

## 🏆 Master Cross-Architecture Comparison Table

| Architecture Paradigm | Model Size | Model Params | Optimal `conf` | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Throughput (Image FPS) | Full Report |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Two-Stage Pipeline** | Nano (`n`) | 0.94M | 0.70 | 0.9330 | 0.8460 | 74.60% | 0.6620 | **662 img/s** | [`benchmark/01_two_stage_resnet.md`](./benchmark/01_two_stage_resnet.md) |
| **Two-Stage Pipeline** | Small (`s`) | 2.60M | 0.70 | 0.9750 | 0.9340 | 75.10% | 0.7170 | **612 img/s** | [`benchmark/01_two_stage_resnet.md`](./benchmark/01_two_stage_resnet.md) |
| **Two-Stage Pipeline** | Medium (`m`) | 9.21M | 0.70 | **0.9900** | **0.9510** | 75.80% | 0.7350 | 156 img/s 🐢 | [`benchmark/01_two_stage_resnet.md`](./benchmark/01_two_stage_resnet.md) |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **Single-Stage Unified** | Nano (`n`) | 0.71M | 0.70 | 0.8835 | 0.8386 | 36.34% | 0.3131 | 634 img/s | [`benchmark/02_single_stage_unified_resnet.md`](./benchmark/02_single_stage_unified_resnet.md) |
| **Single-Stage Unified** | Small (`s`) | 2.52M | 0.65 ⭐ | 0.9192 | 0.9101 | 71.44% | 0.6535 | 608 img/s | [`benchmark/02_single_stage_unified_resnet.md`](./benchmark/02_single_stage_unified_resnet.md) |
| **Single-Stage Unified** | Medium (`m`) | 9.42M | 0.65 ⭐ | 0.9513 | 0.9490 | 83.22% | 0.7907 | **589 img/s** 🚀 | [`benchmark/02_single_stage_unified_resnet.md`](./benchmark/02_single_stage_unified_resnet.md) |
| **Single-Stage Unified** | Large (`l`) | 19.99M | 0.60 ⭐ | 0.9458 | **0.9580** | **87.34%** | **0.8320** | **533 img/s** 🚀 | [`benchmark/02_single_stage_unified_resnet.md`](./benchmark/02_single_stage_unified_resnet.md) |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **Grid Spatial ($K=1$, Focal)** | Nano (`n`) | **0.39M** ⚡ | 0.45 | 0.5696 | 0.5659 | 86.89% | 0.4954 | 345 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| **Grid Spatial ($K=1$, Focal)** | Small (`s`) | **1.55M** ⚡ | 0.50 | 0.5802 | 0.5724 | **87.06%** 🚀 | 0.5047 | 399 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| **Grid Spatial ($K=1$, Focal)** | Medium (`m`) | **6.18M** ⚡ | 0.40 | **0.5851** | **0.5894** | 85.91% | 0.5098 | 377 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **Grid Spatial ($K=1$, BCE)** | Nano (`n`) | **0.39M** ⚡ | 0.95 | 0.5736 | 0.5712 | 86.59% | 0.4980 | 413 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| **Grid Spatial ($K=1$, BCE)** | Small (`s`) | **1.55M** ⚡ | 0.95 | 0.5840 | 0.5842 | **87.17%** 🚀 | **0.5126** | 401 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| **Grid Spatial ($K=1$, BCE)** | Medium (`m`) | **6.18M** ⚡ | 0.95 | **0.5893** | **0.5891** | 85.63% | 0.5105 | 380 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **Multi-Anchor ($K=3$)** | Nano (`n`) | **0.39M** ⚡ | 0.40 | 0.5009 | 0.5243 | 86.34% | 0.4445 | 412 img/s | [`benchmark/04_multi_anchor_spatial_resnet.md`](./benchmark/04_multi_anchor_spatial_resnet.md) |
| **Multi-Anchor ($K=3$)** | Small (`s`) | **1.55M** ⚡ | 0.95 | 0.5192 | 0.5870 | 86.43% | 0.4801 | 395 img/s | [`benchmark/04_multi_anchor_spatial_resnet.md`](./benchmark/04_multi_anchor_spatial_resnet.md) |
| **Multi-Anchor ($K=3$)** | Medium (`m`) | **6.18M** ⚡ | 0.95 | 0.5330 | **0.5960** 🚀 | 85.68% | 0.4878 | 376 img/s | [`benchmark/04_multi_anchor_spatial_resnet.md`](./benchmark/04_multi_anchor_spatial_resnet.md) |

---

## ⚡ Technical Analysis: Single-Stage vs. Two-Stage Throughput

### 📊 Direct End-to-End Throughput & Latency Comparison

| Model Preset | Paradigm | Total Params | End-to-End Latency / Image | End-to-End Throughput (Image FPS) | Speedup vs. Two-Stage |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Nano (`n`)** | Two-Stage | 0.94M | ~1.51 ms | **662 img/s** | Baseline (1.0×) |
| **Nano (`n`)** | **Single-Stage** | 0.71M | ~1.58 ms | 634 img/s | 0.96× |
| ─── | ─── | ─── | ─── | ─── | ─── |
| **Small (`s`)** | Two-Stage | 2.60M | ~1.63 ms | **612 img/s** | Baseline (1.0×) |
| **Small (`s`)** | **Single-Stage** | 2.52M | ~1.64 ms | 608 img/s | 0.99× |
| ─── | ─── | ─── | ─── | ─── | ─── |
| **Medium (`m`)** | Two-Stage | 9.21M | ~6.41 ms | 156 img/s | Baseline (1.0×) |
| **Medium (`m`)** | **Single-Stage** | 9.42M | **~1.70 ms** | **589 img/s** | 🚀 **3.8× Faster** |
| ─── | ─── | ─── | ─── | ─── | ─── |
| **Large (`l`)** | **Single-Stage** | 19.99M | **~1.88 ms** | **533 img/s** | 🚀 **3.4× Faster vs. Two-Stage M** |

---

### 🔍 Architectural Scaling Dynamics: Why Does Single-Stage Win at Scale?

#### 1. Why Two-Stage Nano/Small are Slightly Faster (662 vs. 634 img/s & 612 vs. 608 img/s)
- **Sparse vs. Dense Classification Computation**:
  - In **Two-Stage**, the secondary classifier is extremely tiny (~45k–150k parameters). It **only runs on detected bounding box crops** ($K \approx 8 - 10$ crops per image).
  - In **Single-Stage**, the unified tri-head evaluates character class logits **densely across all 24 spatial grid slots** ($24 \times 47 = 1,128$ class logits per image) during every forward pass.
  - On tiny backbones (Nano/Small) where neural network compute takes $< 0.5\text{ ms}$, evaluating 1,128 dense logits adds a tiny amount of GPU tensor operations. Because Two-Stage Nano/Small only classifies 8–10 crops with a lightweight classifier, it runs slightly faster by ~28 img/s.

#### 2. Why Two-Stage Collapses at Medium Scale (156 img/s) while Single-Stage Remains Fast (589 img/s)
- **Secondary Classifier Scaling Bottleneck**:
  - As model capacity grows, the Two-Stage secondary classifier scales up (~1.2M parameters). Running $K$ crops through a heavy secondary network per image introduces continuous **CPU $\leftrightarrow$ GPU stream synchronizations**, crop slicing overhead, and sequential model passes. This causes Two-Stage throughput to drop dramatically from **612 img/s down to 156 img/s** (4.1ms added latency per image!).
- **Single-Stage Constant-Time Tri-Head**:
  - In **Single-Stage**, the tri-head is baked directly into the backbone output feature map. Scaling the backbone from Nano (0.71M) to Large (19.99M) adds **zero extra forward passes and zero crop extraction overhead**. Latency remains nearly flat (~1.58 ms for Nano $\rightarrow$ 1.70 ms for Medium $\rightarrow$ 1.88 ms for Large), resulting in a **3.8× speedup on Medium (589 img/s vs. 156 img/s)**.

---

## 📈 Density-Stratified Recall Sweep (Recall vs. GT Object Count $n_{gt}$)

Evaluates detection recall drop-off as object density increases per canvas image ($n_{gt}$):

| Density Bucket ($n_{gt}$) | Test Images | Two-Stage Small | Single-Stage Small | Two-Stage Medium | Single-Stage Medium | Single-Stage Large | Grid Spatial Small ($K=1$) | Grid Spatial Medium ($K=1$) | Multi-Anchor Nano ($K=3$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 – 4 objects** | 236 | 96.05% | 96.79% | 93.58% | 94.94% | 89.01% | 99.75% | 99.65% | 74.18% |
| **5 – 8 objects** | 1,244 | 96.57% | 94.43% | 95.86% | 96.89% | 95.06% | 99.78% | 99.70% | 74.25% |
| **9 – 12 objects** | 850 | 95.35% | 90.37% | 95.84% | 94.85% | 95.56% | 99.82% | 99.72% | 74.12% |
| **13 – 16 objects** | 170 | 89.16% | 84.41% | 92.88% | 88.73% | 88.10% | 99.85% | 99.75% | 74.21% |

> **Key Observation**: Single-Stage Medium (`m`) holds **96.89% Recall** on 5–8 object canvases and **94.85% Recall** on 9–12 object canvases, demonstrating strong scale resilience as canvas clutter increases.
>
> > [!WARNING]
> > **Scope Caveat**: Grid Spatial results in the Density Sweep are evaluated exclusively on the `random` layout test set. While spatial grounding perfectly resolves slot competition for sparsely scattered overlapping objects (>99.7% recall), performance collapses significantly on dense, structured layouts like `grid`, `words`, and `line` due to visual crowding (receptive field interference).

---

## 🔬 Architectural Findings

### 1. The Crop Orientation Bug: Transpose Necessity (Two-Stage)
**The Problem**: During Two-Stage inference, the secondary classifier's accuracy on cropped detections unexpectedly collapsed to ~10.6% when tested end-to-end, despite showing ~78% accuracy during standalone training.
**The Root Cause & Fix**: The dataset generator renders characters upright (row-major) on the 224x224 canvases, but raw EMNIST binaries are natively column-major. Because the classifier was trained on raw EMNIST, crops extracted from the canvas had to be explicitly rotated via `.transpose(-1, -2)`.
**The Impact**: Adding the transpose operation instantly restored classifier accuracy. This ablation proved the models were localizing and learning effectively, and the bottleneck was purely a silent data-orientation mismatch.

### 2. Density Degradation vs. Capacity (Single-Stage)
**The Problem**: Early single-stage runs exhibited a pathological "recall floor," where dense canvases (9+ objects) caused bounding box recall to plummet to ~43%.
**The Fix**: This was diagnosed as a loss imbalance. Implementing a unified tri-head loss (`box + obj + class`) and rebalancing `pos_weight` resolved the severe pathological floor.
**The Impact**: The density sweep confirms the pathological signature is gone. However, a milder, capacity-correlated version persists: Nano (0.71M) drops to 75.6% recall on highly dense scenes (13–16 objects), while Large (19.99M) holds 95.5% recall up to 12 objects. This proves that successfully resolving slot competition in dense character clusters requires raw parameter capacity, not just loss tuning.

### 3. Saturated Confidence on Blank Canvas (Single-Stage)
**The Problem**: We observed slots confidently predicting bounding boxes (`conf=1.00`) on completely empty background patches.
**The Root Cause**: The `AdaptiveAvgPool2d((2,2))` in the ResNet backbone collapses the spatial grid into a diffuse, whole-image summary *before* the detection heads make a decision. Because all 24 slots read from the exact same globally-pooled feature vector, they lack local spatial awareness. Confidence scores and box coordinates are statistically correlated by the Hungarian loss, but *architecturally uncoupled* from local spatial visual evidence.
**The Mitigation & Future Fix**: We mitigated this by adding a secondary class-confidence gate (`cls_conf >= 0.30`) to suppress un-grounded false positives post-NMS. However, structurally eliminating this artifact requires migrating from Global Pooling to a **Grid-Based Spatial Head** (YOLO/SSD style) where confidence is bound directly to local spatial receptive fields.

---

## 📁 Individual Architecture Benchmark Reports

- **[`benchmark/01_two_stage_resnet.md`](./benchmark/01_two_stage_resnet.md)** — **Stage 4 Two-Stage ResNet Detection Pipeline**
- **[`benchmark/02_single_stage_unified_resnet.md`](./benchmark/02_single_stage_unified_resnet.md)** — **Stage 5 Single-Stage Unified ResNet Detector**
- **[`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md)** — **Stage 6 Grid-Based Spatial ResNet Detector**
- **[`benchmark/04_multi_anchor_spatial_resnet.md`](./benchmark/04_multi_anchor_spatial_resnet.md)** — **Stage 7 Multi-Anchor ($K=3$) Spatial ResNet Detector**

