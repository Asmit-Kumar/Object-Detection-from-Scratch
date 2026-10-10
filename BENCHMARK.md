# Master Benchmark Index & Architecture Comparison

Welcome to the central **Benchmark Hub** for the Object Detection from Scratch project.

This document maintains the master comparison index across 10,000 placement-stratified test images (`data/OD_benchmark/` across `random`, `grid`, `words`, `line`).

---

## 🏆 Master Cross-Architecture Comparison Table

<!-- BEGIN GENERATED: master_table (scripts/render_benchmark_reports.py) -->
| Architecture Paradigm | Size | Params | `conf` | `conf` Source | Avg Det Precision | Avg Det Recall | Avg Classifier Acc | Avg End-to-End F1 | Eval-Loop Throughput | Pure Inference | Full Report |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Two-Stage Pipeline** | Nano (`n`) | 0.94M | 0.70 | fixed | 0.9331 | 0.8455 | 74.64% | 0.6617 | 662 img/s | — | [`benchmark/01_two_stage_resnet.md`](./benchmark/01_two_stage_resnet.md) |
| **Two-Stage Pipeline** | Small (`s`) | 2.60M | 0.70 | fixed | 0.9748 | 0.9342 | 75.15% | 0.7168 | 612 img/s | — | [`benchmark/01_two_stage_resnet.md`](./benchmark/01_two_stage_resnet.md) |
| **Two-Stage Pipeline** | Medium (`m`) | 9.21M | 0.70 | fixed | 0.9895 | 0.9507 | 75.83% | 0.7353 | 386 img/s | — | [`benchmark/01_two_stage_resnet.md`](./benchmark/01_two_stage_resnet.md) |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **Single-Stage Unified** | Nano (`n`) | 0.71M | 0.70 | hard-coded, source unrecorded | 0.8835 | 0.8386 | 36.34% | 0.3130 | 634 img/s | — | [`benchmark/02_single_stage_unified_resnet.md`](./benchmark/02_single_stage_unified_resnet.md) |
| **Single-Stage Unified** | Small (`s`) | 2.52M | 0.65 | hard-coded, source unrecorded | 0.9192 | 0.9103 | 71.45% | 0.6535 | 608 img/s | — | [`benchmark/02_single_stage_unified_resnet.md`](./benchmark/02_single_stage_unified_resnet.md) |
| **Single-Stage Unified** | Medium (`m`) | 9.42M | 0.65 | hard-coded, source unrecorded | 0.9513 | 0.9490 | 83.22% | 0.7907 | 589 img/s | — | [`benchmark/02_single_stage_unified_resnet.md`](./benchmark/02_single_stage_unified_resnet.md) |
| **Single-Stage Unified** | Large (`l`) | 19.99M | 0.60 | hard-coded, source unrecorded | 0.9458 | 0.9580 | 87.43% | 0.8320 | 532 img/s | — | [`benchmark/02_single_stage_unified_resnet.md`](./benchmark/02_single_stage_unified_resnet.md) |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **Grid Spatial ($K=1$, Focal)** | Nano (`n`) | 0.39M | 0.45 | held-out tune set | 0.5696 | 0.5659 | 86.89% | 0.4954 | 440 img/s | 20,972 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| **Grid Spatial ($K=1$, Focal)** | Small (`s`) | 1.55M | 0.50 | held-out tune set | 0.5802 | 0.5724 | 87.06% | 0.5047 | 416 img/s | 8,039 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| **Grid Spatial ($K=1$, Focal)** | Medium (`m`) | 6.18M | 0.45 | held-out tune set | 0.5883 | 0.5835 | 85.90% | 0.5087 | 394 img/s | 3,547 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **Grid Spatial ($K=1$, BCE)** | Nano (`n`) | 0.39M | 0.90 | held-out tune set | 0.5705 | 0.5756 | 86.59% | 0.4985 | 497 img/s | 21,152 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| **Grid Spatial ($K=1$, BCE)** | Small (`s`) | 1.55M | 0.95 | held-out tune set | 0.5840 | 0.5842 | 87.17% | 0.5126 | 436 img/s | 8,094 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| **Grid Spatial ($K=1$, BCE)** | Medium (`m`) | 6.18M | 0.85 | held-out tune set | 0.5846 | 0.5917 | 85.61% | 0.5095 | 394 img/s | 3,648 img/s | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **Multi-Anchor ($K=3$)** | Nano (`n`) | 0.41M | 0.40 | held-out tune set | 0.5009 | 0.5243 | 86.34% | 0.4445 | 434 img/s | 20,967 img/s | [`benchmark/04_multi_anchor_spatial_resnet.md`](./benchmark/04_multi_anchor_spatial_resnet.md) |
| **Multi-Anchor ($K=3$)** | Small (`s`) | 1.58M | 0.95 | held-out tune set | 0.5192 | 0.5869 | 86.43% | 0.4800 | 416 img/s | 8,006 img/s | [`benchmark/04_multi_anchor_spatial_resnet.md`](./benchmark/04_multi_anchor_spatial_resnet.md) |
| **Multi-Anchor ($K=3$)** | Medium (`m`) | 6.23M | 0.95 | held-out tune set | 0.5330 | 0.5960 | 85.68% | 0.4878 | 398 img/s | 3,575 img/s | [`benchmark/04_multi_anchor_spatial_resnet.md`](./benchmark/04_multi_anchor_spatial_resnet.md) |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **FCOS Anchor-Free** | Nano (`n`) | 0.52M | 0.50 | held-out tune set | 0.9994 | 0.9654 | 92.13% | 0.9048 | 703 img/s | 15,276 img/s | [`benchmark/05_fcos_anchor_free_resnet.md`](./benchmark/05_fcos_anchor_free_resnet.md) |
| **FCOS Anchor-Free** | Small (`s`) | 1.79M | 0.45 | held-out tune set | 0.9972 | 0.9889 | 90.93% | 0.9030 | 650 img/s | 6,868 img/s | [`benchmark/05_fcos_anchor_free_resnet.md`](./benchmark/05_fcos_anchor_free_resnet.md) |
| **FCOS Anchor-Free** | Medium (`m`) | 7.14M | 0.45 | held-out tune set | 0.9972 | 0.9895 | 90.75% | 0.9015 | 603 img/s | 2,988 img/s | [`benchmark/05_fcos_anchor_free_resnet.md`](./benchmark/05_fcos_anchor_free_resnet.md) |
| ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── | ─── |
| **YOLOv8 Anchor-Free** | Nano (`n`) | 0.83M | 0.45 | held-out tune set | 0.9948 | 0.9421 | 93.90% | 0.9087 | 712 img/s | 11,084 img/s | [`benchmark/06_yolov8_anchor_free_resnet.md`](./benchmark/06_yolov8_anchor_free_resnet.md) |
| **YOLOv8 Anchor-Free** | Small (`s`) | 2.51M | 0.45 | held-out tune set | 0.9943 | 0.9614 | 92.70% | 0.9061 | 687 img/s | 4,827 img/s | [`benchmark/06_yolov8_anchor_free_resnet.md`](./benchmark/06_yolov8_anchor_free_resnet.md) |
| **YOLOv8 Anchor-Free** | Medium (`m`) | 6.47M | 0.50 | test-sweep override | 0.9974 | 0.9594 | 91.94% | 0.8990 | 581 img/s | 2,298 img/s | [`benchmark/06_yolov8_anchor_free_resnet.md`](./benchmark/06_yolov8_anchor_free_resnet.md) |

_`conf` source: **held-out tune set** = tuned on `data/OD_benchmark/tune/random`, disjoint from the benchmark; **test-sweep override** = selected directly from the benchmark sweep; **fixed** / **hard-coded** = not tuned on held-out data (provenance of the Single-Stage values is not recorded). Averages are unweighted means over the 4 layouts. FCOS params are counted from model instantiation; other parameter counts come from checkpoint state_dicts._
<!-- END GENERATED: master_table -->

---

## ⚡ Technical Analysis: Throughput & Latency Dynamics

<!-- BEGIN GENERATED: throughput (scripts/render_benchmark_reports.py) -->
| Paradigm | Size | Params | Pure Inference (`model(images)`, batch 128) | Eval-Loop Throughput |
|:---|:---:|:---:|:---:|:---:|
| **Two-Stage Pipeline** | Nano (`n`) | 0.94M | not measured | 662 img/s |
| **Two-Stage Pipeline** | Small (`s`) | 2.60M | not measured | 612 img/s |
| **Two-Stage Pipeline** | Medium (`m`) | 9.21M | not measured | 386 img/s |
| ─── | ─── | ─── | ─── | ─── |
| **Single-Stage Unified** | Nano (`n`) | 0.71M | not measured | 634 img/s |
| **Single-Stage Unified** | Small (`s`) | 2.52M | not measured | 608 img/s |
| **Single-Stage Unified** | Medium (`m`) | 9.42M | not measured | 589 img/s |
| **Single-Stage Unified** | Large (`l`) | 19.99M | not measured | 532 img/s |
| ─── | ─── | ─── | ─── | ─── |
| **Grid Spatial ($K=1$, Focal)** | Nano (`n`) | 0.39M | **20,972 img/s** | 440 img/s |
| **Grid Spatial ($K=1$, Focal)** | Small (`s`) | 1.55M | **8,039 img/s** | 416 img/s |
| **Grid Spatial ($K=1$, Focal)** | Medium (`m`) | 6.18M | **3,547 img/s** | 394 img/s |
| ─── | ─── | ─── | ─── | ─── |
| **Grid Spatial ($K=1$, BCE)** | Nano (`n`) | 0.39M | **21,152 img/s** | 497 img/s |
| **Grid Spatial ($K=1$, BCE)** | Small (`s`) | 1.55M | **8,094 img/s** | 436 img/s |
| **Grid Spatial ($K=1$, BCE)** | Medium (`m`) | 6.18M | **3,648 img/s** | 394 img/s |
| ─── | ─── | ─── | ─── | ─── |
| **Multi-Anchor ($K=3$)** | Nano (`n`) | 0.41M | **20,967 img/s** | 434 img/s |
| **Multi-Anchor ($K=3$)** | Small (`s`) | 1.58M | **8,006 img/s** | 416 img/s |
| **Multi-Anchor ($K=3$)** | Medium (`m`) | 6.23M | **3,575 img/s** | 398 img/s |
| ─── | ─── | ─── | ─── | ─── |
| **FCOS Anchor-Free** | Nano (`n`) | 0.52M | **15,276 img/s** | 703 img/s |
| **FCOS Anchor-Free** | Small (`s`) | 1.79M | **6,868 img/s** | 650 img/s |
| **FCOS Anchor-Free** | Medium (`m`) | 7.14M | **2,988 img/s** | 603 img/s |
| ─── | ─── | ─── | ─── | ─── |
| **YOLOv8 Anchor-Free** | Nano (`n`) | 0.83M | **11,084 img/s** | 712 img/s |
| **YOLOv8 Anchor-Free** | Small (`s`) | 2.51M | **4,827 img/s** | 687 img/s |
| **YOLOv8 Anchor-Free** | Medium (`m`) | 6.47M | **2,298 img/s** | 581 img/s |

_Eval-loop throughput is wall-clock over the whole evaluation loop (PNG decoding with `num_workers=0`, target encoding in the collate function, host↔device copies, post-processing and scipy Hungarian matching); it is dominated by that overhead and varies run to run, so it is not a model-speed comparison. Pure inference times only `model(images)` on GPU-resident batches of 128 (median of repeated, synchronized runs after a ≥2 s warm-up)._ Two-Stage and Single-Stage checkpoints were not re-timed with the pure-inference protocol.
<!-- END GENERATED: throughput -->

---

## 📈 Density-Stratified Recall Sweep (Recall vs. GT Object Count $n_{gt}$)

<!-- BEGIN GENERATED: density (scripts/render_benchmark_reports.py) -->
Recall per ground-truth density bucket on the **`random`** layout (Two-Stage/Single-Stage from `benchmark/density_sweep_results.json`, Grid/Multi-Anchor from `grid_density_sweep_results.json`, FCOS from `fcos_density_sweep_results.json`, and YOLOv8 from `yolov8_density_sweep_results.json`):

| Density Bucket ($n_{gt}$) | Test Images | Two-Stage Medium | Single-Stage Medium | Grid Medium ($K=1$, BCE) | Multi-Anchor Medium ($K=3$) | FCOS Nano | FCOS Small | FCOS Medium | YOLOv8 Nano | YOLOv8 Small | YOLOv8 Medium |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 – 4 objects** | 236 | 93.6% | 94.9% | 98.9% | **99.1%** | 96.7% | 98.6% | **99.1%** | 92.7% | 94.2% | 94.1% |
| **5 – 8 objects** | 1,244 | 95.9% | 96.9% | **99.8%** | 99.5% | 96.2% | 98.5% | 98.4% | 92.2% | 94.1% | 93.9% |
| **9 – 12 objects** | 850 | 95.8% | 94.8% | **99.9%** | 99.9% | 96.2% | 98.8% | 98.6% | 92.5% | 94.5% | 94.5% |
| **13 – 16 objects** | 170 | 92.9% | 88.7% | 99.8% | **99.9%** | 96.1% | 98.3% | 98.5% | 93.1% | 94.9% | 94.5% |

The `random` layout hides the single-scale grid failure. Recall on the structured layouts (Grid/Multi-Anchor: mean over density buckets; FCOS and YOLOv8: overall layout recall):

| Layout | Grid Medium ($K=1$, BCE) | Multi-Anchor Medium ($K=3$) | FCOS Nano | FCOS Small | FCOS Medium | YOLOv8 Nano | YOLOv8 Small | YOLOv8 Medium |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `grid` | 50.13% | 50.59% | 94.96% | 98.09% | 98.44% | 92.50% | 94.45% | 94.13% |
| `words` | 43.52% | 43.73% | 97.50% | 99.37% | 99.47% | 95.93% | 97.88% | 97.78% |
| `line` | 41.59% | 42.85% | 97.49% | 99.47% | 99.34% | 95.92% | 97.87% | 97.62% |
<!-- END GENERATED: density -->

> **Crucial Architectural Breakthrough**:
> While Single-Stage Unified models suffer mild capacity-correlated degradation on dense scenes (dropping to ~84–88%), and single-scale Grid ($K=1, 3$) models collapsed to ~40–50% recall on crowded structured layouts (`grid`, `words`, `line`), **FCOS and YOLOv8 anchor-free detectors keep recall essentially flat across density buckets**. FCOS is strongest on structured-layout recall; YOLOv8 also avoids the single-scale collision collapse (exact values in the tables above).

---

## 🔬 Architectural Findings

### 1. The Crop Orientation Bug: Transpose Necessity (Two-Stage)
**The Problem**: During Two-Stage inference, the secondary classifier's accuracy on cropped detections collapsed to ~10.6% despite 78% standalone accuracy.
**The Root Cause & Fix**: Canvas generator renders upright (row-major), but raw EMNIST binaries are column-major. Adding `.transpose(-1, -2)` instantly restored classifier accuracy.

### 2. Density Degradation vs. Capacity (Single-Stage Unified)
**The Problem**: Early single-stage runs exhibited a pathological recall floor on dense canvases (9+ objects).
**The Fix**: A unified tri-head loss (`box + obj + class`) and rebalancing `pos_weight` resolved the floor, but small models still suffered from capacity saturation in dense scenes.

### 3. Receptive Field Interference in Single-Scale Grid Models (Grid & Multi-Anchor)
**The Problem**: Grid ($K=1$) and Multi-Anchor ($K=3$) models collapsed from ~99.7% recall on `random` down to ~43–50% recall on structured placements (`grid`, `words`, `line`).
**The Root Cause**: Fixed single-scale $14 \times 14$ grid cells (stride 16) suffer from severe spatial receptive field collisions when characters appear side-by-side or stacked closely in lines and word blocks.

### 4. Complete Elimination of Collisions via FCOS Multi-Scale FPN & Centerness
The comparison table shows that FCOS leads average detection precision and recall, while YOLOv8 Nano has the highest average end-to-end F1 (0.9087). Thus the anchor-free families lead different aggregate metrics even though both avoid the single-scale grid collision failure mode.

### 5. YOLOv8 Anchor-Free DFL Head
**The Solution**: The YOLOv8-inspired detector uses a C2f/SPPF/PAN body with separate class logits and distributional LTRB regression at the same two scales (28 × 28 and 14 × 14), with no anchor tensor and no objectness channel.
**The Outcome**: The trained models retain high recall across all four layouts (Nano 94.21%, Small 96.14%, Medium 95.94% average), and density recall stays in the 92–99% range rather than collapsing on structured scenes. This is below FCOS on structured recall but substantially above the single-scale grid baselines.
**The Solution**: FCOS completely abandons preset anchor boxes. It distributes detections across a multi-scale FPN (P3 at $28 \times 28$, stride 8, and P4 at $14 \times 14$, stride 16).
**The Outcome**: Closely-spaced characters are resolved on the fine $28 \times 28$ P3 level while larger structures map to P4. Combined with centerness gating $\sqrt{\sigma(\text{cls}) \times \sigma(\text{cent})}$, FCOS achieves the highest 4-layout average detection precision and recall in the master table. YOLOv8 Nano records the highest average end-to-end F1 (0.9087), so the two anchor-free designs lead different aggregate metrics while both remove the grid collision failure mode.

---

## 📁 Individual Architecture Benchmark Reports

- **[`benchmark/01_two_stage_resnet.md`](./benchmark/01_two_stage_resnet.md)** — **Stage 4 Two-Stage ResNet Detection Pipeline**
- **[`benchmark/02_single_stage_unified_resnet.md`](./benchmark/02_single_stage_unified_resnet.md)** — **Stage 5 Single-Stage Unified ResNet Detector**
- **[`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md)** — **Stage 6 Grid-Based Spatial ResNet Detector**
- **[`benchmark/04_multi_anchor_spatial_resnet.md`](./benchmark/04_multi_anchor_spatial_resnet.md)** — **Stage 7 Multi-Anchor ($K=3$) Spatial ResNet Detector**
- **[`benchmark/05_fcos_anchor_free_resnet.md`](./benchmark/05_fcos_anchor_free_resnet.md)** — **Stage 8 FCOS Anchor-Free ResNet Detector**
- **[`benchmark/06_yolov8_anchor_free_resnet.md`](./benchmark/06_yolov8_anchor_free_resnet.md)** — **YOLOv8-Inspired Anchor-Free Detector**
