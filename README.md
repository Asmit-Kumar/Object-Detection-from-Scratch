# Object Detection from Scratch (PyTorch)

An end-to-end multi-object detection and character recognition system built entirely from scratch in PyTorch — with zero third-party detection frameworks. Given a 224×224 grayscale canvas containing multiple EMNIST characters, the pipeline simultaneously localizes bounding boxes, scores object confidence, and recognizes character classes.

Trained on a custom synthetic dataset generated from EMNIST, running on an **RTX 5070 Ti**.

---

## 🏛️ Pipeline Evolution & Architecture

The project evolved through three major architectural stages:

1. **Stage 4: Two-Stage Pipeline** — Separate ResNet detector predicting bounding boxes followed by crop-slicing and passing to an independent 47-class character classifier.
2. **Stage 5: Single-Stage Unified Detector (FC Tri-Head)** — Integrated bounding box regression, objectness scoring, and class recognition into a single network with global average pooling and Hungarian (bipartite) matching.
3. **Stage 6: Grid-Based Spatial ResNet Detector (YOLO-style)** *(Current SOTA)* — Fully convolutional, anchor-free architecture operating directly on a $14 \times 14$ spatial feature grid, eliminating global pooling and FC projection overhead.

### Stage 6 Grid-Based Spatial Detector Architecture

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

---

## ⚡ Model Parameter Audit & Size Scaling

By transitioning from heavy FC heads to a lightweight $1 \times 1$ spatial convolution head, the Stage 6 Grid Spatial Detector significantly reduced parameter count while dramatically increasing classification accuracy.

| Size Preset | Backbone Channels | Detector Params | Optimal `conf` | Output Tensor Shape | Full Benchmark Report |
|:---|:---|:---:|:---:|:---:|:---|
| **Nano (`n`)** | `[32, 64, 64, 128]` | **0.39M** (392,788) | `0.90` | `(B, 14, 14, 52)` | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| **Small (`s`)** | `[64, 128, 128, 256]` | **1.55M** (1,553,524) | `0.90` | `(B, 14, 14, 52)` | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |
| **Medium (`m`)** | `[128, 256, 256, 512]` | **6.18M** (6,178,996) | `0.90` | `(B, 14, 14, 52)` | [`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md) |

---

## 🔬 Key Architectural Discoveries

### 1. Classification Breakthrough via Spatial Grounding
* **Stage 5 Bottleneck**: Global average pooling (`AdaptiveAvgPool2d((2,2))`) collapsed spatial coordinate maps into abstract 1D vectors before FC projection, leading to an abysmal **36.34% classification accuracy on Nano**.
* **Stage 6 Solution**: Replacing global pooling with a dense $14 \times 14$ spatial Conv head maintains local receptive field grounding, surging Nano classification accuracy to **88.30% (+51.96% gain)** and Small/Medium to **~89.6%**.

### 2. Density Degradation & The Visual Crowding Constraint
* **Sparse / Random Layouts**: On `random` placement, Stage 6 models achieve near-perfect performance: **>99.7% Recall even up to 13–16 objects**, completely resolving slot competition for scattered objects.
* **Structured / Dense Layouts (`grid`, `words`, `line`)**: Detection Precision/Recall drops sharply down to **~40–50%**.
* **Root Cause (Visual Crowding)**: Detailed instrumentation of the `collate_fn` dataset mapping confirmed **exactly 0 grid-cell collisions** across all layouts (target centers never overwrite one another within the same $16 \times 16$ grid cell). Instead, the model's collapse on structured text is caused by **receptive field interference (visual crowding)**. In layouts like `grid`, `words`, and `line`, characters are packed densely next to one another. The ResNet's local receptive field blends these dense visual features together, causing the bounding box and classification heads to fail to disentangle adjacent targets, even though they technically occupy distinct logical grid cells.

---

## 📊 Stage 1 -- Standalone Character Classifier

A lightweight **ResNet** backbone (370,351 parameters / **0.37M**) with a fully-connected classification head. Input: 28×28 grayscale crop. Output: probability distribution over 47 character classes (EMNIST ByMerge).

| Split | Val Accuracy |
|---|---|
| ByClass (62 classes) | ~86.0% |
| **ByMerge (47 classes)** | **~89.9%** |

---

## 🖼️ Dataset & Layout Placement Strategies

The synthetic dataset is generated from scratch using EMNIST characters composited onto blank 224×224 canvases. See [`generator/README.md`](generator/README.md) for full documentation on the generation pipeline, layout strategies, and metadata format.

**Dataset size**: 255,000 training / 45,000 test images.

| Layout Strategy | Sample Scene 1 | Sample Scene 2 | Description |
|:---|:---:|:---:|:---|
| **🎲 Random** | ![](result/dataset/random_1.png) | ![](result/dataset/random_2.png) | Unconstrained random spatial locations across the 224×224 canvas. |
| **📐 Grid** | ![](result/dataset/grid_1.png) | ![](result/dataset/grid_2.png) | Structured rows and columns forming tabular or grid patterns. |
| **📝 Words** | ![](result/dataset/words_1.png) | ![](result/dataset/words_2.png) | Character sequences mimicking multi-character word groupings. |
| **📏 Line** | ![](result/dataset/line_1.png) | ![](result/dataset/line_2.png) | Single horizontal line text layout. |

---

## 📁 Repository & Benchmark Index

- **[`BENCHMARK.md`](./BENCHMARK.md)** — **Master Benchmark Hub & Cross-Stage Comparison**
- **[`benchmark/01_two_stage_resnet.md`](./benchmark/01_two_stage_resnet.md)** — **Stage 4 Two-Stage ResNet Detection Pipeline**
- **[`benchmark/02_single_stage_unified_resnet.md`](./benchmark/02_single_stage_unified_resnet.md)** — **Stage 5 Single-Stage Unified ResNet Detector**
- **[`benchmark/03_grid_based_spatial_resnet.md`](./benchmark/03_grid_based_spatial_resnet.md)** — **Stage 6 Grid-Based Spatial ResNet Detector**

---

## 🛠️ Setup & Execution

```bash
pip install -r requirements.txt
```

To regenerate the synthetic dataset from scratch:
```bash
python -c "from generator.generator import DatasetGenerator; DatasetGenerator().generate()"
```

To run the pipeline verification test:
```bash
python scripts/verify_pipeline.py
```