# Object Detection from Scratch (PyTorch)

An end-to-end, **two-stage** scene understanding pipeline built entirely from scratch in PyTorch -- no off-the-shelf detection frameworks. Given a 224x224 grayscale image containing multiple characters, the pipeline detects every character's bounding box and classifies it.

Trained on a custom synthetic dataset generated from EMNIST, running on an **RTX 5070 Ti**.

> **Note**: The end-to-end inference pipeline (combining detector + classifier into a single runnable script) is planned for a future update. The two stages are currently trained and evaluated independently via the notebooks.

---

## Pipeline Architecture & Workflow

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
    Hungarian Matching                                 Confidence Threshold
            │                                               │
      Detection Loss                                       NMS
 (Huber + BCE w/ pos_weight)                                │
                                                       Final Boxes
                                                            │
                                                   Crop, Pad & Resize
                                                            │
                                                  Character Classifier
                                                      (47 classes)
```

---

## Stage 1 -- Character Classifier

### Architecture
A lightweight **ResNet** backbone (370,351 parameters / **0.37M**) with a fully-connected classification head. Input: 28x28 grayscale crop. Output: probability distribution over 47 character classes.

### EMNIST Split -- Why `bymerge`?

We evaluated both the **ByClass** (62 classes) and **ByMerge** (47 classes) EMNIST splits. ByClass treats visually ambiguous pairs (`C/c`, `O/o`, `S/s`, `V/v`, `W/w`, `X/x`, `Z/z`) as separate classes. At our image scale, these pairs are genuinely indistinguishable, which means training on ByClass forces the model to learn an impossible distinction. ByMerge collapses these pairs into a single class, giving the model a clean, learnable label space.

| Split | Val Accuracy |
|---|---|
| ByClass (62 classes) | ~86.0% |
| **ByMerge (47 classes)** | **~89.9%** |

The ~4% gap is purely from removing the impossible ambiguous pairs -- same model, same hyperparameters.

### Training Journey

We went through several optimizer and loss configurations to push past the 89% ceiling:

- Started with **SGD + StepLR** -- slow convergence, settled around 84%.
- Switched to **Adam + OneCycleLR** -- faster convergence and significantly better generalization.
- Added **GPU-side augmentations** (random affine, erasing) -- improved robustness to varied character styles.
- Final config: `Adam`, `OneCycleLR`, `lr=4e-5`, 30 epochs, CrossEntropyLoss.

**Best result: 89.9% validation accuracy** in 30 epochs (~8.4s/epoch on RTX 5070 Ti).

---

## Stage 2 -- Object Detector

### Architecture
A **ResNet-based** backbone with a configurable multi-slot detection head. The model predicts `N` fixed slots, each producing a `[x, y, w, h, confidence]` vector.

The head was refactored from a single monolithic FC layer into a two-branch structure:

```
Backbone (ResNet blocks)
        |
  AdaptiveAvgPool2d(2x2)
        |
     Flatten
        |
    shared_fc     Linear -> ReLU -> Linear -> ReLU  (shared hidden)
        |
   +---------+
   |         |
box_head   obj_head     (thin final linear projections)
  N x 4      N x 1      [x,y,w,h]  /  confidence logit
   |         |
   +-- cat --+  -> (B, N, 5)
```

Three size variants are available via `get_detector(size=...)`:

| Size | Channels | Detector Params | Combined Pipeline Params (Det + Cls) |
|------|----------|:---:|:---:|
| n (Nano)   | `[32, 64, 64, 128]`   | **0.57M** (565K) | **0.94M** (936K) |
| s (Small)  | `[64, 128, 128, 256]` | **2.23M** (2.2M) | **2.60M** (2.6M) |
| m (Medium) | `[128, 256, 256, 512]` | **8.84M** (8.8M) | **9.21M** (9.2M) |

### From Greedy to Hungarian Matching

The most critical algorithmic improvement was replacing the **greedy argmax assignment** with **Hungarian (bipartite) matching** in the training loss.

#### The Problem with Greedy Matching
The original loss assigned each ground-truth box to the prediction slot with the highest IoU:
```python
_, matched_idx = iou.max(dim=1)   # pure greedy argmax
```
This caused **slot collisions** -- multiple ground-truth objects fighting over the same predicted slot. This is fundamentally a regression task mismatch: the model had no guarantee that each object would get a dedicated slot.

#### Hungarian Matching
Replaced with `scipy.optimize.linear_sum_assignment`, which solves the optimal 1-to-1 bipartite assignment:
```python
pred_idx, gt_idx = linear_sum_assignment(-iou_cost)  # per image in batch
```
This **mathematically guarantees** that every ground-truth object is assigned a unique predicted slot -- zero slot collisions by construction.

#### Impact

| Run | Matching | Val IoU |
|---|---|---|
| Baseline (greedy) | Argmax | 0.833 |
| Greedy (later runs) | Argmax | 0.803-0.810 |
| **Hungarian** | **Bipartite** | **0.903** |

The jump from ~0.81 to **0.903** in a single swap shows just how much the slot collision problem was bottlenecking performance.

### Performance Optimization

The initial Hungarian implementation had a critical bottleneck: running `.cpu().numpy()` and `torch.as_tensor(..., device=device)` **per image inside the batch loop** caused hundreds of GPU synchronization stalls per batch (200ms -> 90ms just from syncs).

Fixed by:
1. **Single bulk transfer**: `iou.detach().cpu().numpy()` once before the loop.
2. **Single bulk push**: `torch.from_numpy(matched_idx_np).to(device)` once after.
3. **Fast-path for single objects**: skip scipy entirely, use `np.argmin` for `n_valid == 1`.
4. **All diagnostics on numpy arrays**: moved the match-ratio diagnostic off GPU entirely.

Result: loss computation time dropped from ~200ms -> **~5ms per batch**. Full epoch time from ~170s -> **~70s**.

### Size Scaling Results

All three variants trained with identical config (AdamW, OneCycleLR, lr=4e-5, 60 epochs, 255k train images).

#### Validation F1 and IoU

| Size | Best Val F1 (conf=0.5) | Best Val IoU |
|------|------------------------|--------------|
| n (Nano)   | 0.8619                 | 0.8034       |
| s (Small)  | 0.9356                 | 0.8745       |
| **m (Medium)** | **0.9399**             | **0.9133**   |

#### Test-Set @ Optimal Confidence Threshold (IoU threshold=0.5, 45k images)

| Size | Conf | P      | R      | F1     |
|------|------|--------|--------|--------|
| n (Nano)   | 0.70 | 0.8990 | 0.8443 | 0.8708 |
| s (Small)  | 0.70 | 0.9611 | 0.9394 | 0.9501 |
| **m (Medium)** | **0.70** | **0.9728** | **0.9500** | **0.9613** |

Increasing backbone capacity consistently improved localisation and detection quality. The Nano model was under-parameterised for the task, while the Medium model achieved the best localisation (0.9133 IoU) and detection performance (0.9613 F1). Performance gains began to diminish beyond the Small model, indicating the onset of diminishing returns.

#### Model Training Curves (Clean Dataset)

**Nano Model (0.57M params)**
![Nano Detector Training Curves](result/nano_new_ds.png)

**Small Model (2.23M params)**
![Small Detector Training Curves](result/small_new_ds.png)

**Medium Model (8.84M params)**
![Medium Detector Training Curves](result/medium_new_ds.png)

### Training Config
- **Model**: `ObjectDetectorResNet` (via `get_detector(size=...)`)
- **Optimizer**: `AdamW`
- **Scheduler**: `OneCycleLR`
- **Loss**: Huber (box regression) + BCE with logits (confidence, dynamic `pos_weight`) with Hungarian assignment
- **Epochs**: 60
- **Device**: RTX 5070 Ti

### Validation Metric & Checkpointing

The training loop monitors **F1 @ conf=0.5, IoU=0.5** as the primary checkpoint metric instead of greedy IoU. Greedy IoU matches the closest prediction to each GT box regardless of confidence, which means a model that predicts 24 low-confidence boxes can score well while being useless in practice. F1 directly penalises both false positives and false negatives, so `best_model.pth` always reflects the most deployable checkpoint.

A **dynamic `pos_weight`** is applied to the BCE confidence loss, computed per-batch as `(total_slots - n_positive) / n_positive`. This rebalances the gradient in sparse scenes (few objects, many background slots) to prevent the model from becoming overly conservative.

---

## Dataset & Layout Placement Strategies

The synthetic dataset is generated from scratch using EMNIST characters composited onto blank 224×224 canvases. See [`generator/README.md`](generator/README.md) for full documentation on the generation pipeline, layout strategies, and metadata format.

**Dataset size**: 255,000 training / 45,000 test images.

### Generated Layout Examples

The generator utilizes 4 placement strategies to emulate real-world scene structures:

| Layout Strategy | Sample Scene 1 | Sample Scene 2 | Description |
|:---|:---:|:---:|:---|
| **🎲 Random** | ![](result/dataset/random_1.png) | ![](result/dataset/random_2.png) | Unconstrained random spatial locations across the 224×224 canvas. |
| **📐 Grid** | ![](result/dataset/grid_1.png) | ![](result/dataset/grid_2.png) | Structured rows and columns forming tabular or grid patterns. |
| **📝 Words** | ![](result/dataset/words_1.png) | ![](result/dataset/words_2.png) | Character sequences mimicking multi-character word groupings. |
| **📏 Line** | ![](result/dataset/line_1.png) | ![](result/dataset/line_2.png) | Single horizontal line text layout. |

---

## Project Structure

```
OD-From-Scratch/
+-- generator/          # Synthetic dataset generation pipeline
|   +-- README.md       # Dataset generation docs
|   +-- generator.py    # DatasetGenerator -- multi-threaded scene creation
|   +-- canvas.py       # Character placement and bbox calculation
|   +-- dataset.py      # EMNIST source data loader
+-- models/             # Model architectures
|   +-- object_detector_res.py        # ResNet multi-slot detector
|   +-- character_classifier_resnet.py # ResNet character classifier
+-- utils/              # Core training utilities
|   +-- dataset.py      # DetectionDataset, DataLoader factories
|   +-- reader.py       # DataReader -- reads PNG images + metadata.jsonl
|   +-- losses.py       # DetectionLoss with Hungarian matching
|   +-- trainer.py      # Training + evaluation loops
|   +-- visualizer.py   # Detection visualization helpers
|   +-- callbacks.py    # Training callbacks (checkpointing etc.)
|   +-- logger.py       # RunLogger -- JSON training logs
+-- notebooks/
|   +-- detection.ipynb     # Detector training + eval notebook
|   +-- classifier.ipynb    # Classifier training notebook
+-- scripts/
|   +-- dataset_setup.py    # One-time EMNIST download and verification
|   +-- verify_pipeline.py  # End-to-end pipeline smoke test
+-- data/OD/            # Generated dataset (not in repo)
+-- logs/               # JSON training run logs (per model)
+-- weights/            # Saved model checkpoints
+-- archive/            # Deprecated / legacy files
+-- requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

To regenerate the dataset from scratch:
```bash
# See generator/README.md for full options
python -c "from generator.generator import DatasetGenerator; DatasetGenerator().generate()"
```

To run the pipeline smoke test:
```bash
python scripts/verify_pipeline.py
```