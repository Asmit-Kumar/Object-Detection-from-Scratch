# Benchmark & Visualization Scripts

This directory contains standalone execution scripts used for generating evaluation benchmarks, dataset visuals, and model prediction grids. 

> **Note**: Diagnostic, scratch, and one-off extraction scripts (e.g., `check_crop_alignment.py`, `eval_diagnostics.py`) are deliberately kept local and excluded from version control via `.gitignore`.

## 📊 Evaluation & Benchmarking

These scripts evaluate models across the `data/OD_benchmark/` dataset (10,000 images stratified across 4 layout types).

- **`run_full_benchmark.py`**
  Evaluates the **Two-Stage** detection pipeline (Detector + independent Classifier). Computes bounding box precision/recall, secondary crop classification accuracy, and end-to-end F1 scores.
  
- **`run_single_stage_benchmark.py`**
  Evaluates the **Single-Stage Unified** detector (Tri-head). Handles dense predictions, optimal confidence thresholds, and computes 3-level evaluation metrics (Det / Cls / E2E).

- **`run_density_sweep.py`**
  A specialized evaluation script that buckets test images by the ground-truth object count to measure how detection recall degrades as canvas clutter and character density increases.

## 🖼️ Visualization Generators

These scripts render output grids with bounding boxes, predicted classes, and confidence scores, saving them to `result/benchmark/`.

- **`generate_dataset_visuals.py`**
  Visualizes raw ground-truth EMNIST canvas images directly from the dataset generator, serving as a baseline sanity check.
  
- **`generate_benchmark_visuals.py`**
  Generates prediction visualization grids for the **Two-Stage** pipeline across varying model sizes (Nano, Small, Medium).

- **`generate_single_stage_visuals.py`**
  Generates prediction visualization grids for the **Single-Stage** models. Displays actual objectness confidence (`det_c`) alongside class confidence (`cls_c`) for NMS-filtered boxes.

## 🧪 Pipeline Utilities

- **`verify_pipeline.py`**
  A quick-run sanity check script used to verify that the end-to-end forward pass, loss calculation, and dataset rendering are functioning correctly without running a full training loop or benchmark suite.
