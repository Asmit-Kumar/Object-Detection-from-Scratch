# Synthetic Dataset Generator

This directory contains the full pipeline for generating the synthetic object detection dataset used by this project.

## Overview

The generator creates **multi-object synthetic scenes** by compositing characters from the [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) dataset onto blank 224x224 grayscale canvases. Each generated image contains between 1 and N characters placed using one of several layout strategies.

The dataset is stored as standard PNG images alongside a `metadata.jsonl` file. Each line of the JSONL file is a self-contained record describing the objects in that image.

## Files

| File | Description |
|---|---|
| `generator.py` | Main entry point. `DatasetGenerator` class that orchestrates multi-threaded scene generation. |
| `canvas.py` | `CanvasGenerator` and `Canvas` classes. Handles character placement logic (random, grid, word-line layouts) and bounding box calculation. |
| `dataset.py` | `DatasetBuilder` -- downloads and wraps the EMNIST source data used during generation. |

## Usage

```python
from generator.generator import DatasetGenerator

gen = DatasetGenerator(
    split="bymerge",       # EMNIST split to draw characters from
    train_len=255_000,     # Number of training images to generate
    test_len=45_000,       # Number of test images to generate
    num_workers=4,         # Parallel generation threads
)
gen.generate()
```

Generated data is saved to `data/OD/train/` and `data/OD/test/`, each containing:
```
data/OD/train/
+-- images/
|   +-- 000000.png
|   +-- 000001.png
|   +-- ...
+-- metadata.jsonl
```

## Metadata Format

Each line in `metadata.jsonl` is a JSON object:
```json
{
  "image": "000042.png",
  "placement": "grid",
  "objects": [
    {"bbox": [x, y, w, h], "label": 12, "char": "C"},
    {"bbox": [x, y, w, h], "label": 38, "char": "m"}
  ]
}
```

## Layout Strategies

The generator uses weighted random placement to ensure a rich variety of scenes:

| Strategy | Probability | Description |
|---|---|---|
| `random` | 50% | Characters placed at uniformly random non-overlapping positions |
| `grid` | 35% | Characters aligned to a regular grid with small jitter |
| `words` | 10% | Characters placed in natural left-to-right word-like sequences |
| `line` | 5% | All characters on a single horizontal line |

## Why `bymerge`?

We use the **EMNIST ByMerge** split (47 classes) rather than ByClass (62 classes) for the source character pool. ByClass separates visually ambiguous character pairs like `C/c`, `O/o`, `S/s`, `W/w`, `X/x`, `Z/z` into distinct classes. Since these characters are genuinely indistinguishable at the small scale used in our synthetic scenes, training a classifier on ByClass forces the model to learn an impossible distinction and hurts real-world accuracy. ByMerge collapses these ambiguous pairs, resulting in a cleaner, more learnable label space and a significantly better-performing classifier (~89.9% vs ~86% accuracy).