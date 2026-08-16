"""
Diagnostic Script: Spatial Grid Cell Coverage & Placement Lattice Analysis.

Instruments the 14x14 spatial grid target assignments across benchmark placement layouts
('random', 'grid', 'words', 'line') in data/OD_benchmark/.

Computes:
  1. Unique active cells (out of 196)
  2. Normalized spatial entropy & Gini coefficient of cell hit distribution
  3. Concentration / Top-10% cell mass ratio
  4. Zero-hit cell count (blind spots)
  5. Generates 4-panel visual heatmap comparing spatial distributions.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from utils.reader import DataReader

BENCHMARK_ROOT = Path("data/OD_benchmark")
PLACEMENTS = ["random", "grid", "words", "line"]
GRID_SIZE = 14  # S = 14
CANVAS_SIZE = 224
CELL_SIZE = CANVAS_SIZE / GRID_SIZE  # 16.0 px


def compute_gini(x: np.ndarray) -> float:
    """Calculate Gini coefficient of an array."""
    x = x.flatten().astype(np.float64)
    if np.all(x == 0):
        return 0.0
    # Values cannot be negative
    diffs = np.abs(x[:, None] - x[None, :])
    return float(np.sum(diffs) / (2.0 * len(x) * np.sum(x)))


def compute_normalized_entropy(x: np.ndarray) -> float:
    """Calculate normalized Shannon entropy in [0, 1]. 1.0 = perfectly uniform."""
    x = x.flatten().astype(np.float64)
    total = np.sum(x)
    if total == 0:
        return 0.0
    p = x / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    max_entropy = np.log2(len(x))
    return float(entropy / max_entropy)


def analyze_layout(placement: str) -> dict:
    """Analyze spatial grid assignment frequency for a single placement layout."""
    data_dir = BENCHMARK_ROOT / placement / "test"
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    reader = DataReader(root=data_dir)
    total_images = len(reader.records)
    total_boxes = 0

    # 14x14 grid accumulation
    heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)

    for rec in reader.records:
        boxes = rec.boxes  # [[x, y, w, h], ...]
        total_boxes += len(boxes)
        for b in boxes:
            cx = b[0] + b[2] / 2.0
            cy = b[1] + b[3] / 2.0
            gx = min(GRID_SIZE - 1, max(0, int(cx // CELL_SIZE)))
            gy = min(GRID_SIZE - 1, max(0, int(cy // CELL_SIZE)))
            heatmap[gy, gx] += 1

    total_cells = GRID_SIZE * GRID_SIZE
    active_cells = int(np.sum(heatmap > 0))
    zero_hit_cells = int(np.sum(heatmap == 0))
    coverage_pct = (active_cells / total_cells) * 100.0

    # Mass concentration: top 10% cells (~20 cells)
    sorted_hits = np.sort(heatmap.flatten())[::-1]
    top_10pct_k = max(1, int(0.10 * total_cells))
    top_10pct_mass = float(np.sum(sorted_hits[:top_10pct_k]) / max(1, total_boxes) * 100.0)
    top_25pct_k = max(1, int(0.25 * total_cells))
    top_25pct_mass = float(np.sum(sorted_hits[:top_25pct_k]) / max(1, total_boxes) * 100.0)

    gini = compute_gini(heatmap)
    entropy_norm = compute_normalized_entropy(heatmap)

    return {
        "placement": placement,
        "total_images": total_images,
        "total_boxes": total_boxes,
        "active_cells": active_cells,
        "zero_hit_cells": zero_hit_cells,
        "coverage_pct": round(coverage_pct, 2),
        "mean_hits_per_cell": round(float(np.mean(heatmap)), 2),
        "std_hits_per_cell": round(float(np.std(heatmap)), 2),
        "max_hits_in_single_cell": int(np.max(heatmap)),
        "min_hits_in_active_cell": int(np.min(heatmap[heatmap > 0])) if active_cells > 0 else 0,
        "top_10pct_cell_mass": round(top_10pct_mass, 2),
        "top_25pct_cell_mass": round(top_25pct_mass, 2),
        "gini_coefficient": round(gini, 4),
        "spatial_entropy": round(entropy_norm, 4),
        "heatmap": heatmap.tolist(),
    }


def generate_heatmap_plot(results: dict, out_path: Path):
    """Render and save 4-panel comparison heatmaps."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    fig.suptitle(
        "Spatial Grid Cell Coverage Frequency Heatmap (14x14 Grid, S=14)",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )

    # Determine global max for consistent color scaling or use per-panel
    for idx, p in enumerate(PLACEMENTS):
        ax = axes[idx]
        data = results[p]
        hm = np.array(data["heatmap"])

        im = ax.imshow(hm, cmap="inferno", interpolation="nearest")
        ax.set_title(
            f"{p.upper()}\nActive: {data['active_cells']}/196 ({data['coverage_pct']}%)\n"
            f"Entropy: {data['spatial_entropy']} | Gini: {data['gini_coefficient']}",
            fontsize=12,
            fontweight="semibold",
        )
        ax.set_xticks(range(0, GRID_SIZE, 2))
        ax.set_yticks(range(0, GRID_SIZE, 2))
        ax.set_xlabel("Grid X (gx)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Grid Y (gy)", fontsize=10)

        # Annotate top stats
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Visual] Saved spatial coverage heatmap to: {out_path}")


def main():
    print("=" * 80)
    print("  SPATIAL GRID CELL COVERAGE & FREQUENCY ANALYSIS")
    print("=" * 80)

    results = {}
    for p in PLACEMENTS:
        res = analyze_layout(p)
        results[p] = res

    # Print summary table
    print(f"\n{'Layout':<10} {'Active Cells':<14} {'Coverage %':<12} {'Zero Hits':<12} {'Entropy (0-1)':<15} {'Gini Coeff':<12} {'Top-10% Mass':<14}")
    print("-" * 90)
    for p in PLACEMENTS:
        r = results[p]
        print(
            f"{p:<10} "
            f"{r['active_cells']}/{GRID_SIZE*GRID_SIZE:<10} "
            f"{r['coverage_pct']:<12.1f} "
            f"{r['zero_hit_cells']:<12} "
            f"{r['spatial_entropy']:<15.4f} "
            f"{r['gini_coefficient']:<12.4f} "
            f"{r['top_10pct_cell_mass']:<14.1f}%"
        )
    print("=" * 90)

    # Save JSON summary
    json_path = Path("benchmark/cell_coverage_results.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw numerical analysis to: {json_path}")

    # Generate visual figure
    img_path = Path("result/analysis/cell_coverage_heatmap.png")
    generate_heatmap_plot(results, img_path)


if __name__ == "__main__":
    main()
