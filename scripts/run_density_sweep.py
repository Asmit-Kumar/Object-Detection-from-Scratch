"""
Density-Stratified Benchmark Sweep Script (Recall vs. GT Objects per Image n_gt).

Evaluates recall drop-off across object density buckets:
  - Bucket 1: 1 - 4 objects/image (sparse)
  - Bucket 2: 5 - 8 objects/image (medium-low)
  - Bucket 3: 9 - 12 objects/image (medium-high)
  - Bucket 4: 13 - 16 objects/image (dense)

Runs across Two-Stage Pipeline models, Single-Stage Unified models, and Grid-Based Spatial models.
"""
import sys
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))

import torch
from models import load_detector
from models.object_detector_res import ObjectDetectorResNet
from utils.dataset import get_detection_loaders
from utils.trainer import evaluate_density_sweep

BENCHMARK_PATH = 'data/OD_benchmark/random'
BUCKETS = [(1, 4), (5, 8), (9, 12), (13, 16)]


def run_full_density_sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_detection_loaders(BENCHMARK_PATH, batch_size=128, test_only=True, num_workers=0)
    
    sweep_results = {'grid_stage': {}}

    print("=" * 80)
    print("  GRID-BASED SPATIAL DETECTOR DENSITY-STRATIFIED SWEEP")
    print("=" * 80)

    # Grid-Based Models (Stage 6)
    print("\n--- GRID-BASED SPATIAL MODELS (conf = 0.90) ---")
    for size in ['n', 's', 'm']:
        stem, ch, blocks, pool = ObjectDetectorResNet.CONFIGS[size]
        model = ObjectDetectorResNet(channels=ch, blocks=blocks).to(device)
        ckpt_path = f'checkpoint/grid_detector_{size}_best.pth'
        sd = torch.load(ckpt_path, map_location=device)
        sd_state = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd_state)
        res = evaluate_density_sweep(model, loader, device=device, conf_threshold=0.90, iou_threshold=0.50, buckets=BUCKETS)
        sweep_results['grid_stage'][size] = res

    # Print Summary Table
    print("\n" + "=" * 80)
    print("  GRID DETECTOR DENSITY-STRATIFIED RECALL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Density Bucket':<16} {'Grid Nano':<14} {'Grid Small':<14} {'Grid Medium':<14}")
    print("-" * 80)

    for i, (min_b, max_b) in enumerate(BUCKETS):
        b_label = f"{min_b}-{max_b} objects"
        r_g_n = sweep_results['grid_stage']['n'][i]['recall']
        r_g_s = sweep_results['grid_stage']['s'][i]['recall']
        r_g_m = sweep_results['grid_stage']['m'][i]['recall']

        print(f"{b_label:<16} {r_g_n:<14.4f} {r_g_s:<14.4f} {r_g_m:<14.4f}")

    print("=" * 80)

    # Save to JSON
    out_path = Path('benchmark/grid_density_sweep_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved raw grid density sweep results to '{out_path}'")


if __name__ == '__main__':
    run_full_density_sweep()
