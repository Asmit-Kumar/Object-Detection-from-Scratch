"""
Density-Stratified Benchmark Sweep Script (Recall vs. GT Objects per Image n_gt).

Evaluates recall drop-off across object density buckets:
  - Bucket 1: 1 - 4 objects/image (sparse)
  - Bucket 2: 5 - 8 objects/image (medium-low)
  - Bucket 3: 9 - 12 objects/image (medium-high)
  - Bucket 4: 13 - 16 objects/image (dense)

Runs across all spatial grid variants:
  - grid_focal_stage: 1_grid_detector_{n, s, m} (conf = 0.50)
  - grid_bce_stage: grid_detector_{n, s, m} (conf = 0.90)
  - multi_anchor_stage: 3_grid_detector_{n, s, m} (conf = 0.95)
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

BENCHMARK_ROOT = 'data/OD_benchmark'
PLACEMENTS = ['random', 'grid', 'words', 'line']
BUCKETS = [(1, 4), (5, 8), (9, 12), (13, 16)]

VARIANTS = [
    {'name': 'grid_focal_stage', 'prefix': '1_grid_detector_', 'conf': 0.50, 'sizes': ['n', 's', 'm']},
    {'name': 'grid_bce_stage', 'prefix': 'grid_detector_', 'conf': 0.90, 'sizes': ['n', 's', 'm']},
    {'name': 'multi_anchor_stage', 'prefix': '3_grid_detector_', 'conf': 0.95, 'sizes': ['n', 's', 'm']},
]

def run_full_density_sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sweep_results = {}

    print("=" * 90)
    print("  FULL BENCHMARK DENSITY-STRATIFIED SWEEP (ALL LAYOUTS x DENSITY BUCKETS)")
    print("=" * 90)

    for v in VARIANTS:
        v_name = v['name']
        prefix = v['prefix']
        conf_t = v['conf']
        v_sizes = v.get('sizes', ['n', 's', 'm'])
        sweep_results[v_name] = {}

        print(f"\n--- {v_name.upper()} MODELS (conf = {conf_t:.2f}) ---")
        for size in v_sizes:
            weights_p = Path(f'weights/{prefix}{size}_best.pth')
            ckpt_p = Path(f'checkpoint/{prefix}{size}_best.pth')
            ckpt_path = str(weights_p) if weights_p.exists() else str(ckpt_p)
            anchors_wh = None
            sd_state = None
            if Path(ckpt_path).exists():
                sd = torch.load(ckpt_path, map_location=device, weights_only=False)
                sd_state = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
                anchors_wh = sd.get('anchors_wh', None) if isinstance(sd, dict) else None
            else:
                print(f"Skipping {ckpt_path} (not found)")
                continue

            num_anchors = anchors_wh.shape[0] if anchors_wh is not None else (3 if prefix.startswith('3_') else 1)
            if anchors_wh is None and num_anchors == 1:
                anchors_wh = torch.tensor([[16.0, 16.0]])

            stem, ch, blocks, pool = ObjectDetectorResNet.CONFIGS[size]
            model = ObjectDetectorResNet(channels=ch, blocks=blocks, num_anchors=num_anchors).to(device)
            model.load_state_dict(sd_state)
            
            sweep_results[v_name][size] = {}
            for p in PLACEMENTS:
                data_path = f"{BENCHMARK_ROOT}/{p}"
                print(f"\n[Evaluating {v_name} {size.upper()} on layout: '{p}']")
                loader = get_detection_loaders(data_path, batch_size=128, test_only=True, num_workers=0, anchors_wh=anchors_wh)
                res = evaluate_density_sweep(model, loader, device=device, conf_threshold=conf_t, iou_threshold=0.50, buckets=BUCKETS)
                sweep_results[v_name][size][p] = res

    # Save to JSON
    out_path = Path('benchmark/grid_density_sweep_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved full benchmark density sweep results to '{out_path}'")


if __name__ == '__main__':
    run_full_density_sweep()


