"""
Density-Stratified Benchmark Sweep Script for Grid and FCOS Detectors.

Evaluates recall drop-off across object density buckets:
  - Bucket 1: 1 - 4 objects/image (sparse)
  - Bucket 2: 5 - 8 objects/image (medium-low)
  - Bucket 3: 9 - 12 objects/image (medium-high)
  - Bucket 4: 13 - 16 objects/image (dense)

Use ``--detector grid`` to run across spatial grid variants:
  - grid_focal_stage: 1_grid_detector_{n, s, m} (conf = 0.50)
  - grid_bce_stage: grid_detector_{n, s, m} (conf = 0.90)
  - multi_anchor_stage: 3_grid_detector_{n, s, m} (conf = 0.95)

Use ``--detector fcos`` to run across FCOS Nano, Small, and Medium models.
"""
import argparse
import sys
import json
from pathlib import Path

import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))

import torch
from models import load_detector, load_fcos_detector
from models.object_detector_res import ObjectDetectorResNet
from utils.dataset import get_detection_loaders
from utils.pipeline import _decode_fcos_candidates
from utils.trainer import evaluate_density_sweep

BENCHMARK_ROOT = 'data/OD_benchmark'
PLACEMENTS = ['random', 'grid', 'words', 'line']
BUCKETS = [(1, 4), (5, 8), (9, 12), (13, 16)]

VARIANTS = [
    {'name': 'grid_focal_stage', 'prefix': '1_grid_detector_', 'conf': 0.50, 'sizes': ['n', 's', 'm']},
    {'name': 'grid_bce_stage', 'prefix': 'grid_detector_', 'conf': 0.90, 'sizes': ['n', 's', 'm']},
    {'name': 'multi_anchor_stage', 'prefix': '3_grid_detector_', 'conf': 0.95, 'sizes': ['n', 's', 'm']},
]


def _pairwise_iou_xywh(box1, box2):
    """Compute IoU between predicted boxes [x, y, w, h] and ground-truth boxes."""
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)
    px = box1[:, 0:1]
    py = box1[:, 1:2]
    pw = box1[:, 2:3]
    ph = box1[:, 3:4]
    gx = box2[:, 0]
    gy = box2[:, 1]
    gw = box2[:, 2]
    gh = box2[:, 3]
    ix1 = np.maximum(px, gx)
    iy1 = np.maximum(py, gy)
    ix2 = np.minimum(px + pw, gx + gw)
    iy2 = np.minimum(py + ph, gy + gh)
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return np.where(union > 0, inter / union, 0.0)


def evaluate_fcos_density_sweep(model, loader, device, conf_threshold=0.50, iou_threshold=0.50, buckets=BUCKETS):
    """Evaluate FCOS recall by GT-object density bucket using decoded candidates."""
    model.eval()
    records = []

    with torch.no_grad():
        for images, targets_batch, annotations in loader:
            gt_boxes_batch, gt_labels_batch = annotations
            images = images.to(device)
            outputs = model(images)
            candidates = _decode_fcos_candidates(model, outputs, score_threshold=conf_threshold, iou_threshold=0.45)

            for b in range(len(candidates)):
                p_boxes, p_scores, p_classes = candidates[b]
                g_boxes = gt_boxes_batch[b].cpu().numpy()
                n_gt = len(g_boxes)
                if n_gt == 0:
                    continue

                preds_boxes = p_boxes.cpu().numpy()
                preds_scores = p_scores.cpu().numpy()
                iou_mat = _pairwise_iou_xywh(preds_boxes, g_boxes)
                records.append({
                    'p_scores': preds_scores,
                    'iou_mat': iou_mat,
                    'n_gt': n_gt,
                })

    bucket_results = []
    for (min_gt, max_gt) in buckets:
        tp = 0
        fp = 0
        fn = 0
        n_images = 0

        for r in records:
            n_gt = r['n_gt']
            if not (min_gt <= n_gt <= max_gt):
                continue
            n_images += 1
            scores = r['p_scores']
            iou_mat = r['iou_mat']
            n_pred = len(scores)

            if n_pred == 0:
                fn += n_gt
                continue

            order = np.argsort(-scores)
            matched = set()
            for idx in order:
                best_j = int(np.argmax(iou_mat[idx]))
                if iou_mat[idx, best_j] >= iou_threshold and best_j not in matched:
                    matched.add(best_j)
                    tp += 1
                else:
                    fp += 1
            fn += n_gt - len(matched)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        bucket_results.append({
            'bucket': f"{min_gt}-{max_gt}",
            'n_images': n_images,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': tp,
            'fp': fp,
            'fn': fn,
        })

    return bucket_results


def run_full_density_sweep(detector: str = 'grid'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sweep_results = {}

    print("=" * 90)
    print("  FULL BENCHMARK DENSITY-STRATIFIED SWEEP (ALL LAYOUTS x DENSITY BUCKETS)")
    print("=" * 90)

    if detector == 'fcos':
        configs = [
            {'size': 'n', 'name': 'FCOS Nano (0.52M)', 'conf': 0.50},
            {'size': 's', 'name': 'FCOS Small (1.79M)', 'conf': 0.45},
            {'size': 'm', 'name': 'FCOS Medium (7.14M)', 'conf': 0.50},
        ]

        for cfg in configs:
            size = cfg['size']
            name = cfg['name']
            conf_t = cfg['conf']
            sweep_results[size] = {'name': name, 'conf': conf_t, 'layouts': {}}
            print(f"\n--- {name.upper()} (conf = {conf_t:.2f}) ---")
            ckpt_path = f'weights/fcos_{size}_28x14_best.pth'
            if not Path(ckpt_path).exists():
                ckpt_path = f'checkpoint/fcos_{size}_28x14_best.pth'
            if not Path(ckpt_path).exists():
                print(f"Skipping {ckpt_path} (not found)")
                continue

            model = load_fcos_detector(ckpt_path, device=device, size=size)
            for p in PLACEMENTS:
                data_path = f"{BENCHMARK_ROOT}/{p}"
                loader = get_detection_loaders(data_path, batch_size=128, test_only=True, num_workers=0, fcos=True)
                res = evaluate_fcos_density_sweep(model, loader, device=device, conf_threshold=conf_t, iou_threshold=0.50, buckets=BUCKETS)
                sweep_results[size]['layouts'][p] = res
                print(f"  [{p:<6}] " + " | ".join([f"{b['bucket']}: R={b['recall']*100:.1f}% ({b['n_images']} imgs)" for b in res]))

        out_path = Path('benchmark/fcos_density_sweep_results.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSaved FCOS density sweep results to '{out_path}'")
        return

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

    out_path = Path('benchmark/grid_density_sweep_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved full benchmark density sweep results to '{out_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the density sweep for the grid or FCOS detector variants.')
    parser.add_argument('--detector', choices=['grid', 'fcos'], default='grid', help='Which detector family to evaluate.')
    args = parser.parse_args()
    run_full_density_sweep(detector=args.detector)

