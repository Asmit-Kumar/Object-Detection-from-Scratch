"""
Single-Stage & Multi-Anchor Spatial Detector Benchmark Evaluation Script.

Evaluates single-stage spatial grid models across all 4 placement layouts
('random', 'grid', 'words', 'line') in data/OD_benchmark/:
  1. Grid (K=1, Focal): 1_grid_detector_{n, s, m}
  2. Grid (K=1, Original BCE): grid_detector_{n, s, m}
  3. Multi-Anchor (K=3): 3_grid_detector_{n, s, m}

Features auto-tuned confidence threshold discovery to maximize End-to-End F1,
and outputs 3-level evaluation metrics (Detection P/R/F1, Classifier Accuracy, E2E F1, Image FPS).
"""
import sys
import time
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from models.object_detector_res import ObjectDetectorResNet
from utils.dataset import get_detection_loaders

BENCHMARK_PATH = 'data/OD_benchmark'
PLACEMENTS = ['random', 'grid', 'words', 'line']
SIZES = ['n', 's', 'm']
SIZE_LABELS = {
    'n': 'Grid Nano (0.39M)',
    's': 'Grid Small (1.55M)',
    'm': 'Grid Medium (6.18M)',
}
OPTIMAL_CONFS = {
    'n': 0.90,
    's': 0.90,
    'm': 0.90,
}


class SingleStagePipeline:
    """
    Inference & Evaluation wrapper for Single-Stage Unified Object Detector (ObjectDetectorResNet v2/v3).
    """

    def __init__(self, size: str = 's', prefix: str = 'grid_detector_', ckpt_path: str = None, device: str = 'cuda', conf_threshold: float = 0.65):
        self.size = size.lower()
        self.device = device
        self.conf_threshold = conf_threshold

        weights_p = Path(f'weights/{prefix}{self.size}_best.pth')
        ckpt_p = Path(f'checkpoint/{prefix}{self.size}_best.pth')
        path = ckpt_path or (str(weights_p) if weights_p.exists() else str(ckpt_p))
        if Path(path).exists():
            sd = torch.load(path, map_location=device, weights_only=False)
            sd_state = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
            self.anchors_wh = sd.get('anchors_wh', None) if isinstance(sd, dict) else None
        else:
            sd_state = None
            self.anchors_wh = None

        if self.anchors_wh is not None:
            num_anchors = self.anchors_wh.shape[0]
        else:
            num_anchors = 3 if prefix.startswith('3_') else 1
            if num_anchors == 1:
                self.anchors_wh = torch.tensor([[16.0, 16.0]])

        stem, ch, blocks, pool = ObjectDetectorResNet.CONFIGS[self.size]
        self.model = ObjectDetectorResNet(channels=ch, blocks=blocks, num_anchors=num_anchors).to(device)

        if sd_state is not None:
            self.model.load_state_dict(sd_state)
        self.model.eval()

    @torch.no_grad()
    def tune_threshold(self, loader, thresholds=np.arange(0.30, 0.96, 0.05)) -> float:
        """Find the confidence threshold that maximizes E2E F1 score on a dataloader."""
        records = []
        for images, targets_batch, labels_batch in loader:
            images = images.to(self.device)
            outputs = self.model(images)
            outputs = outputs.reshape(images.shape[0], -1, 52)
            
            for b in range(images.shape[0]):
                out = outputs[b]
                boxes = out[:, :4]
                confs = torch.sigmoid(out[:, 4])
                cls_logits = out[:, 5:]
                cls_probs = F.softmax(cls_logits, dim=-1)
                cls_preds = torch.argmax(cls_probs, dim=-1)
                
                gt_targets = targets_batch[b].view(-1, 5)
                gt_labels_flat = labels_batch[b].view(-1)
                gt_mask = gt_targets[:, 4] == 1.0
                gt_boxes = gt_targets[gt_mask, :4].cpu().numpy()
                gt_labels = gt_labels_flat[gt_mask].cpu().numpy()
                
                records.append({
                    'boxes': boxes.cpu().numpy(),
                    'confs': confs.cpu().numpy(),
                    'cls_preds': cls_preds.cpu().numpy(),
                    'gt_boxes': gt_boxes,
                    'gt_labels': gt_labels,
                })
        
        best_f1 = -1.0
        best_t = 0.65
        
        for t in thresholds:
            t = round(float(t), 2)
            det_tp = 0
            e2e_tp = 0
            total_preds = 0
            total_gt = 0
            
            for r in records:
                mask = r['confs'] >= t
                preds_boxes = r['boxes'][mask]
                preds_classes = r['cls_preds'][mask]
                gt_boxes = r['gt_boxes']
                gt_labels = r['gt_labels']
                
                n_gt = len(gt_boxes)
                n_pred = len(preds_boxes)
                total_gt += n_gt
                total_preds += n_pred
                
                if n_gt == 0 or n_pred == 0:
                    continue
                
                px = preds_boxes[:, 0:1]; py = preds_boxes[:, 1:2]
                pw = preds_boxes[:, 2:3]; ph = preds_boxes[:, 3:4]
                gx = gt_boxes[:, 0]; gy = gt_boxes[:, 1]
                gw = gt_boxes[:, 2]; gh = gt_boxes[:, 3]
                
                ix1 = np.maximum(px, gx); iy1 = np.maximum(py, gy)
                ix2 = np.minimum(px + pw, gx + gw); iy2 = np.minimum(py + ph, gy + gh)
                inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                union = pw * ph + gw * gh - inter
                iou_mat = np.where(union > 0, inter / union, 0.0)
                
                pred_idx, gt_idx = linear_sum_assignment(-iou_mat)
                for k, m in zip(pred_idx, gt_idx):
                    if iou_mat[k, m] >= 0.50:
                        det_tp += 1
                        if preds_classes[k] == gt_labels[m]:
                            e2e_tp += 1
                            
            e2e_p = e2e_tp / max(1, total_preds)
            e2e_r = e2e_tp / max(1, total_gt)
            e2e_f1 = (2 * e2e_p * e2e_r) / max(1e-6, e2e_p + e2e_r)
            
            if e2e_f1 > best_f1:
                best_f1 = e2e_f1
                best_t = t
                
        return best_t


    @torch.no_grad()
    def evaluate_loader(self, loader, iou_match_thresh: float = 0.50) -> dict:
        total_gt = 0
        total_preds = 0
        total_images = 0
        det_tp = 0
        e2e_tp = 0
        correct_cls = 0

        t0 = time.time()
        for images, targets_batch, labels_batch in loader:
            images = images.to(self.device)
            outputs = self.model(images)  # (B, 14, 14, K, 5 + 47)
            outputs = outputs.reshape(images.shape[0], -1, 52)
            total_images += images.shape[0]

            for b in range(images.shape[0]):
                out = outputs[b]
                boxes = out[:, :4]
                confs = torch.sigmoid(out[:, 4])
                cls_logits = out[:, 5:]
                cls_probs = F.softmax(cls_logits, dim=-1)
                cls_preds = torch.argmax(cls_probs, dim=-1)

                mask = confs >= self.conf_threshold
                preds_boxes = boxes[mask].cpu().numpy()
                preds_confs = confs[mask].cpu().numpy()
                preds_classes = cls_preds[mask].cpu().numpy()

                gt_targets = targets_batch[b].view(-1, 5)
                gt_labels_flat = labels_batch[b].view(-1)
                gt_mask = gt_targets[:, 4] == 1.0

                gt_boxes = gt_targets[gt_mask, :4].cpu().numpy()
                gt_labels = gt_labels_flat[gt_mask].cpu().numpy()

                n_gt = len(gt_boxes)
                n_pred = len(preds_boxes)
                total_gt += n_gt
                total_preds += n_pred

                if n_gt == 0 or n_pred == 0:
                    continue

                px = preds_boxes[:, 0:1]; py = preds_boxes[:, 1:2]
                pw = preds_boxes[:, 2:3]; ph = preds_boxes[:, 3:4]
                gx = gt_boxes[:, 0]; gy = gt_boxes[:, 1]
                gw = gt_boxes[:, 2]; gh = gt_boxes[:, 3]

                ix1 = np.maximum(px, gx); iy1 = np.maximum(py, gy)
                ix2 = np.minimum(px + pw, gx + gw); iy2 = np.minimum(py + ph, gy + gh)
                inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                union = pw * ph + gw * gh - inter
                iou_mat = np.where(union > 0, inter / union, 0.0)

                pred_idx, gt_idx = linear_sum_assignment(-iou_mat)
                for k, m in zip(pred_idx, gt_idx):
                    if iou_mat[k, m] >= iou_match_thresh:
                        det_tp += 1
                        if preds_classes[k] == gt_labels[m]:
                            correct_cls += 1
                            e2e_tp += 1

        elapsed = time.time() - t0
        image_fps = total_images / max(1e-6, elapsed)

        det_p = det_tp / max(1, total_preds)
        det_r = det_tp / max(1, total_gt)
        det_f1 = (2 * det_p * det_r) / max(1e-6, det_p + det_r)

        cls_acc = correct_cls / max(1, det_tp)

        e2e_p = e2e_tp / max(1, total_preds)
        e2e_r = e2e_tp / max(1, total_gt)
        e2e_f1 = (2 * e2e_p * e2e_r) / max(1e-6, e2e_p + e2e_r)

        return {
            'total_gt': total_gt,
            'total_preds': total_preds,
            'total_images': total_images,
            'det_precision': round(det_p, 4),
            'det_recall': round(det_r, 4),
            'det_f1': round(det_f1, 4),
            'classifier_acc': round(cls_acc, 4),
            'e2e_precision': round(e2e_p, 4),
            'e2e_recall': round(e2e_r, 4),
            'e2e_f1': round(e2e_f1, 4),
            'fps': round(image_fps, 1),
            'eval_time_sec': round(elapsed, 2),
        }


VARIANTS = [
    {'name': 'Grid (K=1, Focal)', 'prefix': '1_grid_detector_', 'sizes': ['n', 's', 'm']},
    {'name': 'Grid (K=1, Original BCE)', 'prefix': 'grid_detector_', 'sizes': ['n', 's', 'm']},
    {'name': 'Multi-Anchor (K=3)', 'prefix': '3_grid_detector_', 'sizes': ['n', 's', 'm']},
]

def run_benchmark():
    all_results = {}
    print("=" * 90)
    print("  SINGLE-STAGE & MULTI-ANCHOR UNIFIED DETECTOR BENCHMARK (AUTO-TUNED THRESHOLDS)")
    print("=" * 90)

    for v in VARIANTS:
        v_name = v['name']
        prefix = v['prefix']
        v_sizes = v.get('sizes', SIZES)
        all_results[v_name] = {}

        for sz in v_sizes:
            print(f"\nEvaluating Variant '{v_name}' | Preset: '{sz.upper()}' ({SIZE_LABELS[sz]})...")
            pipe_tune = SingleStagePipeline(size=sz, prefix=prefix, conf_threshold=0.65)
            
            if pipe_tune.anchors_wh is None and prefix.startswith('3_'):
                print(f"  [Skipping] Checkpoint not found for {prefix}{sz}_best.pth")
                continue

            # Auto-tune threshold using random placement split
            tune_loader = get_detection_loaders(
                data_root=f"{BENCHMARK_PATH}/random", batch_size=128, test_only=True, num_workers=0, anchors_wh=pipe_tune.anchors_wh
            )
            opt_conf = pipe_tune.tune_threshold(tune_loader)
            print(f"  [Auto-Tuned] Optimal threshold for {v_name} {sz.upper()}: conf = {opt_conf:.2f}")

            pipe = SingleStagePipeline(size=sz, prefix=prefix, conf_threshold=opt_conf)
            all_results[v_name][sz] = {'opt_conf': opt_conf, 'placements': {}}

            for p in PLACEMENTS:
                data_dir = f"{BENCHMARK_PATH}/{p}"
                loader = get_detection_loaders(
                    data_root=data_dir, batch_size=128, test_only=True, num_workers=0, anchors_wh=pipe.anchors_wh
                )
                res = pipe.evaluate_loader(loader)
                all_results[v_name][sz]['placements'][p] = res
                print(f"  [{p:<6}] Det P={res['det_precision']:.4f} R={res['det_recall']:.4f} | Cls Acc={res['classifier_acc']:.4f} | E2E F1={res['e2e_f1']:.4f} | FPS={res['fps']:.1f} img/s")

    print("\n" + "=" * 90)
    print("  FULL BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Variant':<18} {'Size':<6} {'Conf':<6} {'Layout':<8} {'Det P':<8} {'Det R':<8} {'Cls Acc':<10} {'E2E F1':<9} {'Img FPS':<8}")
    print("-" * 90)

    for v_name, sizes_res in all_results.items():
        for sz, data in sizes_res.items():
            conf_t = data['opt_conf']
            for p, r in data['placements'].items():
                print(f"{v_name:<18} {sz.upper():<6} {conf_t:<6.2f} {p:<8} {r['det_precision']:<8.4f} {r['det_recall']:<8.4f} {r['classifier_acc']:<10.4f} {r['e2e_f1']:<9.4f} {r['fps']:<8.1f}")
    print("=" * 90)

    out_file = Path('benchmark/multi_anchor_results.json')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw evaluation metrics to '{out_file}'")


if __name__ == '__main__':
    run_benchmark()

