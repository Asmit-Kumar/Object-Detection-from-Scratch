"""
Single-Stage Unified Detector Benchmark Evaluation Script.

Evaluates all 4 Single-Stage Unified Detector sizes (Nano 'n', Small 's', Medium 'm', Large 'l')
across all 4 benchmark placement layouts ('random', 'grid', 'words', 'line') in data/OD_benchmark/.

Uses the optimal confidence threshold per size:
  - Nano: conf = 0.70
  - Small: conf = 0.65
  - Medium: conf = 0.65
  - Large: conf = 0.60

Outputs detailed evaluation metrics (Precision, Recall, F1, Classifier Accuracy, End-to-End F1, Image FPS)
and saves summary outputs for benchmark documentation.
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
SIZES = ['n', 's', 'm', 'l']
SIZE_LABELS = {
    'n': 'Nano (0.71M)',
    's': 'Small (2.52M)',
    'm': 'Medium (9.42M)',
    'l': 'Large (19.99M)',
}
OPTIMAL_CONFS = {
    'n': 0.70,
    's': 0.65,
    'm': 0.65,
    'l': 0.60,
}


class SingleStagePipeline:
    """
    Inference & Evaluation wrapper for Single-Stage Unified Object Detector (ObjectDetectorResNet v2).
    """

    def __init__(self, size: str = 's', ckpt_path: str = None, device: str = 'cuda', conf_threshold: float = 0.65):
        self.size = size.lower()
        self.device = device
        self.conf_threshold = conf_threshold

        stem, ch, blocks, pool = ObjectDetectorResNet.CONFIGS[self.size]
        self.model = ObjectDetectorResNet(channels=ch, blocks=blocks, pool_size=pool).to(device)

        path = ckpt_path or f'checkpoint/s_detector_{self.size}_best.pth'
        sd = torch.load(path, map_location=device)
        sd_state = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
        self.model.load_state_dict(sd_state)
        self.model.eval()

    @torch.no_grad()
    def evaluate_loader(self, loader, iou_match_thresh: float = 0.50) -> dict:
        total_gt = 0
        total_preds = 0
        total_images = 0
        det_tp = 0
        e2e_tp = 0
        correct_cls = 0

        t0 = time.time()
        for images, gt_boxes_batch, gt_labels_batch, gt_mask_batch in loader:
            images = images.to(self.device)
            outputs = self.model(images)  # (B, 24, 5 + 47)
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

                gt_mask = gt_mask_batch[b]
                gt_boxes = gt_boxes_batch[b][gt_mask].cpu().numpy()
                gt_labels = gt_labels_batch[b][gt_mask].cpu().numpy()

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
        det_p = det_tp / max(1, total_preds)
        det_r = det_tp / max(1, total_gt)
        det_f1 = (2 * det_p * det_r) / max(1e-6, det_p + det_r)

        cls_acc = correct_cls / max(1, det_tp)

        e2e_p = e2e_tp / max(1, total_preds)
        e2e_r = e2e_tp / max(1, total_gt)
        e2e_f1 = (2 * e2e_p * e2e_r) / max(1e-6, e2e_p + e2e_r)

        image_fps = total_images / max(1e-3, elapsed)

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


def run_benchmark():
    all_results = {}
    print("=" * 80)
    print("  SINGLE-STAGE UNIFIED DETECTOR BENCHMARK (OPTIMAL THRESHOLDS PER SIZE)")
    print("=" * 80)

    for sz in SIZES:
        conf_t = OPTIMAL_CONFS[sz]
        print(f"\nEvaluating Size Preset: '{sz.upper()}' ({SIZE_LABELS[sz]}) @ conf={conf_t:.2f}...")
        pipe = SingleStagePipeline(size=sz, conf_threshold=conf_t)
        all_results[sz] = {}

        for p in PLACEMENTS:
            data_dir = f"{BENCHMARK_PATH}/{p}"
            loader = get_detection_loaders(data_root=data_dir, batch_size=128, test_only=True, num_workers=0)
            res = pipe.evaluate_loader(loader)
            all_results[sz][p] = res
            print(f"  [{p:<6}] Det P={res['det_precision']:.4f} R={res['det_recall']:.4f} | Cls Acc={res['classifier_acc']:.4f} | E2E F1={res['e2e_f1']:.4f} | FPS={res['fps']:.1f} img/s")

    print("\n" + "=" * 80)
    print("  FULL SINGLE-STAGE UNIFIED BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Model Size':<16} {'Conf':<6} {'Layout':<8} {'Det P':<8} {'Det R':<8} {'Cls Acc':<10} {'E2E F1':<9} {'Img FPS':<8}")
    print("-" * 80)

    for sz in SIZES:
        conf_t = OPTIMAL_CONFS[sz]
        for p in PLACEMENTS:
            r = all_results[sz][p]
            print(f"{SIZE_LABELS[sz]:<16} {conf_t:<6.2f} {p:<8} {r['det_precision']:<8.4f} {r['det_recall']:<8.4f} {r['classifier_acc']:<10.4f} {r['e2e_f1']:<9.4f} {r['fps']:<8.1f}")
    print("=" * 80)

    out_file = Path('benchmark/single_stage_results.json')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw evaluation metrics to '{out_file}'")


if __name__ == '__main__':
    run_benchmark()
