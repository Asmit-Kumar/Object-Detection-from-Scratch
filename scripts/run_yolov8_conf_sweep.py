"""Evaluate YOLOv8 checkpoints over a confidence-threshold sweep.

The detector is executed once per model and layout at the minimum threshold;
the cached post-NMS candidates are then re-scored at every requested threshold.
This keeps the benchmark matching protocol identical while avoiding repeated
GPU inference for each confidence value.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))
sys.path.append(str(root_dir / 'scripts'))

from dataio.dataset import get_detection_loaders
from inference.pipeline import _decode_fcos_candidates
from run_single_stage_benchmark import BENCHMARK_PATH, PLACEMENTS, SIZES, YOLOv8Pipeline


DEFAULT_THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(7)]


def _pairwise_iou_xywh(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    px = boxes_a[:, 0:1]
    py = boxes_a[:, 1:2]
    pw = boxes_a[:, 2:3]
    ph = boxes_a[:, 3:4]
    gx = boxes_b[:, 0]
    gy = boxes_b[:, 1]
    gw = boxes_b[:, 2]
    gh = boxes_b[:, 3]

    ix1 = np.maximum(px, gx)
    iy1 = np.maximum(py, gy)
    ix2 = np.minimum(px + pw, gx + gw)
    iy2 = np.minimum(py + ph, gy + gh)
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return np.where(union > 0, inter / union, 0.0)


@torch.no_grad()
def _collect_records(pipe: YOLOv8Pipeline, loader, min_conf: float) -> tuple[list[dict], float]:
    records = []
    if pipe.device.startswith('cuda'):
        torch.cuda.synchronize()
    start = time.perf_counter()

    for images, _, annotations in loader:
        gt_boxes_batch, gt_labels_batch = annotations
        images = images.to(pipe.device)
        outputs = pipe.model(images)
        candidates = _decode_fcos_candidates(
            pipe.model,
            outputs,
            score_threshold=min_conf,
            iou_threshold=0.45,
        )

        for batch_idx, (pred_boxes, pred_scores, pred_classes) in enumerate(candidates):
            records.append({
                'pred_boxes': pred_boxes.cpu().numpy(),
                'pred_scores': pred_scores.cpu().numpy(),
                'pred_classes': pred_classes.cpu().numpy(),
                'gt_boxes': gt_boxes_batch[batch_idx].cpu().numpy(),
                'gt_labels': gt_labels_batch[batch_idx].cpu().numpy(),
            })

    if pipe.device.startswith('cuda'):
        torch.cuda.synchronize()
    return records, time.perf_counter() - start


def _score_records(records: list[dict], threshold: float) -> dict:
    total_gt = total_preds = total_images = 0
    det_tp = e2e_tp = correct_cls = 0

    for record in records:
        mask = record['pred_scores'] >= threshold
        pred_boxes = record['pred_boxes'][mask]
        pred_classes = record['pred_classes'][mask]
        gt_boxes = record['gt_boxes']
        gt_labels = record['gt_labels']

        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        total_gt += n_gt
        total_preds += n_pred
        total_images += 1

        if n_gt == 0 or n_pred == 0:
            continue

        iou_mat = _pairwise_iou_xywh(pred_boxes, gt_boxes)
        pred_idx, gt_idx = linear_sum_assignment(-iou_mat)
        for pred_i, gt_i in zip(pred_idx, gt_idx):
            if iou_mat[pred_i, gt_i] >= 0.50:
                det_tp += 1
                if pred_classes[pred_i] == gt_labels[gt_i]:
                    correct_cls += 1
                    e2e_tp += 1

    det_precision = det_tp / max(1, total_preds)
    det_recall = det_tp / max(1, total_gt)
    det_f1 = (2 * det_precision * det_recall) / max(1e-6, det_precision + det_recall)
    classifier_acc = correct_cls / max(1, det_tp)
    e2e_precision = e2e_tp / max(1, total_preds)
    e2e_recall = e2e_tp / max(1, total_gt)
    e2e_f1 = (2 * e2e_precision * e2e_recall) / max(1e-6, e2e_precision + e2e_recall)

    return {
        'threshold': round(threshold, 2),
        'total_gt': total_gt,
        'total_preds': total_preds,
        'total_images': total_images,
        'det_precision': round(det_precision, 4),
        'det_recall': round(det_recall, 4),
        'det_f1': round(det_f1, 4),
        'classifier_acc': round(classifier_acc, 4),
        'e2e_precision': round(e2e_precision, 4),
        'e2e_recall': round(e2e_recall, 4),
        'e2e_f1': round(e2e_f1, 4),
    }


def _thresholds(start: float, stop: float, step: float) -> list[float]:
    if step <= 0 or stop < start:
        raise ValueError('Expected 0 < step and stop >= start')
    count = int(round((stop - start) / step))
    values = [round(start + i * step, 2) for i in range(count + 1)]
    if values[-1] != round(stop, 2):
        raise ValueError('The range must land exactly on stop after rounding')
    if values[0] < 0.0 or values[-1] > 1.0:
        raise ValueError('Confidence thresholds must be between 0 and 1')
    return values


def run_sweep(start: float = 0.30, stop: float = 0.60, step: float = 0.05,
              output_path: str = 'benchmark/yolov8_conf_sweep_030_060.json') -> None:
    thresholds = _thresholds(start, stop, step)
    min_conf = thresholds[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = {
        'detector': 'yolov8',
        'device': device,
        'thresholds': thresholds,
        'protocol': {
            'benchmark_root': BENCHMARK_PATH,
            'nms_iou_threshold': 0.45,
            'matching_iou_threshold': 0.50,
            'decode_threshold': min_conf,
            'note': 'Candidates are decoded once at the minimum threshold and filtered for each sweep value.',
        },
        'models': {},
    }

    print('=' * 90)
    print(f'  YOLOv8 CONFIDENCE SWEEP ({thresholds[0]:.2f} to {thresholds[-1]:.2f}, step {step:.2f})')
    print(f'  Device: {device} | 10,000 Test Images across 4 Placements')
    print('=' * 90)

    for size in SIZES:
        print(f'\nLoading YOLOv8 {size.upper()}...')
        pipe = YOLOv8Pipeline(size=size, device=device, conf_threshold=min_conf)
        model_result = {
            'architecture': pipe.architecture,
            'device': device,
            'decode_threshold': min_conf,
            'placements': {},
            'averages': {},
        }

        for placement in PLACEMENTS:
            loader = get_detection_loaders(
                data_root=f'{BENCHMARK_PATH}/{placement}',
                batch_size=128,
                test_only=True,
                num_workers=0,
                ltrb=True,
            )
            records, elapsed = _collect_records(pipe, loader, min_conf)
            expected_gt = sum(len(record.labels) for record in loader.dataset.reader.records)
            observed_gt = sum(len(record['gt_boxes']) for record in records)
            if observed_gt != expected_gt:
                raise AssertionError((size, placement, observed_gt, expected_gt))

            layout_results = {}
            for threshold in thresholds:
                metrics = _score_records(records, threshold)
                metrics['cached_inference_fps'] = round(metrics['total_images'] / max(elapsed, 1e-6), 1)
                layout_results[f'{threshold:.2f}'] = metrics
            model_result['placements'][placement] = layout_results
            print(f'  [{size.upper()} | {placement:<6}] cached {elapsed:.1f}s')

        for threshold in thresholds:
            key = f'{threshold:.2f}'
            values = [model_result['placements'][p][key] for p in PLACEMENTS]
            model_result['averages'][key] = {
                'threshold': threshold,
                'det_precision': round(float(np.mean([v['det_precision'] for v in values])), 4),
                'det_recall': round(float(np.mean([v['det_recall'] for v in values])), 4),
                'classifier_acc': round(float(np.mean([v['classifier_acc'] for v in values])), 4),
                'e2e_f1': round(float(np.mean([v['e2e_f1'] for v in values])), 4),
                'total_preds': sum(v['total_preds'] for v in values),
            }
            avg = model_result['averages'][key]
            print(f'  [{size.upper()} | conf={threshold:.2f}] '
                  f'P={avg["det_precision"]:.4f} R={avg["det_recall"]:.4f} '
                  f'Cls={avg["classifier_acc"]:.4f} E2E-F1={avg["e2e_f1"]:.4f}')

        result['models'][size] = model_result
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(f'\nSaved YOLOv8 confidence sweep results to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sweep YOLOv8 confidence thresholds without changing the official benchmark.')
    parser.add_argument('--start', type=float, default=0.30)
    parser.add_argument('--stop', type=float, default=0.60)
    parser.add_argument('--step', type=float, default=0.05)
    parser.add_argument('--output', default='benchmark/yolov8_conf_sweep_030_060.json')
    args = parser.parse_args()
    run_sweep(args.start, args.stop, args.step, args.output)
