"""
FCOS classification-accuracy diagnostics.

Records, per ground-truth object in data/OD_benchmark/ (key = placement, image file,
GT index), whether each FCOS size detected it and classified it correctly. Then:
  1. Compares cls accuracy on each size's own detections vs. on the intersection of
     objects detected by all three sizes (removes the recall selection effect).
  2. Compares train-mix-weighted benchmark cls accuracy against val_cls_acc at the
     best-checkpoint epoch of each training log.

Run after ``run_single_stage_benchmark.py --detector fcos`` (reads opt_conf from
benchmark/fcos_results.json). Output: benchmark/fcos_cls_diagnostics.json
"""
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'scripts'))

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

from run_single_stage_benchmark import BENCHMARK_PATH, PLACEMENTS, SIZES, FCOSPipeline
from utils.dataset import get_detection_loaders
from utils.pipeline import _decode_fcos_candidates
from generator.generator import PLACEMENT_PROBS

RESULTS_PATH = Path('benchmark/fcos_results.json')
OUT_PATH = Path('benchmark/fcos_cls_diagnostics.json')
LOG_DIR = Path('logs/FCOSObjectDetectorResNet')
VAL_CONF = 0.50  # fixed threshold used by evaluate_fcos_detection during training
IOU_MATCH = 0.50


@torch.no_grad()
def per_gt_outcomes(pipe: FCOSPipeline, placement: str) -> dict:
    """Map (placement, image, gt_idx) -> (detected, correct), using evaluate_loader's matching."""
    loader = get_detection_loaders(
        data_root=f"{BENCHMARK_PATH}/{placement}", batch_size=128, test_only=True, num_workers=0, fcos=True
    )
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler), "loader must be unshuffled"
    records = loader.dataset.reader.records

    outcomes = {}
    total_preds = 0
    img_idx = 0
    for images, _, annotations in loader:
        gt_boxes_batch, gt_labels_batch = annotations
        outputs = pipe.model(images.to(pipe.device))
        candidates = _decode_fcos_candidates(pipe.model, outputs, score_threshold=pipe.conf_threshold, iou_threshold=0.45)

        for b in range(len(candidates)):
            p_boxes, _, p_classes = candidates[b]
            preds_boxes = p_boxes.cpu().numpy()
            preds_classes = p_classes.cpu().numpy()
            gt_boxes = gt_boxes_batch[b].cpu().numpy()
            gt_labels = gt_labels_batch[b].cpu().numpy()
            image = records[img_idx].image
            img_idx += 1
            total_preds += len(preds_boxes)

            detected = np.zeros(len(gt_boxes), dtype=bool)
            correct = np.zeros(len(gt_boxes), dtype=bool)
            if len(gt_boxes) and len(preds_boxes):
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
                    if iou_mat[k, m] >= IOU_MATCH:
                        detected[m] = True
                        correct[m] = preds_classes[k] == gt_labels[m]

            for g in range(len(gt_boxes)):
                outcomes[(placement, image, g)] = (bool(detected[g]), bool(correct[g]))

    return outcomes, total_preds


def acc(keys, outcomes) -> float:
    keys = list(keys)
    return round(sum(outcomes[k][1] for k in keys) / max(1, len(keys)), 4)


def weighted(per_placement: dict) -> float:
    return round(sum(PLACEMENT_PROBS[p] * per_placement[p] for p in PLACEMENTS), 4)


def best_epoch_from_log(size: str) -> dict:
    logs = sorted(glob.glob(str(LOG_DIR / f'fcos_{size}_28x14_*.json')))
    log_path = logs[-1]
    with open(log_path) as f:
        log = json.load(f)
    best = [e for e in log['epochs'] if e.get('is_best')][-1]
    return {
        'log_file': Path(log_path).as_posix(),
        'best_epoch': best['epoch'],
        'val_cls_acc': round(best['val_cls_acc'], 4),
        'val_recall': round(best['val_recall'], 4),
        'val_e2e_f1': round(best['val_e2e_f1'], 4),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(RESULTS_PATH) as f:
        fcos_results = json.load(f)

    conf_settings = {
        'opt_conf': {sz: fcos_results[sz]['opt_conf'] for sz in SIZES},
        'val_conf': {sz: VAL_CONF for sz in SIZES},
    }

    out = {
        'protocol': {
            'data': f'{BENCHMARK_PATH}/{{{",".join(PLACEMENTS)}}} (10,000 images)',
            'matching': f'Hungarian, IoU >= {IOU_MATCH}, class-wise NMS IoU 0.45 (same as FCOSPipeline.evaluate_loader)',
            'conf_settings': {
                'opt_conf': 'per-size threshold tuned on data/OD_benchmark/tune/random (from fcos_results.json)',
                'val_conf': f'fixed {VAL_CONF}, the threshold used for val_cls_acc during training',
            },
            'train_mix_weights': PLACEMENT_PROBS,
            'val_note': ('val_cls_acc comes from the data/OD/train validation split (EMNIST-train glyphs), '
                         'computed with greedy score-ordered matching under bf16 autocast at conf 0.50.'),
        },
        'val': {sz: best_epoch_from_log(sz) for sz in SIZES},
    }

    for label, confs in conf_settings.items():
        print(f"\n=== {label}: {confs} ===")
        outcomes = {}
        per_size = {}
        for sz in SIZES:
            pipe = FCOSPipeline(size=sz, device=device, conf_threshold=confs[sz])
            outcomes[sz] = {}
            placements = {}
            for p in PLACEMENTS:
                o, n_preds = per_gt_outcomes(pipe, p)
                outcomes[sz].update(o)
                det_keys = [k for k, v in o.items() if v[0]]
                placements[p] = {
                    'total_gt': len(o),
                    'total_preds': n_preds,
                    'det_tp': len(det_keys),
                    'correct_cls': sum(o[k][1] for k in det_keys),
                    'cls_acc_detected': acc(det_keys, o),
                }
                if label == 'opt_conf':
                    ref = fcos_results[sz]['placements'][p]
                    for key in ('total_gt', 'total_preds', 'det_tp', 'correct_cls'):
                        assert placements[p][key] == ref[key], (sz, p, key, placements[p][key], ref[key])
            cls_by_p = {p: placements[p]['cls_acc_detected'] for p in PLACEMENTS}
            per_size[sz] = {
                'conf': confs[sz],
                'placements': placements,
                'cls_acc_detected_mean': round(float(np.mean(list(cls_by_p.values()))), 4),
                'cls_acc_detected_train_mix': weighted(cls_by_p),
            }
            print(f"  {sz.upper()} conf={confs[sz]:.2f} cls(detected) by placement={cls_by_p}")

        all_keys = list(outcomes[SIZES[0]].keys())
        inter = [k for k in all_keys if all(outcomes[sz][k][0] for sz in SIZES)]
        inter_by_p = {p: [k for k in inter if k[0] == p] for p in PLACEMENTS}
        intersection = {
            'n_objects': len(inter),
            'n_total_gt': len(all_keys),
            'per_placement_n': {p: len(inter_by_p[p]) for p in PLACEMENTS},
            'cls_acc': {},
            'pairwise_mcnemar': {},
        }
        for sz in SIZES:
            by_p = {p: acc(inter_by_p[p], outcomes[sz]) for p in PLACEMENTS}
            intersection['cls_acc'][sz] = {
                'pooled': acc(inter, outcomes[sz]),
                'per_placement': by_p,
                'mean': round(float(np.mean(list(by_p.values()))), 4),
                'train_mix': weighted(by_p),
            }
        for a, b in combinations(SIZES, 2):
            a_only = sum(outcomes[a][k][1] and not outcomes[b][k][1] for k in inter)
            b_only = sum(outcomes[b][k][1] and not outcomes[a][k][1] for k in inter)
            p_val = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
            intersection['pairwise_mcnemar'][f'{a}_vs_{b}'] = {
                f'{a}_correct_{b}_wrong': a_only,
                f'{b}_correct_{a}_wrong': b_only,
                'p_value': float(f'{p_val:.3g}'),
            }

        # Objects found by the larger models but missed by Nano: how hard are they?
        extra = {}
        for sz in SIZES[1:]:
            keys = [k for k in all_keys if outcomes[sz][k][0] and not outcomes['n'][k][0]]
            extra[sz] = {'n_objects': len(keys), 'cls_acc': acc(keys, outcomes[sz])}

        out[label] = {
            'per_size': per_size,
            'intersection': intersection,
            'detected_by_size_missed_by_nano': extra,
            'gap_train_mix_vs_val': {
                sz: round(per_size[sz]['cls_acc_detected_train_mix'] - out['val'][sz]['val_cls_acc'], 4)
                for sz in SIZES
            },
            'gap_intersection_train_mix_vs_val': {
                sz: round(intersection['cls_acc'][sz]['train_mix'] - out['val'][sz]['val_cls_acc'], 4)
                for sz in SIZES
            },
        }
        print(f"  intersection: {len(inter)}/{len(all_keys)} objects | "
              + " ".join(f"{sz.upper()}={intersection['cls_acc'][sz]['pooled']:.4f}" for sz in SIZES))
        print(f"  train-mix gap vs val: {out[label]['gap_train_mix_vs_val']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved FCOS classification diagnostics to '{OUT_PATH}'")


if __name__ == '__main__':
    main()
