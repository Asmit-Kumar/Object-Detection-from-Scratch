"""
Generates benchmark detection visualizations for grid-based, FCOS, and YOLOv8 models
across all sizes (Nano, Small, Medium), 4 placements, and 2 samples.

Use ``--detector grid`` to render:
  - grid_focal_stage: 1_grid_detector_{n, s, m}
  - grid_stage: grid_detector_{n, s, m}
  - 3_grid_stage: 3_grid_detector_{n, s, m}

Use ``--detector fcos`` to render FCOS outputs. Annotated images are saved
under ``result/benchmark/<variant>/<size>/``.
Use ``--detector yolov8`` to render YOLOv8 outputs using the tuned thresholds
saved by ``run_single_stage_benchmark.py``.
"""
import argparse
import json as jsonlib
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))
sys.path.append(str(root_dir / 'scripts'))

import cv2
import torch
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models import load_fcos_detector, load_model
from models.object_detector_res import ObjectDetectorResNet
from generator.dataset import EMNIST_CLASS_NAMES
from inference.pipeline import _decode_fcos_candidates
from run_single_stage_benchmark import SingleStagePipeline

BENCHMARK_PATH = 'data/OD_benchmark'
PLACEMENTS = ['random', 'grid', 'words', 'line']
SIZES = ['n', 's', 'm']
SIZE_LABELS = {'n': 'Grid Nano (0.39M)', 's': 'Grid Small (1.55M)', 'm': 'Grid Medium (6.18M)'}
OPTIMAL_CONFS = {'n': 0.90, 's': 0.90, 'm': 0.90}
CLASS_NAMES = EMNIST_CLASS_NAMES["bymerge"]

SAMPLE_MAP = {
    'random': ['000002.png', '000003.png'],
    'grid':   ['000001.png', '000003.png'],
    'words':  ['000001.png', '000003.png'],
    'line':   ['000002.png', '000003.png'],
}


def _apply_nms(boxes_xywh: torch.Tensor, confs: torch.Tensor, iou_thresh: float = 0.35) -> torch.Tensor:
    if boxes_xywh.shape[0] <= 1:
        return torch.arange(boxes_xywh.shape[0], device=boxes_xywh.device)
    x1 = boxes_xywh[:, 0]
    y1 = boxes_xywh[:, 1]
    x2 = x1 + boxes_xywh[:, 2]
    y2 = y1 + boxes_xywh[:, 3]
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return torchvision.ops.nms(boxes_xyxy, confs, iou_thresh)


VARIANTS = [
    {'name': 'grid_focal_stage', 'prefix': '1_grid_detector_', 'default_conf': 0.50, 'sizes': ['n', 's', 'm']},
    {'name': 'grid_stage', 'prefix': 'grid_detector_', 'default_conf': 0.90, 'sizes': ['n', 's', 'm']},
    {'name': '3_grid_stage', 'prefix': '3_grid_detector_', 'default_conf': 0.95, 'sizes': ['n', 's', 'm']},
]


def generate_visuals(detector: str = 'grid'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if detector in ('fcos', 'yolov8'):
        family_name = 'FCOS' if detector == 'fcos' else 'YOLOv8'
        print(f"Generating {family_name} visual detection grids on {device}...")
        opt_confs = {'n': 0.50, 's': 0.45, 'm': 0.50} if detector == 'fcos' else {}
        results_p = Path(f'benchmark/{detector}_results.json')
        if detector == 'yolov8' and not results_p.exists():
            raise FileNotFoundError(
                f"{results_p} is missing; run scripts/run_single_stage_benchmark.py --detector yolov8 first."
            )
        if results_p.exists():
            try:
                with open(results_p) as f:
                    data = jsonlib.load(f)
                for sz in SIZES:
                    if sz in data and 'opt_conf' in data[sz]:
                        opt_confs[sz] = data[sz]['opt_conf']
                if detector == 'yolov8' and any(sz not in data or 'opt_conf' not in data[sz] for sz in SIZES):
                    raise ValueError(f"{results_p} does not contain tuned thresholds for Nano, Small, and Medium.")
            except Exception:
                if detector == 'yolov8':
                    raise

        for sz in SIZES:
            if detector == 'fcos':
                ckpt_path = f'weights/fcos_{sz}_28x14_best.pth'
                if not Path(ckpt_path).exists():
                    ckpt_path = f'checkpoint/fcos_{sz}_28x14_best.pth'
            else:
                ckpt_path = f'weights/yolov8_{sz}_28x14_best.pth'
                if not Path(ckpt_path).exists():
                    ckpt_path = f'checkpoint/yolov8_{sz}_28x14_best.pth'
            if not Path(ckpt_path).exists():
                print(f"Skipping {ckpt_path} (not found)")
                continue

            conf_thresh = opt_confs[sz]
            out_dir = Path('result/benchmark/fcos_stage' if detector == 'fcos' else 'result/benchmark/yolov8') / sz
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{family_name} {sz.upper()}] Loading model (conf={conf_thresh:.2f})...")
            if detector == 'fcos':
                model = load_fcos_detector(ckpt_path, device=device, size=sz)
            else:
                model = load_model('yolov8', ckpt_path, device=device, size=sz, strict=True)
            model.eval()

            for placement in PLACEMENTS:
                img_dir = Path(BENCHMARK_PATH) / placement / 'test' / 'images'
                sample_files = SAMPLE_MAP[placement]
                for idx, file_name in enumerate(sample_files):
                    sample_path = img_dir / file_name
                    img_gray = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
                    if img_gray is None:
                        print(f"Warning: Could not read {sample_path}")
                        continue
                    # Detectors are trained and benchmarked on [0, 1] images (DataReader); no EMNIST normalization.
                    img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

                    with torch.no_grad():
                        outputs = model(img_tensor)
                        candidates = _decode_fcos_candidates(model, outputs, score_threshold=conf_thresh, iou_threshold=0.45)
                        boxes, scores, classes = candidates[0]
                        valid_boxes = boxes.cpu().numpy()
                        valid_scores = scores.cpu().numpy()
                        valid_classes = classes.cpu().numpy()

                    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
                    ax.imshow(img_gray, cmap='gray', vmin=0, vmax=255)

                    for box, score, cls_idx in zip(valid_boxes, valid_scores, valid_classes):
                        x, y, w, h = box
                        char_label = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else "?"
                        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='#00FFCC', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x, max(0, y - 3), f"{char_label} {float(score):.2f}", color='#FFE03A', fontsize=7, fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.55, ec='none'))

                    ax.set_title(f'{family_name} {sz.upper()} | {placement} | {len(valid_boxes)} detections', fontsize=9, pad=5)
                    ax.axis('off')
                    plt.tight_layout(pad=0.4)
                    out_path = out_dir / f'{placement}_{idx + 1}.png'
                    plt.savefig(out_path, bbox_inches='tight', dpi=130)
                    plt.close()
                    print(f'  Saved: {out_path}')

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f'\nAll {family_name} visual grid images generated successfully!')
        return

    results_path = Path('benchmark/multi_anchor_results.json')
    opt_confs = {}
    if results_path.exists():
        try:
            import json
            with open(results_path) as f:
                data = json.load(f)
                for v_key, v_val in data.items():
                    for sz, val in v_val.items():
                        opt_c = val.get('opt_conf', 0.90)
                        if 'Focal' in v_key:
                            opt_confs[('grid_focal_stage', sz)] = opt_c
                        elif 'Original BCE' in v_key:
                            opt_confs[('grid_stage', sz)] = opt_c
                        elif 'Multi-Anchor' in v_key:
                            opt_confs[('3_grid_stage', sz)] = opt_c
        except Exception:
            pass

    for variant in VARIANTS:
        v_name = str(variant['name'])
        prefix = str(variant['prefix'])
        v_sizes = [str(size) for size in variant.get('sizes', SIZES)]

        for size in v_sizes:
            weights_p = Path(f'weights/{prefix}{size}_best.pth')
            ckpt_p = Path(f'checkpoint/{prefix}{size}_best.pth')
            ckpt_path = str(weights_p) if weights_p.exists() else str(ckpt_p)
            if not Path(ckpt_path).exists():
                print(f"Skipping {ckpt_path} (not found)")
                continue

            out_dir = Path(f'result/benchmark/{v_name}') / size
            out_dir.mkdir(parents=True, exist_ok=True)
            conf_thresh = opt_confs.get((v_name, size), variant['default_conf'])

            print(f"\n[{v_name.upper()} | {SIZE_LABELS[size]}] Loading model (conf threshold: {conf_thresh:.2f})...")

            # Same checkpoint loading as the benchmark (selects the legacy 14x14 architecture when needed).
            model = SingleStagePipeline(size=size, prefix=prefix, ckpt_path=ckpt_path, device=device).model
            model.eval()

            for placement in PLACEMENTS:
                img_dir = Path(BENCHMARK_PATH) / placement / 'test' / 'images'
                sample_files = SAMPLE_MAP[placement]

                for idx, file_name in enumerate(sample_files):
                    sample_path = img_dir / file_name
                    img_gray = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)

                    # Detectors are trained and benchmarked on [0, 1] images (DataReader); no EMNIST normalization.
                    img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

                    with torch.no_grad():
                        output = model(img_tensor).squeeze(0)
                        output = output.reshape(-1, 52)
                        boxes = output[:, :4]
                        confs = torch.sigmoid(output[:, 4])
                        cls_logits = output[:, 5:]
                        cls_probs = F.softmax(cls_logits, dim=-1)
                        cls_preds = torch.argmax(cls_probs, dim=-1)
                        cls_confs = torch.max(cls_probs, dim=-1)[0]
                        mask = confs >= conf_thresh
                        if mask.any():
                            filtered_boxes = boxes[mask]
                            filtered_confs = confs[mask]
                            filtered_det_confs = confs[mask]
                            filtered_classes = cls_preds[mask]
                            filtered_cls_confs = cls_confs[mask]
                            keep = _apply_nms(filtered_boxes, filtered_confs, iou_thresh=0.35)
                            cls_gate = filtered_cls_confs[keep] >= 0.30
                            keep = keep[cls_gate]
                            valid_boxes = filtered_boxes[keep].cpu().numpy()
                            valid_det_confs = filtered_det_confs[keep].cpu().numpy()
                            valid_classes = filtered_classes[keep].cpu().numpy()
                            valid_cls_confs = filtered_cls_confs[keep].cpu().numpy()
                        else:
                            valid_boxes = []
                            valid_det_confs = []
                            valid_classes = []
                            valid_cls_confs = []

                    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
                    ax.imshow(img_gray, cmap='gray', vmin=0, vmax=255)

                    for box, det_c, cls_idx, cls_c in zip(valid_boxes, valid_det_confs, valid_classes, valid_cls_confs):
                        x, y, w, h = box
                        char_label = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else "?"
                        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='#00FFCC', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x, max(0, y - 3), f"{char_label} {float(det_c):.2f}|{float(cls_c):.2f}", color='#FFE03A', fontsize=7, fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.55, ec='none'))

                    ax.set_title(f'{v_name} {size.upper()} | {placement} | {len(valid_boxes)} detections', fontsize=9, pad=5)
                    ax.axis('off')
                    plt.tight_layout(pad=0.4)
                    out_path = out_dir / f'{placement}_{idx + 1}.png'
                    plt.savefig(out_path, bbox_inches='tight', dpi=130)
                    plt.close()
                    print(f'  Saved: {out_path}')

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print('\nAll visual images regenerated successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate benchmark visuals for grid, FCOS, or YOLOv8 detector variants.')
    parser.add_argument('--detector', choices=['grid', 'fcos', 'yolov8'], default='grid', help='Which detector family to render.')
    args = parser.parse_args()
    generate_visuals(detector=args.detector)
