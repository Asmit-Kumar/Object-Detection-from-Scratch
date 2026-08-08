"""
Generates visual benchmark detection grid images for Single-Stage Unified Detector models
across all 4 sizes (Nano, Small, Medium, Large) × 4 placements × 2 samples.
Uses the optimal confidence threshold per model size (Nano: 0.70, Small: 0.65, Medium: 0.65, Large: 0.60).
Saves annotated detection outputs to result/benchmark/single_stage/<model>/<placement>_<idx>.png
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))

import cv2
import torch
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.object_detector_res import ObjectDetectorResNet
from generator.dataset import EMNIST_CLASS_NAMES

BENCHMARK_PATH = 'data/OD_benchmark'
PLACEMENTS = ['random', 'grid', 'words', 'line']
SIZES = ['n', 's', 'm', 'l']
SIZE_LABELS = {'n': 'Nano (0.71M)', 's': 'Small (2.52M)', 'm': 'Medium (9.42M)', 'l': 'Large (19.99M)'}
OPTIMAL_CONFS = {'n': 0.70, 's': 0.65, 'm': 0.65, 'l': 0.60}
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


def generate_visuals():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for size in SIZES:
        out_dir = Path('result/benchmark/single_stage') / size
        out_dir.mkdir(parents=True, exist_ok=True)
        conf_thresh = OPTIMAL_CONFS[size]

        print(f"\n[{SIZE_LABELS[size]}] Loading model (optimal conf threshold: {conf_thresh:.2f})...")
        stem, ch, blocks, pool = ObjectDetectorResNet.CONFIGS[size]
        model = ObjectDetectorResNet(channels=ch, blocks=blocks, pool_size=pool).to(device)

        ckpt_path = f'checkpoint/s_detector_{size}_best.pth'
        sd = torch.load(ckpt_path, map_location=device)
        sd_state = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd_state)
        model.eval()

        for placement in PLACEMENTS:
            img_dir = Path(BENCHMARK_PATH) / placement / 'test' / 'images'
            sample_files = SAMPLE_MAP[placement]

            for idx, file_name in enumerate(sample_files):
                sample_path = img_dir / file_name
                img_gray = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)

                img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                img_tensor = (img_tensor - 0.1307) / 0.3081

                with torch.no_grad():
                    output = model(img_tensor).squeeze(0)  # (24, 52)
                    boxes = output[:, :4]
                    confs = torch.sigmoid(output[:, 4])
                    cls_logits = output[:, 5:]
                    cls_probs = F.softmax(cls_logits, dim=-1)
                    cls_preds = torch.argmax(cls_probs, dim=-1)
                    cls_confs = torch.max(cls_probs, dim=-1)[0]
                    joint_confs = confs * cls_confs

                    # Standard pipeline: Filter by size-specific confidence threshold & apply NMS 0.35
                    mask = confs >= conf_thresh
                    if mask.any():
                        filtered_boxes = boxes[mask]
                        filtered_confs = confs[mask]
                        filtered_det_confs = confs[mask]
                        filtered_classes = cls_preds[mask]
                        filtered_cls_confs = cls_confs[mask]

                        keep = _apply_nms(filtered_boxes, filtered_confs, iou_thresh=0.35)

                        # Secondary gate: require cls_conf >= 0.30 to suppress "object seen, class unknown" false positives
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

                    rect = patches.Rectangle(
                        (x, y), w, h,
                        linewidth=1.5, edgecolor='#00FFCC', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x, max(0, y - 3),
                        f"{char_label} {float(det_c):.2f}|{float(cls_c):.2f}",
                        color='#FFE03A', fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.55, ec='none')
                    )

                n_det = len(valid_boxes)
                ax.set_title(
                    f'{SIZE_LABELS[size]} | {placement} | {n_det} detections',
                    fontsize=9, pad=5
                )
                ax.axis('off')
                plt.tight_layout(pad=0.4)

                out_path = out_dir / f'{placement}_{idx + 1}.png'
                plt.savefig(out_path, bbox_inches='tight', dpi=130)
                plt.close()
                print(f'  Saved: {out_path}')

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\nAll single-stage benchmark visual images regenerated with optimal confidence thresholds successfully!')


if __name__ == '__main__':
    generate_visuals()
