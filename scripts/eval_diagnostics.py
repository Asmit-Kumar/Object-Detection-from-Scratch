import torch
import numpy as np
from collections import defaultdict
from utils.dataset import get_detection_loaders
from models import load_bbox_model
from utils.losses import pairwise_iou

@torch.no_grad()
def recall_by_gt_count(model, loader, device, conf_threshold=0.5, iou_threshold=0.5):
    tp_by_count = defaultdict(int)
    fn_by_count = defaultdict(int)
    use_amp = device.type == "cuda"

    for images, boxes, labels, mask in loader:
        images = images.to(device, non_blocking=True)
        boxes  = boxes.to(device, non_blocking=True)
        mask   = mask.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(images)

        pred_boxes = outputs[..., :4].float()
        pred_conf  = torch.sigmoid(outputs[..., 4].float())
        active     = pred_conf > conf_threshold
        iou = pairwise_iou(pred_boxes, boxes)

        for b in range(images.size(0)):
            gt_idx = mask[b].nonzero(as_tuple=True)[0]
            pr_idx = active[b].nonzero(as_tuple=True)[0]
            n_gt = gt_idx.numel()
            if n_gt == 0:
                continue

            matched = set()
            if pr_idx.numel() > 0:
                sub_iou = iou[b][pr_idx][:, gt_idx]
                order = torch.argsort(pred_conf[b][pr_idx], descending=True)
                for i in order.tolist():
                    j = torch.argmax(sub_iou[i]).item()
                    if sub_iou[i, j] >= iou_threshold and j not in matched:
                        matched.add(j)

            tp_by_count[n_gt] += len(matched)
            fn_by_count[n_gt] += n_gt - len(matched)

    print(f"\n--- Recall by GT Count ---")
    print(f"{'n_gt':>5} {'recall':>8} {'tp':>6} {'fn':>6}")
    for k in sorted(tp_by_count):
        tp, fn = tp_by_count[k], fn_by_count[k]
        print(f"{k:5d} {tp/(tp+fn):8.3f} {tp:6d} {fn:6d}")


@torch.no_grad()
def recall_by_box_area(model, loader, device, conf_threshold=0.5, iou_threshold=0.5, bins=(0, 200, 500, 1000, 5000)):
    tp_bins = np.zeros(len(bins)); fn_bins = np.zeros(len(bins))
    use_amp = device.type == "cuda"

    for images, boxes, labels, mask in loader:
        images = images.to(device, non_blocking=True)
        boxes  = boxes.to(device, non_blocking=True)
        mask   = mask.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(images)

        pred_boxes = outputs[..., :4].float()
        pred_conf  = torch.sigmoid(outputs[..., 4].float())
        active     = pred_conf > conf_threshold
        iou = pairwise_iou(pred_boxes, boxes)
        
        # Convert width and height to pixels. The boxes are [x, y, w, h] normalized by 224
        # Area in pixels = (w * 224) * (h * 224)
        areas = (boxes[..., 2] * 224) * (boxes[..., 3] * 224)

        for b in range(images.size(0)):
            gt_idx = mask[b].nonzero(as_tuple=True)[0]
            pr_idx = active[b].nonzero(as_tuple=True)[0]
            if gt_idx.numel() == 0:
                continue

            matched = set()
            if pr_idx.numel() > 0:
                sub_iou = iou[b][pr_idx][:, gt_idx]
                order = torch.argsort(pred_conf[b][pr_idx], descending=True)
                for i in order.tolist():
                    j = torch.argmax(sub_iou[i]).item()
                    if sub_iou[i, j] >= iou_threshold and j not in matched:
                        matched.add(j)

            for local_j, g in enumerate(gt_idx.tolist()):
                a = areas[b, g].item()
                bin_idx = np.digitize(a, bins) - 1
                if local_j in matched:
                    tp_bins[bin_idx] += 1
                else:
                    fn_bins[bin_idx] += 1

    print(f"\n--- Recall by Box Area ---")
    for i, lo in enumerate(bins):
        hi = bins[i+1] if i+1 < len(bins) else "inf"
        tp, fn = tp_bins[i], fn_bins[i]
        r = tp/(tp+fn) if (tp+fn) else float('nan')
        print(f"area [{lo:>4}, {hi:>4}) pixels: recall={r:.3f}  n={int(tp+fn)}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data
    _, valloader = get_detection_loaders(batch_size=64, num_workers=4)

    # 2. Build Model
    ckpt_path = "checkpoint/detector_s_1_best.pth"
    print(f"Loading checkpoint {ckpt_path}...")
    model = load_bbox_model(ckpt_path, device)

    # 3. Evaluate Diagnostics
    print("Running diagnostics at conf_threshold=0.5...")
    recall_by_gt_count(model, valloader, device, conf_threshold=0.5)
    recall_by_box_area(model, valloader, device, conf_threshold=0.5)
