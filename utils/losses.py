"""
Detection Loss Utilities.

Contains IoU helpers and loss functions for the object detection pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import time


# ── IoU Helpers ───────────────────────────────────────────────────────────────

def pairwise_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU matrix between two sets of boxes.

    Args:
        pred_boxes   : Tensor (N, 4) in [x, y, w, h] format.
        target_boxes : Tensor (M, 4) in [x, y, w, h] format.

    Returns:
        Tensor (N, M) — IoU for every (pred, target) pair.
    """
    has_batch = pred_boxes.dim() == 3
    if not has_batch:
        pred_boxes = pred_boxes.unsqueeze(0)
        target_boxes = target_boxes.unsqueeze(0)

    px1, py1 = pred_boxes[..., 0], pred_boxes[..., 1]
    px2 = px1 + pred_boxes[..., 2]
    py2 = py1 + pred_boxes[..., 3]

    gx1, gy1 = target_boxes[..., 0], target_boxes[..., 1]
    gx2 = gx1 + target_boxes[..., 2]
    gy2 = gy1 + target_boxes[..., 3]

    # Broadcast to (B, N, M)
    ix1 = torch.max(px1.unsqueeze(-1), gx1.unsqueeze(-2))
    iy1 = torch.max(py1.unsqueeze(-1), gy1.unsqueeze(-2))
    ix2 = torch.min(px2.unsqueeze(-1), gx2.unsqueeze(-2))
    iy2 = torch.min(py2.unsqueeze(-1), gy2.unsqueeze(-2))

    inter  = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    p_area = pred_boxes[..., 2]   * pred_boxes[..., 3]
    g_area = target_boxes[..., 2] * target_boxes[..., 3]
    union  = p_area.unsqueeze(-1) + g_area.unsqueeze(-2) - inter
    iou = inter / torch.clamp(union, min=1e-6)
    
    if not has_batch:
        iou = iou.squeeze(0)
    return iou


def mean_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> float:
    """Mean IoU between (N, 4) and (N, 4) paired boxes in [x, y, w, h] format.

    Used by evaluate_regression in trainer.py.
    """
    iou_mat = pairwise_iou(pred_boxes, target_boxes)  # (N, N)
    return iou_mat.diagonal().mean().item()


# ── Detection Loss ────────────────────────────────────────────────────────────

class DetectionLoss(nn.Module):
    """Unified tri-head detection loss for multi-object localization, objectness scoring,
    and character classification.

    For each image in the batch:
      1. **Hungarian match** — assign each GT box to the predicted slot that
         minimizes the joint cost combining IoU distance and classification score
         (via ``scipy.optimize.linear_sum_assignment``).
      2. **Box loss (Huber)** — ``HuberLoss`` on matched (pred_box, gt_box) pairs.
      3. **Objectness loss (BCE)** — ``BCEWithLogitsLoss`` on all slots; matched slots
         get target ``1.0``, background slots get ``0.0``, scaled by ``pos_weight``.
      4. **Class loss (CE)** — ``CrossEntropyLoss`` computed exclusively on matched slots.

    Total loss = box_loss + ``lambda_conf`` * obj_loss + ``lambda_class`` * class_loss

    Args:
        lambda_conf  : Weight for the objectness confidence loss term. Default ``1.0``.
        delta        : Huber loss delta (transition point). Default ``1.0``.
        lambda_class : Weight for the classification loss term. Default ``1.0``.
        pos_weight   : Positive class weight scalar for objectness BCE loss. Default ``2.0``.

    Inputs:
        outputs : ``Tensor (B, MAX_OBJECTS, 5 + num_classes)``
                  Last dim: ``[x, y, w, h, conf_logit, class_logits...]``.
        boxes   : ``list[Tensor(N_gt, 4)]`` — GT boxes per image in ``[x, y, w, h]``.
        labels  : ``Tensor (B, max_gt)`` — GT class ids aligned with ``boxes``.
    """

    def __init__(
        self,
        lambda_conf: float = 1.0,
        delta: float = 1.0,
        use_pos_weight: bool = False,
        lambda_class: float = 1.0,
    ):
        super().__init__()
        self.lambda_conf = lambda_conf
        self.lambda_class = lambda_class
        self.use_pos_weight = use_pos_weight
        self.huber = nn.HuberLoss(reduction='mean', delta=delta)
        self.bce   = nn.BCEWithLogitsLoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(
        self,
        outputs: torch.Tensor,
        padded_gt: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = outputs.size(0)
        device = outputs.device
        
        pred_boxes = outputs[..., :4]
        pred_conf  = outputs[..., 4]
        pred_class = outputs[..., 5:]

        max_gt = padded_gt.size(1)
        if max_gt == 0:
            # Edge case: No objects in any image in the batch
            obj_target = torch.zeros_like(pred_conf)
            return self.lambda_conf * self.bce(pred_conf, obj_target)

        # 2. Compute IoU for the whole batch -> (B, MAX_OBJECTS, max_gt)
        iou = pairwise_iou(pred_boxes.detach(), padded_gt)
        
        # Mask out invalid GT slots so they don't get matched
        iou.masked_fill_(~mask.unsqueeze(1), -1.0)
        
        # 3. Hungarian match: for each GT box, find the optimal predicted slot
        # matched_idx: (B, max_gt)
        iou_np = -iou.detach().cpu().numpy()
        mask_np = mask.cpu().numpy()
        matched_idx_np = np.zeros((B, max_gt), dtype=np.int64)

        t0_hungarian = time.perf_counter()
        for b in range(B):
            valid = mask_np[b]
            n_valid = valid.sum()
            if n_valid == 0:
                continue
            valid_pos = valid.nonzero()[0]
            if n_valid == 1:
                matched_idx_np[b, valid_pos[0]] = iou_np[b][:, valid].argmin()
                continue
            cost = iou_np[b][:, valid]
            pred_idx, gt_idx = linear_sum_assignment(cost)
            matched_idx_np[b, valid_pos[gt_idx]] = pred_idx
        t1_hungarian = time.perf_counter()
        self.last_hungarian_time = (t1_hungarian - t0_hungarian) * 1000

        matched_idx = torch.from_numpy(matched_idx_np).to(device)

        if mask_np.any():
            total_unique = sum(
                len(np.unique(matched_idx_np[b][mask_np[b]]))
                for b in range(B) if mask_np[b].any()
            )
            total_gt = mask_np.sum()
            self.last_matched_ratio = total_unique / total_gt if total_gt > 0 else 1.0
        else:
            self.last_matched_ratio = 1.0

        # 4. Box Loss (Huber)
        # Gather the matched predicted boxes: (B, max_gt, 4)
        gathered_boxes = torch.gather(pred_boxes, 1, matched_idx.unsqueeze(-1).expand(-1, -1, 4))
        
        if mask.any():
            box_loss = self.huber(gathered_boxes[mask], padded_gt[mask])
        else:
            box_loss = outputs.new_tensor(0.0)

        # 5. Confidence Loss (BCE)
        obj_target = torch.zeros_like(pred_conf)
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, max_gt)
        
        # Scatter 1.0 into targets where GT exists
        obj_target[b_idx[mask], matched_idx[mask]] = 1.0

        if self.use_pos_weight:
            n_pos = mask.sum()
            n_slots = obj_target.numel()
            if n_pos > 0:
                pw = (n_slots - n_pos) / n_pos
                pos_weight = torch.tensor([pw], dtype=pred_conf.dtype, device=device)
            else:
                pos_weight = None
            conf_loss = F.binary_cross_entropy_with_logits(pred_conf, obj_target, pos_weight=pos_weight)
        else:
            conf_loss = self.bce(pred_conf, obj_target)

        class_loss = outputs.new_tensor(0.0)
        if labels is not None and mask.any():
            gathered_class_logits = torch.gather(
                pred_class,
                1,
                matched_idx.unsqueeze(-1).expand(-1, -1, pred_class.size(-1)),
            )
            class_loss = self.ce(gathered_class_logits[mask], labels[mask].long())

        return box_loss + self.lambda_conf * conf_loss + self.lambda_class * class_loss
