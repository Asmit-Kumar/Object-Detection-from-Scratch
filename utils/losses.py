"""
Detection Loss Utilities.

Contains IoU helpers, coordinate regression loss, classification loss,
and objectness loss functions (BCE with dynamic positive weighting and Sigmoid Focal Loss)
for the single-stage, multi-anchor, and FCOS spatial detection pipelines.
"""

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import time
from torchvision.ops import sigmoid_focal_loss

from utils.fcos_targets import FCOSTargetGenerator, FCOSTargets


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


def aligned_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute aligned (element-wise) IoU between two tensors of shape (N, 4).

    Boxes are in [x, y, w, h] format.
    """
    x1_1, y1_1 = boxes1[:, 0], boxes1[:, 1]
    x2_1, y2_1 = x1_1 + boxes1[:, 2], y1_1 + boxes1[:, 3]

    x1_2, y1_2 = boxes2[:, 0], boxes2[:, 1]
    x2_2, y2_2 = x1_2 + boxes2[:, 2], y1_2 + boxes2[:, 3]

    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] * boxes1[:, 3]).clamp(min=0)
    area2 = (boxes2[:, 2] * boxes2[:, 3]).clamp(min=0)
    union_area = (area1 + area2 - inter_area).clamp(min=1e-6)

    return inter_area / union_area


def giou_loss_ltrb(pred_ltrb: torch.Tensor, target_ltrb: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU (GIoU) loss for boxes parameterized as [l, t, r, b] distances.

    Args:
        pred_ltrb: (N, 4) tensor of [l, t, r, b].
        target_ltrb: (N, 4) tensor of [l, t, r, b].

    Returns:
        (N,) tensor of GIoU loss values: 1 - GIoU.
    """
    pred_l, pred_t, pred_r, pred_b = pred_ltrb[:, 0], pred_ltrb[:, 1], pred_ltrb[:, 2], pred_ltrb[:, 3]
    gt_l, gt_t, gt_r, gt_b = target_ltrb[:, 0], target_ltrb[:, 1], target_ltrb[:, 2], target_ltrb[:, 3]

    pred_w = (pred_l + pred_r).clamp_min(0)
    pred_h = (pred_t + pred_b).clamp_min(0)
    pred_area = pred_w * pred_h

    gt_w = (gt_l + gt_r).clamp_min(0)
    gt_h = (gt_t + gt_b).clamp_min(0)
    gt_area = gt_w * gt_h

    inter_w = (torch.minimum(pred_l, gt_l) + torch.minimum(pred_r, gt_r)).clamp_min(0)
    inter_h = (torch.minimum(pred_t, gt_t) + torch.minimum(pred_b, gt_b)).clamp_min(0)
    inter_area = inter_w * inter_h

    union_area = (pred_area + gt_area - inter_area).clamp_min(1e-6)
    iou = inter_area / union_area

    convex_w = torch.maximum(pred_l, gt_l) + torch.maximum(pred_r, gt_r)
    convex_h = torch.maximum(pred_t, gt_t) + torch.maximum(pred_b, gt_b)
    convex_area = (convex_w * convex_h).clamp_min(1e-6)

    giou = iou - (convex_area - union_area) / convex_area
    return 1.0 - giou


# ── Detection Loss ────────────────────────────────────────────────────────────

class DetectionLoss(nn.Module):
    """
    Multi-Scale and Single-Scale Detection Loss Function.
    """

    def __init__(
            self,
            lambda_conf: float = 1.0,
            lambda_class: float = 1.0,
            lambda_iou: float = 1.0,
            delta: float = 1.0,
            pos_weight: float | None = None,
            use_pos_weight: bool = True,
            max_pos_weight: float = 50.0,
            use_focal_loss: bool = False,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_conf = lambda_conf
        self.lambda_class = lambda_class
        self.lambda_iou = lambda_iou
        self.use_pos_weight = use_pos_weight
        self.max_pos_weight = max_pos_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.huber = nn.HubenLoss(delta=delta) if hasattr(nn, "HubenLoss") else nn.HuberLoss(delta=delta)
        self.ce = nn.CrossEntropyLoss()

        if pos_weight is not None:
            self.register_buffer("fixed_pos_weight", torch.tensor([pos_weight]))
        else:
            self.fixed_pos_weight = None

    def forward(
            self,
            outputs: torch.Tensor | Mapping[int, torch.Tensor],
            targets: torch.Tensor | Mapping[int, torch.Tensor],
            labels: torch.Tensor | Mapping[int, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        if isinstance(outputs, Mapping):
            scale_losses = []
            for grid_size, scale_outputs in outputs.items():
                scale_targets = targets[grid_size]
                scale_labels = labels[grid_size] if labels is not None else None
                scale_losses.append(
                    self._forward_single(scale_outputs, scale_targets, scale_labels)
                )
            loss = torch.stack(scale_losses).mean()
            return loss, {"loss": loss.item()}

        loss = self._forward_single(outputs, targets, labels)
        return loss, {"loss": loss.item()}

    def _forward_single(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor,
            labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_boxes = outputs[..., :4]
        pred_conf = outputs[..., 4]
        pred_class = outputs[..., 5:]

        gt_boxes = targets[..., :4]
        gt_objectness = targets[..., 4]

        positive = gt_objectness == 1.0

        if self.fixed_pos_weight is not None:
            pos_weight = self.fixed_pos_weight.to(outputs.device)
        elif self.use_pos_weight:
            num_pos = positive.sum().float()
            num_neg = (gt_objectness == 0.0).sum().float()
            pw = (num_neg / (num_pos + 1e-6)).clamp(max=self.max_pos_weight)
            pos_weight = pw.view(1)
        else:
            pos_weight = None

        if self.use_focal_loss:
            conf_loss = sigmoid_focal_loss(
                pred_conf,
                gt_objectness,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="mean",
            )
        else:
            conf_loss = F.binary_cross_entropy_with_logits(
                pred_conf,
                gt_objectness,
                pos_weight=pos_weight,
            )

        if positive.any():
            box_loss = self.huber(
                pred_boxes[positive],
                gt_boxes[positive],
            )
            iou = aligned_iou(pred_boxes[positive], gt_boxes[positive])
            iou_loss = 1.0 - iou.mean()
        else:
            box_loss = outputs.new_tensor(0.0)
            iou_loss = outputs.new_tensor(0.0)

        if labels is not None and positive.any():
            class_loss = self.ce(
                pred_class[positive],
                labels[positive],
            )
        else:
            class_loss = outputs.new_tensor(0.0)

        return (
                (box_loss + self.lambda_iou * iou_loss)
                + self.lambda_conf * conf_loss
                + self.lambda_class * class_loss
        )


# ── FCOS Loss ─────────────────────────────────────────────────────────────────

class FCOSLoss(nn.Module):
    """
    FCOS Multi-Scale Loss.

    Computes:
      - Focal Loss on multi-class classification across all feature locations.
      - Centerness-weighted GIoU Loss on positive locations.
      - Binary Cross-Entropy Loss on centerness for positive locations.

    Args:
        lambda_reg: Loss multiplier for bounding box GIoU regression. Default 1.0.
        lambda_centerness: Loss multiplier for centerness BCE. Default 1.0.
        focal_alpha: Sigmoid focal loss alpha. Default 0.25.
        focal_gamma: Sigmoid focal loss gamma. Default 2.0.
    """

    def __init__(
            self,
            lambda_reg: float = 1.0,
            lambda_centerness: float = 1.0,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            grid_sizes: tuple[int, int] = (28, 14),
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_centerness = lambda_centerness
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.grid_sizes = grid_sizes
        self.target_generator = FCOSTargetGenerator(grid_sizes=grid_sizes)

    def forward(
            self,
            preds_by_scale: Mapping[int, Mapping[str, torch.Tensor]],
            targets: FCOSTargets | tuple[list[torch.Tensor], list[torch.Tensor]],
            labels: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute total FCOS loss across all pyramid scales.

        Args:
            preds_by_scale: Dict {28: {'cls_logits', 'reg_ltrb', 'centerness_logits'}, 14: ...}
            targets: Either an FCOSTargets instance, or a list/tuple of bounding boxes [x, y, w, h].
            labels: List of label tensors if targets is a list of bounding boxes.

        Returns:
            (total_loss, loss_dict)
        """
        device = next(iter(preds_by_scale.values()))["cls_logits"].device

        if not isinstance(targets, FCOSTargets):
            # Accept either separate (boxes, labels) arguments or the documented
            # ``(boxes_batch, labels_batch)`` tuple form.
            if labels is None and isinstance(targets, (tuple, list)) and len(targets) == 2:
                boxes_batch, labels_batch = targets
            else:
                boxes_batch = targets
                labels_batch = labels if labels is not None else []
            fcos_targets = self.target_generator.generate_targets(boxes_batch, labels_batch, device)
        else:
            fcos_targets = targets

        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        total_cent_loss = torch.tensor(0.0, device=device)

        total_num_pos = 0.0
        total_cent_gt_sum = torch.tensor(0.0, device=device)

        for grid_size in self.grid_sizes:
            preds = preds_by_scale[grid_size]
            cls_logits = preds["cls_logits"].float()          # (B, H, W, num_classes)
            reg_ltrb = preds["reg_ltrb"]                      # (B, H, W, 4)
            cent_logits = preds["centerness_logits"]          # (B, H, W)

            cls_gt = fcos_targets.cls_targets[grid_size]      # (B, H, W)
            reg_gt = fcos_targets.reg_targets[grid_size]      # (B, H, W, 4)
            cent_gt = fcos_targets.centerness_targets[grid_size]  # (B, H, W)

            num_classes = cls_logits.shape[-1]
            pos_mask = cls_gt >= 0                            # (B, H, W)
            num_pos = pos_mask.sum().float()
            total_num_pos += num_pos.item()

            # 1. Classification Focal Loss
            # One-hot encoding of targets (background is all zeros)
            cls_one_hot = torch.zeros_like(cls_logits)
            if pos_mask.any():
                pos_labels = cls_gt[pos_mask]
                cls_one_hot[pos_mask] = F.one_hot(pos_labels, num_classes=num_classes).float()

            cls_loss = sigmoid_focal_loss(
                cls_logits,
                cls_one_hot,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="sum",
            )
            total_cls_loss = total_cls_loss + cls_loss

            # 2. Regression & Centerness Loss on positive locations
            if pos_mask.any():
                pred_ltrb_pos = reg_ltrb[pos_mask]            # (N_pos, 4)
                gt_ltrb_pos = reg_gt[pos_mask]                # (N_pos, 4)
                gt_cent_pos = cent_gt[pos_mask]                # (N_pos,)
                pred_cent_pos = cent_logits[pos_mask]         # (N_pos,)

                # GIoU loss weighted by ground-truth centerness
                giou_losses = giou_loss_ltrb(pred_ltrb_pos, gt_ltrb_pos)  # (N_pos,)
                reg_loss = (giou_losses * gt_cent_pos).sum()
                total_reg_loss = total_reg_loss + reg_loss
                total_cent_gt_sum = total_cent_gt_sum + gt_cent_pos.sum()

                # Centerness BCE loss
                cent_loss = F.binary_cross_entropy_with_logits(
                    pred_cent_pos,
                    gt_cent_pos,
                    reduction="sum",
                )
                total_cent_loss = total_cent_loss + cent_loss

        normalizer = max(total_num_pos, 1.0)
        cls_loss_norm = total_cls_loss / normalizer
        reg_loss_norm = total_reg_loss / normalizer
        cent_loss_norm = total_cent_loss / normalizer

        total_loss = (
            cls_loss_norm
            + self.lambda_reg * reg_loss_norm
            + self.lambda_centerness * cent_loss_norm
        )

        return total_loss, {
            "loss": total_loss.item(),
            "cls_loss": cls_loss_norm.item(),
            "reg_loss": reg_loss_norm.item(),
            "cent_loss": cent_loss_norm.item(),
            "num_pos": total_num_pos,
        }


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing DetectionLoss and FCOSLoss...")
    fcos_loss = FCOSLoss()
    print("FCOSLoss initialized successfully.")
