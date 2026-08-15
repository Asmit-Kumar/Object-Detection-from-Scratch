"""
Detection Loss Utilities.

Contains IoU helpers, coordinate regression loss, classification loss,
and objectness loss functions (BCE with dynamic positive weighting and Sigmoid Focal Loss)
for the single-stage and multi-anchor spatial detection pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import time
from torchvision.ops import sigmoid_focal_loss


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

def aligned_iou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """Compute IoU for corresponding boxes in two equally shaped tensors.

    Args:
        pred_boxes: Tensor ``(..., 4)`` in ``[x, y, w, h]`` format.
        target_boxes: Tensor ``(..., 4)`` in ``[x, y, w, h]`` format.

    Returns:
        Tensor ``(...)`` containing the IoU for each corresponding pair.
    """
    px1, py1 = pred_boxes[..., 0], pred_boxes[..., 1]
    px2 = px1 + pred_boxes[..., 2]
    py2 = py1 + pred_boxes[..., 3]

    tx1, ty1 = target_boxes[..., 0], target_boxes[..., 1]
    tx2 = tx1 + target_boxes[..., 2]
    ty2 = ty1 + target_boxes[..., 3]

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)

    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    pred_area = pred_boxes[..., 2] * pred_boxes[..., 3]
    target_area = target_boxes[..., 2] * target_boxes[..., 3]
    union = pred_area + target_area - inter

    return inter / torch.clamp(union, min=1e-6)


# ── Detection Loss ────────────────────────────────────────────────────────────

class DetectionLoss(nn.Module):
    """Unified tri-head detection loss for multi-object localization, objectness scoring,
    and character classification across K anchor slots per spatial cell.

    For each image in the batch:
      1. **Spatial Target Mapping** — assigns GT boxes to corresponding $14 \times 14 \times K$
         grid-anchor slots during collate_fn.
      2. **Box loss (Huber)** — ``HuberLoss`` on matched (pred_box, gt_box) positive slots.
      3. **IoU loss** — ``1.0 - mean(aligned_iou)`` on matched positive slots.
      4. **Objectness loss (Focal or BCE)** — ``sigmoid_focal_loss`` or ``BCEWithLogitsLoss``
         on all spatial/anchor slots; matched slots get target ``1.0``, background slots get ``0.0``.
      5. **Class loss (CE)** — ``CrossEntropyLoss`` computed exclusively on matched slots.

    Total loss = (box_loss + lambda_iou * iou_loss) + lambda_conf * conf_loss + lambda_class * class_loss

    Args:
        lambda_conf    : Weight for the objectness confidence loss term. Default ``1.0``.
        delta          : Huber loss delta (transition point). Default ``1.0``.
        use_pos_weight : Whether to compute positive class weight dynamically for BCE. Default ``False``.
        lambda_class   : Weight for the classification loss term. Default ``1.0``.
        lambda_iou     : Weight for the IoU loss term. Default ``1.0``.
        use_focal_loss : Whether to use Sigmoid Focal Loss for objectness. Default ``True``.
        focal_alpha    : Focal loss alpha balancing factor. Default ``0.25``.
        focal_gamma    : Focal loss gamma focusing parameter. Default ``2.0``.
        pos_weight_cap : Maximum cap for BCE dynamic positive weight. Default ``15.0``.

    Inputs:
        outputs   : ``Tensor (B, 14, 14, K, 5 + num_classes)``
                    Last dim: ``[x, y, w, h, conf_logit, class_logits...]``.
        target_gt : ``Tensor (B, 14, 14, K, 5)``
                    Last dim: ``[x, y, w, h, objectness]``.
        labels    : ``Tensor (B, 14, 14, K)`` — GT class ids aligned with ``target_gt``.
    """

    def __init__(
        self,
        lambda_conf: float = 1.0,
        delta: float = 1.0,
        use_pos_weight: bool = False,
        lambda_class: float = 1.0,
        lambda_iou: float = 1.0,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight_cap: float = 15.0,
    ):
        super().__init__()
        self.lambda_conf = lambda_conf
        self.lambda_class = lambda_class
        self.lambda_iou = lambda_iou
        self.use_pos_weight = use_pos_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight_cap = pos_weight_cap
        self.huber = nn.HuberLoss(reduction='mean', delta=delta)
        self.bce   = nn.BCEWithLogitsLoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(
            self,
            outputs: torch.Tensor,
            target_gt: torch.Tensor,
            labels: torch.Tensor | None = None,
    ) -> torch.Tensor:

        pred_boxes = outputs[..., :4]
        pred_conf = outputs[..., 4]
        pred_class = outputs[..., 5:]

        gt_boxes = target_gt[..., :4]
        gt_objectness = target_gt[..., 4]

        positive = gt_objectness.bool()

        if self.use_pos_weight and not self.use_focal_loss:
            n_pos = positive.sum()
            n_cells = positive.numel()

            if n_pos > 0:
                pw = (n_cells - n_pos) / n_pos
                pw = torch.clamp(pw, max=self.pos_weight_cap)
                pos_weight = torch.tensor(
                    [pw],
                    dtype=pred_conf.dtype,
                    device=pred_conf.device,
                )
            else:
                pos_weight = None
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


if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LAMBDA_CONF = 1.0
    NUM_CLASSES = 10
    B = 2
    K_ANCHORS = 3
    
    criterion = DetectionLoss(lambda_conf=LAMBDA_CONF, delta=1.0, use_pos_weight=True, use_focal_loss=True)
    print('DetectionLoss ready.')

    S = 14
    NUM_CLASSES = 47
    print(f"Creating sample inputs for batch of {B}...")

    # dummy_out: (B, S, S, K, 5 + NUM_CLASSES)
    dummy_out = torch.randn(B, S, S, K_ANCHORS, 5 + NUM_CLASSES, device=DEVICE)
    # target_gt: (B, S, S, K, 5) where last dim is [x, y, w, h, objectness]
    target_gt = torch.zeros(B, S, S, K_ANCHORS, 5, device=DEVICE)
    # populate some random ground truth objects
    target_gt[:, 3, 3, 0, 4] = 1.0  # object at cell (3,3), anchor 0
    target_gt[:, 5, 5, 1, 4] = 1.0  # object at cell (5,5), anchor 1

    # labels_v: (B, S, S, K)
    labels_v = torch.randint(0, NUM_CLASSES, (B, S, S, K_ANCHORS), device=DEVICE)

    with torch.no_grad():
        loss_val = criterion(dummy_out, target_gt, labels_v)
    print(f'Smoke-test loss: {loss_val.item():.4f}')
