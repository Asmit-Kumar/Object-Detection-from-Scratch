"""Prediction-aware task-aligned assignment across all detection scales."""

from dataclasses import dataclass

import torch

from dataio.ltrb_targets import LTRBTargetGenerator


def flatten_ltrb_predictions(preds_by_scale, strides, grid_sizes=(28, 14)):
    logits, distances, points = [], [], []
    for grid in grid_sizes:
        pred = preds_by_scale[grid]
        batch, height, width, _ = pred["reg_ltrb"].shape
        logits.append(pred["cls_logits"].float().reshape(batch, -1, pred["cls_logits"].shape[-1]))
        distances.append(pred["reg_ltrb"].float().reshape(batch, -1, 4))
        points.append(LTRBTargetGenerator.get_grid_centers(
            height, width, strides[grid], pred["reg_ltrb"].device,
        ).reshape(-1, 2))
    logits, distances, points = torch.cat(logits, 1), torch.cat(distances, 1), torch.cat(points)
    boxes = torch.cat((points - distances[..., :2], points + distances[..., 2:]), -1)
    return logits, distances, points, boxes


@dataclass
class TaskAlignedTargets:
    scores: torch.Tensor
    ltrb: torch.Tensor
    matched_gt: torch.Tensor


class TaskAlignedAssigner:
    def __init__(self, topk: int, alpha: float = 0.5, beta: float = 6.0):
        if topk < 1 or alpha <= 0 or beta <= 0:
            raise ValueError("topk, alpha and beta must be positive")
        self.topk, self.alpha, self.beta = topk, alpha, beta

    @torch.no_grad()
    def __call__(self, logits, boxes_xyxy, points, boxes_batch, labels_batch):
        batch, locations, classes = logits.shape
        if len(boxes_batch) != batch or len(labels_batch) != batch:
            raise ValueError("Annotation batch size must match predictions")
        scores = torch.zeros_like(logits, dtype=torch.float32)
        ltrb = torch.zeros_like(boxes_xyxy, dtype=torch.float32)
        matched_gt = torch.full((batch, locations), -1, device=logits.device, dtype=torch.long)
        max_gt = max((len(boxes) for boxes in boxes_batch), default=0)
        if not max_gt:
            if any(len(labels) for labels in labels_batch):
                raise ValueError("Each ground-truth box needs a class label")
            return TaskAlignedTargets(scores, ltrb, matched_gt)
        gt = logits.new_zeros((batch, max_gt, 4), dtype=torch.float32)
        gt_labels = torch.zeros((batch, max_gt), device=logits.device, dtype=torch.long)
        valid = torch.zeros((batch, max_gt), device=logits.device, dtype=torch.bool)
        for b, (boxes, labels) in enumerate(zip(boxes_batch, labels_batch)):
            boxes = torch.as_tensor(boxes, device=logits.device, dtype=torch.float32).reshape(-1, 4)
            labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long).reshape(-1)
            if len(boxes) != len(labels):
                raise ValueError("Each ground-truth box needs a class label")
            count = len(boxes)
            gt[b, :count] = torch.cat((boxes[:, :2], boxes[:, :2] + boxes[:, 2:]), -1)
            gt_labels[b, :count] = labels
            valid[b, :count] = True
        if ((gt_labels < 0) | (gt_labels >= classes)).any():
            raise ValueError("Ground-truth class is out of range")
        gt_dist = torch.cat((points[None, None] - gt[:, :, None, :2],
                             gt[:, :, None, 2:] - points[None, None]), -1)
        inside = (gt_dist.amin(-1) > 0) & valid[..., None]
        intersection = (torch.minimum(gt[:, :, None, 2:], boxes_xyxy[:, None, :, 2:])
                        - torch.maximum(gt[:, :, None, :2], boxes_xyxy[:, None, :, :2])).clamp_min(0).prod(-1)
        gt_area = (gt[..., 2:] - gt[..., :2]).clamp_min(0).prod(-1)
        pred_area = (boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]).clamp_min(0).prod(-1)
        overlaps = intersection / (gt_area[..., None] + pred_area[:, None] - intersection).clamp_min(1e-7)
        class_scores = logits.sigmoid().transpose(1, 2).gather(
            1, gt_labels[..., None].expand(-1, -1, locations),
        )
        metric = class_scores.pow(self.alpha) * overlaps.pow(self.beta) * inside
        values, indices = metric.topk(min(self.topk, locations), dim=2)
        positive = torch.zeros_like(inside)
        positive.scatter_(2, indices, values > 0)
        # Conflicting locations belong only to the highest-IoU candidate GT.
        winner = overlaps.masked_fill(~positive, -1).argmax(1)
        foreground = positive.any(1)
        positive &= torch.nn.functional.one_hot(winner, max_gt).transpose(1, 2).bool()
        pos_metric = metric * positive
        quality = (
            pos_metric * (overlaps * positive).amax(2, keepdim=True)
            / pos_metric.amax(2, keepdim=True).clamp_min(1e-12)
        ).amax(1)
        b, index = torch.where(foreground)
        matched_gt[b, index] = winner[b, index]
        scores[b, index, gt_labels[b, winner[b, index]]] = quality[b, index]
        ltrb[b, index] = gt_dist[b, winner[b, index], index]
        return TaskAlignedTargets(scores, ltrb, matched_gt)
