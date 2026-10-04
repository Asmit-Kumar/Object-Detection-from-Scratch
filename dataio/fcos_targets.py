"""
FCOS Target Generator & Center Sampling.

Computes ground-truth target tensors for multi-scale anchor-free detection:
  - Per-pixel distance regression targets: [l, t, r, b]
  - Centerness targets: sqrt(min(l, r)/max(l, r) * min(t, b)/max(t, b))
  - Multi-scale scale division with center-sampling radius
"""
from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class FCOSTargets:
    """Container for FCOS training targets partitioned by grid size."""
    cls_targets: dict[int, torch.Tensor]         # (B, H, W) class index or -1 for background
    reg_targets: dict[int, torch.Tensor]         # (B, H, W, 4) [l, t, r, b] in pixels
    centerness_targets: dict[int, torch.Tensor]  # (B, H, W) float [0, 1]

    def to(self, device: torch.device | str, non_blocking: bool = False) -> FCOSTargets:
        return FCOSTargets(
            cls_targets={k: v.to(device, non_blocking=non_blocking) for k, v in self.cls_targets.items()},
            reg_targets={k: v.to(device, non_blocking=non_blocking) for k, v in self.reg_targets.items()},
            centerness_targets={k: v.to(device, non_blocking=non_blocking) for k, v in self.centerness_targets.items()},
        )


class FCOSTargetGenerator:
    """
    Generates multi-scale FCOS training targets with center sampling and scale partitioning.

    Args:
        grid_sizes: Tuple of grid sizes, e.g. (28, 14).
        strides: Mapping of grid size to stride in pixels, e.g. {28: 8.0, 14: 16.0}.
        scale_ranges: Mapping of grid size to [min_dist, max_dist] for scale assignment.
        center_sampling_radius: Radius multiplier relative to stride for center sampling.
        canvas_size: Square canvas spatial size. Default 224.
    """

    DEFAULT_SCALE_RANGES = {
        28: (0.0, 32.0),
        14: (24.0, float("inf")),
    }

    DEFAULT_STRIDES = {
        28: 8.0,
        14: 16.0,
    }

    def __init__(
            self,
            grid_sizes: tuple[int, int] = (28, 14),
            strides: dict[int, float] | None = None,
            scale_ranges: dict[int, tuple[float, float]] | None = None,
            center_sampling_radius: float = 1.5,
            canvas_size: int = 224,
    ):
        self.grid_sizes = grid_sizes
        self.strides = strides or self.DEFAULT_STRIDES
        self.scale_ranges = scale_ranges or self.DEFAULT_SCALE_RANGES
        self.center_sampling_radius = center_sampling_radius
        self.canvas_size = canvas_size

    @staticmethod
    def get_grid_centers(height: int, width: int, stride: float, device: torch.device) -> torch.Tensor:
        """Return (H, W, 2) grid center locations (xc, yc) in pixel space."""
        shift_x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
        shift_y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1)

    def generate_targets(
            self,
            boxes_batch: list[torch.Tensor] | torch.Tensor,
            labels_batch: list[torch.Tensor] | torch.Tensor,
            device: torch.device,
    ) -> FCOSTargets:
        """
        Generate FCOS targets for a batch of ground-truth annotations.

        Args:
            boxes_batch: List of (N_i, 4) tensors in absolute [x, y, w, h] format.
            labels_batch: List of (N_i,) integer class label tensors.
            device: Target torch device.

        Returns:
            FCOSTargets container containing cls_targets, reg_targets, and centerness_targets.
        """
        batch_size = len(boxes_batch)
        if batch_size == 0:
            raise ValueError("boxes_batch must contain at least one image")

        # Pad the variable-length annotations once, then evaluate every image,
        # location, and candidate box with batched tensor operations. This keeps
        # the CUDA path efficient by avoiding many small per-image kernel launches.
        normalized_boxes = []
        normalized_labels = []
        max_boxes = 0
        for boxes, labels in zip(boxes_batch, labels_batch):
            boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
            if boxes.numel() == 0:
                boxes = boxes.reshape(0, 4)
            normalized_boxes.append(boxes)
            normalized_labels.append(labels.reshape(-1))
            max_boxes = max(max_boxes, boxes.shape[0])

        cls_targets_by_scale = {}
        reg_targets_by_scale = {}
        cent_targets_by_scale = {}
        for grid_size in self.grid_sizes:
            cls_targets_by_scale[grid_size] = torch.full(
                (batch_size, grid_size, grid_size), -1, dtype=torch.long, device=device
            )
            reg_targets_by_scale[grid_size] = torch.zeros(
                (batch_size, grid_size, grid_size, 4), dtype=torch.float32, device=device
            )
            cent_targets_by_scale[grid_size] = torch.zeros(
                (batch_size, grid_size, grid_size), dtype=torch.float32, device=device
            )

        if max_boxes == 0:
            return FCOSTargets(
                cls_targets=cls_targets_by_scale,
                reg_targets=reg_targets_by_scale,
                centerness_targets=cent_targets_by_scale,
            )

        boxes_padded = torch.zeros(
            (batch_size, max_boxes, 4), dtype=torch.float32, device=device
        )
        labels_padded = torch.zeros(
            (batch_size, max_boxes), dtype=torch.long, device=device
        )
        valid_boxes = torch.zeros(
            (batch_size, max_boxes), dtype=torch.bool, device=device
        )
        for batch_idx, (boxes, labels) in enumerate(zip(normalized_boxes, normalized_labels)):
            count = boxes.shape[0]
            if count:
                boxes_padded[batch_idx, :count] = boxes
                labels_padded[batch_idx, :count] = labels[:count]
                valid_boxes[batch_idx, :count] = True

        x1 = boxes_padded[..., 0].view(batch_size, 1, 1, max_boxes)
        y1 = boxes_padded[..., 1].view(batch_size, 1, 1, max_boxes)
        x2 = (boxes_padded[..., 0] + boxes_padded[..., 2]).view(batch_size, 1, 1, max_boxes)
        y2 = (boxes_padded[..., 1] + boxes_padded[..., 3]).view(batch_size, 1, 1, max_boxes)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        areas = (boxes_padded[..., 2] * boxes_padded[..., 3]).view(batch_size, 1, 1, max_boxes)
        valid_boxes = valid_boxes.view(batch_size, 1, 1, max_boxes)

        centers_by_grid = {
            grid_size: self.get_grid_centers(
                grid_size, grid_size, self.strides[grid_size], device
            )
            for grid_size in self.grid_sizes
        }

        for grid_size in self.grid_sizes:
            stride = self.strides[grid_size]
            min_scale, max_scale = self.scale_ranges[grid_size]
            centers = centers_by_grid[grid_size]
            xc = centers[..., 0].view(1, grid_size, grid_size, 1)
            yc = centers[..., 1].view(1, grid_size, grid_size, 1)

            l = xc - x1
            t = yc - y1
            r = x2 - xc
            b = y2 - yc
            ltrb = torch.stack((l, t, r, b), dim=-1)

            radius = self.center_sampling_radius * stride
            cx1 = torch.maximum(x1, cx - radius)
            cy1 = torch.maximum(y1, cy - radius)
            cx2 = torch.minimum(x2, cx + radius)
            cy2 = torch.minimum(y2, cy + radius)
            cl = xc - cx1
            ct = yc - cy1
            cr = cx2 - xc
            cb = cy2 - yc
            in_subbox = torch.minimum(torch.minimum(cl, ct), torch.minimum(cr, cb)) > 0

            max_dist = ltrb.amax(dim=-1)
            in_scale = (max_dist >= min_scale) & (max_dist <= max_scale)
            valid_mask = in_subbox & in_scale & valid_boxes

            box_areas_masked = torch.where(valid_mask, areas, torch.full_like(areas, float("inf")))
            min_area, matched_gt_idx = box_areas_masked.min(dim=-1)
            pos_mask = torch.isfinite(min_area)

            gather_idx = matched_gt_idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, 1, 4
            )
            pos_ltrb = torch.gather(ltrb, dim=3, index=gather_idx).squeeze(3)
            labels_grid = labels_padded.view(batch_size, 1, 1, max_boxes).expand(
                -1, grid_size, grid_size, -1
            )
            pos_labels = torch.gather(
                labels_grid, dim=3, index=matched_gt_idx.unsqueeze(-1)
            ).squeeze(-1)

            cls_targets_by_scale[grid_size] = torch.where(
                pos_mask, pos_labels, cls_targets_by_scale[grid_size]
            )
            reg_targets_by_scale[grid_size] = torch.where(
                pos_mask.unsqueeze(-1), pos_ltrb, reg_targets_by_scale[grid_size]
            )

            pl, pt, pr, pb = [pos_ltrb[..., idx].clamp_min(1e-6) for idx in range(4)]
            cent = torch.sqrt(
                (torch.minimum(pl, pr) / torch.maximum(pl, pr))
                * (torch.minimum(pt, pb) / torch.maximum(pt, pb))
            ).clamp(0.0, 1.0)
            cent_targets_by_scale[grid_size] = torch.where(
                pos_mask, cent, cent_targets_by_scale[grid_size]
            )

        return FCOSTargets(
            cls_targets=cls_targets_by_scale,
            reg_targets=reg_targets_by_scale,
            centerness_targets=cent_targets_by_scale,
        )
