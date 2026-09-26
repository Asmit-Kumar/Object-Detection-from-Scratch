"""
utils/pipeline.py — End-to-End Object Detection & Character Recognition Pipeline.

Combines:
  1. ObjectDetectorResNet / FCOSObjectDetectorResNet (Stage 1: Multi-slot / Anchor-free character detection)
  2. CharacterClassifierResNet (Stage 2: 47-class EMNIST ByMerge character recognition)

Features:
  - High-throughput batched inference
  - Vectorized GPU batch crop extraction
  - Single-pass parallel classifier execution
  - Concurrent multi-stream evaluation across benchmark placement layouts (random, grid, words, line)
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
import concurrent.futures

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from generator.dataset import EMNIST_CLASS_NAMES


def _apply_nms(boxes_xywh: torch.Tensor, confs: torch.Tensor, iou_thresh: float = 0.35) -> torch.Tensor:
    """
    Apply Non-Maximum Suppression (NMS) on predicted bounding boxes [x, y, w, h].
    Returns indices of boxes to keep.
    """
    if boxes_xywh.shape[0] <= 1:
        return torch.arange(boxes_xywh.shape[0], device=boxes_xywh.device)
    x1 = boxes_xywh[:, 0]
    y1 = boxes_xywh[:, 1]
    x2 = x1 + boxes_xywh[:, 2]
    y2 = y1 + boxes_xywh[:, 3]
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return torchvision.ops.nms(boxes_xyxy, confs, iou_thresh)


def _decode_fcos_candidates(
        detector,
        outputs: dict[int, dict[str, torch.Tensor]],
        score_threshold: float,
        iou_threshold: float = 0.45,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Decode FCOS outputs, gate scores with centerness, and apply class-wise NMS.

    Returns one ``(boxes_xywh, scores, classes)`` tuple per batch item. The
    highest-scoring class at each location is retained because the second
    pipeline stage performs the final character classification.
    """
    first_reg = next(iter(outputs.values()))["reg_ltrb"]
    batch_size = first_reg.size(0)
    num_classes = next(iter(outputs.values()))["cls_logits"].size(-1)
    all_boxes = []
    all_scores = []
    all_classes = []
    all_batch_ids = []

    # Decode every image and pyramid level in one tensor pass. NMS is then
    # performed once for the whole batch, grouped by (image, class), instead
    # of launching one NMS call per image.
    for grid_size, preds in outputs.items():
        reg_ltrb = preds["reg_ltrb"].float()
        cls_probs = preds["cls_logits"].float().sigmoid()
        cent_probs = preds["centerness_logits"].float().sigmoid().unsqueeze(-1)
        scores, classes = torch.sqrt((cls_probs * cent_probs).clamp_min(1e-12)).max(dim=-1)

        height, width = reg_ltrb.shape[1:3]
        centers = detector.generate_grid_centers(
            height, width, detector.STRIDES[grid_size], reg_ltrb.device
        )
        x1 = centers[..., 0].view(1, height, width) - reg_ltrb[..., 0]
        y1 = centers[..., 1].view(1, height, width) - reg_ltrb[..., 1]
        x2 = centers[..., 0].view(1, height, width) + reg_ltrb[..., 2]
        y2 = centers[..., 1].view(1, height, width) + reg_ltrb[..., 3]
        boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=-1).clamp(0.0, 224.0)

        active = scores >= score_threshold
        if active.any():
            all_boxes.append(boxes_xyxy[active])
            all_scores.append(scores[active])
            all_classes.append(classes[active])
            all_batch_ids.append(
                torch.arange(batch_size, device=reg_ltrb.device)
                .view(batch_size, 1, 1)
                .expand_as(scores)[active]
            )

    if not all_boxes:
        empty = first_reg.new_zeros((0, 4))
        return [
            (empty, empty.new_zeros((0,)), empty.new_zeros((0,), dtype=torch.long))
            for _ in range(batch_size)
        ]

    boxes_xyxy = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    classes = torch.cat(all_classes, dim=0)
    batch_ids = torch.cat(all_batch_ids, dim=0)
    nms_groups = batch_ids * num_classes + classes
    keep = torchvision.ops.batched_nms(boxes_xyxy, scores, nms_groups, iou_threshold)

    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    classes = classes[keep]
    batch_ids = batch_ids[keep]
    candidates = []
    for batch_idx in range(batch_size):
        selected = batch_ids == batch_idx
        boxes = boxes_xyxy[selected]
        boxes_xywh = torch.cat(
            (boxes[:, :2], (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0)),
            dim=-1,
        )
        candidates.append((boxes_xywh, scores[selected], classes[selected]))
    return candidates


def _flatten_detection_outputs(outputs: torch.Tensor | dict[int, torch.Tensor]) -> torch.Tensor:
    """Flatten a one- or multi-scale detector result to ``(B, N, D)``."""
    if isinstance(outputs, dict):
        return torch.cat(
            [output.reshape(output.size(0), -1, output.size(-1)) for output in outputs.values()],
            dim=1,
        )
    return outputs.reshape(outputs.size(0), -1, outputs.size(-1))


def _flatten_detection_targets(targets: torch.Tensor | dict[int, torch.Tensor]) -> torch.Tensor:
    """Flatten one- or multi-scale target grids to ``(B, N, 5)``."""
    return _flatten_detection_outputs(targets)


def _flatten_detection_labels(labels: torch.Tensor | dict[int, torch.Tensor]) -> torch.Tensor:
    """Flatten one- or multi-scale target label grids to ``(B, N)``."""
    if isinstance(labels, dict):
        return torch.cat(
            [label.reshape(label.size(0), -1) for label in labels.values()],
            dim=1,
        )
    return labels.reshape(labels.size(0), -1)

CLASSIFIER_MODEL_PATH = 'weights/classifier_resent_bymerge_s_best.pth'
DETECTOR_MODEL_PATH = {
    'n': 'weights/detector_n_new_best.pth',
    's': 'weights/detector_s_new_best.pth',
    'm': 'weights/detector_m_new_best.pth',
}
BENCHMARK_PATH = 'data/OD_benchmark'
PLACEMENTS = ['random', 'grid', 'words', 'line']

_EMNIST_MEAN = 0.1307
_EMNIST_STD  = 0.3081


@dataclass
class DetectionResult:
    bbox: List[float]
    detector_conf: float
    class_idx: int
    char_label: str
    classifier_conf: float
    joint_conf: float


class DetectionPipeline:
    """
    Two-Stage End-to-End Detection Pipeline:
      Stage 1: ObjectDetectorResNet / FCOSObjectDetectorResNet — detects character bounding boxes.
      Stage 2: CharacterClassifierResNet — 47-class EMNIST ByMerge recognition.
    """

    def __init__(
        self,
        detector_size: str = 's',
        detector_weights: str = None,
        classifier_weights: str = CLASSIFIER_MODEL_PATH,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        conf_threshold: float = 0.70,
        iou_threshold: float = 0.45,
        num_classes: int = 47,
        multi_scale_detector: bool = False,
        fcos_detector: bool = False,
    ):
        self.detector_size   = detector_size
        self.conf_threshold  = conf_threshold
        self.iou_threshold   = iou_threshold
        self.device          = device
        self.num_classes     = num_classes
        self.multi_scale_detector = multi_scale_detector
        self.fcos_detector = fcos_detector

        self.detector   = self._load_detector(detector_weights)
        self.classifier = self._load_classifier(classifier_weights, num_classes)
        self.class_names = EMNIST_CLASS_NAMES["bymerge"]

    def _load_detector(self, weights_path: str = None):
        if self.fcos_detector:
            from models import load_fcos_detector
            loader = load_fcos_detector
        elif self.multi_scale_detector:
            from models import load_multiscale_detector
            loader = load_multiscale_detector
        else:
            from models import load_detector
            loader = load_detector

        path = weights_path or DETECTOR_MODEL_PATH.get(self.detector_size.lower(), '')
        print(f"[DetectionPipeline] Loading Detector ('{self.detector_size}') from '{path}'...")
        if path and Path(path).exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "anchors_wh" in ckpt:
                self.anchors_wh = ckpt["anchors_wh"]
            else:
                self.anchors_wh = None
        else:
            self.anchors_wh = None

        return loader(path=path if path and Path(path).exists() else None, device=self.device, size=self.detector_size)

    def _load_classifier(self, weights_path: str = None, num_classes: int = 47):
        from models import load_classifier
        path = weights_path or CLASSIFIER_MODEL_PATH
        print(f"[DetectionPipeline] Loading Classifier from '{path}'...")
        return load_classifier(path=path, device=self.device, num_classes=num_classes)

    def _preprocess_image_input(self, image_input: Any) -> torch.Tensor:
        """
        Normalize input image into a (1, 1, 224, 224) float tensor in [0, 1] on self.device.
        """
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not load image at {img_path}")
            tensor = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                tensor = torch.from_numpy(image_input).float().unsqueeze(0).unsqueeze(0)
            elif image_input.ndim == 3:
                tensor = torch.from_numpy(image_input[..., 0]).float().unsqueeze(0).unsqueeze(0)
            else:
                tensor = torch.from_numpy(image_input).float()
            if tensor.max() > 1.0:
                tensor = tensor.div(255.0)
        elif isinstance(image_input, torch.Tensor):
            tensor = image_input.float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.max() > 1.0:
                tensor = tensor.div(255.0)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if tensor.shape[-2:] != (224, 224):
            tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)

        return tensor.to(self.device)

    def crop_and_preprocess_patch(
        self,
        image_tensor: torch.Tensor,
        bbox: List[float],
        margin: int = 1,
    ) -> torch.Tensor:
        """
        Extract a single bbox patch from a (1, 224, 224) image tensor, apply background cleaning,
        contrast normalization, square aspect-ratio padding, 28x28 resizing, and EMNIST tensor transposition.

        Returns:
            torch.Tensor: (1, 28, 28) CUDA tensor normalized to EMNIST mean/std and transposed to match EMNIST ubyte layout.
        """
        _, h_img, w_img = image_tensor.shape
        x, y, w, h = bbox

        x1 = max(0, int(round(x - margin)))
        y1 = max(0, int(round(y - margin)))
        x2 = min(w_img, int(round(x + w + margin)))
        y2 = min(h_img, int(round(y + h + margin)))

        # Fallback for degenerate box
        if x2 <= x1 or y2 <= y1:
            crop = torch.zeros((1, 28, 28), device=self.device)
            return ((crop - _EMNIST_MEAN) / _EMNIST_STD).transpose(-1, -2)

        crop = image_tensor[:, y1:y2, x1:x2]  # (1, H_crop, W_crop)

        # Background cleaning & contrast normalization
        border_vals = torch.cat([
            crop[0, 0, :], crop[0, -1, :],
            crop[0, :, 0], crop[0, :, -1]
        ])
        bg_est = torch.median(border_vals)
        crop_clean = torch.clamp(crop - bg_est, min=0.0)
        max_val = crop_clean.max()
        if max_val > 0.02:
            crop_clean = crop_clean / max_val

        # Aspect-ratio preserving square pad
        _, c_h, c_w = crop_clean.shape
        max_dim = max(c_h, c_w)
        pad_top = (max_dim - c_h) // 2
        pad_bottom = max_dim - c_h - pad_top
        pad_left = (max_dim - c_w) // 2
        pad_right = max_dim - c_w - pad_left

        crop_padded = F.pad(crop_clean, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        crop_resized = F.interpolate(
            crop_padded.unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False
        ).squeeze(0)

        # Normalize with EMNIST stats & transpose (-1, -2) to match EMNIST classifier expected layout
        crop_normalized = (crop_resized - _EMNIST_MEAN) / _EMNIST_STD
        return crop_normalized.transpose(-1, -2)

    @torch.no_grad()
    def _detector_candidates(
        self,
        images_tensor: torch.Tensor,
        conf_threshold: float,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Run detector decoding and NMS for every image in a batch."""
        if self.fcos_detector:
            raw_out = self.detector(images_tensor)
            return _decode_fcos_candidates(
                self.detector,
                raw_out,
                score_threshold=conf_threshold,
                iou_threshold=self.iou_threshold,
            )

        det_outputs = _flatten_detection_outputs(self.detector(images_tensor))
        candidates = []
        for det_output in det_outputs:
            det_output = det_output.reshape(-1, det_output.shape[-1])
            boxes = det_output[:, :4]
            confs = torch.sigmoid(det_output[:, 4])
            mask = confs >= conf_threshold
            if not mask.any():
                candidates.append((boxes.new_zeros((0, 4)), confs.new_zeros((0,)), torch.zeros((0,), dtype=torch.long, device=boxes.device)))
                continue
            filtered_boxes = boxes[mask]
            filtered_confs = confs[mask]
            keep = _apply_nms(filtered_boxes, filtered_confs, iou_thresh=self.iou_threshold)
            candidates.append((filtered_boxes[keep], filtered_confs[keep]))
        return candidates

    @torch.no_grad()
    def predict_image(
        self,
        image_input: Any,
        conf_threshold: float | None = None,
    ) -> List[DetectionResult]:
        """
        Run end-to-end pipeline on a single image.

        1. Stage 1: Detector output -> filter confidence >= conf_threshold.
        2. Stage 2: Parallel crop extraction -> single-pass batched classifier pass.
        3. Returns List[DetectionResult].
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        img_tensor = self._preprocess_image_input(image_input)  # (1, 1, 224, 224)

        # Stage 1: Detector, decoding, centerness gating, and NMS.
        boxes, confs, _ = self._detector_candidates(img_tensor, conf_threshold)[0]
        if boxes.numel() == 0:
            return []

        valid_boxes = boxes.cpu().numpy().tolist()
        valid_confs = confs.cpu().numpy().tolist()

        # Stage 2: Parallel Crop Extraction
        img_single = img_tensor.squeeze(0)  # (1, 224, 224)
        crops = [
            self.crop_and_preprocess_patch(img_single, bbox)
            for bbox in valid_boxes
        ]
        crops_batch = torch.stack(crops, dim=0).to(self.device)  # (K, 1, 28, 28)

        # Stage 2: Parallel Classifier Forward Pass
        cls_logits = self.classifier(crops_batch)  # (K, 47)
        cls_probs = F.softmax(cls_logits, dim=-1)  # (K, 47)
        top_probs, top_classes = torch.max(cls_probs, dim=-1)

        top_probs_list = top_probs.cpu().numpy().tolist()
        top_classes_list = top_classes.cpu().numpy().tolist()

        results = []
        for bbox, det_c, cls_idx, cls_c in zip(valid_boxes, valid_confs, top_classes_list, top_probs_list):
            char_str = self.class_names[cls_idx] if 0 <= cls_idx < len(self.class_names) else "?"
            results.append(
                DetectionResult(
                    bbox=[round(v, 2) for v in bbox],
                    detector_conf=round(float(det_c), 4),
                    class_idx=int(cls_idx),
                    char_label=char_str,
                    classifier_conf=round(float(cls_c), 4),
                    joint_conf=round(float(det_c * cls_c), 4),
                )
            )

        return results

    @torch.no_grad()
    def predict_batch(
        self,
        images_tensor: torch.Tensor,
        conf_threshold: float | None = None,
    ) -> List[List[DetectionResult]]:
        """
        Run vectorized parallel pipeline on a batch of images (B, 1, 224, 224).

        Performs a single parallel detector pass, extracts all valid crops across the entire batch,
        and executes a SINGLE GPU-vectorized classifier forward pass on all crops simultaneously!
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        images_tensor = images_tensor.to(self.device)
        B = images_tensor.shape[0]

        # Stage 1: Detector Batch Forward Pass, decoding, centerness gating, and NMS.
        detector_candidates = self._detector_candidates(images_tensor, conf_threshold)

        batch_crops = []
        batch_crop_metadata = []  # (batch_idx, bbox, det_conf)

        for b in range(B):
            img_single = images_tensor[b]
            boxes, confs, _ = detector_candidates[b]
            if boxes.numel() == 0:
                continue

            valid_boxes = boxes.cpu().numpy().tolist()
            valid_confs = confs.cpu().numpy().tolist()

            for bbox, det_c in zip(valid_boxes, valid_confs):
                crop = self.crop_and_preprocess_patch(img_single, bbox)
                batch_crops.append(crop)
                batch_crop_metadata.append((b, bbox, det_c))

        batch_results: List[List[DetectionResult]] = [[] for _ in range(B)]

        if not batch_crops:
            return batch_results

        # Stage 2: Vectorized Parallel Classifier Forward Pass (All crops in batch at once)
        all_crops_tensor = torch.stack(batch_crops, dim=0).to(self.device)  # (N_total, 1, 28, 28)
        cls_logits = self.classifier(all_crops_tensor)  # (N_total, 47)
        cls_probs = F.softmax(cls_logits, dim=-1)
        top_probs, top_classes = torch.max(cls_probs, dim=-1)

        top_probs_list = top_probs.cpu().numpy().tolist()
        top_classes_list = top_classes.cpu().numpy().tolist()

        for (b_idx, bbox, det_c), cls_idx, cls_c in zip(batch_crop_metadata, top_classes_list, top_probs_list):
            char_str = self.class_names[cls_idx] if 0 <= cls_idx < len(self.class_names) else "?"
            batch_results[b_idx].append(
                DetectionResult(
                    bbox=[round(v, 2) for v in bbox],
                    detector_conf=round(float(det_c), 4),
                    class_idx=int(cls_idx),
                    char_label=char_str,
                    classifier_conf=round(float(cls_c), 4),
                    joint_conf=round(float(det_c * cls_c), 4),
                )
            )

        return batch_results

    def evaluate_loader(
        self,
        loader: Any,
        conf_threshold: float | None = None,
        iou_match_thresh: float = 0.50,
        matching_mode: str = "hungarian",
    ) -> Dict[str, float]:
        """
        Evaluate end-to-end detection and classification metrics on a DataLoader.
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        total_gt_boxes = 0
        total_pred_boxes = 0
        detector_tp = 0
        end2end_tp = 0
        correct_chars_on_matched_boxes = 0

        t0 = time.time()
        for batch in loader:
            images = batch[0]
            batch_results = self.predict_batch(images, conf_threshold=conf_threshold)

            for b in range(len(batch_results)):
                preds = batch_results[b]
                if len(batch) == 4:
                    gt_boxes_batch, gt_labels_batch, gt_mask_batch = batch[1], batch[2], batch[3]
                    mask = gt_mask_batch[b]
                    gt_boxes = gt_boxes_batch[b][mask].cpu().numpy()  # (M, 4)
                    gt_labels = gt_labels_batch[b][mask].cpu().numpy()  # (M,)
                elif len(batch) == 3 and isinstance(batch[2], tuple):
                    boxes_list, labels_list = batch[2]
                    gt_boxes = boxes_list[b].cpu().numpy()
                    gt_labels = labels_list[b].cpu().numpy()
                else:  # (images, targets, labels) grid format
                    targets_batch = _flatten_detection_targets(batch[1])
                    gt_labels_grid = _flatten_detection_labels(batch[2])
                    gt_obj = targets_batch[b, ..., 4].bool()
                    gt_boxes = targets_batch[b, ..., :4][gt_obj].cpu().numpy()
                    gt_labels = gt_labels_grid[b][gt_obj].cpu().numpy()

                n_gt = len(gt_boxes)
                total_gt_boxes += n_gt
                total_pred_boxes += len(preds)

                if n_gt == 0 or len(preds) == 0:
                    continue

                # Match predicted boxes to GT boxes via IoU
                pred_boxes   = np.array([p.bbox for p in preds], dtype=np.float32)  # (K, 4)
                pred_classes = np.array([p.class_idx for p in preds])               # (K,)

                # --- Vectorized IoU (K x M) via broadcasting ---
                px = pred_boxes[:, 0:1]; py = pred_boxes[:, 1:2]   # (K, 1)
                pw = pred_boxes[:, 2:3]; ph = pred_boxes[:, 3:4]   # (K, 1)
                gx = gt_boxes[:, 0];     gy = gt_boxes[:, 1]       # (M,)
                gw = gt_boxes[:, 2];     gh = gt_boxes[:, 3]       # (M,)

                ix1 = np.maximum(px, gx);          iy1 = np.maximum(py, gy)            # (K, M)
                ix2 = np.minimum(px + pw, gx + gw); iy2 = np.minimum(py + ph, gy + gh) # (K, M)
                inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)            # (K, M)
                union = pw * ph + gw * gh - inter                                        # (K, M)
                iou_mat = np.where(union > 0, inter / union, 0.0)                       # (K, M)

                if matching_mode == "hungarian":
                    from scipy.optimize import linear_sum_assignment
                    pred_indices, gt_indices = linear_sum_assignment(-iou_mat)
                    for k, m in zip(pred_indices, gt_indices):
                        if iou_mat[k, m] >= iou_match_thresh:
                            detector_tp += 1
                            if pred_classes[k] == gt_labels[m]:
                                correct_chars_on_matched_boxes += 1
                                end2end_tp += 1
                else:
                    sort_order = np.argsort([-p.detector_conf for p in preds])
                    matched_gt = set()
                    for k in sort_order:
                        best_m   = int(np.argmax(iou_mat[k]))
                        best_iou = iou_mat[k, best_m]

                        if best_iou >= iou_match_thresh and best_m not in matched_gt:
                            matched_gt.add(best_m)
                            detector_tp += 1

                            if pred_classes[k] == gt_labels[best_m]:
                                correct_chars_on_matched_boxes += 1
                                end2end_tp += 1

        elapsed = time.time() - t0
        det_p = detector_tp / max(1, total_pred_boxes)
        det_r = detector_tp / max(1, total_gt_boxes)
        det_f1 = (2 * det_p * det_r) / max(1e-6, det_p + det_r)

        cls_acc = correct_chars_on_matched_boxes / max(1, detector_tp)

        e2e_p = end2end_tp / max(1, total_pred_boxes)
        e2e_r = end2end_tp / max(1, total_gt_boxes)
        e2e_f1 = (2 * e2e_p * e2e_r) / max(1e-6, e2e_p + e2e_r)

        return {
            "total_gt": total_gt_boxes,
            "total_preds": total_pred_boxes,
            "det_precision": round(det_p, 4),
            "det_recall": round(det_r, 4),
            "det_f1": round(det_f1, 4),
            "classifier_acc": round(cls_acc, 4),
            "e2e_precision": round(e2e_p, 4),
            "e2e_recall": round(e2e_r, 4),
            "e2e_f1": round(e2e_f1, 4),
            "eval_time_sec": round(elapsed, 2),
            "fps": round(total_gt_boxes / max(1e-3, elapsed), 2),
        }

    def evaluate_benchmark_parallel(
        self,
        benchmark_dir: str | Path = "data/OD_benchmark",
        batch_size: int = 128,
        num_workers: int = 2,
        matching_mode: str = "hungarian",
    ) -> Dict[str, Dict[str, float]]:
        """
        Run multi-stream CUDA parallel evaluation across all 4 benchmark placement layouts
        ('random', 'grid', 'words', 'line') simultaneously!
        """
        from utils.dataset import get_detection_loaders

        benchmark_path = Path(benchmark_dir)
        placements = ["random", "grid", "words", "line"]
        results = {}

        print(f"\n[Pipeline] Starting Multi-Stream Parallel Evaluation on benchmark directory: '{benchmark_path}' (mode='{matching_mode}')...")
        t0 = time.time()

        def _eval_placement(placement_name: str) -> Tuple[str, Dict[str, float]]:
            p_dir = benchmark_path / placement_name
            if not p_dir.exists():
                return placement_name, {"error": "directory_not_found"}

            test_loader = get_detection_loaders(
                data_root=p_dir,
                batch_size=batch_size,
                num_workers=0,
                test_only=True,
                fcos=self.fcos_detector,
                anchors_wh=getattr(self, "anchors_wh", None),
                grid_sizes=getattr(self.detector, "grid_sizes", (28,)),
                anchors_per_scale=getattr(self.detector, "anchors_per_scale", None),
            )

            # Assign a dedicated CUDA Stream for concurrent GPU execution
            stream = torch.cuda.Stream() if torch.cuda.is_available() else None
            if stream:
                with torch.cuda.stream(stream):
                    metrics = self.evaluate_loader(test_loader, matching_mode=matching_mode)
                stream.synchronize()
            else:
                metrics = self.evaluate_loader(test_loader, matching_mode=matching_mode)

            return placement_name, metrics

        # Execute 4 placement benchmark evaluations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_placement = {
                executor.submit(_eval_placement, p): p for p in placements
            }

            for future in concurrent.futures.as_completed(future_to_placement):
                p_name, metrics = future.result()
                results[p_name] = metrics

        total_elapsed = time.time() - t0
        print(f"[Pipeline] Completed 4-way Parallel Benchmark in {total_elapsed:.2f} seconds!\n")
        return results


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    if str(_root / "generator") not in sys.path:
        sys.path.insert(0, str(_root / "generator"))

    print("=" * 65)
    print("  EndToEndPipeline Standalone Execution & Verification")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pipeline = DetectionPipeline(
        detector_size='s',
        device=device,
        conf_threshold=0.70,
    )

    sample_path = Path("data/OD_benchmark/words/test/images/000000.png")
    if not sample_path.exists():
        sample_path = Path("data/OD/test/images/000000.png")

    if sample_path.exists():
        print(f"\nRunning prediction on sample image: '{sample_path}'")
        detections = pipeline.predict_image(sample_path)
        print(f"Detected {len(detections)} characters:")
        for i, det in enumerate(detections, 1):
            print(
                f"  [{i:2d}] bbox={det.bbox} | char='{det.char_label}' | "
                f"det_conf={det.detector_conf:.4f} | cls_conf={det.classifier_conf:.4f} | "
                f"joint_conf={det.joint_conf:.4f}"
            )
    else:
        print("\nNo sample image found to run single-image prediction.")
