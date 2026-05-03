import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label
import json
from datetime import datetime
import os


# Pre-compute the denorm scale vector on CPU once (avoids rebuilding it per batch)
_BBOX_DENORM = None


class DigitDetectionPipeline:
    """
    Pipeline for end-to-end digit detection and recognition.

    Integrates a PyTorch bounding box detection model (or classical fallback)
    with a PyTorch digit classification model.
    """

    def __init__(
        self,
        bbox_model=None,
        classifier_model=None,
        image_shape=(128, 128),
        use_classical_fallback=True,
        normalize_bbox=False,
        device=None
    ):
        """
        Initialize the pipeline.

        Args:
            bbox_model (torch.nn.Module, optional): Trained model for bbox regression.
            classifier_model (torch.nn.Module, optional): Trained model for digit classification.
            image_shape (tuple, optional): Input image shape (H, W). Defaults to (128, 128).
            use_classical_fallback (bool, optional): Whether to use Connected Components if CNN fails.
            normalize_bbox (bool, optional): If True, denormalize [0,1] bbox coords to pixel space.
            device (torch.device, optional): Target device for inference. Auto-detects if None.
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bbox_model = bbox_model
        if self.bbox_model is not None:
            self.bbox_model.to(self.device)
            self.bbox_model.eval()

        self.classifier_model = classifier_model
        if self.classifier_model is not None:
            self.classifier_model.to(self.device)
            self.classifier_model.eval()

        self.image_shape = image_shape
        self.use_classical_fallback = use_classical_fallback
        self.normalize_bbox = normalize_bbox

        # Pre-compute denorm scale on CPU to avoid re-allocating per batch
        h, w = image_shape
        self._denorm_scale = np.array([w, h, w, h], dtype=np.float32)
        # Threshold for "oversized" bbox: 90% of image area
        self._max_area = 0.9 * h * w

    # ------------------------------------------------------------------
    # Classical fallback
    # ------------------------------------------------------------------

    @staticmethod
    def find_digit_bbox(img, threshold=0.15, margin=2):
        """
        Find bounding box using connected components (Classical Fallback).

        Args:
            img (torch.Tensor | np.ndarray): Input image, any shape with a single spatial plane.
            threshold (float): Binarization threshold.
            margin (int): Pixel margin around bbox.

        Returns:
            list[int]: [x_min, y_min, x_max, y_max] or None.
        """
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        img = img.squeeze()
        labels_map, num = label(img > threshold)

        if num == 0:
            return None

        largest = max(range(1, num + 1), key=lambda i: np.sum(labels_map == i))
        ys, xs = np.where(labels_map == largest)

        H, W = img.shape
        x_min = int(max(0, xs.min() - margin))
        y_min = int(max(0, ys.min() - margin))
        x_max = int(min(W - 1, xs.max() + margin))
        y_max = int(min(H - 1, ys.max() + margin))

        return [x_min, y_min, x_max, y_max]

    # ------------------------------------------------------------------
    # BBox validation (pure-Python, no numpy alloc for the common case)
    # ------------------------------------------------------------------

    def is_valid_bbox(self, bbox):
        """
        Validate if a bounding box is reasonable.

        Checks: non-null, valid coords, positive area, within image, < 90% image area.
        """
        if bbox is None:
            return False

        if isinstance(bbox, torch.Tensor):
            bbox = bbox.detach().cpu().tolist()
        elif isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()

        x_min, y_min, x_max, y_max = bbox

        if any(v != v for v in bbox):  # fast NaN check without numpy
            return False

        if x_max <= x_min or y_max <= y_min:
            return False

        if x_min < 0 or y_min < 0:
            return False

        H, W = self.image_shape
        if x_max > W or y_max > H:
            return False

        if (x_max - x_min) * (y_max - y_min) > self._max_area:
            return False

        return True

    # ------------------------------------------------------------------
    # Single-image helpers
    # ------------------------------------------------------------------

    def get_bbox(self, image):
        """Get bounding box for a single image (CNN → fallback)."""
        self.last_bbox_source = None

        if self.bbox_model is not None:
            with torch.no_grad():
                pred = self.bbox_model(image.unsqueeze(0).to(self.device))[0]
                cnn_bbox = pred.cpu().tolist()

            if self.normalize_bbox:
                cnn_bbox = [v * s for v, s in zip(cnn_bbox, self._denorm_scale)]

            if self.is_valid_bbox(cnn_bbox):
                self.last_bbox_source = "cnn"
                return cnn_bbox

        if self.use_classical_fallback:
            self.last_bbox_source = "classical"
            return self.find_digit_bbox(image)

        return None

    def _crop_and_resize(self, image, bbox, size=(28, 28)):
        """
        Crop a (1, H, W) tensor to bbox and bilinearly resize to `size`.

        Stays on CPU — the crop is small (sub-image), GPU transfer happens
        only when the batch is assembled.
        """
        x_min, y_min, x_max, y_max = (int(round(float(v))) for v in bbox)

        H = image.shape[-2]
        W = image.shape[-1]
        x_min = max(0, min(x_min, W - 1))
        x_max = max(0, min(x_max, W))
        y_min = max(0, min(y_min, H - 1))
        y_max = max(0, min(y_max, H))

        if x_max <= x_min or y_max <= y_min:
            return None

        # image is (1, H, W) — slice directly, no numpy round-trip
        crop = image[:, y_min:y_max, x_min:x_max]  # (1, h, w)

        if crop.numel() == 0:
            return None

        # interpolate expects (N, C, H, W)
        crop_resized = F.interpolate(crop.unsqueeze(0), size=size, mode='bilinear', align_corners=False)
        return crop_resized[0]  # (1, 28, 28)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def init_logger(self, log_path="logs/experiments.json"):
        self.log_path = log_path
        self.logs = []

    def _log_result(self, image_id, result, bbox_source):
        self.logs.append({
            "image_id": image_id,
            "bbox_source": bbox_source,
            "bbox": [float(v) for v in result["bbox"]],
            "digit": result["digit"],
            "confidence": result["confidence"],
        })

    def save_logs(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump({
                "created_at": datetime.now().isoformat(),
                "num_samples": len(self.logs),
                "results": self.logs
            }, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image, image_id=None):
        """
        Run full pipeline on a single image: Detect Box -> Crop -> Classify.

        Args:
            image (torch.Tensor): Shape (1, H, W), float32, [0, 1].
            image_id (str, optional): Identifier used for logging.

        Returns:
            dict | None: {"bbox", "digit", "confidence"} or None on failure.
        """
        bbox = self.get_bbox(image)
        if bbox is None:
            return None

        crop = self._crop_and_resize(image, bbox)
        if crop is None:
            return None

        if self.classifier_model is not None:
            with torch.no_grad():
                probs = F.softmax(self.classifier_model(crop.unsqueeze(0).to(self.device))[0], dim=0)
            digit = int(torch.argmax(probs).item())
            confidence = float(torch.max(probs).item())
        else:
            digit, confidence = -1, 0.0

        result = {"bbox": bbox, "digit": digit, "confidence": confidence}

        if image_id is not None and hasattr(self, "logs"):
            self._log_result(image_id, result, self.last_bbox_source)

        return result

    def predict_batch(self, images, batch_size=256, image_ids=None):
        """
        Run the full pipeline on a batch of images efficiently.

        Stage 1 – batched GPU inference for bounding boxes.
        Stage 2 – classical fallback only for images with invalid CNN boxes (CPU).
        Stage 3 – vectorised bbox validation on the full set of raw predictions.
        Stage 4 – serial crop-and-resize on CPU (crops are small; avoids N GPU allocs).
        Stage 5 – batched GPU inference for digit classification.

        Args:
            images (torch.Tensor): Shape (N, 1, H, W), float32, [0, 1].
            batch_size (int): Mini-batch size for model calls. Defaults to 256.
            image_ids (list[str], optional): Identifiers for logging. Uses str(i) if None.

        Returns:
            list[dict | None]: Per-image prediction dicts (None on detection failure).
        """
        n = len(images)
        results = [None] * n
        bboxes = [None] * n
        bbox_sources = [""] * n

        # ---- Stage 1: Batched BBox CNN ----
        if self.bbox_model is not None:
            raw_parts = []
            with torch.no_grad():
                for start in range(0, n, batch_size):
                    batch = images[start:start + batch_size].to(self.device)
                    raw_parts.append(self.bbox_model(batch).cpu())
            raw_bboxes = torch.cat(raw_parts, dim=0).numpy()  # (N, 4)

            if self.normalize_bbox:
                raw_bboxes = raw_bboxes * self._denorm_scale  # broadcast, in-place-style

            for i in range(n):
                bbox = raw_bboxes[i].tolist()
                if self.is_valid_bbox(bbox):
                    bboxes[i] = bbox
                    bbox_sources[i] = "cnn"

        # ---- Stage 2: Classical fallback (only for misses) ----
        if self.use_classical_fallback:
            for i in range(n):
                if bboxes[i] is None:
                    fb = self.find_digit_bbox(images[i])
                    if fb is not None:
                        bboxes[i] = fb
                        bbox_sources[i] = "classical"

        # ---- Stage 3: Crop-and-resize (CPU, stays on (1,H,W) tensors) ----
        crops = []
        crop_indices = []
        for i in range(n):
            if bboxes[i] is None:
                continue
            crop = self._crop_and_resize(images[i], bboxes[i])
            if crop is not None:
                crops.append(crop)
                crop_indices.append(i)

        if not crops:
            return results

        crops_tensor = torch.stack(crops)  # (M, 1, 28, 28) — still CPU

        # ---- Stage 4: Batched Digit Classification ----
        if self.classifier_model is not None:
            prob_parts = []
            with torch.no_grad():
                for start in range(0, len(crops_tensor), batch_size):
                    batch = crops_tensor[start:start + batch_size].to(self.device)
                    prob_parts.append(F.softmax(self.classifier_model(batch), dim=1).cpu())
            final_probs = torch.cat(prob_parts, dim=0)  # (M, 10)

            digits = torch.argmax(final_probs, dim=1).tolist()
            confidences = torch.max(final_probs, dim=1).values.tolist()

            log_enabled = hasattr(self, "logs")
            ids = image_ids if image_ids is not None else [str(i) for i in range(n)]

            for j, i in enumerate(crop_indices):
                results[i] = {
                    "bbox": bboxes[i],
                    "digit": int(digits[j]),
                    "confidence": float(confidences[j]),
                    "bbox_source": bbox_sources[i],
                }
                if log_enabled:
                    self._log_result(ids[i], results[i], bbox_sources[i])
        else:
            for j, i in enumerate(crop_indices):
                results[i] = {
                    "bbox": bboxes[i],
                    "digit": -1,
                    "confidence": 0.0,
                    "bbox_source": bbox_sources[i],
                }

        return results


if __name__ == "__main__":
    print("PyTorch Pipeline ready. Import DigitDetectionPipeline to use.")
