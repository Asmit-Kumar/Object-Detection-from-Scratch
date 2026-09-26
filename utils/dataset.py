"""
Data Pipeline Utilities.

Provides:
  - AugmentedTensorDataset: tensor-backed Dataset with on-the-fly transforms.
  - DatasetBuilder: file-based loaders for bbox regression and classification
    pipelines (used by the custom PNG-based generation workflow).
  - get_emnist_transforms(): build train/eval transform pipelines for EMNIST.
  - get_emnist_dataloaders(): single-call train/val/test DataLoader factory
    for EMNIST ByClass (62 classes).
"""
import math
import os
from dataclasses import replace
from pathlib import Path
import random
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from utils.reader import Record, DataReader
from utils.fcos_targets import FCOSTargetGenerator, FCOSTargets
import time

try:
    from torchvision.datasets import EMNIST
    import torchvision.transforms as T
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'OD'
S = 28  # Grid size for grid based detection
K = 1  # Set once per training run; anchors are fitted from the training split.
YOLO_GRID_SIZES = (28, 14)
YOLO_ANCHORS_PER_SCALE = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_default_data_root() -> str:
    """Resolve ./data relative to the project root (parent of utils/)."""
    return str(Path(__file__).resolve().parent.parent / "data")


def _auto_workers() -> int:
    """Return a sensible default worker count (half of CPU cores, capped at 8)."""
    return min(8, max(1, (os.cpu_count() or 2) // 2))


def _auto_detection_workers() -> int:
    """Return the measured-fast default for this detection pipeline."""
    return _auto_workers()


def _wh_iou(box_wh: torch.Tensor, anchors_wh: torch.Tensor) -> torch.Tensor:
    """Return IoU between one ``[w, h]`` box shape and every anchor shape."""
    box_wh = box_wh.reshape(1, 2)
    intersection = torch.minimum(box_wh, anchors_wh).prod(dim=-1)
    union = box_wh.prod(dim=-1) + anchors_wh.prod(dim=-1) - intersection
    return intersection / union.clamp_min(1e-6)


def _pairwise_wh_iou(boxes_wh: torch.Tensor, anchors_wh: torch.Tensor) -> torch.Tensor:
    """Return width/height IoU for every box/anchor pair."""
    boxes = boxes_wh[:, None, :].float()
    anchors = anchors_wh[None, :, :].float()
    intersection = torch.minimum(boxes, anchors).prod(dim=-1)
    union = boxes.prod(dim=-1) + anchors.prod(dim=-1) - intersection
    return intersection / union.clamp_min(1e-6)


def fit_anchors_wh(
        boxes_wh: torch.Tensor,
        k: int,
        max_iterations: int = 100,
        seed: int = 42,
) -> torch.Tensor:
    """Fit ``k`` width/height anchors with IoU-distance k-means.

    The medoid-style update uses the per-dimension median, as is conventional
    for anchor fitting. Inputs and outputs are in the source box coordinate
    system, currently pixels.
    """
    if boxes_wh.ndim != 2 or boxes_wh.size(1) != 2:
        raise ValueError("boxes_wh must have shape (N, 2)")
    if k < 1 or k > len(boxes_wh):
        raise ValueError(f"k must be in [1, {len(boxes_wh)}], got {k}")

    boxes_wh = boxes_wh.to(dtype=torch.float32, device="cpu")
    valid = (boxes_wh > 0).all(dim=1)
    boxes_wh = boxes_wh[valid]
    if len(boxes_wh) < k:
        raise ValueError("Not enough positive-width/height boxes to fit anchors")

    generator = torch.Generator().manual_seed(seed)
    initial = torch.randperm(len(boxes_wh), generator=generator)[:k]
    anchors_wh = boxes_wh[initial].clone()

    for _ in range(max_iterations):
        assignments = _pairwise_wh_iou(boxes_wh, anchors_wh).argmax(dim=1)
        updated = anchors_wh.clone()

        for anchor_idx in range(k):
            members = boxes_wh[assignments == anchor_idx]
            if len(members):
                updated[anchor_idx] = members.median(dim=0).values
            else:
                replacement = int(torch.randint(
                    len(boxes_wh), (1,), generator=generator
                ).item())
                updated[anchor_idx] = boxes_wh[replacement]

        if torch.equal(updated, anchors_wh):
            break
        anchors_wh = updated

    return anchors_wh[anchors_wh.prod(dim=1).argsort()]


def _collect_box_wh(reader: DataReader, indices: torch.Tensor) -> torch.Tensor:
    """Collect box widths/heights from metadata without loading image files."""
    boxes_wh = []
    for idx in indices.tolist():
        boxes = torch.from_numpy(reader.records[idx].boxes)
        if len(boxes):
            boxes_wh.append(boxes[:, 2:4])

    if not boxes_wh:
        raise ValueError("Cannot fit anchors: the training split has no boxes")
    return torch.cat(boxes_wh, dim=0)


# ── Tensor-backed Dataset ─────────────────────────────────────────────────────

class AugmentedTensorDataset(Dataset):
    """
    Dataset that wraps pre-loaded image and label tensors and applies
    transforms on-the-fly during __getitem__.

    Args:
        images (Tensor): Float image tensor of shape (N, C, H, W).
        labels (Tensor): Label tensor of shape (N, ...).
        transform (callable | None): Applied to each image at access time.
        class_names (Sequence | None): Optional display names indexed by class id.
    """
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None, class_names=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.classes = list(class_names) if class_names is not None else None
        self.class_names = self.classes

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, index):
        x = self.images[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[index]


# ── File-based loaders (custom PNG pipeline) ──────────────────────────────────

class DatasetBuilder:
    """
    File-based dataset builder for the custom PNG generation pipeline.
    Loads images and labels from directories produced by the bbox/class
    generation scripts.
    """

    @staticmethod
    def as_dataloader(
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        transform=None,
    ) -> DataLoader:
        """
        Wrap tensors into a DataLoader with optional on-the-fly transforms.
        """
        dataset = AugmentedTensorDataset(images, labels, transform=transform)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=(num_workers > 0 and torch.cuda.is_available()),
            persistent_workers=(num_workers > 0),
        )

    @staticmethod
    def create_dataloaders(
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 128,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
        drop_last: bool = False,
        num_workers: int = 0,
        train_transform=None,
        eval_transform=None,
    ) -> DataLoader | tuple:
        """
        Split tensors into train/val/test DataLoaders in one call.
        """
        n = len(images)
        test_size  = int(test_split * n)
        val_size   = int(val_split * n)
        train_size = n - val_size - test_size

        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n, generator=g)

        train_idx = indices[:train_size]
        val_idx   = indices[train_size:train_size + val_size]
        test_idx  = indices[train_size + val_size:]

        pin = num_workers > 0 and torch.cuda.is_available()
        pw  = num_workers > 0

        def _make(idx, transform, shuffle):
            ds = AugmentedTensorDataset(images[idx], labels[idx], transform=transform)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                              drop_last=drop_last, num_workers=num_workers,
                              pin_memory=pin, persistent_workers=pw)

        loaders = []
        if train_size > 0:
            loaders.append(_make(train_idx, train_transform, shuffle=True))
        if val_size > 0:
            loaders.append(_make(val_idx, eval_transform, shuffle=False))
        if test_size > 0:
            loaders.append(_make(test_idx, eval_transform, shuffle=False))

        return loaders[0] if len(loaders) == 1 else tuple(loaders)


class DetectionAugment:
    """Joint geometric augmentation for detection: scale + rotation only.
    Also includes photometric augmentations (blur, brightness, contrast).
    """

    def __init__(
            self, scale_range=(0.90, 1.10), max_rotation=15.0,
            p_rotate=0.25, p_scale=0.25,
            p_blur=0.2, p_brightness=0.2, p_contrast=0.2
    ):
        self.scale_range = scale_range
        self.max_rotation = max_rotation
        self.p_rotate = p_rotate
        self.p_scale = p_scale
        self.p_blur = p_blur
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast

    def __call__(self, record: Record) -> Record:
        image = record.image
        _, H, W = image.shape
        boxes = record.boxes.clone()

        if torch.rand(1) < self.p_blur:
            if torch.rand(1) < 0.3:
                image = F.gaussian_blur(image, kernel_size=[5, 5], sigma=[0.1, 0.8])
            else:
                image = F.gaussian_blur(image, kernel_size=[3, 3], sigma=[0.1, 0.8])

        if torch.rand(1) < self.p_brightness:
            factor = random.uniform(0.7, 1.3)
            image = F.adjust_brightness(image, factor)

        if torch.rand(1) < self.p_contrast:
            factor = random.uniform(0.7, 1.3)
            image = F.adjust_contrast(image, factor)

        if torch.rand(1) < self.p_scale:
            s = random.uniform(*self.scale_range)
            new_h, new_w = int(H * s), int(W * s)
            image = F.resize(image, [new_h, new_w], antialias=True)

            if s < 1.0:
                ph, pw = H - new_h, W - new_w
                image = F.pad(image, [pw // 2, ph // 2, pw - pw // 2, ph - ph // 2])
                dx, dy = pw // 2, ph // 2
            else:
                oh, ow = (new_h - H) // 2, (new_w - W) // 2
                image = image[:, oh:oh + H, ow:ow + W]
                dx, dy = -ow, -oh

            boxes[:, 0] = boxes[:, 0] * s + dx  # x
            boxes[:, 1] = boxes[:, 1] * s + dy  # y
            boxes[:, 2] = boxes[:, 2] * s  # w
            boxes[:, 3] = boxes[:, 3] * s  # h

        if torch.rand(1) < self.p_rotate:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            image = F.rotate(image, angle, fill=0)

            rad = math.radians(-angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            cx, cy = W / 2.0, H / 2.0

            new_boxes = []
            for box in boxes:
                x, y, w, h = box.tolist()
                corners = torch.tensor([
                    [x, y], [x + w, y],
                    [x, y + h], [x + w, y + h],
                ], dtype=torch.float32)
                corners -= torch.tensor([cx, cy])
                rot = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
                corners = (rot @ corners.T).T + torch.tensor([cx, cy])
                x1, y1 = corners[:, 0].min(), corners[:, 1].min()
                x2, y2 = corners[:, 0].max(), corners[:, 1].max()
                new_boxes.append([x1, y1, x2 - x1, y2 - y1])
            boxes = torch.tensor(new_boxes, dtype=torch.float32)

        boxes[:, 0].clamp_(0, W - 1)
        boxes[:, 1].clamp_(0, H - 1)

        max_w = W - boxes[:, 0]
        max_h = H - boxes[:, 1]
        boxes[:, 2].clamp_(min=1.0)
        boxes[:, 3].clamp_(min=1.0)
        boxes[:, 2] = torch.minimum(boxes[:, 2], max_w)
        boxes[:, 3] = torch.minimum(boxes[:, 3], max_h)

        return replace(
            record,
            image=image,
            boxes=boxes,
        )


class DetectionDataset(Dataset):
    """
    Dataset wrapping DataReader records for single-stage, multi-anchor, and anchor-free (FCOS) detection.
    """

    def __init__(
            self, reader: DataReader | None = None,
            transform: DetectionAugment | None = None,
            normalize_boxes=False, indices=None,
            anchors_wh: torch.Tensor | None = None,
            grid_sizes: tuple[int, ...] = (S,),
            anchors_per_scale: int | None = None,
            fcos: bool = False,
            defer_fcos_targets: bool = False,
    ):
        self.fcos = fcos
        self.defer_fcos_targets = defer_fcos_targets
        self.grid_sizes = tuple(grid_sizes)

        if not fcos:
            if anchors_wh is None:
                raise ValueError("DetectionDataset requires anchors_wh fitted from the training split")
            if anchors_wh.ndim != 2 or anchors_wh.size(1) != 2:
                raise ValueError("anchors_wh must have shape (K, 2)")
            if not grid_sizes or any(grid_size < 1 for grid_size in grid_sizes):
                raise ValueError("grid_sizes must contain one or more positive grid sizes")

            if anchors_per_scale is None:
                if anchors_wh.size(0) % len(grid_sizes) != 0:
                    raise ValueError(
                        "anchors_wh cannot be divided evenly across the requested grid sizes"
                    )
                anchors_per_scale = anchors_wh.size(0) // len(grid_sizes)
            expected_anchors = len(grid_sizes) * anchors_per_scale
            if anchors_wh.size(0) != expected_anchors:
                raise ValueError(
                    "anchors_wh must contain anchors_per_scale anchors for every grid size; "
                    f"expected {expected_anchors}, got {anchors_wh.size(0)}"
                )
            self.anchors_wh = anchors_wh.detach().to(dtype=torch.float32, device="cpu").contiguous()
            self.anchors_per_scale = anchors_per_scale
            self.fcos_target_gen = None
        else:
            if defer_fcos_targets and not fcos:
                raise ValueError("defer_fcos_targets requires fcos=True")
            self.anchors_wh = None
            self.anchors_per_scale = 1
            self.fcos_target_gen = FCOSTargetGenerator(grid_sizes=self.grid_sizes)

        self.reader = reader if reader is not None else DataReader(root=ROOT_DIR / "train")
        self.transform = transform
        self.normalize_boxes = normalize_boxes
        self.indices = indices

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.reader)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]

        record = self.reader.get_record(idx)

        if self.transform:
            record = self.transform(record)

        if self.normalize_boxes:
            h, w = record.image.shape[-2:]
            record = replace(
                record,
                boxes=record.boxes / torch.tensor([w, h, w, h])
            )

        return record

    def collate_fn(self, batch):
        """
        Collate records into detection targets.
        In FCOS mode, generates FCOSTargets directly.
        """
        if self.fcos:
            images = torch.stack([r.image for r in batch])
            boxes = [r.boxes for r in batch]
            labels = [r.labels for r in batch]
            if self.defer_fcos_targets:
                return images, None, (boxes, labels)
            fcos_targets = self.fcos_target_gen.generate_targets(boxes, labels, device=torch.device("cpu"))
            return images, fcos_targets, (boxes, labels)

        images = torch.stack([r.image for r in batch])
        B = len(batch)
        targets_by_scale = {
            grid_size: torch.zeros(
                B, grid_size, grid_size, self.anchors_per_scale, 5,
                dtype=torch.float32,
            )
            for grid_size in self.grid_sizes
        }
        labels_by_scale = {
            grid_size: torch.zeros(
                B, grid_size, grid_size, self.anchors_per_scale,
                dtype=torch.int64,
            )
            for grid_size in self.grid_sizes
        }
        matched_ious_by_scale = {
            grid_size: torch.full(
                (B, grid_size, grid_size, self.anchors_per_scale),
                -1.0,
                dtype=torch.float32,
            )
            for grid_size in self.grid_sizes
        }
        _, h, w = batch[0].image.shape

        for i, r in enumerate(batch):
            if r.boxes.numel() == 0:
                continue

            box_wh = r.boxes[:, 2:4]
            inter = torch.minimum(box_wh.unsqueeze(1), self.anchors_wh.unsqueeze(0)).prod(dim=-1)
            union = box_wh.prod(dim=-1, keepdim=True) + self.anchors_wh.prod(dim=-1, keepdim=True).T - inter
            all_anchor_ious = inter / union.clamp_min(1e-6)

            for box_idx, (b, l) in enumerate(zip(r.boxes, r.labels)):
                anchor_ious = all_anchor_ious[box_idx]
                global_anchor_idx = int(anchor_ious.argmax().item())
                scale_idx, anchor_idx = divmod(
                    global_anchor_idx,
                    self.anchors_per_scale,
                )
                grid_size = self.grid_sizes[scale_idx]
                cell_w = w / grid_size
                cell_h = h / grid_size
                cx = b[0] + b[2] / 2
                cy = b[1] + b[3] / 2
                gx = min(grid_size - 1, max(0, int(cx // cell_w)))
                gy = min(grid_size - 1, max(0, int(cy // cell_h)))

                matched_ious = matched_ious_by_scale[grid_size]
                if anchor_ious[global_anchor_idx] > matched_ious[i, gy, gx, anchor_idx]:
                    targets = targets_by_scale[grid_size]
                    labels = labels_by_scale[grid_size]
                    targets[i, gy, gx, anchor_idx, :4] = b
                    targets[i, gy, gx, anchor_idx, 4] = 1.0
                    labels[i, gy, gx, anchor_idx] = l
                    matched_ious[i, gy, gx, anchor_idx] = anchor_ious[global_anchor_idx]

        if len(self.grid_sizes) == 1:
            grid_size = self.grid_sizes[0]
            return images, targets_by_scale[grid_size], labels_by_scale[grid_size]
        return images, targets_by_scale, labels_by_scale


def get_detection_loaders(
        data_root: Path = ROOT_DIR,
        batch_size: int = 128,
        shuffle: bool = True,
        test: bool = False,
        test_only: bool = False,
        image_size: tuple = (224, 224),
        val_size: float | int = 15_000,
        num_workers: int | None = None,
        pin_memory: bool = torch.cuda.is_available(),
        transform: DetectionAugment | None = DetectionAugment(),
        normalize_boxes: bool = False,
        seed: int = 42,
        prefetch_factor: int = 2,
        num_anchors: int = K,
        anchors_wh: torch.Tensor | None = None,
        grid_sizes: tuple[int, ...] = (S,),
        anchors_per_scale: int | None = None,
        fcos: bool = False,
        defer_fcos_targets: bool = False,
) -> tuple | DataLoader:
    """
    Factory function to construct train, val, or test DataLoaders for detection.
    """
    if num_workers is None:
        num_workers = _auto_detection_workers()
    if not grid_sizes or any(grid_size < 1 for grid_size in grid_sizes):
        raise ValueError("grid_sizes must contain one or more positive grid sizes")

    if not fcos:
        if anchors_per_scale is None:
            anchors_per_scale = (
                YOLO_ANCHORS_PER_SCALE
                if tuple(grid_sizes) == YOLO_GRID_SIZES
                else num_anchors
            )
        total_anchors = len(grid_sizes) * anchors_per_scale
    else:
        anchors_per_scale = 1
        total_anchors = 0

    def _get_loader(
            reader,
            transform,
            shuffle,
            indices=None,
    ):
        dataset = DetectionDataset(
            reader=reader,
            transform=transform,
            normalize_boxes=normalize_boxes,
            indices=indices,
            anchors_wh=anchors_wh,
            grid_sizes=grid_sizes,
            anchors_per_scale=anchors_per_scale,
            fcos=fcos,
            defer_fcos_targets=defer_fcos_targets,
        )

        loader_kwargs = dict(
            dataset=dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=dataset.collate_fn,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        return DataLoader(**loader_kwargs)

    # Fast path: load ONLY the test set without reading train set metadata
    if test_only:
        if not fcos and anchors_wh is None:
            raise ValueError(
                "test_only requires anchors_wh fitted during training; "
                "load the tensor saved with the matching checkpoint."
            )
        test_root = Path(data_root) / "test"
        if not test_root.exists():
            raise FileNotFoundError(f"Test directory not found at: {test_root}")
        test_reader = DataReader(
            root=test_root,
            image_size=image_size,
        )
        return _get_loader(test_reader, transform=None, shuffle=False)

    train_root = Path(data_root) / "train"
    if not train_root.exists():
        raise FileNotFoundError(train_root)

    train_reader = DataReader(
        root=train_root,
        image_size=image_size,
    )

    ds_size = len(train_reader)
    if isinstance(val_size, float):
        val_len = int(ds_size * val_size)
    else:
        val_len = val_size

    train_len = ds_size - val_len

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(ds_size, generator=g)

    loaders = []

    train_idx = indices[:train_len]
    if not fcos:
        if anchors_wh is None:
            anchors_wh = fit_anchors_wh(
                _collect_box_wh(train_reader, train_idx),
                k=total_anchors,
                seed=seed,
            )
        else:
            anchors_wh = anchors_wh.detach().to(dtype=torch.float32, device="cpu")
            if anchors_wh.ndim != 2 or anchors_wh.shape != (total_anchors, 2):
                raise ValueError(
                    f"anchors_wh must have shape ({total_anchors}, 2), got {tuple(anchors_wh.shape)}"
                )
    else:
        anchors_wh = None

    loaders.append(_get_loader(
        train_reader,
        transform,
        shuffle,
        train_idx,
    ))

    if val_len > 0:
        val_idx = indices[train_len:]
        loaders.append(_get_loader(
            train_reader,
            None,
            False,
            val_idx,
        ))

    test_root = Path(data_root) / "test"
    if test:
        if not test_root.exists():
            raise FileNotFoundError(test_root)
        loaders.append(_get_loader(
            reader=DataReader(
                root=test_root,
                image_size=image_size,
            ),
            transform=None,
            shuffle=False,
        ))

    return loaders[0] if len(loaders) == 1 else tuple(loaders)


# ── EMNIST ByClass convenience loaders ────────────────────────────────────────

_EMNIST_MEAN = 0.1307
_EMNIST_STD  = 0.3081


def get_emnist_transforms(
    degrees: float = 30.0,
    translate: tuple = (0.1, 0.1),
    scale: tuple = (0.9, 1.1),
    erasing_prob: float = 0.1,
    augment: bool = True,
):
    """
    Build train and eval transform pipelines for EMNIST ByClass.
    """
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required. pip install torchvision")

    if augment:
        ops = [
            T.RandomAffine(degrees=degrees, translate=translate, scale=scale, fill=0),
        ]
        if erasing_prob > 0:
            ops.append(T.RandomErasing(p=erasing_prob, scale=(0.02, 0.15)))
        train_tf = T.Compose(ops)
    else:
        train_tf = None

    return train_tf, None


def get_emnist_gpu_transform(
    degrees: float = 30.0,
    translate: tuple = (0.1, 0.1),
    scale: tuple = (0.9, 1.1),
    erasing_prob: float = 0.1,
):
    """
    Build a GPU-compatible batch augmentation transform for EMNIST.
    """
    try:
        import torchvision.transforms.v2 as T2
    except ImportError:
        return None

    ops = [
        T2.RandomAffine(degrees=degrees, translate=translate, scale=scale, fill=0),
    ]
    if erasing_prob > 0:
        ops.append(T2.RandomErasing(p=erasing_prob, scale=(0.02, 0.15)))
    return T2.Compose(ops)


def _load_emnist_tensors(
    data_root: str,
    split: str,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    raw = EMNIST(root=data_root, split=split, train=train, download=True)
    images = raw.data.to(torch.float32).div(255.0)
    images = (images - _EMNIST_MEAN) / _EMNIST_STD
    images = images.unsqueeze(1)
    labels = raw.targets.to(torch.int64)
    return images, labels, list(raw.classes)


def get_emnist_dataloaders(
    batch_size: int = 256,
    num_workers: int | None = None,
    val_split: float = 0.1,
    augment: bool = True,
    gpu_augment: bool = True,
    data_root: str | None = None,
    split: str = "byclass",
    seed: int = 42,
    degrees: float = 30.0,
    translate: tuple = (0.1, 0.1),
    scale: tuple = (0.9, 1.1),
    erasing_prob: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create EMNIST ByClass train/val/test DataLoaders in a single call.
    """
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required. pip install torchvision")

    if num_workers is None:
        num_workers = _auto_workers()

    if data_root is None:
        data_root = _get_default_data_root()

    train_data, train_labels, classes = _load_emnist_tensors(data_root, split, train=True)
    test_data, test_labels, _ = _load_emnist_tensors(data_root, split, train=False)

    n_total = len(train_data)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    use_cpu_aug = augment and not gpu_augment
    train_tf, eval_tf = get_emnist_transforms(
        degrees=degrees, translate=translate, scale=scale,
        erasing_prob=erasing_prob, augment=use_cpu_aug,
    )

    pin = num_workers > 0 and torch.cuda.is_available()
    pw = num_workers > 0

    train_ds = AugmentedTensorDataset(train_data[train_idx], train_labels[train_idx],
                                      transform=train_tf, class_names=classes)
    val_ds = AugmentedTensorDataset(train_data[val_idx], val_labels[val_idx],
                                    transform=eval_tf, class_names=classes)
    test_ds = AugmentedTensorDataset(test_data, test_labels,
                                     transform=eval_tf, class_names=classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=pw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin, persistent_workers=pw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin, persistent_workers=pw)

    return train_loader, val_loader, test_loader
