"""
Training and Evaluation Utilities.

Provides reusable functions for training loops, validation, testing, and
metrics computation (both classification and regression) with AMP support.

BF16 + GradScaler note:
    bfloat16 has a wider dynamic range than float16, so it does NOT require
    loss scaling in practice. However, GradScaler is accepted here for API
    uniformity — when device_type='cuda' and dtype=bfloat16, PyTorch
    automatically makes the scaler a no-op (scale stays at 1.0, no overflow
    checks). Pass  scaler=torch.amp.GradScaler('cuda')  if you want the
    unified API, or simply pass  scaler=None  for bf16 runs.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from .losses import pairwise_iou, mean_iou, DetectionLoss  # noqa: F401
try:
    import wandb
except ImportError:
    wandb = None

# Default AMP dtype used across all training helpers.
_AMP_DTYPE = torch.bfloat16


def train_one_epoch(
        model, loader, criterion, optimizer,
        device, scheduler=None, scaler=None,
        clip_grad_norm: float = 0.0,
        step_scheduler_per_batch: bool = True,
        gpu_augment=None,
):
    """
    Run one full training epoch.

    Uses bfloat16 autocast when a CUDA device is detected.

    Args:
        clip_grad_norm: If > 0, clip gradient norms to this value before the
            optimizer step. Requires scaler.unscale_() which is handled
            automatically. Defaults to 0.0 (disabled).
        step_scheduler_per_batch: If True (default), step the scheduler after
            every mini-batch (e.g. OneCycleLR). If False, the caller (fit())
            is responsible for stepping once per epoch (e.g. CosineAnnealingLR).
        gpu_augment (callable | None): A torchvision.transforms.v2 transform
            applied to the batch ON the GPU after .to(device). Use
            get_emnist_gpu_transform() to create this. Much faster than
            per-sample CPU augmentation inside the DataLoader. Defaults to None.

    GPU throughput notes:
      - non_blocking=True on .to(device) overlaps CPU→GPU transfer with compute,
        requires pin_memory=True in the DataLoader (set by get_emnist_dataloaders).
      - set_to_none=True on zero_grad is faster than zeroing (avoids memset).
      - On Windows: num_workers > 0 requires training to run inside
        `if __name__ == '__main__':` — otherwise workers silently fall back to 0.
    """
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    use_amp = device.type == "cuda"

    for inputs, labels in loader:
        # non_blocking=True: async CPU→GPU copy while GPU runs previous batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # GPU batch augmentation — single vectorised CUDA op, far faster than
        # per-sample CPU augmentation inside the DataLoader workers.
        if gpu_augment is not None:
            with torch.no_grad():
                inputs = gpu_augment(inputs)

        # set_to_none is faster than zeroing — skips the memset
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Record scale before stepping to detect overflow
        prev_scale = scaler.get_scale() if scaler is not None else None

        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        if scheduler is not None and step_scheduler_per_batch:
            overflow = (
                scaler is not None
                and prev_scale is not None
                and scaler.get_scale() < prev_scale
            )
            if not overflow:
                scheduler.step()

        epoch_loss += loss.item()
        if wandb is not None and wandb.run is not None:
            wandb.log({"train/batch_loss": loss.item()})

        n_batches += 1

    return epoch_loss / n_batches


def _infer_class_names_from_loader(loader):
    """Return class labels exposed by a DataLoader dataset, if available."""
    dataset = getattr(loader, "dataset", None)
    return _infer_class_names_from_dataset(dataset, seen=set())


def _infer_class_names_from_dataset(dataset, seen):
    if dataset is None or id(dataset) in seen:
        return None
    seen.add(id(dataset))

    for attr in ("class_names", "classes"):
        values = getattr(dataset, attr, None)
        if values is not None:
            return list(values)

    class_to_idx = getattr(dataset, "class_to_idx", None)
    if class_to_idx:
        return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    nested_dataset = getattr(dataset, "dataset", None)
    names = _infer_class_names_from_dataset(nested_dataset, seen)
    if names is not None:
        return names

    for child in getattr(dataset, "datasets", ()):
        names = _infer_class_names_from_dataset(child, seen)
        if names is not None:
            return names

    return None


def evaluate_classification(
    model, loader, criterion, device,
    print_per_class_accuracy: bool = False,
    class_names=None,
):
    """
    Evaluate model on a classification dataset.

    Args:
        print_per_class_accuracy (bool): If True, print accuracy for each class
            after evaluation. Defaults to False.
        class_names (Sequence | None): Optional display names indexed by class id.
            When None, names are inferred from loader.dataset when available.

    Returns:
        (avg_loss, accuracy): Average loss and overall accuracy percentage.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = None
    class_total = None

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
                    outputs = model(imgs)
                    total_loss += criterion(outputs, labels).item()
            else:
                outputs = model(imgs)
                total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if print_per_class_accuracy:
                num_classes = outputs.size(1)
                if class_correct is None:
                    class_correct = torch.zeros(num_classes, dtype=torch.long)
                    class_total = torch.zeros(num_classes, dtype=torch.long)

                labels_cpu = labels.detach().view(-1).to("cpu")
                correct_mask_cpu = (predicted == labels).detach().view(-1).to("cpu")

                class_total += torch.bincount(labels_cpu, minlength=num_classes)
                class_correct += torch.bincount(
                    labels_cpu[correct_mask_cpu],
                    minlength=num_classes,
                )

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    if print_per_class_accuracy and class_total is not None:
        if class_names is None:
            class_names = _infer_class_names_from_loader(loader)
        elif not isinstance(class_names, list):
            class_names = list(class_names)

        print("Per-class accuracy:")
        for class_idx in range(class_total.numel()):
            class_count = class_total[class_idx].item()
            class_label = (
                class_names[class_idx]
                if class_names is not None and class_idx < len(class_names)
                else f"Class {class_idx}"
            )
            if class_count == 0:
                print(f"  {class_label}: n/a (0/0)")
                continue

            class_hits = class_correct[class_idx].item()
            class_accuracy = 100.0 * class_hits / class_count
            print(f"  {class_label}: {class_accuracy:.2f}% ({class_hits}/{class_count})")

    return avg_loss, accuracy

def compute_iou(pred_boxes, target_boxes):
    """Mean IoU between paired (N,4) boxes in [x, y, w, h] format.

    Thin wrapper around ``losses.mean_iou`` kept for backward compatibility
    with ``evaluate_regression``.
    """
    return mean_iou(pred_boxes, target_boxes)


# _pairwise_iou is imported from .losses as pairwise_iou
_pairwise_iou = pairwise_iou


def evaluate_regression(model, loader, criterion, device):
    """
    Evaluate model on a bounding box regression dataset.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_iou = 0.0

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for imgs, bboxes in loader:
            imgs   = imgs.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
                    outputs = model(imgs)
            else:
                outputs = model(imgs)

            # Cast back to float32 for loss/metric computation
            outputs = outputs.float()
            total_loss += criterion(outputs, bboxes).item()
            total_mae += F.l1_loss(outputs, bboxes).item()
            total_iou += compute_iou(outputs, bboxes)

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    avg_iou = total_iou / n_batches

    return avg_loss, avg_mae, avg_iou



def records_to_padded_tensors(records, device):
    """Converts a batch of Record dataclasses to padded tensors on the GPU."""
    counts = [r.boxes.size(0) for r in records]
    max_gt = max(counts, default=0)
    B = len(records)
    
    # Pre-allocate on CPU (fast zeroing and slicing)
    boxes = torch.zeros((B, max_gt, 4), dtype=torch.float32)
    labels = torch.zeros((B, max_gt), dtype=torch.int64)
    mask = torch.zeros((B, max_gt), dtype=torch.bool)
    
    classes_list = []
    for i, r in enumerate(records):
        n = r.boxes.size(0)
        if n > 0:
            boxes[i, :n] = r.boxes
            labels[i, :n] = r.labels
            mask[i, :n] = True
        
    # Bulk transfer to GPU once
    boxes = boxes.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    
    return boxes, labels, mask

def train_one_epoch_detection(
    model, loader, criterion, optimizer, device,
    scheduler=None, scaler=None,
    clip_grad_norm: float = 0.0,
    step_scheduler_per_batch: bool = True,
):
    """Training loop for multi-object detection.

    Expects loader to yield (images, list[Record]).
    Uses records_to_padded_tensors internally before loss computation.
    """
    model.train()
    epoch_loss = 0.0
    epoch_match_ratio = 0.0
    n_batches  = 0
    use_amp    = device.type == "cuda"

    loader_iter = iter(loader)
    while True:
        is_timing = n_batches < 5
        if is_timing:
            torch.cuda.synchronize()
            t_start = time.perf_counter()

        try:
            images, targets, labels = next(loader_iter)
        except StopIteration:
            break

        if is_timing:
            torch.cuda.synchronize()
            t_data = time.perf_counter()

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_timing:
            torch.cuda.synchronize()
            t_pad = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        
        if is_timing:
            t0 = t_pad

        with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
            outputs = model(images)
            if is_timing: torch.cuda.synchronize(); t1 = time.perf_counter()
            loss = criterion(outputs, targets, labels)
            if is_timing: torch.cuda.synchronize(); t2 = time.perf_counter()

        prev_scale = scaler.get_scale() if scaler is not None else None

        if scaler is not None:
            scaler.scale(loss).backward()
            if is_timing: torch.cuda.synchronize(); t3 = time.perf_counter()
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if is_timing: torch.cuda.synchronize(); t3 = time.perf_counter()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        if is_timing:
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            print(
                f"Batch {n_batches+1:2d} | "
                f"data={(t_data-t_start)*1000:.1f}ms "
                f"pad={(t_pad-t_data)*1000:.1f}ms "
                f"forward={(t1-t0)*1000:.1f}ms "
                f"loss={(t2-t1)*1000:.1f}ms "
                f"backward={(t3-t2)*1000:.1f}ms "
                f"step={(t4-t3)*1000:.1f}ms"
            )

        if scheduler is not None and step_scheduler_per_batch:
            overflow = (
                scaler is not None
                and prev_scale is not None
                and scaler.get_scale() < prev_scale
            )
            if not overflow:
                scheduler.step()

        epoch_loss += loss.item()
        if wandb is not None and wandb.run is not None:
            wandb.log({
                "train/batch_loss": loss.item(),
            })

        n_batches += 1

    return epoch_loss / n_batches


def evaluate_detection(
    model, loader, criterion, device, conf_threshold=0.5, iou_threshold=0.5
):
    """Evaluate a detection model on a single validation or test pass.

    Computes detection, classification, and end-to-end metrics using confidence-sorted
    greedy IoU matching against ground-truth annotations.

    Returns:
        dict: Hierarchical metrics evaluation dictionary containing:
            - ``"loss"`` (float): Average validation loss over all batches.
            - ``"detection"`` (dict): Localization-only metrics (P/R/F1/IoU/TP/FP/FN).
              A TP is any predicted box whose best-matching GT IoU >= iou_threshold.
            - ``"classification"`` (dict): Secondary metrics on matched detections
              (``accuracy``, ``correct``, ``total``, ``confusion_matrix``, and
              ``per_class`` breakdown).
            - ``"end_to_end"`` (dict): Combined metrics (``precision``, ``recall``, ``f1``,
              ``tp``, ``fp``, ``fn``) requiring both box localization IoU >= iou_threshold
              AND correct character class prediction.

    For full P/R/F1 threshold sweeps, use ``evaluate_detection_sweep()``.
    Expects loader to yield (images, targets, labels).
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n_images = 0
    tp = fp = fn = 0
    e2e_tp = e2e_fp = e2e_fn = 0
    class_correct = 0
    class_total = 0
    confusion_matrix = None
    per_class_e2e_tp = per_class_e2e_fp = per_class_e2e_fn = None
    class_names = _infer_class_names_from_loader(loader)
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for images, targets, labels in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
                    outputs = model(images)
                    loss = criterion(outputs, targets, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets, labels)

            total_loss += loss.item()

            gt_obj = targets[..., 4].bool()
            gt_bxs = targets[..., :4]
            pred_conf_grid = torch.sigmoid(outputs[..., 4].float())
            pred_boxes_grid = outputs[..., :4].float()
            pred_class_grid = outputs[..., 5:].float()
            if pred_class_grid.size(-1) == 0:
                raise ValueError(
                    "evaluate_detection requires model outputs with class logits."
                )
            pred_labels_grid = pred_class_grid.argmax(dim=-1)

            pred_conf_np = pred_conf_grid.cpu().numpy()
            pred_boxes_np = pred_boxes_grid.cpu().numpy()
            pred_labels_np = pred_labels_grid.cpu().numpy()
            gt_obj_np = gt_obj.cpu().numpy()
            gt_bxs_np = gt_bxs.cpu().numpy()
            gt_labels_np = labels.cpu().numpy()

            if confusion_matrix is None:
                num_classes = pred_class_grid.size(-1)
                confusion_matrix = np.zeros(
                    (num_classes, num_classes), dtype=np.int64
                )
                per_class_e2e_tp = np.zeros(num_classes, dtype=np.int64)
                per_class_e2e_fp = np.zeros(num_classes, dtype=np.int64)
                per_class_e2e_fn = np.zeros(num_classes, dtype=np.int64)

            for b in range(images.size(0)):
                pos_mask_b = gt_obj_np[b]
                n_gt = int(pos_mask_b.sum())
                gt_boxes_b = gt_bxs_np[b][pos_mask_b]
                gt_labels_b = gt_labels_np[b][pos_mask_b]

                conf_flat = pred_conf_np[b].flatten()
                boxes_flat = pred_boxes_np[b].reshape(-1, 4)
                pred_lbl_flat = pred_labels_np[b].flatten()

                if n_gt > 0:
                    iou_np_b = _pairwise_iou(
                        torch.from_numpy(boxes_flat),
                        torch.from_numpy(gt_boxes_b)
                    ).numpy()
                    total_iou += float(iou_np_b.max(axis=0).mean())
                    n_images += 1
                else:
                    iou_np_b = np.zeros((len(conf_flat), 0), dtype=np.float32)

                pr_idx = np.where(conf_flat > conf_threshold)[0]
                if len(pr_idx) == 0:
                    fn += n_gt
                    e2e_fn += n_gt
                    for gi in range(n_gt):
                        per_class_e2e_fn[int(gt_labels_b[gi])] += 1
                    continue

                order = pr_idx[np.argsort(-conf_flat[pr_idx])]
                matched = set()

                for i in order:
                    if n_gt == 0:
                        fp += 1
                        e2e_fp += 1
                        per_class_e2e_fp[int(pred_lbl_flat[i])] += 1
                        continue

                    j = int(np.argmax(iou_np_b[i]))
                    if iou_np_b[i, j] >= iou_threshold and j not in matched:
                        matched.add(j)
                        tp += 1

                        gt_label = int(gt_labels_b[j])
                        pred_label = int(pred_lbl_flat[i])
                        if not 0 <= gt_label < confusion_matrix.shape[0]:
                            raise ValueError(
                                f"Ground-truth class {gt_label} is outside the "
                                f"model's class range [0, {confusion_matrix.shape[0]})."
                            )

                        confusion_matrix[gt_label, pred_label] += 1
                        class_total += 1
                        if pred_label == gt_label:
                            class_correct += 1
                            e2e_tp += 1
                            per_class_e2e_tp[gt_label] += 1
                        else:
                            e2e_fp += 1
                            e2e_fn += 1
                            per_class_e2e_fp[pred_label] += 1
                            per_class_e2e_fn[gt_label] += 1
                    else:
                        fp += 1
                        e2e_fp += 1
                        per_class_e2e_fp[int(pred_lbl_flat[i])] += 1

                fn += n_gt - len(matched)
                e2e_fn += n_gt - len(matched)
                for local_j in set(range(n_gt)) - matched:
                    per_class_e2e_fn[int(gt_labels_b[local_j])] += 1

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    e2e_precision = float(
        e2e_tp / (e2e_tp + e2e_fp) if (e2e_tp + e2e_fp) else 0.0
    )
    e2e_recall = float(e2e_tp / (e2e_tp + e2e_fn)) if (e2e_tp + e2e_fn) else 0.0
    e2e_f1 = float(
        2 * e2e_precision * e2e_recall / (e2e_precision + e2e_recall)
        if (e2e_precision + e2e_recall)
        else 0.0
    )

    avg_loss = float(total_loss / len(loader))
    mean_iou = float(total_iou / max(n_images, 1))

    per_class = {}
    if confusion_matrix is not None:
        for class_idx in range(confusion_matrix.shape[0]):
            class_count = int(confusion_matrix[class_idx].sum())
            class_hits = int(confusion_matrix[class_idx, class_idx])

            e2e_tp_k  = int(per_class_e2e_tp[class_idx])
            e2e_fp_k  = int(per_class_e2e_fp[class_idx])
            e2e_fn_k  = int(per_class_e2e_fn[class_idx])
            e2e_p_k   = e2e_tp_k / (e2e_tp_k + e2e_fp_k) if (e2e_tp_k + e2e_fp_k) else 0.0
            e2e_r_k   = e2e_tp_k / (e2e_tp_k + e2e_fn_k) if (e2e_tp_k + e2e_fn_k) else 0.0
            e2e_f1_k  = (
                2 * e2e_p_k * e2e_r_k / (e2e_p_k + e2e_r_k)
                if (e2e_p_k + e2e_r_k) else 0.0
            )

            class_metrics = {
                "accuracy": float(class_hits / class_count) if class_count else 0.0,
                "correct":  class_hits,
                "total":    class_count,
                "e2e_tp":        e2e_tp_k,
                "e2e_fp":        e2e_fp_k,
                "e2e_fn":        e2e_fn_k,
                "e2e_precision": round(float(e2e_p_k),  4),
                "e2e_recall":    round(float(e2e_r_k),  4),
                "e2e_f1":        round(float(e2e_f1_k), 4),
            }
            if class_names is not None and class_idx < len(class_names):
                class_metrics["name"] = class_names[class_idx]
            per_class[class_idx] = class_metrics

    return {
        "loss": avg_loss,
        "detection": {
            "iou": mean_iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        },
        "classification": {
            "accuracy": float(class_correct / class_total) if class_total else 0.0,
            "per_class": per_class,
            "confusion_matrix": (
                confusion_matrix.tolist() if confusion_matrix is not None else []
            ),
        },
        "end_to_end": {
            "precision": e2e_precision,
            "recall": e2e_recall,
            "f1": e2e_f1,
            "tp": int(e2e_tp),
            "fp": int(e2e_fp),
            "fn": int(e2e_fn),
        },
    }


def evaluate_detection_sweep(
    model, loader, device,
    conf_thresholds: np.ndarray | list[float] | tuple = np.arange(0.2, 0.9, 0.1),
    iou_threshold: float = 0.5,
    print_table: bool = True,
):
    """Run a confidence threshold sweep and return a list of metric dicts.

    Unlike evaluate_detection, this does NOT compute loss — it performs a
    single forward pass, caches raw predictions on CPU, then sweeps all
    thresholds without re-running the model.

    Returns per-threshold detection P/R/F1 AND overall classification accuracy
    + per-class E2E F1, mirroring the full metric surface of evaluate_detection.

    Intended for use on the test set after training completes. Results are
    returned as a list of dicts (one per threshold) suitable for saving into
    RunLogger via logger.log_test_results().

    Args:
        model:             Trained detection model.
        loader:            Test DataLoader (yields images, targets, labels).
        device:            torch.device.
        conf_thresholds:   Iterable of confidence thresholds to sweep.
        iou_threshold:     IoU threshold for a TP match.
        print_table:       If True, print a formatted metric table.

    Returns:
        List of dicts: [{"conf": 0.2, "precision": ..., "recall": ..., "f1": ...,
                         "tp": ..., "fp": ..., "fn": ...,
                         "cls_accuracy": ..., "per_class_e2e": {class_idx: {"f1":...}}},
                        ...]
    """
    thresholds = sorted(conf_thresholds)
    model.eval()
    use_amp = device.type == "cuda"
    class_names = _infer_class_names_from_loader(loader)

    records = []
    num_classes = None

    with torch.no_grad():
        for images, targets, labels in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
                    outputs = model(images)
            else:
                outputs = model(images)

            pred_conf_grid   = torch.sigmoid(outputs[..., 4].float())
            pred_boxes_grid  = outputs[..., :4].float()
            pred_class_grid  = outputs[..., 5:].float()
            pred_labels_grid = pred_class_grid.argmax(dim=-1)
            gt_obj = targets[..., 4].bool()
            gt_bxs = targets[..., :4]

            if num_classes is None:
                num_classes = pred_class_grid.size(-1)

            pred_conf_np   = pred_conf_grid.cpu().numpy()
            pred_boxes_np  = pred_boxes_grid.cpu().numpy()
            pred_labels_np = pred_labels_grid.cpu().numpy()
            gt_obj_np      = gt_obj.cpu().numpy()
            gt_bxs_np      = gt_bxs.cpu().numpy()
            gt_labels_np   = labels.cpu().numpy()

            for b in range(images.size(0)):
                pos_b = gt_obj_np[b]
                n_gt = int(pos_b.sum())
                if n_gt == 0:
                    continue
                gt_boxes_b  = gt_bxs_np[b][pos_b]
                gt_labels_b = gt_labels_np[b][pos_b]

                conf_flat   = pred_conf_np[b].flatten()
                boxes_flat  = pred_boxes_np[b].reshape(-1, 4)
                lbl_flat    = pred_labels_np[b].flatten()

                iou_b = _pairwise_iou(
                    torch.from_numpy(boxes_flat),
                    torch.from_numpy(gt_boxes_b)
                ).numpy()

                records.append((
                    conf_flat,
                    iou_b,
                    lbl_flat,
                    gt_labels_b,
                    n_gt,
                ))

    results = []
    for thr in thresholds:
        tp = fp = fn = 0
        cls_correct = cls_total = 0
        per_class_e2e_tp = np.zeros(num_classes, dtype=np.int64)
        per_class_e2e_fp = np.zeros(num_classes, dtype=np.int64)
        per_class_e2e_fn = np.zeros(num_classes, dtype=np.int64)

        for conf, sub_iou, pred_lbl, gt_lbl, n_gt in records:
            pr_indices = np.where(conf > thr)[0]

            if len(pr_indices) == 0:
                fn += n_gt
                for k in gt_lbl:
                    per_class_e2e_fn[int(k)] += 1
                continue

            order = pr_indices[np.argsort(-conf[pr_indices])]
            matched = set()
            for i in order:
                best_j = int(np.argmax(sub_iou[i]))
                if sub_iou[i, best_j] >= iou_threshold and best_j not in matched:
                    matched.add(best_j)
                    tp += 1
                    p_lbl = int(pred_lbl[i])
                    g_lbl = int(gt_lbl[best_j])
                    cls_total += 1
                    if p_lbl == g_lbl:
                        cls_correct += 1
                        per_class_e2e_tp[g_lbl] += 1
                    else:
                        per_class_e2e_fp[p_lbl] += 1
                        per_class_e2e_fn[g_lbl] += 1
                else:
                    fp += 1
                    per_class_e2e_fp[int(pred_lbl[i])] += 1
            fn += n_gt - len(matched)
            for local_j in set(range(n_gt)) - matched:
                per_class_e2e_fn[int(gt_lbl[local_j])] += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        cls_acc = cls_correct / cls_total if cls_total else 0.0

        per_class_e2e = {}
        for k in range(num_classes):
            tp_k = int(per_class_e2e_tp[k])
            fp_k = int(per_class_e2e_fp[k])
            fn_k = int(per_class_e2e_fn[k])
            p_k  = tp_k / (tp_k + fp_k) if (tp_k + fp_k) else 0.0
            r_k  = tp_k / (tp_k + fn_k) if (tp_k + fn_k) else 0.0
            f_k  = 2 * p_k * r_k / (p_k + r_k) if (p_k + r_k) else 0.0
            entry = {"e2e_precision": round(p_k, 4), "e2e_recall": round(r_k, 4),
                     "e2e_f1": round(f_k, 4), "e2e_tp": tp_k, "e2e_fp": fp_k, "e2e_fn": fn_k}
            if class_names is not None and k < len(class_names):
                entry["name"] = class_names[k]
            per_class_e2e[k] = entry

        results.append({
            "conf":         round(thr, 2),
            "precision":    round(precision, 4),
            "recall":       round(recall, 4),
            "f1":           round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "cls_accuracy": round(cls_acc, 4),
            "per_class_e2e": per_class_e2e,
        })

    if print_table:
        print(f"\n{'Conf':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  {'ClsAcc':>7}  {'TP':>7}  {'FP':>7}  {'FN':>7}")
        print("-" * 66)
        for r in results:
            print(
                f"{r['conf']:6.2f}  {r['precision']:6.4f}  {r['recall']:6.4f}  "
                f"{r['f1']:6.4f}  {r['cls_accuracy']:7.4f}  "
                f"{r['tp']:7d}  {r['fp']:7d}  {r['fn']:7d}"
            )

    return results


def evaluate_density_sweep(
    model, loader, device,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    buckets: list[tuple[int, int]] = [(1, 3), (4, 7), (8, 24)]
):
    """Run a single-threshold evaluation, stratified by ground-truth object count.

    Useful for diagnosing density-dependent recall drop-off.
    Args:
        buckets: List of (min_gt, max_gt) inclusive tuples.
    """
    model.eval()
    use_amp = device.type == "cuda"
    
    # Store records: (pred_conf_np, iou_np, n_gt)
    records = []
    
    with torch.no_grad():
        for images, targets, labels in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
                    outputs = model(images)
            else:
                outputs = model(images)

            pred_boxes_grid = outputs[..., :4].float()
            pred_conf_grid  = torch.sigmoid(outputs[..., 4].float())
            gt_obj = targets[..., 4].bool()
            gt_bxs = targets[..., :4]

            pred_conf_np  = pred_conf_grid.cpu().numpy()
            pred_boxes_np = pred_boxes_grid.cpu().numpy()
            gt_obj_np     = gt_obj.cpu().numpy()
            gt_bxs_np     = gt_bxs.cpu().numpy()

            for b in range(images.size(0)):
                pos_b = gt_obj_np[b]
                n_gt = int(pos_b.sum())
                if n_gt == 0:
                    continue
                gt_boxes_b = gt_bxs_np[b][pos_b]
                conf_flat  = pred_conf_np[b].flatten()
                boxes_flat = pred_boxes_np[b].reshape(-1, 4)

                iou_b = _pairwise_iou(
                    torch.from_numpy(boxes_flat),
                    torch.from_numpy(gt_boxes_b)
                ).numpy()

                records.append((
                    conf_flat,
                    iou_b,
                    n_gt
                ))

    import numpy as np
    
    bucket_results = []
    for (min_gt, max_gt) in buckets:
        tp = fp = fn = 0
        n_images = 0
        for conf, sub_iou, n_gt in records:
            if not (min_gt <= n_gt <= max_gt):
                continue
            n_images += 1
            
            pr_mask = conf > conf_threshold
            pr_indices = np.where(pr_mask)[0]
            n_pr = len(pr_indices)

            if n_pr == 0:
                fn += n_gt
                continue

            order = pr_indices[np.argsort(-conf[pr_indices])]
            matched = set()
            for i in order:
                best_j = int(np.argmax(sub_iou[i]))
                if sub_iou[i, best_j] >= iou_threshold and best_j not in matched:
                    matched.add(best_j)
                    tp += 1
                else:
                    fp += 1
            fn += n_gt - len(matched)
            
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        
        bucket_results.append({
            "bucket": f"{min_gt}-{max_gt}",
            "n_images": n_images,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn
        })

    print(f"\nStratified Evaluation (Conf: {conf_threshold}, IoU: {iou_threshold})")
    print(f"{'Bucket':>8}  {'Images':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print("-" * 62)
    for r in bucket_results:
        print(f"{r['bucket']:>8}  {r['n_images']:6d}  {r['precision']:6.4f}  {r['recall']:6.4f}  {r['f1']:6.4f}  {r['tp']:5d}  {r['fp']:5d}  {r['fn']:5d}")
        
    return bucket_results


def fit(
    model, trainloader, valloader, criterion,
    optimizer, device, epochs,
    scheduler=None, scaler=None, checkpoint=None,
    log=False, log_dir=None,
    clip_grad_norm: float = 1.0,
    step_scheduler_per_batch: bool = True,
    verbose: int = 1,
    model_label: str = None,
    resume: bool = False,
):
    """
    Full training loop with per-epoch logging, validation, and optional checkpointing.
    Detection-only: uses train_one_epoch_detection and evaluate_detection.

    Uses bfloat16 autocast automatically on CUDA devices. Pass a GradScaler
    for API uniformity (it behaves as a no-op with bf16).

    Args:
        model (nn.Module): The model to train.
        trainloader (DataLoader): Training DataLoader.
        valloader (DataLoader): Validation DataLoader.
        criterion (callable): Detection loss (e.g. DetectionLoss).
        optimizer (Optimizer): PyTorch optimizer (e.g. SGD, Adam).
        device (torch.device): Target device ('cuda' or 'cpu').
        epochs (int): Number of epochs to train.
        scheduler (LRScheduler, optional): LR scheduler. Defaults to None.
            - step_scheduler_per_batch=True (default): stepped every mini-batch
              (e.g. OneCycleLR, CyclicLR).
            - step_scheduler_per_batch=False: stepped once per epoch after
              validation (e.g. CosineAnnealingLR, StepLR).
        scaler (GradScaler, optional): torch.amp.GradScaler for AMP training.
            For bf16 this is a no-op but is accepted for API compatibility.
            Defaults to None.
        clip_grad_norm (float): Clip gradient norms to this value before the
            optimizer step. Set to 0.0 to disable. Defaults to 1.0.
        checkpoint (ModelCheckpoint, optional): Callback invoked after each epoch.
            Monitors Val F1. Defaults to None.
        log (bool): If True, record this run with RunLogger. Each run is saved as
            its own timestamped JSON file — nothing is ever overwritten.
            Defaults to False.
        log_dir (str | None): Directory to store run JSON files.
            When None (default), auto-derived from the model class name.
        verbose (int): 0 = silent, 1 = log on new best (default), 2 = every epoch.
        model_label (str | None): Optional label added to the run config.
        resume (bool): If True, resume from the last checkpoint. Defaults to False.

    Returns:
        tuple: (history, logger)
            - history (dict): per-epoch metric lists with keys:
              'train_loss', 'val_loss', 'val_metric',
              'val_iou', 'val_f1', 'val_precision', 'val_recall',
              'val_cls_acc' (classification accuracy on matched detections),
              'val_e2e_f1', 'val_e2e_precision', 'val_e2e_recall'
              (end-to-end: correct localisation + correct class)
            - logger (RunLogger | None): the logger instance when log=True,
              otherwise None. Use it after fit() to call e.g.
              logger.log_test_results(sweep_results).
    """

    history = {
        'train_loss':       [],
        'val_loss':         [],
        'val_metric':       [],
        'val_iou':          [],
        'val_f1':           [],
        'val_precision':    [],
        'val_recall':       [],
        'val_cls_acc':      [],
        'val_e2e_f1':       [],
        'val_e2e_precision': [],
        'val_e2e_recall':   [],
    }

    amp_info = "bf16 autocast" if device.type == "cuda" else "fp32 (CPU)"
    print(f"Training with {amp_info} | Epochs: {epochs}")

    # ── Set up RunLogger ──────────────────────────────────────────────────────
    logger = None
    if log:
        from .logger import RunLogger
        model_name = type(model).__name__
        resolved_log_dir = log_dir if log_dir is not None else f"logs/{model_name}"
        logger = RunLogger(log_dir=resolved_log_dir, verbose=verbose)
        config = {
            "model":     model_name,
            "optimizer": type(optimizer).__name__,
            "scheduler": type(scheduler).__name__ if scheduler else None,
            "lr":        optimizer.param_groups[0]["lr"],
            "epochs":    epochs,
            "device":    str(device),
            "clip_grad_norm": clip_grad_norm,
            "step_scheduler_per_batch": step_scheduler_per_batch,
        }
        if model_label: config["label"] = model_label
        logger.start(config=config)

    total_start = time.time()
    best_val_metric = float('-inf')
    start_epoch = 0

    if resume and checkpoint is not None:
        try:
            start_epoch = checkpoint.resume_training(optimizer, scheduler, scaler)
            print(f"Resuming training from epoch {start_epoch}")
        except FileNotFoundError as e:
            print(f"Could not resume training: {e}")

    for epoch in range(start_epoch, epochs):
        train_start = time.perf_counter()

        # ── Train ─────────────────────────────────────────────────────────────
        avg_train_loss = train_one_epoch_detection(
            model, trainloader, criterion, optimizer, device,
            scheduler, scaler,
            clip_grad_norm=clip_grad_norm,
            step_scheduler_per_batch=step_scheduler_per_batch,
        )
        history['train_loss'].append(avg_train_loss)
        train_end = time.perf_counter()

        # Step scheduler per epoch (e.g. CosineAnnealingLR)
        if scheduler is not None and not step_scheduler_per_batch:
            scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]

        metrics           = evaluate_detection(model, valloader, criterion, device)
        avg_val_loss      = metrics["loss"]
        det               = metrics["detection"]
        cls               = metrics["classification"]
        e2e               = metrics["end_to_end"]
        val_iou           = det["iou"]
        val_f1            = det["f1"]
        val_precision     = det["precision"]
        val_recall        = det["recall"]
        val_cls_acc       = cls["accuracy"]
        val_e2e_f1        = e2e["f1"]
        val_e2e_precision = e2e["precision"]
        val_e2e_recall    = e2e["recall"]
        val_metric        = val_f1

        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(val_iou)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_cls_acc'].append(val_cls_acc)
        history['val_e2e_f1'].append(val_e2e_f1)
        history['val_e2e_precision'].append(val_e2e_precision)
        history['val_e2e_recall'].append(val_e2e_recall)
        history['val_metric'].append(val_metric)

        val_end = time.perf_counter()
        elapsed = train_end - train_start
        print(f"Train: {train_end - train_start:.1f}s | Val: {val_end - train_end:.1f}s")
        print(
            f"  --> Det  | PR: {val_precision:.3f} | Rec: {val_recall:.3f} | F1: {val_f1:.3f} "
            f"| TP: {det['tp']} FP: {det['fp']} FN: {det['fn']}"
        )
        print(
            f"  --> Cls  | Acc: {val_cls_acc:.3f}"
        )
        print(
            f"  --> E2E  | F1: {val_e2e_f1:.3f} "
            f"| PR: {e2e['precision']:.3f} | Rec: {e2e['recall']:.3f}"
        )

        # ── Checkpoint / best tracking ────────────────────────────────────────
        if checkpoint is not None:
            is_best = checkpoint(
                val_metric, epoch, optimizer, scheduler, scaler,
                metrics={
                    "val_loss": avg_val_loss, "val_iou": val_iou,
                    "val_f1": val_f1, "val_precision": val_precision,
                    "val_recall": val_recall, "val_cls_acc": val_cls_acc,
                    "val_e2e_f1": val_e2e_f1,
                    "val_e2e_precision": val_e2e_precision,
                    "val_e2e_recall": val_e2e_recall,
                }
            )
        else:
            is_best = val_metric > best_val_metric
        if is_best:
            best_val_metric = val_metric

        # ── Logging ───────────────────────────────────────────────────────────
        if wandb is not None and wandb.run is not None:
            wandb.log({
                "train/epoch_loss":   avg_train_loss,
                "val/loss":           avg_val_loss,
                "val/iou":            val_iou,
                "val/precision":      val_precision,
                "val/recall":         val_recall,
                "val/f1":             val_f1,
                "val/cls_acc":        val_cls_acc,
                "val/e2e_f1":         val_e2e_f1,
                "val/e2e_precision":  val_e2e_precision,
                "val/e2e_recall":     val_e2e_recall,
                "epoch":              epoch + 1,
            })

        if logger is not None:
            logger.log_epoch(
                epoch, train_loss=avg_train_loss, val_loss=avg_val_loss,
                val_metric=val_metric, lr=current_lr,
                epoch_time=elapsed, metric_label="val_f1",
                val_iou=val_iou, val_precision=val_precision, val_recall=val_recall,
                val_cls_acc=val_cls_acc,
                val_e2e_f1=val_e2e_f1,
                val_e2e_precision=val_e2e_precision,
                val_e2e_recall=val_e2e_recall,
            )
        elif verbose >= 2 or verbose is True or (verbose == 1 and is_best):
            best_marker = "  ★ NEW BEST" if is_best else ""
            print(
                f'Epoch [{epoch + 1:2d}/{epochs}] | Train Loss: {avg_train_loss:.4f} '
                f'| Val Loss: {avg_val_loss:.4f} | Det F1: {val_f1:.4f} '
                f'| Cls Acc: {val_cls_acc:.4f} | E2E F1: {val_e2e_f1:.4f} '
                f'| IoU: {val_iou:.4f} | Time: {elapsed:.2f}s{best_marker}'
            )

    total_minutes = (time.time() - total_start) / 60
    if verbose >= 2 or verbose is True:
        print(f'\nFinished Training in {total_minutes:.2f} minutes')

    best = None
    if checkpoint is not None:
        best = checkpoint.best_score
        print(f'Best Validation Metric: {best:.4f}')

    if logger is not None:
        logger.finish(best_val_metric=best)

    return history, logger

