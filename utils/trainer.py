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
            images, boxes, labels, mask = next(loader_iter)
        except StopIteration:
            break

        if is_timing:
            torch.cuda.synchronize()
            t_data = time.perf_counter()

        images = images.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if is_timing:
            torch.cuda.synchronize()
            t_pad = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        
        if is_timing:
            t0 = t_pad

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
                outputs = model(images)
                if is_timing: torch.cuda.synchronize(); t1 = time.perf_counter()
                loss = criterion(outputs, boxes, mask)
                if is_timing: torch.cuda.synchronize(); t2 = time.perf_counter()
        else:
            outputs = model(images)
            if is_timing: torch.cuda.synchronize(); t1 = time.perf_counter()
            loss = criterion(outputs, boxes, mask)
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
            
        epoch_match_ratio += getattr(criterion, 'last_matched_ratio', 1.0)

        if is_timing:
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            print(
                f"Batch {n_batches+1:2d} | "
                f"match={getattr(criterion, 'last_matched_ratio', 1.0):.2f} "
                f"hungar={getattr(criterion, 'last_hungarian_time', 0.0):.1f}ms "
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
                "train/match_ratio": getattr(criterion, 'last_matched_ratio', 1.0)
            })

        n_batches += 1

    if hasattr(criterion, 'last_matched_ratio'):
        print(f"  --> Epoch Average Match Ratio: {epoch_match_ratio / n_batches:.4f}")

    return epoch_loss / n_batches


def evaluate_detection(
    model, loader, criterion, device,
):
    """Evaluate a detection model on a single pass.

    Computes greedy mean IoU (the training/validation metric).
    For full P/R/F1 threshold analysis, use evaluate_detection_sweep().
    Expects loader to yield (images, boxes, labels, mask).
    """
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    n_images   = 0
    use_amp    = device.type == "cuda"

    with torch.no_grad():
        for images, boxes, labels, mask in loader:
            images = images.to(device, non_blocking=True)
            boxes  = boxes.to(device, non_blocking=True)
            mask   = mask.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
                    outputs = model(images)
                    loss    = criterion(outputs, boxes, mask)
            else:
                outputs = model(images)
                loss    = criterion(outputs, boxes, mask)

            total_loss += loss.item()

            pred_boxes = outputs[..., :4].float()

            iou = _pairwise_iou(pred_boxes, boxes)   # (B, N_pred, max_gt)
            iou.masked_fill_(~mask.unsqueeze(1), -1.0)
            matched_iou = iou.max(dim=1).values       # (B, max_gt)

            for b in range(images.size(0)):
                b_mask = mask[b]
                if b_mask.any():
                    total_iou += matched_iou[b, b_mask].mean().item()
                    n_images  += 1

    avg_loss = total_loss / len(loader)
    mean_iou = total_iou / max(n_images, 1)

    return {"val_loss": avg_loss, "val_iou": mean_iou}


def evaluate_detection_sweep(
    model, loader, device,
    conf_thresholds: np.ndarray | list[float] | tuple = np.arange(0.2, 0.9, 0.1),
    iou_threshold: float = 0.5,
    print_table: bool = True,
):
    """Run a confidence threshold sweep and return a list of P/R/F1 dicts.

    Unlike evaluate_detection, this does NOT compute loss — it only computes
    detection metrics at multiple thresholds in a single forward pass.

    Intended for use on the test set after training completes. Results are
    returned as a list of dicts (one per threshold) suitable for saving into
    RunLogger via logger.log_test_results().

    Args:
        model:             Trained detection model.
        loader:            Test DataLoader.
        device:            torch.device.
        conf_thresholds:   Iterable of confidence thresholds to sweep.
        iou_threshold:     IoU threshold for a TP match.
        print_table:       If True, print a formatted P/R/F1 table.

    Returns:
        List of dicts: [{"conf": 0.2, "precision": ..., "recall": ..., "f1": ...,
                         "tp": ..., "fp": ..., "fn": ...}, ...]
    """
    thresholds = sorted(conf_thresholds)
    model.eval()
    use_amp = device.type == "cuda"

    # Collect (pred_conf_cpu, pred_boxes_cpu, iou_cpu, n_gt) per image across
    # the whole loader — then sweep thresholds on CPU. This avoids re-running
    # the model N times.
    records = []  # list of (pred_conf np.ndarray, iou np.ndarray, n_gt int)

    with torch.no_grad():
        for images, boxes, labels, mask in loader:
            images = images.to(device, non_blocking=True)
            boxes  = boxes.to(device, non_blocking=True)
            mask   = mask.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=_AMP_DTYPE):
                    outputs = model(images)
            else:
                outputs = model(images)

            pred_boxes = outputs[..., :4].float()
            pred_conf  = torch.sigmoid(outputs[..., 4].float())
            iou        = _pairwise_iou(pred_boxes, boxes)  # (B, N_pred, max_gt)

            # Pull everything off GPU in one shot per batch
            pred_conf_np = pred_conf.cpu().numpy()
            iou_np       = iou.cpu().numpy()
            mask_np      = mask.cpu().numpy()

            for b in range(images.size(0)):
                gt_valid = mask_np[b]  # (max_gt,) bool
                n_gt = int(gt_valid.sum())
                if n_gt == 0:
                    continue
                records.append((
                    pred_conf_np[b],            # (N_pred,)
                    iou_np[b][:, gt_valid],     # (N_pred, n_gt) — only valid GT cols
                    n_gt,
                ))

    import numpy as np

    results = []
    for thr in thresholds:
        tp = fp = fn = 0
        for conf, sub_iou, n_gt in records:
            pr_mask = conf > thr
            pr_indices = np.where(pr_mask)[0]
            n_pr = len(pr_indices)

            if n_pr == 0:
                fn += n_gt
                continue

            # Sort active preds by confidence descending (COCO convention)
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
        results.append({
            "conf": round(thr, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        })

    if print_table:
        print(f"\n{'Conf':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>7}  {'FP':>7}  {'FN':>7}")
        print("-" * 56)
        for r in results:
            print(
                f"{r['conf']:6.2f}  {r['precision']:6.4f}  {r['recall']:6.4f}  "
                f"{r['f1']:6.4f}  {r['tp']:7d}  {r['fp']:7d}  {r['fn']:7d}"
            )

    return results



def fit(
    model, trainloader, valloader, criterion,
    optimizer, device, epochs, task_type="classification",
    scheduler=None, scaler=None, checkpoint=None,
    log=False, log_dir=None,
    clip_grad_norm: float = 1.0,
    step_scheduler_per_batch: bool = True,
    gpu_augment=None,
    verbose: int = 1,
    print_per_class_accuracy: bool = False,
    class_names=None,
    model_label: str = None,
    resume: bool = False,
):
    """
    Full training loop with per-epoch logging, validation, and optional checkpointing.
    Supports both classification and bounding box regression tasks.

    Uses bfloat16 autocast automatically on CUDA devices. Pass a GradScaler
    for API uniformity (it behaves as a no-op with bf16).

    Args:
        model (nn.Module): The model to train.
        trainloader (DataLoader): Training DataLoader. Must be produced by DatasetBuilder.
        valloader (DataLoader): Validation DataLoader.
        criterion (callable): Loss function (e.g. nn.CrossEntropyLoss, nn.HuberLoss).
        optimizer (Optimizer): PyTorch optimizer (e.g. SGD, Adam).
        device (torch.device): Target device ('cuda' or 'cpu').
        epochs (int): Number of epochs to train.
        task_type (str): One of "classification" or "regression".
            - "classification": Tracks Val Loss and Val Accuracy (%).
            - "regression": Tracks Val Loss, Val MAE, and Val IoU.
        scheduler (LRScheduler, optional): LR scheduler. Defaults to None.
            - step_scheduler_per_batch=True (default): stepped every mini-batch
              (e.g. OneCycleLR, CyclicLR).
            - step_scheduler_per_batch=False: stepped once per epoch after
              validation (e.g. CosineAnnealingLR, StepLR).
        scaler (GradScaler, optional): torch.amp.GradScaler for AMP training.
            For bf16 this is a no-op but is accepted for API compatibility.
            Defaults to None.
        clip_grad_norm (float): Clip gradient norms to this value before the
            optimizer step. Matches the CIFAR reference (default 1.0).
            Set to 0.0 to disable. Defaults to 1.0.
        gpu_augment (callable | None): GPU batch augmentation from
            get_emnist_gpu_transform(). Applied on-device before forward pass.
            Defaults to None.
        checkpoint (ModelCheckpoint, optional): Callback invoked after each epoch.
            For classification, monitors Val Accuracy.
            For regression, monitors Val IoU.
            Defaults to None.
        log (bool): If True, record this run with RunLogger. Each run is saved as
            its own timestamped JSON file — nothing is ever overwritten.
            Defaults to False.
        log_dir (str | None): Directory to store run JSON files.
            When None (default), auto-derived from the model class name::

                CharacterClassifier  →  logs/CharacterClassifier/
                ObjectDetectorRes    →  logs/ObjectDetectorRes/
        print_per_class_accuracy (bool): If True, print per-class validation
            accuracy for classification runs. Defaults to False.
        class_names (Sequence | None): Optional display names indexed by class id
            for per-class accuracy output. When None, inferred from the loader.

    Returns:
        tuple: (history, logger)
            - history (dict): per-epoch metric lists.
              Keys: 'train_loss', 'val_loss', 'val_metric'
              Detection adds: 'val_iou' | Regression adds: 'val_mae', 'val_iou'
            - logger (RunLogger | None): the logger instance when log=True,
              otherwise None. Use it after fit() to call e.g.
              logger.log_test_results(sweep_results).
    """
    history = {'train_loss': [], 'val_loss': [], 'val_metric': []}
    if task_type == "regression":
        history['val_mae'] = []
        history['val_iou'] = []
    elif task_type == "detection":
        history['val_iou'] = []

    amp_info = "bf16 autocast" if device.type == "cuda" else "fp32 (CPU)"
    metric_unit = "%" if task_type == "classification" else ""
    print(f"Training with {amp_info} | Task: {task_type} | Epochs: {epochs}")

    # ── Set up RunLogger ──────────────────────────────────────────────────────
    logger = None
    if log:
        from .logger import RunLogger
        model_name = type(model).__name__
        resolved_log_dir = log_dir if log_dir is not None else f"logs/{model_name}"
        logger = RunLogger(log_dir=resolved_log_dir, verbose=verbose, metric_unit=metric_unit)
        config = {
            "model":     model_name,
            "task":      task_type,
            "optimizer": type(optimizer).__name__,
            "scheduler": type(scheduler).__name__ if scheduler else None,
            "lr":        optimizer.param_groups[0]["lr"],
            "epochs":    epochs,
            "device":    str(device),
            "clip_grad_norm": clip_grad_norm,
            "step_scheduler_per_batch": step_scheduler_per_batch,
            "gpu_augment": gpu_augment is not None,
            "print_per_class_accuracy": print_per_class_accuracy,
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

        # Train
        if task_type == "detection":
            avg_train_loss = train_one_epoch_detection(
                model, trainloader, criterion, optimizer, device,
                scheduler, scaler,
                clip_grad_norm=clip_grad_norm,
                step_scheduler_per_batch=step_scheduler_per_batch,
            )
        else:
            avg_train_loss = train_one_epoch(
                model, trainloader, criterion, optimizer, device,
                scheduler, scaler,
                clip_grad_norm=clip_grad_norm,
                step_scheduler_per_batch=step_scheduler_per_batch,
                gpu_augment=gpu_augment,
            )
        history['train_loss'].append(avg_train_loss)
        
        train_end = time.perf_counter()

        # Step scheduler per epoch (e.g. CosineAnnealingLR)
        if scheduler is not None and not step_scheduler_per_batch:
            scheduler.step()

        # Validate
        current_lr = optimizer.param_groups[0]["lr"]

        if task_type == "classification":
            avg_val_loss, val_metric = evaluate_classification(
                model, valloader, criterion, device,
                print_per_class_accuracy=print_per_class_accuracy,
                class_names=class_names,
            )
            val_end = time.perf_counter()
            elapsed = train_end - train_start
            
            print(f"Train: {train_end - train_start:.1f}s | Val: {val_end - train_end:.1f}s")
            
            history['val_loss'].append(avg_val_loss)
            history['val_metric'].append(val_metric)  # Accuracy

            if checkpoint is not None:
                is_best = checkpoint(val_metric, epoch, optimizer, scheduler, scaler)
            else:
                is_best = val_metric > best_val_metric
            if is_best:
                best_val_metric = val_metric

            if wandb is not None and wandb.run is not None:
                wandb.log({
                    "train/epoch_loss": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "val/accuracy": val_metric,
                    "epoch": epoch + 1
                })

            if logger is not None:
                logger.log_epoch(
                    epoch, train_loss=avg_train_loss, val_loss=avg_val_loss,
                    val_metric=val_metric, lr=current_lr,
                    epoch_time=elapsed, metric_label="val_acc",
                )
            elif verbose >= 2 or verbose is True or (verbose == 1 and is_best):
                best_marker = "  ★ NEW BEST" if is_best else ""
                print(f'Epoch [{epoch + 1:2d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_metric:.2f}% | Time: {elapsed:.2f}s{best_marker}')

        elif task_type == "regression":
            avg_val_loss, val_mae, val_iou = evaluate_regression(model, valloader, criterion, device)
            val_end = time.perf_counter()
            elapsed = train_end - train_start
            
            print(f"Train: {train_end - train_start:.1f}s | Val: {val_end - train_end:.1f}s")
            
            history['val_loss'].append(avg_val_loss)
            history['val_mae'].append(val_mae)
            history['val_iou'].append(val_iou)

            # Using validation IoU as the checkpointing metric
            if checkpoint is not None:
                is_best = checkpoint(val_iou, epoch, optimizer, scheduler, scaler)
            else:
                is_best = val_iou > best_val_metric
            if is_best:
                best_val_metric = val_iou

            if wandb is not None and wandb.run is not None:
                wandb.log({
                    "train/epoch_loss": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "val/mae": val_mae,
                    "val/iou": val_iou,
                    "epoch": epoch + 1
                })

            if logger is not None:
                logger.log_epoch(
                    epoch, train_loss=avg_train_loss, val_loss=avg_val_loss,
                    val_metric=val_iou, lr=current_lr,
                    epoch_time=elapsed, metric_label="val_iou",
                    val_mae=val_mae,
                )
            elif verbose >= 2 or verbose is True or (verbose == 1 and is_best):
                best_marker = "  ★ NEW BEST" if is_best else ""
                print(f'Epoch [{epoch + 1:2d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae:.4f} | Val IoU: {val_iou:.4f} | Time: {elapsed:.2f}s{best_marker}')

        elif task_type == "detection":
            metrics     = evaluate_detection(model, valloader, criterion, device)
            avg_val_loss = metrics["val_loss"]
            val_iou     = metrics["val_iou"]
            val_end     = time.perf_counter()
            elapsed     = train_end - train_start

            print(f"Train: {train_end - train_start:.1f}s | Val: {val_end - train_end:.1f}s")

            history['val_loss'].append(avg_val_loss)
            history['val_iou'].append(val_iou)

            if checkpoint is not None:
                is_best = checkpoint(val_iou, epoch, optimizer, scheduler, scaler)
            else:
                is_best = val_iou > best_val_metric
            if is_best:
                best_val_metric = val_iou

            if wandb is not None and wandb.run is not None:
                wandb.log({
                    "train/epoch_loss": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "val/iou": val_iou,
                    "epoch": epoch + 1
                })

            if logger is not None:
                logger.log_epoch(
                    epoch, train_loss=avg_train_loss, val_loss=avg_val_loss,
                    val_metric=val_iou, lr=current_lr,
                    epoch_time=elapsed, metric_label="val_iou",
                )
            elif verbose >= 2 or verbose is True or (verbose == 1 and is_best):
                best_marker = "  * NEW BEST" if is_best else ""
                print(f'Epoch [{epoch + 1:2d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {val_iou:.4f} | Time: {elapsed:.2f}s{best_marker}')


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

