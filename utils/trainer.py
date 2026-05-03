"""
Training and Evaluation Utilities.

Provides reusable functions for training loops, validation, testing, and
metrics computation (both classification and regression) with AMP support.
"""

import time
import torch
import torch.nn.functional as F

def train_one_epoch(
        model, loader, criterion, optimizer,
        device, scheduler=None, scaler=None
):
    """
    Run one full training epoch.
    """
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            # Assumes scheduler is stepped per batch (e.g. OneCycleLR)
            scheduler.step()

        epoch_loss += loss.item()
        n_batches += 1

    return epoch_loss / n_batches


def evaluate_classification(model, loader, criterion, device):
    """
    Evaluate model on a classification dataset.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def compute_iou(pred_bboxes, target_bboxes):
    """
    Computes Intersection over Union (IoU) between predicted and target bounding boxes.
    Assumes bounding boxes are in format [x_min, y_min, x_max, y_max].
    """
    x1 = torch.max(pred_bboxes[:, 0], target_bboxes[:, 0])
    y1 = torch.max(pred_bboxes[:, 1], target_bboxes[:, 1])
    x2 = torch.min(pred_bboxes[:, 2], target_bboxes[:, 2])
    y2 = torch.min(pred_bboxes[:, 3], target_bboxes[:, 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    pred_area = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
    target_area = (target_bboxes[:, 2] - target_bboxes[:, 0]) * (target_bboxes[:, 3] - target_bboxes[:, 1])
    
    union_area = pred_area + target_area - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)
    
    return iou.mean().item()

def evaluate_regression(model, loader, criterion, device):
    """
    Evaluate model on a bounding box regression dataset.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for imgs, bboxes in loader:
            imgs, bboxes = imgs.to(device), bboxes.to(device)
            outputs = model(imgs)
            
            total_loss += criterion(outputs, bboxes).item()
            total_mae += F.l1_loss(outputs, bboxes).item()
            total_iou += compute_iou(outputs, bboxes)

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    avg_iou = total_iou / n_batches
    
    return avg_loss, avg_mae, avg_iou


def fit(
    model, trainloader, valloader, criterion,
    optimizer, device, epochs, task_type="classification",
    scheduler=None, scaler=None, checkpoint=None,
):
    """
    Full training loop with per-epoch logging, validation, and optional checkpointing.
    Supports both classification and bounding box regression tasks.

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
        scheduler (LRScheduler, optional): LR scheduler stepped per mini-batch (e.g. OneCycleLR).
            If None, no LR scheduling is applied. Defaults to None.
        scaler (GradScaler, optional): torch.amp.GradScaler for AMP mixed precision training.
            If None, training runs in full float32 precision. Defaults to None.
        checkpoint (ModelCheckpoint, optional): Callback invoked after each epoch.
            For classification, monitors Val Accuracy.
            For regression, monitors Val IoU.
            Defaults to None.

    Returns:
        dict: history dict with keys:
            - Always present: 'train_loss', 'val_loss', 'val_metric'
            - Regression only: 'val_mae', 'val_iou'
    """
    history = {'train_loss': [], 'val_loss': [], 'val_metric': []}
    if task_type == "regression":
        history['val_mae'] = []
        history['val_iou'] = []

    total_start = time.time()

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        avg_train_loss = train_one_epoch(
            model, trainloader, criterion, optimizer, device, scheduler, scaler
        )
        history['train_loss'].append(avg_train_loss)

        # Validate
        if task_type == "classification":
            avg_val_loss, val_metric = evaluate_classification(model, valloader, criterion, device)
            history['val_loss'].append(avg_val_loss)
            history['val_metric'].append(val_metric)  # Accuracy
            
            if checkpoint is not None:
                checkpoint(val_metric, epoch, optimizer, scheduler, scaler)
                
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch + 1:2d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_metric:.2f}% | Time: {elapsed:.2f}s')

        elif task_type == "regression":
            avg_val_loss, val_mae, val_iou = evaluate_regression(model, valloader, criterion, device)
            history['val_loss'].append(avg_val_loss)
            history['val_mae'].append(val_mae)
            history['val_iou'].append(val_iou)
            
            # Using validation IoU as the checkpointing metric
            if checkpoint is not None:
                checkpoint(val_iou, epoch, optimizer, scheduler, scaler)
                
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch + 1:2d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae:.4f} | Val IoU: {val_iou:.4f} | Time: {elapsed:.2f}s')

    total_minutes = (time.time() - total_start) / 60
    print(f'\nFinished Training in {total_minutes:.2f} minutes')
    if checkpoint is not None:
        print(f'Best Validation Metric: {checkpoint.best_score:.4f}')

    return history
