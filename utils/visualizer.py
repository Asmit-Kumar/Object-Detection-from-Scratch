import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


class Visualizer:
    """
    Utility class for visualizing object detection results.
    """

    # ── Internal helper ───────────────────────────────────────────────────────
    @staticmethod
    def _to_numpy(image):
        """Convert a torch.Tensor or ndarray image to a 2-D numpy array."""
        if isinstance(image, torch.Tensor):
            return image.detach().cpu().squeeze().numpy()
        return np.squeeze(image)

    # ── Single-image visualisations ───────────────────────────────────────────

    @staticmethod
    def visualize_detection(image, pred_bbox, actual_bbox=None,
                            digit_class=None, conf=None, pred_class=None,
                            class_names=None, img_size=128):
        """
        Visualize a single detection with bounding boxes, confidence, and
        predicted class label.

        Args:
            image: torch.Tensor or ndarray, shape (1, H, W) or (H, W).
            pred_bbox: [x_min, y_min, x_max, y_max] (normalised 0-1).
            actual_bbox: optional GT box in same format.
            digit_class: optional GT class label shown in the title.
            conf: optional float confidence score (0-1) for the predicted box.
            pred_class: optional predicted class index to annotate on the box.
            class_names: optional list mapping class index → display name.
                If None and pred_class is an int, the raw index is shown.
            img_size: pixel size the normalised bbox coords map to. Default 128.
        """
        img_np = Visualizer._to_numpy(image)
        ax = plt.figure(figsize=(10, 10)).gca()
        ax.imshow(img_np, cmap='gray')

        def _add_rect(bbox, color, legend_label):
            x0 = bbox[0] * img_size
            y0 = bbox[1] * img_size
            w  = (bbox[2] - bbox[0]) * img_size
            h  = (bbox[3] - bbox[1]) * img_size
            ax.add_patch(patches.Rectangle(
                (x0, y0), w, h, edgecolor=color,
                facecolor='none', linewidth=2, label=legend_label))
            return x0, y0, w, h

        x0, y0, _, _ = _add_rect(pred_bbox, 'red', 'Predicted')

        # Build annotation text for the predicted box
        parts = []
        if pred_class is not None:
            cls_str = (class_names[pred_class]
                       if class_names is not None and pred_class < len(class_names)
                       else str(pred_class))
            parts.append(cls_str)
        if conf is not None:
            parts.append(f'{float(conf):.2f}')
        if parts:
            ax.text(
                x0 + 2, max(0, y0 - 4),
                ' | '.join(parts),
                color='red', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                          alpha=0.55, edgecolor='none'),
            )

        if actual_bbox is not None:
            _add_rect(actual_bbox, 'lime', 'GT')

        ax.legend(loc='upper left')
        title = 'Detection'
        if digit_class is not None:
            title = f'GT: {digit_class}'
        if pred_class is not None:
            cls_str = (class_names[pred_class]
                       if class_names is not None and pred_class < len(class_names)
                       else str(pred_class))
            title += f'   Pred: {cls_str}'
            if conf is not None:
                title += f' ({float(conf):.2f})'
        ax.set_title(title, fontsize=13)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_multi_detection_batch(
        imgs_v, preds, boxes_v, mask_v, conf_thresh=0.65, class_names=None
    ):

        """
        Visualize up to 10 images from a batch with GT boxes and annotated
        predicted boxes (confidence + predicted class if available).

        Args:
            imgs_v: (B, 1, H, W) image tensor (CPU).
            preds:  (B, MAX_OBJ, 5+C) raw model output
                    [x, y, w, h, conf_logit, cls_logit₀ … cls_logitₙ].
                    A pure-localisation output (C=0) is also accepted.
            boxes_v: (B, MAX_OBJ, 4) GT boxes.
            mask_v:  (B, MAX_OBJ) boolean GT mask.
            conf_thresh: sigmoid confidence threshold to filter slots.
            class_names: optional list[str] mapping class index → label.
                If None, raw class indices are shown.
        """
        N_SHOW = min(10, imgs_v.size(0))
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        has_classes = preds.shape[-1] > 5   # unified model output

        for i, ax in enumerate(axes.flat[:N_SHOW]):
            img        = imgs_v[i].squeeze().numpy()
            slots      = preds[i]                        # (MAX_OBJ, 5+C)
            gt_boxes   = boxes_v[i][mask_v[i]]

            conf_scores = torch.sigmoid(slots[:, 4])    # (MAX_OBJ,)
            keep_mask   = conf_scores >= conf_thresh      # boolean mask
            kept_boxes  = slots[keep_mask, :4]          # (K, 4)
            kept_conf   = conf_scores[keep_mask]        # (K,)

            if has_classes:
                kept_cls = slots[keep_mask, 5:].argmax(dim=-1)  # (K,)
            else:
                kept_cls = None

            # Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
            if kept_boxes.shape[0] > 1:
                x1 = kept_boxes[:, 0]
                y1 = kept_boxes[:, 1]
                x2 = x1 + kept_boxes[:, 2]
                y2 = y1 + kept_boxes[:, 3]
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                nms_keep = torchvision.ops.nms(boxes_xyxy, kept_conf, iou_threshold=0.35)
                kept_boxes = kept_boxes[nms_keep]
                kept_conf  = kept_conf[nms_keep]
                if kept_cls is not None:
                    kept_cls = kept_cls[nms_keep]

            ax.imshow(img, cmap='gray')

            # ── GT boxes (lime, solid) ─────────────────────────────────────
            for box in gt_boxes:
                x, y, w, h = box.tolist()
                ax.add_patch(patches.Rectangle(
                    (x, y), w, h, linewidth=1.5, edgecolor='lime',
                    facecolor='none'))


            # ── Predicted boxes (red, dashed) + annotation ────────────────
            for k in range(len(kept_boxes)):
                x, y, w, h = kept_boxes[k].tolist()
                ax.add_patch(patches.Rectangle(
                    (x, y), w, h, linewidth=1.5, edgecolor='red',
                    linestyle='--', facecolor='none'))

                # Build label: 'cls | conf'
                label_parts = []
                if kept_cls is not None:
                    cls_idx = int(kept_cls[k].item())
                    cls_str = (class_names[cls_idx]
                               if class_names is not None and cls_idx < len(class_names)
                               else str(cls_idx))
                    label_parts.append(cls_str)
                label_parts.append(f'{kept_conf[k].item():.2f}')

                ax.text(
                    x + 1, max(0, y - 3),
                    ' | '.join(label_parts),
                    color='red', fontsize=6, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black',
                              alpha=0.5, edgecolor='none'),
                )

            # GT count / pred count as subtitle
            ax.set_title(
                f'GT:{int(mask_v[i].sum())}  Pred:{len(kept_boxes)}',
                fontsize=7, color='white', pad=2,
            )
            ax.axis('off')

        fig.patch.set_facecolor('#1a1a1a')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_pipeline_detections(image, detections, title='End-to-End Pipeline Detections'):
        """
        Visualize pipeline detections with class labels and confidence scores.

        Args:
            image: torch.Tensor or ndarray input image.
            detections: list of DetectionResult objects (with .bbox, .char_label, .joint_conf).
            title: plot title.
        """
        img_np = Visualizer._to_numpy(image)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np, cmap='gray')

        for det in detections:
            x, y, w, h = det.bbox
            ax.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor='cyan', facecolor='none'))
            ax.text(
                x, max(0, y - 4),
                f'{det.char_label} ({det.joint_conf:.2f})',
                color='yellow', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                          alpha=0.6, edgecolor='none'),
            )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    # ── Training history ──────────────────────────────────────────────────────

    @staticmethod
    def visualize_training_curves(train_losses, val_losses, val_metric, metric_label='Val Metric'):
        """
        Plot train/val loss and one validation metric side by side.

        For the full detection dashboard use visualize_detection_history().

        Args:
            train_losses: per-epoch training losses.
            val_losses:   per-epoch validation losses.
            val_metric:   per-epoch metric values.
            metric_label: axis/title label for the metric panel.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses,   label='Val Loss')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(val_metric, label=metric_label, color='green')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel(metric_label)
        ax2.set_title(f'Validation {metric_label}')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_detection_history(history):
        """
        Full 2×3 dashboard from the history dict returned by fit().

        Panels: Loss | Det P/R/F1 | Det IoU
                Cls Acc | E2E F1 | Det F1 / Cls Acc / E2E F1 overlay

        Args:
            history (dict): keys expected:
                'train_loss', 'val_loss',
                'val_f1', 'val_precision', 'val_recall',
                'val_iou', 'val_cls_acc', 'val_e2e_f1'
        """
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Detection Training History', fontsize=15, fontweight='bold')

        def _panel(ax, title, ylabel, ylim=None):
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
            if ylim: ax.set_ylim(*ylim)
            ax.legend(); ax.grid(True, alpha=0.3)

        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], label='Train Loss', color='royalblue')
        ax.plot(epochs, history['val_loss'],   label='Val Loss',   color='tomato')
        _panel(ax, 'Loss', 'Loss')

        # Det P / R / F1
        ax = axes[0, 1]
        ax.plot(epochs, history['val_f1'],       label='F1',        color='mediumseagreen')
        ax.plot(epochs, history['val_precision'], label='Precision', color='cornflowerblue', linestyle='--')
        ax.plot(epochs, history['val_recall'],    label='Recall',    color='salmon',         linestyle='--')
        _panel(ax, 'Detection P / R / F1', 'Score', (0, 1))

        # Det IoU
        ax = axes[0, 2]
        ax.plot(epochs, history['val_iou'], label='IoU', color='darkorange')
        _panel(ax, 'Detection IoU', 'IoU', (0, 1))

        # Cls Acc
        ax = axes[1, 0]
        ax.plot(epochs, history['val_cls_acc'], label='Cls Acc', color='mediumpurple')
        _panel(ax, 'Classification Accuracy\n(on matched detections)', 'Accuracy', (0, 1))

        # E2E F1
        ax = axes[1, 1]
        ax.plot(epochs, history['val_e2e_f1'], label='E2E F1', color='teal')
        if 'val_e2e_precision' in history and len(history['val_e2e_precision']) == len(epochs):
            ax.plot(epochs, history['val_e2e_precision'], label='E2E Precision', color='cadetblue', linestyle='--')
        if 'val_e2e_recall' in history and len(history['val_e2e_recall']) == len(epochs):
            ax.plot(epochs, history['val_e2e_recall'], label='E2E Recall', color='darkturquoise', linestyle='--')
        _panel(ax, 'End-to-End P / R / F1\n(box + class correct)', 'F1', (0, 1))

        # Overlay
        ax = axes[1, 2]
        ax.plot(epochs, history['val_f1'],      label='Det F1',  color='mediumseagreen')
        ax.plot(epochs, history['val_cls_acc'], label='Cls Acc', color='mediumpurple')
        ax.plot(epochs, history['val_e2e_f1'],  label='E2E F1',  color='teal')
        _panel(ax, 'Det F1 / Cls Acc / E2E F1', 'Score', (0, 1))

        plt.tight_layout()
        plt.show()

    # ── Evaluation results ────────────────────────────────────────────────────

    @staticmethod
    def visualize_per_class_e2e(metrics, title='Per-Class End-to-End Metrics'):
        """
        Grouped bar chart of per-class E2E Precision / Recall / F1.

        Args:
            metrics (dict): full dict from evaluate_detection().
                Uses metrics['classification']['per_class'].
            title: figure title.
        """
        per_class = metrics['classification']['per_class']
        if not per_class:
            print('No per-class data available.')
            return

        class_ids  = sorted(per_class.keys())
        labels     = [per_class[k].get('name', f'class_{k}') for k in class_ids]
        precisions = [per_class[k]['e2e_precision'] for k in class_ids]
        recalls    = [per_class[k]['e2e_recall']    for k in class_ids]
        f1s        = [per_class[k]['e2e_f1']        for k in class_ids]

        x, w = np.arange(len(labels)), 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))
        ax.bar(x - w, precisions, w, label='E2E Precision', color='cornflowerblue')
        ax.bar(x,     recalls,    w, label='E2E Recall',    color='salmon')
        ax.bar(x + w, f1s,        w, label='E2E F1',        color='mediumseagreen')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_sweep_results(sweep_results, title='Confidence Threshold Sweep'):
        """
        Plot Det P/R/F1 and Cls Accuracy across confidence thresholds.

        Args:
            sweep_results (list[dict]): output of evaluate_detection_sweep().
                Each dict must contain 'conf', 'precision', 'recall', 'f1', 'cls_accuracy'.
            title: figure title.
        """
        confs      = [r['conf']                    for r in sweep_results]
        precisions = [r['precision']               for r in sweep_results]
        recalls    = [r['recall']                  for r in sweep_results]
        f1s        = [r['f1']                      for r in sweep_results]
        cls_accs   = [r.get('cls_accuracy', 0.0)   for r in sweep_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontsize=13, fontweight='bold')

        ax1.plot(confs, f1s,        marker='o', label='Det F1',       color='mediumseagreen')
        ax1.plot(confs, precisions, marker='s', label='Det Precision', color='cornflowerblue', linestyle='--')
        ax1.plot(confs, recalls,    marker='^', label='Det Recall',    color='salmon',          linestyle='--')
        ax1.set_xlabel('Confidence Threshold'); ax1.set_ylabel('Score')
        ax1.set_title('Detection P / R / F1 vs Threshold')
        ax1.set_ylim(0, 1); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(confs, cls_accs, marker='D', color='mediumpurple', label='Cls Accuracy')
        ax2.set_xlabel('Confidence Threshold'); ax2.set_ylabel('Accuracy')
        ax2.set_title('Classification Accuracy vs Threshold\n(on matched detections)')
        ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
