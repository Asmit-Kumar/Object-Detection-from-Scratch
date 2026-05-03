import matplotlib.pyplot as plt
import torch


class Visualizer:
    """
    Utility class for visualizing object detection results.
    """

    @staticmethod
    def visualize_detection(image, pred_bbox, actual_bbox=None, digit_class=None):
        """
        Visualize detection results with bounding boxes.

        Args:
            image (torch.Tensor or numpy.ndarray): The image tensor to display. Shape should be (1, H, W) or (H, W).
            pred_bbox (list): Predicted bounding box [x_min, y_min, x_max, y_max] (normalized or relative).
                              The code assumes these are normalized to 0-1 or relative to 128x128.
            actual_bbox (list, optional): Ground truth bounding box [x_min, y_min, x_max, y_max] (normalized). Defaults to None.
            digit_class (int/str, optional): Detected digit class to display in title. Defaults to None.
        """
        plt.figure(figsize=(10, 10))
        
        # Ensure tensor is on CPU and converted to numpy for matplotlib
        if isinstance(image, torch.Tensor):
            img_np = image.detach().cpu().squeeze().numpy()
        else:
            import numpy as np
            img_np = np.squeeze(image)
            
        plt.imshow(img_np, cmap="gray")

        # Draw predicted bbox
        rect = plt.Rectangle(
            (pred_bbox[0] * 128, pred_bbox[1] * 128),
            (pred_bbox[2] - pred_bbox[0]) * 128,
            (pred_bbox[3] - pred_bbox[1]) * 128,
            edgecolor='red',
            facecolor='none',
            linewidth=2,
            label="Predicted"
        )
        plt.gca().add_patch(rect)

        # Draw actual bbox if provided
        if actual_bbox is not None:
            rect = plt.Rectangle(
                (actual_bbox[0] * 128, actual_bbox[1] * 128),
                (actual_bbox[2] - actual_bbox[0]) * 128,
                (actual_bbox[3] - actual_bbox[1]) * 128,
                edgecolor='green',
                facecolor='none',
                linewidth=2,
                label="Actual"
            )
            plt.gca().add_patch(rect)

        plt.legend(loc="upper left")
        if digit_class is not None:
            plt.title(f"Detected Digit: {digit_class}")
        plt.axis('off')
        plt.show()


def plot_training_curves(train_losses, val_losses, val_metric, metric_label="Val Metric"):
    """
    Plot training/validation loss and a validation metric side by side.

    Args:
        train_losses (list): Per-epoch training losses.
        val_losses (list): Per-epoch validation losses.
        val_metric (list): Per-epoch validation metric (e.g. accuracy or IoU).
        metric_label (str): Y-axis label and title for the metric plot.
            Defaults to "Val Metric". Use "Accuracy (%)" for classification
            or "IoU" for regression.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_metric, label=metric_label, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_label)
    ax2.set_title(f'Validation {metric_label}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


