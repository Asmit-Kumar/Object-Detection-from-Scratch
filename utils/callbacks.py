import torch
from pathlib import Path


class EarlyStopping:
    """
    Early stops the training if a monitored metric doesn't improve after a given patience.

    Args:
        patience (int): How many epochs to wait after last improvement before stopping.
        min_delta (float): Minimum change to qualify as an improvement.
        mode (str): 'min' for loss (lower is better), 'max' for accuracy (higher is better).
        verbose (bool): Print a message for each epoch where metric doesn't improve.
    """
    def __init__(self, patience=5, min_delta=0.001, mode='min', verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False

        if self.mode == 'min':
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.best_score = float('-inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, current_score):
        if self.mode == 'min':
            is_improvement = current_score < (self.best_score - self.min_delta)
        else:
            is_improvement = current_score > (self.best_score + self.min_delta)

        if is_improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


class ModelCheckpoint:
    """
    Saves the best model weights and maintains a crash-protection checkpoint
    that includes the full training state (optimizer, scheduler, scaler).

    Args:
        model (torch.nn.Module): The model to checkpoint.
        checkpoint_path (str): Path for the crash-protection checkpoint (saved every epoch).
        best_model_path (str): Path for the best model weights.
        mode (str): 'min' for loss (lower is better), 'max' for accuracy (higher is better).
        verbose (bool): Print a message when a new best model is saved.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: str = "./checkpoint/model.pth",
        best_model_path: str = "./checkpoint/best_model.pth",
        mode: str = "min",
        verbose: bool = True,
    ):
        self.model = model
        self.checkpoint_path = Path(checkpoint_path)
        self.best_model_path = Path(best_model_path)
        self.mode = mode
        self.verbose = verbose

        # Ensure checkpoint directories exist at init time
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.mode == 'min':
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.best_score = float('-inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, current_score: float, epoch: int = 0, optimizer=None, scheduler=None, scaler=None) -> bool:
        """
        Save the latest checkpoint and conditionally update the best model.

        Args:
            current_score (float): The metric to monitor (val_loss or val_acc).
            epoch (int): Current epoch number (stored in checkpoint for resume).
            optimizer: The optimizer (state saved for crash protection).
            scheduler: The LR scheduler (state saved for crash protection).
            scaler: The AMP GradScaler (state saved for crash protection).

        Returns:
            True if a new best was found.
        """
        # 1. Crash Protection: Save the full training state every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state'] = scheduler.state_dict()
        if scaler is not None:
            checkpoint['scaler_state'] = scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_path)

        # 2. Save Best Model
        if self.mode == 'min':
            is_best = current_score < self.best_score
        else:
            is_best = current_score > self.best_score

        if is_best:
            self.best_score = current_score
            torch.save(self.model.state_dict(), self.best_model_path)
            if self.verbose:
                print(f"[ModelCheckpoint] New best ({self.mode}): {current_score:.4f} — saved to {self.best_model_path}")

        return is_best

    def restore_best_weights(self) -> None:
        """Load the best saved weights back into the model (in-place)."""
        if not self.best_model_path.exists():
            raise FileNotFoundError(
                f"No best-model checkpoint found at '{self.best_model_path}'. "
                "Make sure training has run at least one epoch."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=device, weights_only=True)
        )
        if self.verbose:
            print(f"[ModelCheckpoint] Restored best weights from {self.best_model_path} (score={self.best_score:.4f})")

    def resume_training(self, optimizer=None, scheduler=None, scaler=None) -> int:
        """
        Load the latest checkpoint to resume training after a crash.

        Returns:
            The epoch number to resume from (next epoch after the saved one).
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at '{self.checkpoint_path}'. "
                "Cannot resume training."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)

        self.model.load_state_dict(ckpt['model_state'])
        if optimizer is not None and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if scheduler is not None and 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        if scaler is not None and 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])

        resume_epoch = ckpt['epoch'] + 1
        if self.verbose:
            print(f"[ModelCheckpoint] Resumed from epoch {ckpt['epoch'] + 1}, starting at epoch {resume_epoch + 1}")
        return resume_epoch
