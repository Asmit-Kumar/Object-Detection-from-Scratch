import torch
from pathlib import Path


class ModelCheckpoint:
    """
    Saves the best model weights and maintains a crash-protection checkpoint
    that includes the full training state (optimizer, scheduler, scaler, fitted anchors_wh).

    Args:
        model (torch.nn.Module): The model to checkpoint.
        checkpoint_path (str): Path for the crash-protection checkpoint (saved every epoch).
        best_model_path (str): Path for the best model weights.
        mode (str): 'min' for loss (lower is better), 'max' for accuracy (higher is better).
        verbose (bool): Print a message when a new best model is saved.
        config (dict | None): Optional dictionary of training hyperparameters/settings to persist.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: str = "./checkpoint/model.pth",
        best_model_path: str = "./checkpoint/best_model.pth",
        mode: str = "max",
        verbose: bool = True,
        config: dict = None,
    ):
        self.model = model
        self.checkpoint_path = Path(checkpoint_path)
        self.best_model_path = Path(best_model_path)
        self.mode = mode
        self.verbose = verbose
        self.config = config

        # Ensure checkpoint directories exist at init time
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.mode == 'min':
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.best_score = float('-inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

        # Safely inspect existing checkpoint/best_model to set the comparison baseline
        # without loading or touching any model weights or optimizer states.
        self._peek_existing_best_score()

    def _peek_existing_best_score(self) -> None:
        """Safely peek at existing files to initialize best_score baseline."""
        for path in (self.best_model_path, self.checkpoint_path):
            if path.exists():
                try:
                    data = torch.load(path, map_location="cpu", weights_only=False)
                    if isinstance(data, dict):
                        score = data.get("best_score", data.get("best_metric", None))
                        if score is not None and isinstance(score, (int, float)):
                            self.best_score = float(score)
                            if self.verbose:
                                print(f"[ModelCheckpoint] Initialized best_score baseline: {self.best_score:.4f} from '{path.name}'")
                            return
                except Exception:
                    pass

    def __call__(
        self,
        current_score: float,
        epoch: int = 0,
        optimizer=None,
        scheduler=None,
        scaler=None,
        metrics: dict = None,
        anchors_wh: torch.Tensor | None = None,
    ) -> bool:
        """
        Save the latest checkpoint and conditionally update the best model.

        Args:
            current_score (float): The metric to monitor (val_loss or val_acc).
            epoch (int): Current epoch number (stored in checkpoint for resume).
            optimizer: The optimizer (state saved for crash protection).
            scheduler: The LR scheduler (state saved for crash protection).
            scaler: The AMP GradScaler (state saved for crash protection).
            metrics: Optional dictionary of additional metrics to store.
            anchors_wh: Optional Tensor of fitted width/height anchor dimensions (K, 2).

        Returns:
            True if a new best was found.
        """
        # 1. Crash Protection: Save the full training state every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'best_score': self.best_score,
            'current_score': current_score,
        }
        if anchors_wh is not None:
            checkpoint['anchors_wh'] = anchors_wh
        if self.config is not None:
            checkpoint['config'] = self.config
        if metrics is not None:
            checkpoint['metrics'] = metrics
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
            save_payload = {
                'model_state_dict': self.model.state_dict(),
                'best_score': self.best_score,
            }
            if anchors_wh is not None:
                save_payload['anchors_wh'] = anchors_wh
            if self.config is not None:
                save_payload['config'] = self.config
            torch.save(save_payload, self.best_model_path)
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
        state = torch.load(self.best_model_path, map_location=device, weights_only=False)
        model_state = state.get('model_state_dict', state) if isinstance(state, dict) else state
        self.model.load_state_dict(model_state)
        if self.verbose:
            print(f"[ModelCheckpoint] Restored best weights from {self.best_model_path} (score={self.best_score:.4f})")

    def resume_training(self, optimizer=None, scheduler=None, scaler=None) -> tuple[int, torch.Tensor | None]:
        """
        Load the latest checkpoint to resume training after a crash.

        Returns:
            Tuple of (resume_epoch, anchors_wh).
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
            
        if 'best_score' in ckpt:
            self.best_score = ckpt['best_score']

        anchors_wh = ckpt.get('anchors_wh', None)
        resume_epoch = ckpt['epoch'] + 1
        if self.verbose:
            print(f"[ModelCheckpoint] Resumed from epoch {ckpt['epoch'] + 1}, starting at epoch {resume_epoch + 1}")
            if 'best_score' in ckpt:
                print(f"[ModelCheckpoint] Restored previous best score: {self.best_score:.4f}")
        return resume_epoch, anchors_wh
