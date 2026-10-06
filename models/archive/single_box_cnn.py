"""
Archived: single-object bounding-box regressor, the first PyTorch detector (May 2026).

Originally `ObjectDetector` in the root `models.py` (commit 788cc42, removed in 161ffb4), trained in
`BBox_detector.ipynb`. One box per image, no classes.

Checkpoints: weights/bbox_model_torch.pth, checkpoint/od_checkpoint_light.pth,
checkpoint/best_od_checkpoint_light.pth.

    python -m models.archive.single_box_cnn
"""
import torch
from torch import nn


class SingleBoxCNN(nn.Module):
    """
    Lightweight BBox Regressor (~700k params): 4-stage CNN backbone + 1x1 bottleneck + FC head.

    Outputs (B, 4): normalised [x_min, y_min, x_max, y_max] in [0, 1].
    """

    INPUT_SHAPE = (1, 128, 128)  # four 2x max-pools down to the 8x8 map the head expects

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 32, 1), nn.ReLU())   # 64 -> 32 channels before flattening
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.block1(x)       # (B, 32, 64, 64)
        x = self.block2(x)       # (B, 64, 32, 32)
        x = self.block3(x)       # (B, 128, 16, 16)
        x = self.block4(x)       # (B, 64, 8, 8)
        x = self.bottleneck(x)   # (B, 32, 8, 8)
        return self.head(x)      # (B, 4)


if __name__ == "__main__":
    from models.summary import print_summary

    print_summary("archive.single_box_cnn")
