"""
Shared model definitions for the Object Detection & Digit Recognition pipeline.
Import these instead of re-defining the architectures in every script.
"""

import torch
from torch import nn


class ObjectDetector(nn.Module):
    """
    Lightweight BBox Regressor (~700k params).
    4-stage CNN backbone + 1x1 bottleneck + FC head.
    Outputs normalised [x_min, y_min, x_max, y_max] in [0, 1].
    """
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 32, 1), nn.ReLU())
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.bottleneck(x)
        return self.head(x)


class DigitClassifier(nn.Module):
    """
    Ultra-light Digit Classifier (~130k params).
    3-stage CNN backbone compressing to 3x3 spatial resolution before head.
    Outputs raw logits for 10 digit classes.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def load_bbox_model(path, device):
    """Load ObjectDetector weights from path onto device."""
    model = ObjectDetector().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_digit_model(path, device):
    """Load DigitClassifier weights from path onto device."""
    model = DigitClassifier().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
