"""
Character Classifier Network Architecture.

Provides a compact ResNet-style CNN model (CharacterClassifierResNet) for 28x28 grayscale
EMNIST character recognition across 47 classes (ByMerge split) or 62 classes (ByClass split).

Total Parameters: 370,351 (~0.37M)
"""
import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions, BatchNorm, ReLU, and shortcut projection.

    Args:
        in_channels (int): Input feature map channel depth.
        out_channels (int): Output feature map channel depth.
        stride (int): Stride for downsampling in the first convolution. Default: 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying residual connection and ReLU activation."""
        return F.relu(self.conv(x) + self.skip(x))


class CharacterClassifierResNet(nn.Module):
    """
    Compact ResNet-based Character Classifier for 28x28 grayscale character crops.

    Architecture:
      - Stem: 1 -> 32 channels, 3x3 Conv + BN + ReLU
      - Block 1: 32 -> 64 channels (stride 2 downsampling to 14x14)
      - Block 2: 64 -> 64 channels (stride 2 downsampling to 7x7)
      - Block 3: 64 -> 64 channels (residual processing)
      - Classifier Head: AdaptiveAvgPool2d(3x3) -> Flatten -> Linear(576, 256) -> ReLU -> Dropout(0.5) -> Linear(256, num_classes)

    Args:
        num_classes (int): Number of target output classes (47 for EMNIST ByMerge, 62 for ByClass). Default: 62.
    """

    def __init__(self, num_classes: int = 62):
        super(CharacterClassifierResNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block1 = BasicBlock(32, 64, stride=2)
        self.block2 = BasicBlock(64, 64, stride=2)
        self.block3 = BasicBlock(64, 64)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for character classification.

        Args:
            x (torch.Tensor): Grayscale image tensor of shape (B, 1, 28, 28).

        Returns:
            torch.Tensor: Unnormalized class logits of shape (B, num_classes).
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = CharacterClassifierResNet(num_classes=47)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
