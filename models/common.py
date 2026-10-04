"""
Building blocks shared by the detector architectures.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SimpleResBlock(nn.Module):
    """
    Standard ResNet residual block with 3x3 convolutions, BatchNorm, and shortcut projection.

    Args:
        in_channels (int): Input feature channels.
        out_channels (int): Output feature channels.
        stride (int): Downsampling stride for the first convolution. Default: 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying residual connection and ReLU activation."""
        out  = F.relu(self.bn1(self.conv1(x)))
        out  = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ScaleExp(nn.Module):
    """Learnable scale multiplier with exp activation to enforce positive distances."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x * self.scale)


class ResNetDetectorBase(nn.Module):
    """
    Stride-2 stem plus four residual stages shared by every detector.

    Subclasses call ``super().__init__`` and then add their necks and heads. The attribute names
    (conv1, bn1, layer1..layer4) are the state_dict keys of every saved checkpoint, so they must
    stay at the top level of the model and must not be renamed.

    Args:
        channels (list[int]): Output channels of the four stages; the stem width is channels[0].
        blocks (list[int]): Residual blocks per stage.
        strides (tuple[int, ...]): Stride of the first block in each stage.
        stem_padding (int): Padding of the 3x3 stem convolution.
    """

    def __init__(
            self,
            channels: list[int],
            blocks: list[int],
            strides: tuple[int, int, int, int],
            stem_padding: int = 0,
    ):
        super().__init__()
        if len(channels) != 4 or len(blocks) != 4:
            raise ValueError("channels and blocks must each contain four entries (one per res-layer)")
        if any(n < 1 for n in blocks):
            raise ValueError("each res-layer must contain at least one block")

        stem_out = channels[0]
        self.conv1 = nn.Conv2d(1, stem_out, kernel_size=3, stride=2, padding=stem_padding, bias=False)
        self.bn1   = nn.BatchNorm2d(stem_out)

        self.in_channels = stem_out
        self.layer1 = self._make_layer(SimpleResBlock, channels[0], blocks[0], stride=strides[0])
        self.layer2 = self._make_layer(SimpleResBlock, channels[1], blocks[1], stride=strides[1])
        self.layer3 = self._make_layer(SimpleResBlock, channels[2], blocks[2], stride=strides[2])
        self.layer4 = self._make_layer(SimpleResBlock, channels[3], blocks[3], stride=strides[3])

    def _forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn1(self.conv1(x)))

    def _make_layer(self, block, planes, blocks, stride=1):
        # A single-block stage is stored bare (keys 'layerN.conv1...', not 'layerN.0.conv1...').
        layers = [block(self.in_channels, planes, stride=stride)]
        self.in_channels = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes, stride=1))
        return layers[0] if blocks == 1 else nn.Sequential(*layers)
