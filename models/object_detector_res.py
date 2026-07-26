"""
ResNet Object Detector Architecture.

Provides the Stage 1 ObjectDetectorResNet model for multi-object bounding box localization
and confidence prediction (N=24 fixed output slots).

Supported Presets:
  - Nano  ('n'): 0.57M parameters (565,784)
  - Small ('s'): 2.23M parameters (2,227,128) [Recommended]
  - Medium('m'): 8.84M parameters (8,836,856) [Best Accuracy]
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


class ObjectDetectorResNet(nn.Module):
    """
    ResNet-style Object Detector for multi-object localization and presence scoring.

    Outputs (B, max_objects, 5) tensor containing [x, y, w, h, confidence_logit] for each slot.

    Args:
        max_objects (int): Maximum number of predicted slots per scene. Default: 24.
        channels (list[int] | None): Channel width list for the 4 residual layers.
        pool_size (int): Adaptive average pooling grid size before FC head. Default: 2 (2x2 grid).
    """

    # Named size presets: (stem_out, [layer1..4 channels])
    CONFIGS = {
        "n": (32, [32,  64,  64,  128]),   # Nano   ~0.57M params
        "s": (64, [64,  128, 128, 256]),   # Small  ~2.23M params (default)
        "m": (64, [128, 256, 256, 512]),   # Medium ~8.84M params
    }

    def __init__(self, max_objects: int = 24, channels: list[int] | None = None, pool_size: int = 2):
        super().__init__()
        if channels is None:
            channels = self.CONFIGS["s"][1]
            stem_out = self.CONFIGS["s"][0]
        else:
            stem_out = channels[0]

        assert len(channels) == 4, "channels must have exactly 4 entries (one per res-layer)"

        self.max_objects = max_objects

        # Stem
        self.conv1 = nn.Conv2d(1, stem_out, kernel_size=3, stride=2, bias=False)
        self.bn1   = nn.BatchNorm2d(stem_out)

        # Residual layers
        c = channels
        self.layer1 = SimpleResBlock(stem_out,  c[0], stride=2)
        self.layer2 = SimpleResBlock(c[0],      c[1], stride=2)
        self.layer3 = SimpleResBlock(c[1],      c[2], stride=2)
        self.layer4 = SimpleResBlock(c[2],      c[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Shared FC head
        shared_dim = c[3] * 2
        head_dim   = shared_dim // 2
        fc_in      = c[3] * pool_size * pool_size
        self.shared_fc = nn.Sequential(
            nn.Linear(fc_in, shared_dim),
            nn.ReLU(inplace=True),
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(inplace=True),
        )

        # Branch heads
        self.box_head = nn.Linear(head_dim, max_objects * 4)   # [x, y, w, h]
        self.obj_head = nn.Linear(head_dim, max_objects * 1)   # confidence logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for object detection.

        Args:
            x (torch.Tensor): Input grayscale image tensor of shape (B, 1, 224, 224).

        Returns:
            torch.Tensor: Tensor of shape (B, max_objects, 5) where last dim is [x, y, w, h, logit].
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        shared = self.shared_fc(x)

        boxes = self.box_head(shared).view(-1, self.max_objects, 4)
        confs = self.obj_head(shared).view(-1, self.max_objects, 1)

        return torch.cat([boxes, confs], dim=-1)


if __name__ == "__main__":
    for size in ("n", "s", "m"):
        stem, ch = ObjectDetectorResNet.CONFIGS[size]
        m = ObjectDetectorResNet(channels=ch)
        n = sum(p.numel() for p in m.parameters())
        print(f"  {size}  channels={ch}  params={n:,}")
