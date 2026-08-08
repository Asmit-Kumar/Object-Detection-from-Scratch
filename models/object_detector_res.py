"""
ResNet Object Detector Architecture.

Provides the Stage 5 Single-Stage Unified ObjectDetectorResNet model for simultaneous
multi-object bounding box localization, objectness scoring, and class recognition
using a tri-head design (N=24 fixed output slots).

Supported Presets:
  - Nano  ('n'): 0.71M parameters (711,296)
  - Small ('s'): 2.52M parameters (2,517,024) [Recommended]
  - Medium('m'): 9.42M parameters (9,415,520)
  - Large ('l'): 19.99M parameters (19,989,088) [Best Accuracy]
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
    Single-Stage Unified ResNet Detector for multi-object localization, objectness scoring,
    and character classification in a single forward pass.

    Outputs (B, max_objects, 5 + num_classes) tensor containing box coordinates [x, y, w, h],
    objectness logit [obj], and class logits [cls_0 ... cls_N] for each output slot.

    Args:
        max_objects (int): Maximum number of predicted slots per scene. Default: 24.
        channels (list[int] | None): Channel width list for the 4 residual layers.
        pool_size (int): Adaptive average pooling grid size before FC head. Default: 2 (2x2 grid).
        num_classes (int): Number of class logits per slot. Default: 47.
        blocks (list[int] | None): Number of residual blocks in each layer.
    """

    # Named size presets: (stem_out, [layer1..4 channels], [block1..4], default_pool_size)
    CONFIGS = {
        "n": (32, [32,  64,  64,  128], [1,1,1,1], 2),   # Nano   ~0.71M params
        "s": (64, [64,  128, 128, 256], [1,1,1,1], 2),   # Small  ~2.52M params (default)
        "m": (64, [128, 256, 256, 512], [1,1,1,1], 2),   # Medium ~9.42M params
        "l": (64, [128, 256, 384, 512], [2,2,2,2], 2),   # Large  ~19.99M params
    }

    def __init__(
            self, max_objects: int = 24, channels: list[int] | None = None,
            pool_size: int = 2, num_classes: int = 47, blocks: list[int] | None = None
    ):
        super().__init__()
        if channels is None:
            stem_out, channels, default_blocks, _ = self.CONFIGS["s"]
            if blocks is None:
                blocks = default_blocks
        else:
            stem_out = channels[0]
            if blocks is None:
                blocks = [1, 1, 1, 1]

        assert len(channels) == 4, "channels must have exactly 4 entries (one per res-layer)"
        assert len(blocks) == 4, "blocks must have exactly 4 entries (one per res-layer)"
        assert all(n >= 1 for n in blocks), "each res-layer must contain at least one block"

        self.max_objects = max_objects
        self.num_classes = num_classes

        # Stem
        self.conv1 = nn.Conv2d(1, stem_out, kernel_size=3, stride=2, bias=False)
        self.bn1   = nn.BatchNorm2d(stem_out)

        # Residual layers
        c = channels
        self.in_channels = stem_out
        self.layer1 = self._make_layer(SimpleResBlock, c[0], blocks[0], stride=2)
        self.layer2 = self._make_layer(SimpleResBlock, c[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(SimpleResBlock, c[2], blocks[2], stride=2)
        self.layer4 = self._make_layer(SimpleResBlock, c[3], blocks[3], stride=2)

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
        self.class_head = nn.Linear(head_dim, max_objects * num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for object detection.

        Args:
            x (torch.Tensor): Input grayscale image tensor of shape (B, 1, 224, 224).

        Returns:
            torch.Tensor: Tensor of shape (B, max_objects, 5 + num_classes) where the last
                dimension is [x, y, w, h, objectness_logit, class_logits...].
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
        objectness = self.obj_head(shared).view(-1, self.max_objects, 1)

        classes = self.class_head(shared).view(-1, self.max_objects, self.num_classes)

        return torch.cat([boxes, objectness, classes], dim=-1)

    def _make_layer(self, block, planes, blocks, stride=1):
        """Build one residual stage from the configured block count."""
        layers = [block(self.in_channels, planes, stride=stride)]
        self.in_channels = planes

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes, stride=1))

        return layers[0] if blocks == 1 else nn.Sequential(*layers)


if __name__ == "__main__":
    for size in ("n", "s", "m", "l"):
        stem, ch, blocks, pool = ObjectDetectorResNet.CONFIGS[size]
        m = ObjectDetectorResNet(channels=ch, blocks=blocks, pool_size=pool)
        n = sum(p.numel() for p in m.parameters())
        print(f"  {size}  channels={ch}  params={n:,}   blocks={blocks}  pool={pool}")
