"""
ResNet Object Detector Architecture.

Provides the Stage 6 Single-Stage Unified ObjectDetectorResNet model for simultaneous
multi-object bounding box localization, objectness scoring, and class recognition
using an anchor-free $14 \times 14$ spatial grid head.

Supported Presets:
  - Nano  ('n'): 0.39M parameters (392,788)
  - Small ('s'): 1.55M parameters (1,553,524) [Recommended]
  - Medium('m'): 6.18M parameters (6,178,996)
  - Large ('l'): 16.75M parameters (16,752,564)
"""
import torch
from torch import nn
from torch.nn import functional as F
from utils.dataset import K


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

    Outputs (B, 14, 14, 5 + num_classes) spatial tensor containing box coordinates [x, y, w, h],
    objectness logit [obj], and class logits [cls_0 ... cls_N] for each spatial cell.

    Args:
        channels (list[int] | None): Channel width list for the 4 residual layers.
        num_classes (int): Number of class logits per slot. Default: 47.
        blocks (list[int] | None): Number of residual blocks in each layer.
    """

    # Named size presets: (stem_out, [layer1..4 channels], [block1..4], default_pool_size)
    CONFIGS = {
        "n": (32, [32,  64,  64,  128], [1,1,1,1], 14),   # Nano   ~0.39M params
        "s": (64, [64,  128, 128, 256], [1,1,1,1], 14),   # Small  ~1.55M params (default)
        "m": (64, [128, 256, 256, 512], [1,1,1,1], 14),   # Medium ~6.18M params
        "l": (64, [128, 256, 384, 512], [2,2,2,2], 14),   # Large  ~16.75M params
    }

    def __init__(
            self, channels: list[int] | None = None,
            num_classes: int = 47, blocks: list[int] | None = None,
            num_anchors: int = K,
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

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Stem
        self.conv1 = nn.Conv2d(1, stem_out, kernel_size=3, stride=2, bias=False)
        self.bn1   = nn.BatchNorm2d(stem_out)

        # Residual layers
        c = channels
        self.in_channels = stem_out
        self.layer1 = self._make_layer(SimpleResBlock, c[0], blocks[0], stride=2)
        self.layer2 = self._make_layer(SimpleResBlock, c[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(SimpleResBlock, c[2], blocks[2], stride=2)
        self.layer4 = self._make_layer(SimpleResBlock, c[3], blocks[3], stride=1)

        self.grid_head = nn.Conv2d(c[3], num_anchors * (num_classes + 5), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for object detection.

        Args:
            x (torch.Tensor): Input grayscale image tensor of shape (B, 1, 224, 224).

        Returns:
            torch.Tensor: Tensor of shape (B, 14, 14, K, 5 + num_classes) where the last
                dimension is [x, y, w, h, objectness_logit, class_logits...].
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.grid_head(x)
        x = x.permute(0, 2, 3, 1)
        B, H, W, _ = x.shape
        x = x.reshape(B, H, W, self.num_anchors, 5 + self.num_classes)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        """Build one residual stage from the configured block count."""
        layers = [block(self.in_channels, planes, stride=stride)]
        self.in_channels = planes

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes, stride=1))

        return layers[0] if blocks == 1 else nn.Sequential(*layers)


if __name__ == "__main__":
    print("=" * 80)
    print("ResNet Object Detector - Forward Pass Validation")
    print("=" * 80)
    
    for size in ("n", "s", "m", "l"):
        print(f"\n{'─' * 80}")
        print(f"Model Size: {size.upper()} (Config: {['Nano', 'Small', 'Medium', 'Large'][['n', 's', 'm', 'l'].index(size)]})")
        print(f"{'─' * 80}")
        
        stem, ch, blocks, grid_size = ObjectDetectorResNet.CONFIGS[size]
        model = ObjectDetectorResNet(channels=ch, blocks=blocks)
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"Architecture:")
        print(f"  Stem output channels:   {stem}")
        print(f"  Layer channels:         {ch}")
        print(f"  Blocks per layer:       {blocks}")
        print(f"  Output grid size:       {grid_size}×{grid_size}")
        print(f"  Total parameters:       {param_count:,}")
        print(f"  Number of classes:      {model.num_classes}")
        print(f"  Number of anchors:      {model.num_anchors}")
        
        # Forward pass testing with different batch sizes
        for batch_size in [1, 2, 4]:
            print(f"\n  Batch size: {batch_size}")
            input_tensor = torch.randn(batch_size, 1, 224, 224)
            
            print(f"    Input shape:  {tuple(input_tensor.shape)}")
            output = model(input_tensor)
            print(f"    Output shape: {tuple(output.shape)}")
            
            actual_k = output.shape[3]
            actual_out_dim = output.shape[4]
            print(f"    Anchors (K):  {actual_k}")
            print(f"    Output dim:   {actual_out_dim} (= 5 + {model.num_classes})")
            
            # Validate shape: (B, grid_size, grid_size, K, 5+num_classes)
            expected_shape = (batch_size, grid_size, grid_size, model.num_anchors, 5 + model.num_classes)
            assert output.shape == expected_shape, \
                f"Shape mismatch! Got {tuple(output.shape)}, expected {expected_shape}"
            
            # Validate output value ranges
            assert not torch.isnan(output).any(), "Output contains NaN values!"
            assert not torch.isinf(output).any(), "Output contains Inf values!"
            
            print(f"    ✓ Forward pass validated")
        
        print(f"\n  ✓ All forward pass tests passed for size '{size}'!")
    
    print(f"\n{'=' * 80}")
    print("✓ All models validated successfully!")
    print("=" * 80)
