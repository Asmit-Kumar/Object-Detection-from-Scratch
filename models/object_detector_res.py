"""
ResNet Object Detector Architecture.

Provides:
  1. Single-Stage Unified ObjectDetectorResNet (K-slot anchor grid detector)
  2. MultiScaleObjectDetectorResNet (Two-scale FPN grid detector)
  3. FCOSObjectDetectorResNet (Anchor-Free Fully Convolutional One-Stage Detector with FPN,
     centerness branch, and decoupled per-pixel regression towers)

Supported Presets:
  - Nano  ('n'): ~0.39M - 0.45M params
  - Small ('s'): ~1.55M - 1.82M params [Recommended]
  - Medium('m'): ~6.18M - 6.50M params
  - Large ('l'): ~16.75M - 17.5M params
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from utils.dataset import K, S, YOLO_ANCHORS_PER_SCALE, YOLO_GRID_SIZES


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

    Outputs (B, 28, 28, K, 5 + num_classes) spatial tensor containing box coordinates [x, y, w, h],
    objectness logit [obj], and class logits [cls_0 ... cls_N] for each spatial cell and anchor slot.
    """

    CONFIGS = {
        "n": (32, [32,  64,  64,  128], [1, 1, 1, 1], S),   # Nano   ~0.39M params
        "s": (64, [64,  128, 128, 256], [1, 1, 1, 1], S),   # Small  ~1.55M params (default)
        "m": (64, [128, 256, 256, 512], [1, 1, 1, 1], S),   # Medium ~6.18M params
        "l": (64, [128, 256, 384, 512], [2, 2, 2, 2], S),   # Large  ~16.75M params
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
        self.layer3 = self._make_layer(SimpleResBlock, c[2], blocks[2], stride=1)
        self.layer4 = self._make_layer(SimpleResBlock, c[3], blocks[3], stride=1)

        self.grid_head = nn.Conv2d(c[3], num_anchors * (num_classes + 5), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        layers = [block(self.in_channels, planes, stride=stride)]
        self.in_channels = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes, stride=1))
        return layers[0] if blocks == 1 else nn.Sequential(*layers)


class MultiScaleObjectDetectorResNet(nn.Module):
    """YOLO-style detector with two spatial prediction heads."""

    CONFIGS = ObjectDetectorResNet.CONFIGS

    def __init__(
            self,
            channels: list[int] | None = None,
            num_classes: int = 47,
            blocks: list[int] | None = None,
            anchors_per_scale: int = YOLO_ANCHORS_PER_SCALE,
            grid_sizes: tuple[int, int] = YOLO_GRID_SIZES,
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

        if len(channels) != 4 or len(blocks) != 4:
            raise ValueError("channels and blocks must each contain four entries")
        if tuple(grid_sizes) != YOLO_GRID_SIZES:
            raise ValueError(
                f"grid_sizes must be {YOLO_GRID_SIZES} for the current 224px backbone"
            )
        if anchors_per_scale < 1:
            raise ValueError("anchors_per_scale must be positive")

        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        self.grid_sizes = tuple(grid_sizes)
        self.num_anchors = len(self.grid_sizes) * anchors_per_scale

        self.conv1 = nn.Conv2d(1, stem_out, kernel_size=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_out)

        self.in_channels = stem_out
        self.layer1 = self._make_layer(SimpleResBlock, channels[0], blocks[0], stride=2)
        self.layer2 = self._make_layer(SimpleResBlock, channels[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(SimpleResBlock, channels[2], blocks[2], stride=2)
        self.layer4 = self._make_layer(SimpleResBlock, channels[3], blocks[3], stride=1)

        route_channels = channels[2]
        self.lateral_medium = nn.Conv2d(channels[3], route_channels, kernel_size=1, bias=False)
        self.smooth_fine = self._smooth_block(channels[1] + route_channels, route_channels)

        prediction_channels = anchors_per_scale * (num_classes + 5)
        self.head_fine = nn.Conv2d(route_channels, prediction_channels, kernel_size=1)
        self.head_medium = nn.Conv2d(route_channels, prediction_channels, kernel_size=1)

    @staticmethod
    def _smooth_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        fine_features = self.layer2(x)      # 28 x 28
        medium_features = self.layer4(self.layer3(fine_features))  # 14 x 14

        medium = self.lateral_medium(medium_features)
        medium_upsampled = F.interpolate(
            medium, size=fine_features.shape[-2:], mode="nearest"
        )
        fine = self.smooth_fine(torch.cat((fine_features, medium_upsampled), dim=1))

        features_by_scale = {
            28: fine,
            14: medium,
        }
        heads_by_scale = {
            28: self.head_fine,
            14: self.head_medium,
        }
        return {
            grid_size: self._reshape_head(heads_by_scale[grid_size](features_by_scale[grid_size]))
            for grid_size in self.grid_sizes
        }

    def _reshape_head(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x = x.permute(0, 2, 3, 1)
        return x.reshape(
            batch_size,
            height,
            width,
            self.anchors_per_scale,
            5 + self.num_classes,
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(self.in_channels, planes, stride=stride)]
        self.in_channels = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes, stride=1))
        return layers[0] if blocks == 1 else nn.Sequential(*layers)


class ScaleExp(nn.Module):
    """Learnable scale multiplier with exp activation to enforce positive distances."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x * self.scale)


class FCOSObjectDetectorResNet(nn.Module):
    """
    Anchor-Free Fully Convolutional ResNet Detector (FCOS) with Feature Pyramid Network (FPN),
    decoupled classification/regression towers, centerness branch, and learnable scale factors.

    Outputs per scale {28: {...}, 14: {...}}:
        - 'cls_logits':        (B, H, W, num_classes)
        - 'reg_ltrb':          (B, H, W, 4) in canvas pixel distances [l, t, r, b]
        - 'centerness_logits': (B, H, W)
    """

    CONFIGS = {
        "n": (32, [32,  64,  64,  128], [1, 1, 1, 1], 48),    # Nano   ~0.42M params
        "s": (64, [64,  128, 128, 256], [1, 1, 1, 1], 64),    # Small  ~1.65M params (default)
        "m": (64, [128, 256, 256, 512], [1, 1, 1, 1], 128),   # Medium ~6.5M params
        "l": (64, [128, 256, 384, 512], [2, 2, 2, 2], 128),   # Large  ~17.5M params
    }

    GRID_SIZES = (28, 14)
    STRIDES = {28: 8.0, 14: 16.0}

    def __init__(
            self,
            channels: list[int] | None = None,
            num_classes: int = 47,
            blocks: list[int] | None = None,
            fpn_channels: int | None = None,
            head_convs: int = 2,
    ):
        super().__init__()

        if channels is None:
            stem_out, channels, default_blocks, default_fpn = self.CONFIGS["s"]
            if blocks is None:
                blocks = default_blocks
            if fpn_channels is None:
                fpn_channels = default_fpn
        else:
            stem_out = channels[0]
            if blocks is None:
                blocks = [1, 1, 1, 1]
            if fpn_channels is None:
                fpn_channels = channels[2]

        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.grid_sizes = self.GRID_SIZES

        # ── ResNet Backbone ───────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(1, stem_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(stem_out)

        self.in_channels = stem_out
        self.layer1 = self._make_layer(SimpleResBlock, channels[0], blocks[0], stride=2)  # 56x56
        self.layer2 = self._make_layer(SimpleResBlock, channels[1], blocks[1], stride=2)  # 28x28 (C3)
        self.layer3 = self._make_layer(SimpleResBlock, channels[2], blocks[2], stride=2)  # 14x14 (C4)
        self.layer4 = self._make_layer(SimpleResBlock, channels[3], blocks[3], stride=1)  # 14x14 (deep C4)

        # ── FPN Lateral & Smoothing Layers ───────────────────────────────────
        self.lateral4 = nn.Conv2d(channels[3], fpn_channels, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(channels[1], fpn_channels, kernel_size=1, bias=False)

        self.smooth3 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(16, fpn_channels), num_channels=fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth4 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(16, fpn_channels), num_channels=fpn_channels),
            nn.ReLU(inplace=True),
        )

        # ── Decoupled Shared FCOS Towers ──────────────────────────────────────
        cls_tower_layers = []
        reg_tower_layers = []
        for _ in range(head_convs):
            cls_tower_layers.extend([
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(16, fpn_channels), num_channels=fpn_channels),
                nn.ReLU(inplace=True),
            ])
            reg_tower_layers.extend([
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(16, fpn_channels), num_channels=fpn_channels),
                nn.ReLU(inplace=True),
            ])

        self.cls_tower = nn.Sequential(*cls_tower_layers)
        self.reg_tower = nn.Sequential(*reg_tower_layers)

        # ── Prediction Heads ──────────────────────────────────────────────────
        self.cls_head = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(fpn_channels, 4, kernel_size=1)
        self.centerness_head = nn.Conv2d(fpn_channels, 1, kernel_size=1)

        # Learnable scale multiplier per FPN pyramid level (P3: 28x28, P4: 14x14)
        self.scale_p3 = ScaleExp(init_value=1.0)
        self.scale_p4 = ScaleExp(init_value=1.0)

        self._init_weights()

    def _init_weights(self):
        """Standard FCOS / RetinaNet initialization with focal prior bias."""
        for m in [self.cls_tower, self.reg_tower, self.smooth3, self.smooth4]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

        for layer in [self.lateral3, self.lateral4]:
            nn.init.kaiming_uniform_(layer.weight, a=1)

        nn.init.normal_(self.reg_head.weight, std=0.01)
        nn.init.constant_(self.reg_head.bias, 0.0)

        nn.init.normal_(self.centerness_head.weight, std=0.01)
        nn.init.constant_(self.centerness_head.bias, 0.0)

        # Initialize classification bias so initial sigmoid output is ~pi = 0.01
        pi = 0.01
        cls_bias_init = -math.log((1.0 - pi) / pi)
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.constant_(self.cls_head.bias, cls_bias_init)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(self.in_channels, planes, stride=stride)]
        self.in_channels = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes, stride=1))
        return layers[0] if blocks == 1 else nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Grayscale canvas tensor (B, 1, 224, 224).

        Returns:
            dict[int, dict[str, torch.Tensor]] keyed by grid size (28 and 14):
                - 'cls_logits': (B, H, W, num_classes)
                - 'reg_ltrb': (B, H, W, 4) in canvas pixel distance units
                - 'centerness_logits': (B, H, W)
        """
        # Backbone forward
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        c3 = self.layer2(x)                     # (B, C1, 28, 28)
        c4 = self.layer4(self.layer3(c3))       # (B, C3, 14, 14)

        # FPN forward
        p4 = self.smooth4(self.lateral4(c4))    # (B, fpn_channels, 14, 14)
        p3_fused = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3_fused)             # (B, fpn_channels, 28, 28)

        # Level P3 (28x28, stride 8)
        cls_feat_p3 = self.cls_tower(p3)
        reg_feat_p3 = self.reg_tower(p3)

        cls_p3 = self.cls_head(cls_feat_p3).permute(0, 2, 3, 1).contiguous()
        cent_p3 = self.centerness_head(reg_feat_p3).squeeze(1).contiguous()
        raw_reg_p3 = self.reg_head(reg_feat_p3).permute(0, 2, 3, 1).contiguous()
        reg_p3 = self.scale_p3(raw_reg_p3) * self.STRIDES[28]

        # Level P4 (14x14, stride 16)
        cls_feat_p4 = self.cls_tower(p4)
        reg_feat_p4 = self.reg_tower(p4)

        cls_p4 = self.cls_head(cls_feat_p4).permute(0, 2, 3, 1).contiguous()
        cent_p4 = self.centerness_head(reg_feat_p4).squeeze(1).contiguous()
        raw_reg_p4 = self.reg_head(reg_feat_p4).permute(0, 2, 3, 1).contiguous()
        reg_p4 = self.scale_p4(raw_reg_p4) * self.STRIDES[14]

        return {
            28: {
                "cls_logits": cls_p3,
                "reg_ltrb": reg_p3,
                "centerness_logits": cent_p3,
            },
            14: {
                "cls_logits": cls_p4,
                "reg_ltrb": reg_p4,
                "centerness_logits": cent_p4,
            },
        }

    @staticmethod
    def generate_grid_centers(height: int, width: int, stride: float, device: torch.device) -> torch.Tensor:
        """Generate (H, W, 2) grid center coordinates (xc, yc) in canvas pixel space."""
        shift_x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
        shift_y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1)

    def decode_predictions(
        self,
        outputs: dict[int, dict[str, torch.Tensor]],
        clamp_canvas_size: float | None = 224.0,
    ) -> dict[int, torch.Tensor]:
        """
        Decode FCOS distance outputs [l, t, r, b] to [x, y, w, h, centerness_logit, class_logits...]
        matching the repository's standard detection tensor format (B, H, W, 1, 5 + num_classes).
        """
        decoded = {}
        for grid_size, preds in outputs.items():
            stride = self.STRIDES[grid_size]
            reg_ltrb = preds["reg_ltrb"]                      # (B, H, W, 4)
            cls_logits = preds["cls_logits"]                  # (B, H, W, num_classes)
            cent_logits = preds["centerness_logits"].unsqueeze(-1)  # (B, H, W, 1)

            B, H, W, _ = reg_ltrb.shape
            centers = self.generate_grid_centers(H, W, stride, reg_ltrb.device)
            xc = centers[..., 0].unsqueeze(0)
            yc = centers[..., 1].unsqueeze(0)

            l = reg_ltrb[..., 0]
            t = reg_ltrb[..., 1]
            r = reg_ltrb[..., 2]
            b = reg_ltrb[..., 3]

            x1 = xc - l
            y1 = yc - t
            x2 = xc + r
            y2 = yc + b

            if clamp_canvas_size is not None:
                x1 = x1.clamp(0.0, clamp_canvas_size)
                y1 = y1.clamp(0.0, clamp_canvas_size)
                x2 = x2.clamp(0.0, clamp_canvas_size)
                y2 = y2.clamp(0.0, clamp_canvas_size)

            x = x1
            y = y1
            w = (x2 - x1).clamp_min(0.0)
            h = (y2 - y1).clamp_min(0.0)

            boxes = torch.stack((x, y, w, h), dim=-1)
            slot_tensor = torch.cat((boxes, cent_logits, cls_logits), dim=-1).unsqueeze(3)
            decoded[grid_size] = slot_tensor

        return decoded


if __name__ == "__main__":
    print("=" * 80)
    print("ResNet Object Detector - Forward Pass Validation")
    print("=" * 80)

    for size in ("n", "s", "m", "l"):
        print(f"\n{'-' * 80}")
        print(f"Model Size: {size.upper()} (Config: {['Nano', 'Small', 'Medium', 'Large'][['n', 's', 'm', 'l'].index(size)]})")
        print(f"{'-' * 80}")

        stem, ch, blocks, grid_size = ObjectDetectorResNet.CONFIGS[size]
        model = ObjectDetectorResNet(channels=ch, blocks=blocks)
        param_count = sum(p.numel() for p in model.parameters())

        print(f"Architecture:")
        print(f"  Stem output channels:   {stem}")
        print(f"  Layer channels:         {ch}")
        print(f"  Blocks per layer:       {blocks}")
        print(f"  Output grid size:       {grid_size}x{grid_size}")
        print(f"  Total parameters:       {param_count:,}")
        print(f"  Number of classes:      {model.num_classes}")
        print(f"  Number of anchors:      {model.num_anchors}")

        for batch_size in [1, 2, 4]:
            print(f"\n  Batch size: {batch_size}")
            input_tensor = torch.randn(batch_size, 1, 224, 224)
            out = model(input_tensor)
            print(f"  Output shape: {out.shape}")

    print("\n" + "=" * 80)
    print("FCOS Object Detector - Validation")
    print("=" * 80)
    for size in ("n", "s", "m", "l"):
        fcos = FCOSObjectDetectorResNet(channels=ObjectDetectorResNet.CONFIGS[size][1])
        params = sum(p.numel() for p in fcos.parameters())
        print(f"FCOS {size.upper()} parameters: {params:,}")
