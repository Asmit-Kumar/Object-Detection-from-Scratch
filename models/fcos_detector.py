"""
FCOS anchor-free detector (Stage 8): FPN over 28x28 / 14x14, decoupled towers, centerness branch
and per-pixel ltrb distances.

    python -m models.fcos_detector     # per-size parameters and output shapes
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import ResNetDetectorBase, ScaleExp
from .configs import FCOS_PRESETS, resolve_backbone_args


class FCOSObjectDetectorResNet(ResNetDetectorBase):
    """
    Anchor-Free Fully Convolutional ResNet Detector (FCOS) with Feature Pyramid Network (FPN),
    decoupled classification/regression towers, centerness branch, and learnable scale factors.

    Outputs per scale {28: {...}, 14: {...}}:
        - 'cls_logits':        (B, H, W, num_classes)
        - 'reg_ltrb':          (B, H, W, 4) in canvas pixel distances [l, t, r, b]
        - 'centerness_logits': (B, H, W)

    Args:
        channels (list[int] | None): Output channels of the four res-layers; None uses the 's' preset.
        num_classes (int): Class logits per location. Default: 47 (EMNIST ByMerge).
        blocks (list[int] | None): Residual blocks per res-layer. Default: one each.
        fpn_channels (int | None): FPN / tower width. Default: the 's' preset's 64 when channels is
            None, otherwise channels[2].
        head_convs (int): 3x3 conv + GroupNorm + ReLU layers in each tower. Default: 2.
    """

    INPUT_SHAPE = (1, 224, 224)  # grayscale canvas

    # Legacy (stem, channels, blocks, fpn_channels) tuples; the presets live in models/configs.py.
    CONFIGS = {size: (p.stem, list(p.channels), list(p.blocks), p.fpn_channels) for size, p in FCOS_PRESETS.items()}

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
        if fpn_channels is None:
            fpn_channels = FCOS_PRESETS["s"].fpn_channels if channels is None else channels[2]
        channels, blocks = resolve_backbone_args(FCOS_PRESETS, channels, blocks)

        # ── ResNet Backbone ───────────────────────────────────────────────────
        # layer1 56x56, layer2 28x28 (C3), layer3 14x14 (C4), layer4 14x14 (deep C4)
        super().__init__(channels, blocks, strides=(2, 2, 2, 1), stem_padding=1)

        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.grid_sizes = self.GRID_SIZES

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
        x = self._forward_stem(x)
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
    from models.summary import print_summary

    print_summary("fcos")
