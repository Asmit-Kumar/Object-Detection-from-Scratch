"""
Single-scale grid detector (Stage 6): one 28x28 prediction grid with K anchor slots per cell.
"""
import torch
from torch import nn
from .common import ResNetDetectorBase
from .configs import GRID_PRESETS, K, S, resolve_backbone_args


class ObjectDetectorResNet(ResNetDetectorBase):
    """
    Single-Stage Unified ResNet Detector for multi-object localization, objectness scoring,
    and character classification in a single forward pass.

    Outputs (B, 28, 28, K, 5 + num_classes) spatial tensor containing box coordinates [x, y, w, h],
    objectness logit [obj], and class logits [cls_0 ... cls_N] for each spatial cell and anchor slot.
    """

    # Legacy (stem, channels, blocks, grid_size) tuples; the presets live in models/configs.py.
    CONFIGS = {size: (p.stem, list(p.channels), list(p.blocks), S) for size, p in GRID_PRESETS.items()}

    def __init__(
            self, channels: list[int] | None = None,
            num_classes: int = 47, blocks: list[int] | None = None,
            num_anchors: int = K,
    ):
        channels, blocks = resolve_backbone_args(GRID_PRESETS, channels, blocks)
        super().__init__(channels, blocks, strides=(2, 2, 1, 1), stem_padding=0)

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.grid_head = nn.Conv2d(channels[3], num_anchors * (num_classes + 5), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.grid_head(x)
        x = x.permute(0, 2, 3, 1)
        B, H, W, _ = x.shape
        x = x.reshape(B, H, W, self.num_anchors, 5 + self.num_classes)
        return x
