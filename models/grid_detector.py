"""
Single-scale grid detector (Stage 6/7 design, 28x28 grid): every cell of one 28x28 prediction grid
predicts K anchor slots of [x, y, w, h, objectness, class logits].

    python -m models.grid_detector     # per-size parameters and output shapes
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

    Args:
        channels (list[int] | None): Output channels of the four res-layers; None uses the 's' preset.
        num_classes (int): Class logits per slot. Default: 47 (EMNIST ByMerge).
        blocks (list[int] | None): Residual blocks per res-layer. Default: one each.
        num_anchors (int): Anchor slots per cell (K). Default: dataio.dataset.K.
    """

    INPUT_SHAPE = (1, 224, 224)  # grayscale canvas

    # Legacy (stem, channels, blocks, grid_size) tuples; the presets live in models/configs.py.
    CONFIGS = {size: (p.stem, list(p.channels), list(p.blocks), S) for size, p in GRID_PRESETS.items()}

    def __init__(
            self, channels: list[int] | None = None,
            num_classes: int = 47, blocks: list[int] | None = None,
            num_anchors: int = K,
    ):
        channels, blocks = resolve_backbone_args(GRID_PRESETS, channels, blocks)
        # layer3/layer4 keep stride 1 so the grid stays at 28x28 (224 / 8).
        super().__init__(channels, blocks, strides=(2, 2, 1, 1), stem_padding=0)

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # One 1x1 conv predicts every slot of every cell: K * (4 box + 1 objectness + classes).
        self.grid_head = nn.Conv2d(channels[3], num_anchors * (num_classes + 5), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_stem(x)      # (B, C0, 111, 111)
        x = self.layer1(x)             # (B, C0, 56, 56)
        x = self.layer2(x)             # (B, C1, 28, 28)
        x = self.layer3(x)             # (B, C2, 28, 28)
        x = self.layer4(x)             # (B, C3, 28, 28)
        x = self.grid_head(x)          # (B, K * (5 + num_classes), 28, 28)
        x = x.permute(0, 2, 3, 1)
        B, H, W, _ = x.shape
        x = x.reshape(B, H, W, self.num_anchors, 5 + self.num_classes)
        return x


if __name__ == "__main__":
    from models.summary import print_summary

    print_summary("grid")
