"""
Archived: Stage 6 grid detector (K=1) and Stage 7 multi-anchor detector (K=3) on a 14x14 grid.

The frozen snapshot of models/object_detector_res.py at commit 2be0b81, before 2ac08da moved the grid
to 28x28 (layer3 stride 2 -> 1). Previously kept as scripts/legacy_models/grid_detector_14x14.py,
which now re-exports this class. Benchmark results for these checkpoints depend on this exact
architecture: tests/test_archive.py pins its state_dict layout.

Checkpoints (14x14 grid, K anchors, anchors_wh stored alongside):
    K=1: weights/{1_,}grid_detector_{n,s,m}_best.pth, checkpoint/{1_,}grid_detector_{n,s,m}_{best,latest}.pth
    K=3: weights/3_grid_detector_{n,s,m}_best.pth, checkpoint/3_grid_detector_{n,s,m}_{best,latest}.pth

    python -m models.archive.grid14_detector
"""
import torch
from torch import nn
from models.common import ResNetDetectorBase
from models.configs import GRID_PRESETS, K, resolve_backbone_args

PRESETS = GRID_PRESETS
GRID_SIZE = 14


class Grid14Detector(ResNetDetectorBase):
    """
    Single-Stage Unified ResNet Detector for multi-object localization, objectness scoring,
    and character classification in a single forward pass.

    Outputs (B, 14, 14, K, 5 + num_classes) spatial tensor containing box coordinates [x, y, w, h],
    objectness logit [obj], and class logits [cls_0 ... cls_N] for each spatial cell and anchor slot.

    Args:
        channels (list[int] | None): Channel width list for the 4 residual layers; None uses the 's' preset.
        num_classes (int): Number of class logits per slot. Default: 47.
        blocks (list[int] | None): Number of residual blocks in each layer.
        num_anchors (int): Number of anchor slots per grid cell (K). Default: dataio.dataset.K.
    """

    INPUT_SHAPE = (1, 224, 224)  # grayscale canvas

    # Legacy (stem, channels, blocks, grid_size) tuples, as scripts/run_single_stage_benchmark.py unpacks them.
    CONFIGS = {size: (p.stem, list(p.channels), list(p.blocks), GRID_SIZE) for size, p in PRESETS.items()}

    def __init__(
            self, channels: list[int] | None = None,
            num_classes: int = 47, blocks: list[int] | None = None,
            num_anchors: int = K,
    ):
        channels, blocks = resolve_backbone_args(PRESETS, channels, blocks)
        # layer3 still downsamples (stride 2), which is what makes the grid 14x14 instead of 28x28.
        super().__init__(channels, blocks, strides=(2, 2, 2, 1), stem_padding=0)

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.grid_head = nn.Conv2d(channels[3], num_anchors * (num_classes + 5), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_stem(x)    # (B, C0, 111, 111)
        x = self.layer1(x)           # (B, C0, 56, 56)
        x = self.layer2(x)           # (B, C1, 28, 28)
        x = self.layer3(x)           # (B, C2, 14, 14)
        x = self.layer4(x)           # (B, C3, 14, 14)
        x = self.grid_head(x)        # (B, K * (5 + num_classes), 14, 14)
        x = x.permute(0, 2, 3, 1)
        B, H, W, _ = x.shape
        x = x.reshape(B, H, W, self.num_anchors, 5 + self.num_classes)

        return x


if __name__ == "__main__":
    from models.summary import print_summary

    print_summary("archive.grid14", num_anchors=1)   # Stage 6
    print_summary("archive.grid14", num_anchors=3)   # Stage 7 (multi-anchor, retired)
