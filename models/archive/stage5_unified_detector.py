"""
Archived: Stage 5 single-stage unified detector (FC tri-head), August 2026.

Originally `ObjectDetectorResNet` in models/object_detector_res.py from commit 78d0485 to the
`stage5-single-stage` tag (edee5a1), replaced by the Stage 6 grid detector in 7a3a1d4. Boxes,
objectness and classes come from three FC heads on a globally pooled feature, so spatial detail is
lost before the heads (see the Stage 5 benchmark: 36% classification accuracy on Nano).

Checkpoints: weights/s_detector_{n,s,m,l}_best.pth, checkpoint/s_detector_{n,s,m,l}_best.pth,
checkpoint/s_detector_{s,m,l}_latest.pth, checkpoint/s_detector_s_new_best.pth.

    python -m models.archive.stage5_unified_detector
"""
import torch
from torch import nn
from ..common import ResNetDetectorBase
from ..configs import GRID_PRESETS, resolve_backbone_args

PRESETS = GRID_PRESETS  # n / s / m / l, the same widths and depths as the grid detectors


class Stage5UnifiedDetector(ResNetDetectorBase):
    """
    ResNet-style Object Detector for multi-object localization, presence scoring and classification.

    Outputs (B, max_objects, 5 + num_classes): [x, y, w, h, objectness_logit, class_logits...] per slot.

    Args:
        max_objects (int): Maximum number of predicted slots per scene. Default: 24.
        channels (list[int] | None): Channel width list for the 4 residual layers; None uses the 's' preset.
        pool_size (int): Adaptive average pooling grid size before FC head. Default: 2 (2x2 grid).
        num_classes (int): Number of class logits per slot. Default: 47.
        blocks (list[int] | None): Number of residual blocks in each layer.
    """

    INPUT_SHAPE = (1, 224, 224)  # grayscale canvas

    def __init__(
            self, max_objects: int = 24, channels: list[int] | None = None,
            pool_size: int = 2, num_classes: int = 47, blocks: list[int] | None = None
    ):
        channels, blocks = resolve_backbone_args(PRESETS, channels, blocks)
        super().__init__(channels, blocks, strides=(2, 2, 2, 2), stem_padding=0)
        self.max_objects = max_objects
        self.num_classes = num_classes

        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Shared FC trunk: C3 * pool^2 -> 2 * C3 -> C3
        c = channels
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
        self.obj_head = nn.Linear(head_dim, max_objects * 1)   # objectness logit
        self.class_head = nn.Linear(head_dim, max_objects * num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_stem(x)    # (B, C0, 111, 111)
        x = self.layer1(x)           # (B, C0, 56, 56)
        x = self.layer2(x)           # (B, C1, 28, 28)
        x = self.layer3(x)           # (B, C2, 14, 14)
        x = self.layer4(x)           # (B, C3, 7, 7)
        x = self.avgpool(x)          # (B, C3, pool, pool)
        x = torch.flatten(x, 1)

        shared = self.shared_fc(x)

        boxes = self.box_head(shared).view(-1, self.max_objects, 4)
        objectness = self.obj_head(shared).view(-1, self.max_objects, 1)
        classes = self.class_head(shared).view(-1, self.max_objects, self.num_classes)

        return torch.cat([boxes, objectness, classes], dim=-1)


if __name__ == "__main__":
    from models.summary import print_summary

    print_summary("archive.stage5")
