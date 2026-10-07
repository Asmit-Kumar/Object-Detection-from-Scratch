"""
Archived: Stage 4 (two-stage pipeline) box detector with a shared FC trunk, July 2026.

Originally `ObjectDetectorResNet` in models/object_detector_res.py from commit 2292f24 to the
`stage4-two-stage` tag (9625739). It predicts 24 box slots plus a confidence per slot; the crops are
then classified by CharacterClassifierResNet.

Checkpoints: weights/detector_{n,s,m}{,_new_best}.pth (the DetectionPipeline defaults),
checkpoint/detector_{n,s,m}_new_best.pth, checkpoint/detector_{nano,small,medium}_best.pth,
checkpoint/detector_n_v2_best.pth, checkpoint/detector_S_V2_best.pth, checkpoint/detector_m_best.pth.

    python -m models.archive.stage4_box_detector
"""
import torch
from torch import nn
from models.common import ResNetDetectorBase
from models.configs import GRID_PRESETS, resolve_backbone_args

PRESETS = {size: GRID_PRESETS[size] for size in ("n", "s", "m")}


class Stage4BoxDetector(ResNetDetectorBase):
    """
    ResNet-style Object Detector for multi-object localization and presence scoring.

    Outputs (B, max_objects, 5): [x, y, w, h, confidence_logit] for each slot.

    Args:
        max_objects (int): Maximum number of predicted slots per scene. Default: 24.
        channels (list[int] | None): Channel width list for the 4 residual layers; None uses the 's' preset.
        pool_size (int): Adaptive average pooling grid size before FC head. Default: 2 (2x2 grid).
        blocks (list[int] | None): Residual blocks per res-layer. Default: one each (as originally).
    """

    INPUT_SHAPE = (1, 224, 224)  # grayscale canvas

    def __init__(
            self, max_objects: int = 24, channels: list[int] | None = None, pool_size: int = 2,
            blocks: list[int] | None = None,
    ):
        channels, blocks = resolve_backbone_args(PRESETS, channels, blocks)
        super().__init__(channels, blocks, strides=(2, 2, 2, 2), stem_padding=0)
        self.max_objects = max_objects

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
        self.obj_head = nn.Linear(head_dim, max_objects * 1)   # confidence logit

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
        confs = self.obj_head(shared).view(-1, self.max_objects, 1)

        return torch.cat([boxes, confs], dim=-1)


if __name__ == "__main__":
    from models.summary import print_summary

    print_summary("archive.stage4")
