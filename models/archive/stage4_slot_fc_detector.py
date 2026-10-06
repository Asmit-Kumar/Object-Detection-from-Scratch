"""
Archived: Stage 4 (two-stage pipeline), first multi-object box detector, July 2026.

Originally `ObjectDetectorResNet` in models/object_detector_res.py at commit 161ffb4, replaced in
2292f24 by the shared-FC version (stage4_box_detector.py). It predicts a fixed set of 24 box slots
from one flattened 7x7 feature map; classes come from the separate character classifier.
The original hard-coded the 's' widths; 'n' is the nano width its checkpoints were trained with.

Checkpoints: weights/detector_{n,s}_best.pth, checkpoint/detector_{n,s}_best.pth,
checkpoint/detector_s_{1,11,123,1sfg}_best.pth, checkpoint/detector_resnet_s{,_1,_11}_best.pth.

    python -m models.archive.stage4_slot_fc_detector
"""
import torch
from torch import nn
from ..common import ResNetDetectorBase
from ..configs import GRID_PRESETS, resolve_backbone_args

PRESETS = {size: GRID_PRESETS[size] for size in ("n", "s")}


class Stage4SlotDetectorFC(ResNetDetectorBase):
    """
    Multi-object box detector with a single FC head over a 7x7 pooled map.

    Outputs (B, max_objects, 5): [x, y, w, h, confidence_logit] per slot.

    Args:
        channels (list[int] | None): Output channels of the four res-layers; None uses the 's' preset.
        blocks (list[int] | None): Residual blocks per res-layer. Default: one each.
        max_objects (int): Predicted slots per image. Default: 24.
    """

    INPUT_SHAPE = (1, 224, 224)  # grayscale canvas

    def __init__(self, channels: list[int] | None = None, blocks: list[int] | None = None, max_objects: int = 24):
        channels, blocks = resolve_backbone_args(PRESETS, channels, blocks)
        super().__init__(channels, blocks, strides=(2, 2, 2, 2), stem_padding=0)
        self.max_objects = max_objects

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(channels[3] * 7 * 7, max_objects * 5)   # all slots from one linear layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_stem(x)    # (B, C0, 111, 111)
        x = self.layer1(x)           # (B, C0, 56, 56)
        x = self.layer2(x)           # (B, C1, 28, 28)
        x = self.layer3(x)           # (B, C2, 14, 14)
        x = self.layer4(x)           # (B, C3, 7, 7)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(-1, self.max_objects, 5)

        return x


if __name__ == "__main__":
    from models.summary import print_summary

    print_summary("archive.stage4_fc")
