"""
Two-scale grid detector: 28x28 and 14x14 prediction heads joined by a small FPN.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .common import ResNetDetectorBase
from .configs import GRID_PRESETS, YOLO_ANCHORS_PER_SCALE, YOLO_GRID_SIZES, resolve_backbone_args
from .grid_detector import ObjectDetectorResNet


class MultiScaleObjectDetectorResNet(ResNetDetectorBase):
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
        channels, blocks = resolve_backbone_args(GRID_PRESETS, channels, blocks)
        if tuple(grid_sizes) != YOLO_GRID_SIZES:
            raise ValueError(
                f"grid_sizes must be {YOLO_GRID_SIZES} for the current 224px backbone"
            )
        if anchors_per_scale < 1:
            raise ValueError("anchors_per_scale must be positive")
        super().__init__(channels, blocks, strides=(2, 2, 2, 1), stem_padding=0)

        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        self.grid_sizes = tuple(grid_sizes)
        self.num_anchors = len(self.grid_sizes) * anchors_per_scale

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
        x = self._forward_stem(x)
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
