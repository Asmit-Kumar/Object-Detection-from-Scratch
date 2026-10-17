"""
YOLOv8-inspired two-scale anchor-free detector.

The architecture uses YOLOv8-style Conv-BN-SiLU blocks, C2f stages, an SPPF
context block, a PAN neck, and decoupled box / class branches. It predicts
per-location [l, t, r, b] distances at 28x28 and 14x14 without anchor boxes or an
objectness branch, so it trains with the shared ltrb pipeline (LTRBTargetGenerator
and LTRBLoss in dataio/ltrb_targets.py and training/losses.py, also used by FCOS).

    python -m models.yolov8_detector     # per-size parameters and output shapes
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.common import ScaleExp
from models.configs import (
    YOLOV8_PRESETS,
    YOLO_GRID_SIZES,
    resolve_backbone_args, DetectorPreset,
)
from models.yolov8_detector import YOLOv8ObjectDetector


class ConvBNAct(nn.Module):
    """YOLOv8-style convolution followed by BatchNorm and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Two-convolution residual bottleneck used inside C2f blocks."""

    def __init__(self, channels: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_channels = max(1, int(channels * expansion))
        self.conv1 = ConvBNAct(channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBNAct(hidden_channels, channels, kernel_size=3)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """YOLOv8 C2f block: split features, cascade bottlenecks, then fuse."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            repeats: int,
            shortcut: bool = True,
            expansion: float = 0.5,
    ):
        super().__init__()
        if repeats < 1:
            raise ValueError("C2f repeats must be positive")
        hidden_channels = max(1, int(out_channels * expansion))
        self.conv1 = ConvBNAct(in_channels, 2 * hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            Bottleneck(hidden_channels, shortcut=shortcut, expansion=1.0)
            for _ in range(repeats)
        )
        self.conv2 = ConvBNAct((2 + repeats) * hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = list(self.conv1(x).chunk(2, dim=1))
        for block in self.blocks:
            features.append(block(features[-1]))
        return self.conv2(torch.cat(features, dim=1))


class SPPF(nn.Module):
    """Fast spatial-pyramid pooling block used at the deepest feature level."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = max(1, in_channels // 2)
        self.conv1 = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = ConvBNAct(hidden_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.conv2(torch.cat((x, y1, y2, self.pool(y2)), dim=1))


class DecoupledDetectionHead(nn.Module):
    """Separate box and class branches for one feature scale (anchor-free, no objectness)."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        branch_channels = max(16, min(in_channels, 64))
        self.num_classes = num_classes

        self.box = nn.Sequential(
            ConvBNAct(in_channels, branch_channels),
            ConvBNAct(branch_channels, branch_channels),
            nn.Conv2d(branch_channels, 4, kernel_size=1),
        )
        self.classes = nn.Sequential(
            ConvBNAct(in_channels, branch_channels),
            ConvBNAct(branch_channels, branch_channels),
            nn.Conv2d(branch_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        boxes = self.box(x).permute(0, 2, 3, 1).contiguous()  # (B, H, W, 4) raw
        classes = self.classes(x).permute(0, 2, 3, 1).contiguous()  # (B, H, W, C) logits
        return boxes, classes


def make_aux_preset(
    main: DetectorPreset,
    width_ratio: float = 0.5,
    depth_ratio: float = 0.5,
) -> DetectorPreset:
    def scale_channels(c: int) -> int:
        return max(1, int(c * width_ratio + 0.5))

    def scale_depth(d: int) -> int:
        return max(1, int(d * depth_ratio + 0.5))

    return DetectorPreset(
        stem=scale_channels(main.stem),
        channels=tuple(scale_channels(c) for c in main.channels),
        blocks=tuple(scale_depth(d) for d in main.blocks),
    )


class AuxiliaryDetectionBranch(nn.Module):
    def __init__(self, main: DetectorPreset, auxiliary: DetectorPreset, num_classes: int):
        super().__init__()
        c3, c4 = main.channels[2:]
        a3, a4 = auxiliary.channels[2:]
        self.project_p3 = ConvBNAct(c3, a3, kernel_size=1)
        self.project_p4 = ConvBNAct(c4, a4, kernel_size=1)
        self.reduce_p4 = ConvBNAct(a4, a3, kernel_size=1)
        self.top_down = C2f(2 * a3, a3, repeats=auxiliary.blocks[2], shortcut=False)
        self.down_p3 = ConvBNAct(a3, a4, kernel_size=3, stride=2)
        self.bottom_up = C2f(2 * a4, a4, repeats=auxiliary.blocks[3], shortcut=False)
        self.head_p3 = DecoupledDetectionHead(a3, num_classes)
        self.head_p4 = DecoupledDetectionHead(a4, num_classes)
        self.scale_p3 = ScaleExp(init_value=1.0)
        self.scale_p4 = ScaleExp(init_value=1.0)

    def forward(
            self, p3_backbone: torch.Tensor, p4_backbone: torch.Tensor,
    ) -> dict[int, dict[str, torch.Tensor]]:
        p3_backbone = self.project_p3(p3_backbone)
        p4_backbone = self.project_p4(p4_backbone)
        p3 = self.top_down(torch.cat((
            p3_backbone,
            F.interpolate(self.reduce_p4(p4_backbone), size=p3_backbone.shape[-2:], mode="nearest"),
        ), dim=1))
        p4 = self.bottom_up(torch.cat((p4_backbone, self.down_p3(p3)), dim=1))
        box_p3, cls_p3 = self.head_p3(p3)
        box_p4, cls_p4 = self.head_p4(p4)
        return {
            28: {
                "cls_logits": cls_p3,
                "reg_ltrb": self.scale_p3(box_p3) * YOLOv8ObjectDetector.STRIDES[28],
            },
            14: {
                "cls_logits": cls_p4,
                "reg_ltrb": self.scale_p4(box_p4) * YOLOv8ObjectDetector.STRIDES[14],
            },
        }


class AuxiliaryObjectDetector(nn.Module):
    """
    YOLOv8-inspired anchor-free detector with 28x28 and 14x14 prediction heads.

    Outputs per scale {28: {...}, 14: {...}}, the same ltrb contract as FCOS minus centerness:
        - 'cls_logits': (B, H, W, num_classes)
        - 'reg_ltrb':   (B, H, W, 4) in canvas pixel distances [l, t, r, b] from the cell centre

    There is no objectness / centerness branch (as in YOLOv8), so the shared loss and decoder
    score a location by its class probability alone. ``STRIDES`` maps grid size to stride.

    Args:
        channels: Four backbone channel widths; None selects the ``s`` preset.
        num_classes: Number of class logits per grid cell.
        blocks: C2f repeat counts for the four backbone stages.
        grid_sizes: Must be the repository's two-scale 28x28 / 14x14 contract.
    """

    INPUT_SHAPE = (1, 224, 224)
    CONFIGS = {
        size: (preset.stem, list(preset.channels), list(preset.blocks))
        for size, preset in YOLOV8_PRESETS.items()
    }

    STRIDES = {28: 8.0, 14: 16.0}

    def __init__(
            self,
            channels: list[int] | None = None,
            num_classes: int = 47,
            blocks: list[int] | None = None,
            grid_sizes: tuple[int, int] = YOLO_GRID_SIZES,
    ):
        channels, blocks = resolve_backbone_args(YOLOV8_PRESETS, channels, blocks)
        if len(channels) != 4 or len(blocks) != 4:
            raise ValueError("channels and blocks must each contain four entries")
        if any(width < 1 for width in channels) or any(repeats < 1 for repeats in blocks):
            raise ValueError("channels and blocks must contain positive values")
        if tuple(grid_sizes) != YOLO_GRID_SIZES:
            raise ValueError(
                f"grid_sizes must be {YOLO_GRID_SIZES} for the current 224px backbone"
            )

        super().__init__()
        c1, c2, c3, c4 = channels
        self.num_classes = num_classes
        self.grid_sizes = tuple(grid_sizes)
        self.main_preset = DetectorPreset(c1, tuple(channels), tuple(blocks))

        # Backbone: 224 -> 112 -> 56 -> 28 (P3) -> 14 (P4).
        self.stem = ConvBNAct(1, c1, kernel_size=3, stride=2)
        self.stage1 = C2f(c1, c1, repeats=blocks[0])
        self.down1 = ConvBNAct(c1, c2, kernel_size=3, stride=2)
        self.stage2 = C2f(c2, c2, repeats=blocks[1])
        self.down2 = ConvBNAct(c2, c3, kernel_size=3, stride=2)
        self.stage3 = C2f(c3, c3, repeats=blocks[2])
        self.down3 = ConvBNAct(c3, c4, kernel_size=3, stride=2)
        self.stage4 = C2f(c4, c4, repeats=blocks[3])
        self.sppf = SPPF(c4, c4)

        # PAN neck: top-down P4 -> P3 fusion followed by bottom-up P3 -> P4 fusion.
        self.reduce_p4 = ConvBNAct(c4, c3, kernel_size=1)
        self.top_down = C2f(2 * c3, c3, repeats=blocks[2], shortcut=False)
        self.down_p3 = ConvBNAct(c3, c4, kernel_size=3, stride=2)
        self.bottom_up = C2f(2 * c4, c4, repeats=blocks[3], shortcut=False)

        self.head_p3 = DecoupledDetectionHead(c3, num_classes)
        self.head_p4 = DecoupledDetectionHead(c4, num_classes)

        # Learnable scale multiplier per pyramid level (P3: 28x28, P4: 14x14), as in FCOS.
        self.scale_p3 = ScaleExp(init_value=1.0)
        self.scale_p4 = ScaleExp(init_value=1.0)

        self.auxiliary_network = self._build_auxiliary_network(self.main_preset)
        self._configure_batch_norm()
        self._init_heads()

    def _configure_batch_norm(self) -> None:
        """Use the BatchNorm hyperparameters used by the YOLO family."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03

    def _build_auxiliary_network(self, preset: DetectorPreset) -> AuxiliaryDetectionBranch:
        return AuxiliaryDetectionBranch(
            preset, make_aux_preset(preset), self.num_classes,
        )

    def _init_heads(self) -> None:
        """Small head weights and a focal prior (p = 0.01) on the class bias, as in FCOS / RetinaNet."""
        cls_bias_init = -math.log((1.0 - 0.01) / 0.01)
        for head in (
                self.head_p3, self.head_p4,
                self.auxiliary_network.head_p3, self.auxiliary_network.head_p4,
        ):
            nn.init.normal_(head.box[-1].weight, std=0.01)
            nn.init.constant_(head.box[-1].bias, 0.0)
            nn.init.normal_(head.classes[-1].weight, std=0.01)
            nn.init.constant_(head.classes[-1].bias, cls_bias_init)

    def forward(
            self, x: torch.Tensor,
    ) -> dict[int, dict[str, torch.Tensor]] | dict[str, dict[int, dict[str, torch.Tensor]]]:
        x = self.stage1(self.stem(x))
        x = self.stage2(self.down1(x))
        p3_backbone = self.stage3(self.down2(x))
        p4_backbone = self.sppf(self.stage4(self.down3(p3_backbone)))

        p3 = self.top_down(torch.cat((
            p3_backbone,
            F.interpolate(self.reduce_p4(p4_backbone), size=p3_backbone.shape[-2:], mode="nearest"),
        ), dim=1))
        p4 = self.bottom_up(torch.cat((p4_backbone, self.down_p3(p3)), dim=1))

        box_p3, cls_p3 = self.head_p3(p3)
        box_p4, cls_p4 = self.head_p4(p4)

        main = {
            28: {
                "cls_logits": cls_p3,
                "reg_ltrb": self.scale_p3(box_p3) * self.STRIDES[28],
            },
            14: {
                "cls_logits": cls_p4,
                "reg_ltrb": self.scale_p4(box_p4) * self.STRIDES[14],
            },
        }
        if self.training:
            return {"main": main, "aux": self.auxiliary_network(p3_backbone, p4_backbone)}
        return main

    def to_inference(self) -> YOLOv8ObjectDetector:
        main = YOLOv8ObjectDetector(
            channels=list(self.main_preset.channels),
            blocks=list(self.main_preset.blocks),
            num_classes=self.num_classes,
            grid_sizes=self.grid_sizes,
        )
        reference = next(self.parameters())
        main.to(device=reference.device, dtype=reference.dtype)
        main.load_state_dict({
            key: value for key, value in self.state_dict().items()
            if not key.startswith("auxiliary_network.")
        })
        return main.eval()


if __name__ == "__main__":
    from models.summary import print_summary

    print_summary("auxiliary")
