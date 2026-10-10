"""
Size presets (n / s / m / l) for the detector architectures.

This is the single place the models package reads dataset-level constants from.
"""
from dataclasses import dataclass

from dataio.dataset import K, S, YOLO_ANCHORS_PER_SCALE, YOLO_GRID_SIZES

DEFAULT_SIZE = "s"


@dataclass(frozen=True)
class DetectorPreset:
    """
    Backbone widths and depths for one size preset.

    Args:
        stem (int): Stem width as listed in the legacy CONFIGS tuples. The models build the stem
            with channels[0], so this value is informational only.
        channels (tuple[int, ...]): Output channels of the four residual stages.
        blocks (tuple[int, ...]): Residual blocks per stage.
        fpn_channels (int | None): FPN width (FCOS only).
    """

    stem: int
    channels: tuple[int, int, int, int]
    blocks: tuple[int, int, int, int]
    fpn_channels: int | None = None


# Shared by ObjectDetectorResNet and MultiScaleObjectDetectorResNet.
GRID_PRESETS = {
    "n": DetectorPreset(32, (32,  64,  64,  128), (1, 1, 1, 1)),   # Nano   ~0.39M params
    "s": DetectorPreset(64, (64,  128, 128, 256), (1, 1, 1, 1)),   # Small  ~1.55M params (default)
    "m": DetectorPreset(64, (128, 256, 256, 512), (1, 1, 1, 1)),   # Medium ~6.18M params
    "l": DetectorPreset(64, (128, 256, 384, 512), (2, 2, 2, 2)),   # Large  ~16.75M params
}

FCOS_PRESETS = {
    "n": DetectorPreset(32, (32,  64,  64,  128), (1, 1, 1, 1), fpn_channels=48),    # Nano   ~0.42M params
    "s": DetectorPreset(64, (64,  128, 128, 256), (1, 1, 1, 1), fpn_channels=64),    # Small  ~1.65M params (default)
    "m": DetectorPreset(64, (128, 256, 256, 512), (1, 1, 1, 1), fpn_channels=128),   # Medium ~6.5M params
    "l": DetectorPreset(64, (128, 256, 384, 512), (2, 2, 2, 2), fpn_channels=128),   # Large  ~17.5M params
}

# YOLOv8-inspired backbone and PAN neck widths/depths. These presets are
# deliberately separate from GRID_PRESETS because their C2f layout is distinct.
YOLOV8_PRESETS = {
    "n": DetectorPreset(16, (16,  32,  64,  128), (1, 1, 1, 1)),
    "s": DetectorPreset(32, (32,  64,  128, 256), (1, 1, 2, 1)),
    "m": DetectorPreset(48, (48,  96,  192, 384), (1, 2, 4, 2)),
    "l": DetectorPreset(64, (64, 128, 256, 512), (2, 3, 6, 3)),
}


def get_preset(presets: dict[str, DetectorPreset], size: str) -> DetectorPreset:
    """Look up a size preset, case-insensitively."""
    size = size.lower()
    if size not in presets:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(presets)}")
    return presets[size]


def resolve_backbone_args(
        presets: dict[str, DetectorPreset],
        channels: list[int] | None,
        blocks: list[int] | None,
) -> tuple[list[int], list[int]]:
    """Apply the constructor defaults: the 's' preset when channels is None, else one block per stage."""
    if channels is None:
        preset = presets[DEFAULT_SIZE]
        return list(preset.channels), list(preset.blocks) if blocks is None else blocks
    return channels, [1, 1, 1, 1] if blocks is None else blocks
