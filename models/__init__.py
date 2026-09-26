"""
models — shared model definitions for the Object Detection & Classification pipeline.

Usage
-----
# Named size preset (recommended)
from models import get_detector, get_multiscale_detector, get_fcos_detector, get_classifier

detector      = get_detector(size="s")
fpn_detector  = get_multiscale_detector(size="s")
fcos_detector = get_fcos_detector(size="s")
classifier    = get_classifier(num_classes=62)

# Direct class import (still works)
from models import ObjectDetectorResNet, MultiScaleObjectDetectorResNet, FCOSObjectDetectorResNet, CharacterClassifierResNet
"""

import torch

from .object_detector_res import (
    FCOSObjectDetectorResNet,
    MultiScaleObjectDetectorResNet,
    ObjectDetectorResNet,
)
from .character_classifier_resnet import CharacterClassifierResNet
from utils.dataset import S

__all__ = [
    "ObjectDetectorResNet",
    "MultiScaleObjectDetectorResNet",
    "FCOSObjectDetectorResNet",
    "CharacterClassifierResNet",
    "get_detector",
    "get_multiscale_detector",
    "get_fcos_detector",
    "get_classifier",
    "load_detector",
    "load_multiscale_detector",
    "load_fcos_detector",
    "load_classifier",
]
S = S

# ---------------------------------------------------------------------------
# Getter functions
# ---------------------------------------------------------------------------

def get_detector(size: str = "s", **kwargs) -> ObjectDetectorResNet:
    """Return an ObjectDetectorResNet for the given size preset.

    Size presets
    ------------
    n  — nano   : channels [32,  64,  64,  128]  ~0.39M params
    s  — small  : channels [64,  128, 128, 256]  ~1.55M params (default / recommended)
    m  — medium : channels [128, 256, 256, 512]  ~6.18M params
    l  — large  : channels [128, 256, 384, 512]  ~16.75M params
    """
    size = size.lower()
    if size not in ObjectDetectorResNet.CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(ObjectDetectorResNet.CONFIGS)}")

    _, preset_channels, preset_blocks, _ = ObjectDetectorResNet.CONFIGS[size]
    channels = kwargs.pop("channels", preset_channels)   # caller can override
    blocks = kwargs.pop("blocks", preset_blocks)         # caller can override
    kwargs.pop("pool_size", None)

    return ObjectDetectorResNet(channels=channels, blocks=blocks, **kwargs)


def get_multiscale_detector(size: str = "s", **kwargs) -> MultiScaleObjectDetectorResNet:
    """Return the two-scale FPN detector for the requested size preset."""
    size = size.lower()
    if size not in MultiScaleObjectDetectorResNet.CONFIGS:
        raise ValueError(
            f"Unknown size '{size}'. Choose from: {list(MultiScaleObjectDetectorResNet.CONFIGS)}"
        )

    _, preset_channels, preset_blocks, _ = MultiScaleObjectDetectorResNet.CONFIGS[size]
    channels = kwargs.pop("channels", preset_channels)
    blocks = kwargs.pop("blocks", preset_blocks)
    return MultiScaleObjectDetectorResNet(channels=channels, blocks=blocks, **kwargs)


def get_fcos_detector(size: str = "s", **kwargs) -> FCOSObjectDetectorResNet:
    """Return an FCOSObjectDetectorResNet for the requested size preset.

    Size presets
    ------------
    n  — nano   : channels [32,  64,  64,  128], fpn=48  ~0.42M params
    s  — small  : channels [64,  128, 128, 256], fpn=64  ~1.65M params (default)
    m  — medium : channels [128, 256, 256, 512], fpn=128 ~6.5M params
    l  — large  : channels [128, 256, 384, 512], fpn=128 ~17.5M params
    """
    size = size.lower()
    if size not in FCOSObjectDetectorResNet.CONFIGS:
        raise ValueError(
            f"Unknown size '{size}'. Choose from: {list(FCOSObjectDetectorResNet.CONFIGS)}"
        )

    _, preset_channels, preset_blocks, preset_fpn = FCOSObjectDetectorResNet.CONFIGS[size]
    channels = kwargs.pop("channels", preset_channels)
    blocks = kwargs.pop("blocks", preset_blocks)
    fpn_channels = kwargs.pop("fpn_channels", preset_fpn)

    return FCOSObjectDetectorResNet(
        channels=channels,
        blocks=blocks,
        fpn_channels=fpn_channels,
        **kwargs,
    )


def get_classifier(num_classes: int = 62, **kwargs) -> CharacterClassifierResNet:
    """Return a CharacterClassifierResNet.

    Args:
        num_classes: Number of output classes. Default 62 (0-9, A-Z, a-z).
        **kwargs:    Forwarded to CharacterClassifierResNet.
    """
    return CharacterClassifierResNet(num_classes=num_classes, **kwargs)


# ---------------------------------------------------------------------------
# Weight loaders (convenience — load weights from disk)
# ---------------------------------------------------------------------------

def load_detector(path: str, device, size: str = "m", **kwargs) -> ObjectDetectorResNet:
    """Instantiate and load ObjectDetectorResNet weights from *path*."""
    model = get_detector(size=size, **kwargs).to(device)
    raw = torch.load(path, map_location=device, weights_only=False)
    state_dict = raw.get('model_state_dict', raw) if isinstance(raw, dict) else raw
    if isinstance(raw, dict) and 'anchors_wh' in raw:
        model.anchors_wh = raw['anchors_wh']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_multiscale_detector(path: str, device, size: str = "s", **kwargs) -> MultiScaleObjectDetectorResNet:
    """Instantiate and load MultiScaleObjectDetectorResNet weights from *path*."""
    model = get_multiscale_detector(size=size, **kwargs).to(device)
    raw = torch.load(path, map_location=device, weights_only=False)
    state_dict = raw.get('model_state_dict', raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_fcos_detector(path: str, device, size: str = "s", **kwargs) -> FCOSObjectDetectorResNet:
    """Instantiate and load FCOSObjectDetectorResNet weights from *path*."""
    model = get_fcos_detector(size=size, **kwargs).to(device)
    raw = torch.load(path, map_location=device, weights_only=False)
    state_dict = raw.get('model_state_dict', raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_classifier(path: str, device, **kwargs) -> CharacterClassifierResNet:
    """Instantiate and load CharacterClassifierResNet weights from *path*."""
    model = get_classifier(**kwargs).to(device)
    raw = torch.load(path, map_location=device, weights_only=False)
    state_dict = raw.get('model_state_dict', raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
