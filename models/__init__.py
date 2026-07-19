"""
models — shared model definitions for the Object Detection & Classification pipeline.

Usage
-----
# Named size preset (recommended)
from models import get_detector, get_classifier

detector   = get_detector(size="m", max_objects=24)
classifier = get_classifier(num_classes=62)

# Direct class import (still works)
from models import ObjectDetectorResNet, CharacterClassifierResNet
"""

import torch

from .object_detector_res      import ObjectDetectorResNet
from .character_classifier_resnet import CharacterClassifierResNet

__all__ = [
    "ObjectDetectorResNet",
    "CharacterClassifierResNet",
    "get_detector",
    "get_classifier",
    "load_detector",
    "load_classifier",
]


# ---------------------------------------------------------------------------
# Getter functions
# ---------------------------------------------------------------------------

def get_detector(size: str = "s", max_objects: int = 24, **kwargs) -> ObjectDetectorResNet:
    """Return an ObjectDetectorResNet for the given size preset.

    Size presets
    ------------
    n  — nano   : channels [32,  64,  64,  128]  ~1.1M params
    s  — small  : channels [64,  128, 128, 256]  ~3.0M params  (default / v1)
    m  — medium : channels [128, 256, 256, 512]  ~9.2M params

    Any kwarg accepted by ObjectDetectorResNet can override the preset, e.g.::

        get_detector("n", max_objects=32)
        get_detector("s", channels=[64, 64, 128, 128])   # custom channels

    Args:
        size:        One of ``"n"``, ``"s"``, ``"m"``. Case-insensitive.
        max_objects: Maximum predicted slots. Default 24.
        **kwargs:    Forwarded to ObjectDetectorResNet.
                     Notable: ``channels=[...]`` to override preset widths,
                     ``pool_size=1`` or ``pool_size=2`` (default) to control
                     the AdaptiveAvgPool2d size before the FC head.
    """
    size = size.lower()
    if size not in ObjectDetectorResNet.CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(ObjectDetectorResNet.CONFIGS)}")

    _, preset_channels = ObjectDetectorResNet.CONFIGS[size]
    channels = kwargs.pop("channels", preset_channels)   # caller can override

    return ObjectDetectorResNet(max_objects=max_objects, channels=channels, **kwargs)


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

def load_detector(path: str, device, size: str = "m", max_objects: int = 24, **kwargs) -> ObjectDetectorResNet:
    """Instantiate and load ObjectDetectorResNet weights from *path*.

    Args:
        path:        Path to a ``.pth`` state-dict file.
        device:      ``torch.device`` to map weights onto.
        size:        Size preset used when the model was trained. Default ``"m"``.
        max_objects: Must match the value used during training.
        **kwargs:    Forwarded to ``get_detector``.
    """
    model = get_detector(size=size, max_objects=max_objects, **kwargs).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_classifier(path: str, device, num_classes: int = 62, **kwargs) -> CharacterClassifierResNet:
    """Instantiate and load CharacterClassifierResNet weights from *path*.

    Args:
        path:        Path to a ``.pth`` state-dict file.
        device:      ``torch.device`` to map weights onto.
        num_classes: Must match the value used during training.
        **kwargs:    Forwarded to ``get_classifier``.
    """
    model = get_classifier(num_classes=num_classes, **kwargs).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
