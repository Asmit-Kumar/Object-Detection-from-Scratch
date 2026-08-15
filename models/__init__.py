"""
models — shared model definitions for the Object Detection & Classification pipeline.

Usage
-----
# Named size preset (recommended)
from models import get_detector, get_classifier

detector   = get_detector(size="s")
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
S = 14

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

    Any kwarg accepted by ObjectDetectorResNet can override the preset, e.g.::

        get_detector("n")
        get_detector("s", channels=[64, 64, 128, 128])   # custom channels

    Args:
        size:        One of ``"n"``, ``"s"``, ``"m"``, ``"l"``. Case-insensitive.
        **kwargs:    Forwarded to ObjectDetectorResNet.
                     Notable: ``channels=[...]`` to override preset widths.
    """
    size = size.lower()
    if size not in ObjectDetectorResNet.CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(ObjectDetectorResNet.CONFIGS)}")

    _, preset_channels, preset_blocks, _ = ObjectDetectorResNet.CONFIGS[size]
    channels = kwargs.pop("channels", preset_channels)   # caller can override
    blocks = kwargs.pop("blocks", preset_blocks)         # caller can override
    kwargs.pop("pool_size", None)

    return ObjectDetectorResNet(channels=channels, blocks=blocks, **kwargs)


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
    """Instantiate and load ObjectDetectorResNet weights from *path*.

    Args:
        path:        Path to a ``.pth`` state-dict file.
        device:      ``torch.device`` to map weights onto.
        size:        Size preset used when the model was trained ("n", "s", "m", "l"). Default ``"m"``.
        **kwargs:    Forwarded to ``get_detector``.
    """
    model = get_detector(size=size, **kwargs).to(device)
    raw = torch.load(path, map_location=device, weights_only=False)
    state_dict = raw.get('model_state_dict', raw) if isinstance(raw, dict) else raw
    if isinstance(raw, dict) and 'anchors_wh' in raw:
        model.anchors_wh = raw['anchors_wh']
    model.load_state_dict(state_dict, strict=False)
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
