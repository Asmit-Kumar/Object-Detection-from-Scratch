"""
models — shared model definitions for the Object Detection & Classification pipeline.

Usage
-----
# One factory for every architecture (recommended)
from models import build_model, load_model

detector   = build_model("fcos", size="s")                  # "grid" | "multiscale" | "fcos" | "classifier"
detector   = build_model("grid", size="m", num_anchors=3)   # keyword arguments override the preset
classifier = build_model("classifier", num_classes=47)
detector   = load_model("fcos", "weights/fcos_s_28x14_best.pth", device, size="s")

# Per-architecture helpers (still work)
from models import get_detector, get_multiscale_detector, get_fcos_detector, get_classifier

# Direct class import (still works)
from models import ObjectDetectorResNet, MultiScaleObjectDetectorResNet, FCOSObjectDetectorResNet, CharacterClassifierResNet

Layout
------
configs.py                     size presets (n / s / m / l) and dataset-level defaults
common.py                      SimpleResBlock, ScaleExp, ResNetDetectorBase (stem + four res-layers)
grid_detector.py               ObjectDetectorResNet           ("grid")
multiscale_detector.py         MultiScaleObjectDetectorResNet ("multiscale")
fcos_detector.py               FCOSObjectDetectorResNet       ("fcos")
character_classifier_resnet.py the character classifier (own BasicBlock, kept separate on purpose:
                               its layers differ from SimpleResBlock and its checkpoints depend on them)
object_detector_res.py         re-exports the three detectors for older imports
archive/                       earlier architectures restored from git history ("archive.*" names),
                               still able to load their checkpoints; see models/archive/__init__.py
summary.py                     print_summary(name): per-size parameters and output shapes

Every model file prints its summary when run: python -m models.fcos_detector

Adding a model: put it in its own file (reuse common.py), add presets to configs.py if it has
sizes, register it in _REGISTRY below, and pin its state_dict signature in tests/test_models.py.
Retiring a model: move its file to archive/ and rename its registry entry to "archive.<name>".
"""
import warnings

import torch
from torch import nn

from models.configs import FCOS_PRESETS, GRID_PRESETS, get_preset
from models.grid_detector import ObjectDetectorResNet
from models.multiscale_detector import MultiScaleObjectDetectorResNet
from models.fcos_detector import FCOSObjectDetectorResNet
from models.character_classifier_resnet import CharacterClassifierResNet
from models.archive import (
    DigitClassifierCNN,
    Grid14Detector,
    SingleBoxCNN,
    SingleBoxResNet,
    Stage4BoxDetector,
    Stage4SlotDetectorFC,
    Stage5UnifiedDetector,
)
from models.archive import grid14_detector, stage4_box_detector, stage4_slot_fc_detector, stage5_unified_detector

__all__ = [
    "ObjectDetectorResNet",
    "MultiScaleObjectDetectorResNet",
    "FCOSObjectDetectorResNet",
    "CharacterClassifierResNet",
    "SingleBoxCNN",
    "SingleBoxResNet",
    "DigitClassifierCNN",
    "Stage4SlotDetectorFC",
    "Stage4BoxDetector",
    "Stage5UnifiedDetector",
    "Grid14Detector",
    "MODEL_NAMES",
    "build_model",
    "load_model",
    "get_detector",
    "get_multiscale_detector",
    "get_fcos_detector",
    "get_classifier",
    "load_detector",
    "load_multiscale_detector",
    "load_fcos_detector",
    "load_classifier",
]

# name -> (class, size presets). Single-size models (the classifiers) have no presets.
_REGISTRY = {
    "grid": (ObjectDetectorResNet, GRID_PRESETS),
    "multiscale": (MultiScaleObjectDetectorResNet, GRID_PRESETS),
    "fcos": (FCOSObjectDetectorResNet, FCOS_PRESETS),
    "classifier": (CharacterClassifierResNet, None),
    # Earlier architectures (models/archive/), kept so their checkpoints still load.
    "archive.single_box_cnn": (SingleBoxCNN, None),
    "archive.single_box_resnet": (SingleBoxResNet, None),
    "archive.digit_cnn": (DigitClassifierCNN, None),
    "archive.stage4_fc": (Stage4SlotDetectorFC, stage4_slot_fc_detector.PRESETS),
    "archive.stage4": (Stage4BoxDetector, stage4_box_detector.PRESETS),
    "archive.stage5": (Stage5UnifiedDetector, stage5_unified_detector.PRESETS),
    "archive.grid14": (Grid14Detector, grid14_detector.PRESETS),
}
MODEL_NAMES = tuple(_REGISTRY)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(name: str, size: str = "s", **overrides) -> nn.Module:
    """Return an untrained model for the given architecture and size preset.

Args:
        name:        One of MODEL_NAMES: "grid", "multiscale", "fcos", "classifier", or an
                     archived architecture such as "archive.stage4" (see models/archive/).
        size:        Size preset "n" | "s" | "m" | "l" (ignored for single-size models).
        **overrides: Constructor arguments; they take precedence over the preset
                     (e.g. channels=[...], blocks=[...], fpn_channels=..., num_classes=...).
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_REGISTRY)}")
    cls, presets = _REGISTRY[name]
    if presets is None:
        return cls(**overrides)

    preset = get_preset(presets, size)
    kwargs = {"channels": list(preset.channels), "blocks": list(preset.blocks)}
    if preset.fpn_channels is not None:
        kwargs["fpn_channels"] = preset.fpn_channels
    kwargs.update(overrides)
    return cls(**kwargs)


def _extract_state_dict(checkpoint) -> dict:
    """Return the weights from a bare state_dict, a best-model file or a full training checkpoint."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_model(
        name: str,
        path: str,
        device,
        size: str = "s",
        strict: bool = True,
        **overrides,
) -> nn.Module:
    """Build a model with build_model(), load weights from *path* and switch it to eval mode.

    With strict=True a checkpoint that does not match the architecture raises. With strict=False
    the mismatch is reported as a warning and the unmatched layers keep their random init.
    """
    model = build_model(name, size=size, **overrides).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "anchors_wh" in checkpoint:
        model.anchors_wh = checkpoint["anchors_wh"]

    result = model.load_state_dict(_extract_state_dict(checkpoint), strict=strict)
    if result.missing_keys or result.unexpected_keys:
        label = name if _REGISTRY[name][1] is None else f"{name}/{size}"
        warnings.warn(
            f"'{path}' does not match {label}: "
            f"{len(result.missing_keys)} missing keys {result.missing_keys[:5]}, "
            f"{len(result.unexpected_keys)} unexpected keys {result.unexpected_keys[:5]}. "
            "The missing layers keep their random initialization.",
            stacklevel=2,
        )
    model.eval()
    return model


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
    kwargs.pop("pool_size", None)
    return build_model("grid", size=size, **kwargs)


def get_multiscale_detector(size: str = "s", **kwargs) -> MultiScaleObjectDetectorResNet:
    """Return the two-scale FPN detector for the requested size preset."""
    return build_model("multiscale", size=size, **kwargs)


def get_fcos_detector(size: str = "s", **kwargs) -> FCOSObjectDetectorResNet:
    """Return an FCOSObjectDetectorResNet for the requested size preset.

    Size presets
    ------------
    n  — nano   : channels [32,  64,  64,  128], fpn=48  ~0.42M params
    s  — small  : channels [64,  128, 128, 256], fpn=64  ~1.65M params (default)
    m  — medium : channels [128, 256, 256, 512], fpn=128 ~6.5M params
    l  — large  : channels [128, 256, 384, 512], fpn=128 ~17.5M params
    """
    return build_model("fcos", size=size, **kwargs)


def get_classifier(num_classes: int = 62, **kwargs) -> CharacterClassifierResNet:
    """Return a CharacterClassifierResNet.

    Args:
        num_classes: Number of output classes. Default 62 (0-9, A-Z, a-z).
        **kwargs:    Forwarded to CharacterClassifierResNet.
    """
    return build_model("classifier", num_classes=num_classes, **kwargs)


# ---------------------------------------------------------------------------
# Weight loaders (convenience — load weights from disk)
#
# These keep their original non-strict loading so existing callers behave the same;
# a checkpoint/architecture mismatch now emits a warning instead of passing silently.
# ---------------------------------------------------------------------------

def load_detector(path: str, device, size: str = "m", **kwargs) -> ObjectDetectorResNet:
    """Instantiate and load ObjectDetectorResNet weights from *path*."""
    kwargs.pop("pool_size", None)
    return load_model("grid", path, device, size=size, strict=False, **kwargs)


def load_multiscale_detector(path: str, device, size: str = "s", **kwargs) -> MultiScaleObjectDetectorResNet:
    """Instantiate and load MultiScaleObjectDetectorResNet weights from *path*."""
    return load_model("multiscale", path, device, size=size, strict=False, **kwargs)


def load_fcos_detector(path: str, device, size: str = "s", **kwargs) -> FCOSObjectDetectorResNet:
    """Instantiate and load FCOSObjectDetectorResNet weights from *path*."""
    return load_model("fcos", path, device, size=size, strict=False, **kwargs)


def load_classifier(path: str, device, num_classes: int = 62, **kwargs) -> CharacterClassifierResNet:
    """Instantiate and load CharacterClassifierResNet weights from *path*."""
    return load_model("classifier", path, device, strict=False, num_classes=num_classes, **kwargs)
