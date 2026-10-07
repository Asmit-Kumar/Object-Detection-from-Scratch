"""
ResNet Object Detector Architecture.

Provides:
  1. Single-Stage Unified ObjectDetectorResNet (K-slot anchor grid detector)
  2. MultiScaleObjectDetectorResNet (Two-scale FPN grid detector)
  3. YOLOv8ObjectDetector (YOLOv8-inspired two-scale PAN grid detector)
  4. FCOSObjectDetectorResNet (Anchor-Free Fully Convolutional One-Stage Detector with FPN,
     centerness branch, and decoupled per-pixel regression towers)

Supported Presets:
  - Nano  ('n'): ~0.39M - 0.45M params
  - Small ('s'): ~1.55M - 1.82M params [Recommended]
  - Medium('m'): ~6.18M - 6.50M params
  - Large ('l'): ~16.75M - 17.5M params
"""
from models.common import ResNetDetectorBase, ScaleExp, SimpleResBlock  # noqa: F401  (re-exported for old imports)
from models.fcos_detector import FCOSObjectDetectorResNet
from models.grid_detector import ObjectDetectorResNet
from models.multiscale_detector import MultiScaleObjectDetectorResNet
from models.yolov8_detector import YOLOv8ObjectDetector

# The detectors now live in grid_detector.py, multiscale_detector.py, yolov8_detector.py and fcos_detector.py;
# this module re-exports them so `models.object_detector_res` imports keep working.


if __name__ == "__main__":
    from models.summary import print_summary

    for name in ("grid", "multiscale", "yolov8", "fcos"):
        print_summary(name)
