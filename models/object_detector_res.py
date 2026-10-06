"""
ResNet Object Detector Architecture.

Provides:
  1. Single-Stage Unified ObjectDetectorResNet (K-slot anchor grid detector)
  2. MultiScaleObjectDetectorResNet (Two-scale FPN grid detector)
  3. FCOSObjectDetectorResNet (Anchor-Free Fully Convolutional One-Stage Detector with FPN,
     centerness branch, and decoupled per-pixel regression towers)

Supported Presets:
  - Nano  ('n'): ~0.39M - 0.45M params
  - Small ('s'): ~1.55M - 1.82M params [Recommended]
  - Medium('m'): ~6.18M - 6.50M params
  - Large ('l'): ~16.75M - 17.5M params
"""
import torch
from .common import ResNetDetectorBase, ScaleExp, SimpleResBlock  # noqa: F401  (re-exported for old imports)
from .fcos_detector import FCOSObjectDetectorResNet
from .grid_detector import ObjectDetectorResNet
from .multiscale_detector import MultiScaleObjectDetectorResNet

# The detectors now live in grid_detector.py, multiscale_detector.py and fcos_detector.py;
# this module re-exports them so `models.object_detector_res` imports keep working.


if __name__ == "__main__":
    print("=" * 80)
    print("ResNet Object Detector - Forward Pass Validation")
    print("=" * 80)

    for size in ("n", "s", "m", "l"):
        print(f"\n{'-' * 80}")
        print(f"Model Size: {size.upper()} (Config: {['Nano', 'Small', 'Medium', 'Large'][['n', 's', 'm', 'l'].index(size)]})")
        print(f"{'-' * 80}")

        stem, ch, blocks, grid_size = ObjectDetectorResNet.CONFIGS[size]
        model = ObjectDetectorResNet(channels=ch, blocks=blocks)
        param_count = sum(p.numel() for p in model.parameters())

        print(f"Architecture:")
        print(f"  Stem output channels:   {stem}")
        print(f"  Layer channels:         {ch}")
        print(f"  Blocks per layer:       {blocks}")
        print(f"  Output grid size:       {grid_size}x{grid_size}")
        print(f"  Total parameters:       {param_count:,}")
        print(f"  Number of classes:      {model.num_classes}")
        print(f"  Number of anchors:      {model.num_anchors}")

        for batch_size in [1, 2, 4]:
            print(f"\n  Batch size: {batch_size}")
            input_tensor = torch.randn(batch_size, 1, 224, 224)
            out = model(input_tensor)
            print(f"  Output shape: {out.shape}")

    print("\n" + "=" * 80)
    print("FCOS Object Detector - Validation")
    print("=" * 80)
    for size in ("n", "s", "m", "l"):
        fcos = FCOSObjectDetectorResNet(channels=ObjectDetectorResNet.CONFIGS[size][1])
        params = sum(p.numel() for p in fcos.parameters())
        print(f"FCOS {size.upper()} parameters: {params:,}")
