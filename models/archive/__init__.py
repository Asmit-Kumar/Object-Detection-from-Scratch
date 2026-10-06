"""
models.archive — earlier architectures, restored from git history and kept loadable.

They are not used by the current pipeline, but each one still loads its old checkpoints, so past
results can be reproduced and compared. Build or load them through the same factory as the active
models, using the "archive." names:

    from models import build_model, load_model
    model = load_model("archive.stage4", "weights/detector_s_new_best.pth", device, size="s")

| build_model name           | Class                  | Era / stage                      | Source commit       |
|----------------------------|------------------------|----------------------------------|---------------------|
| archive.single_box_cnn     | SingleBoxCNN           | single-object box regressor      | 788cc42 (models.py) |
| archive.single_box_resnet  | SingleBoxResNet        | single-object box regressor      | 161ffb4 (commented) |
| archive.digit_cnn          | DigitClassifierCNN     | 10-class digit classifier        | 788cc42 (models.py) |
| archive.stage4_fc          | Stage4SlotDetectorFC   | Stage 4, first multi-object      | 161ffb4             |
| archive.stage4             | Stage4BoxDetector      | Stage 4, two-stage box detector  | 2292f24 .. 9625739  |
| archive.stage5             | Stage5UnifiedDetector  | Stage 5, FC tri-head             | 78d0485 .. edee5a1  |
| archive.grid14             | Grid14Detector         | Stage 6 (K=1) / Stage 7 (K=3)    | 2be0b81             |

Each module's docstring lists the checkpoints it loads; `python -m models.archive.<module>` prints
its per-size parameters and output shapes. The TensorFlow models of the first versions lived in
notebooks and are kept by the v1/v2/v3-tensorflow tags rather than here.
"""
from .digit_classifier_cnn import DigitClassifierCNN
from .grid14_detector import Grid14Detector
from .single_box_cnn import SingleBoxCNN
from .single_box_resnet import SingleBoxResNet
from .stage4_box_detector import Stage4BoxDetector
from .stage4_slot_fc_detector import Stage4SlotDetectorFC
from .stage5_unified_detector import Stage5UnifiedDetector

__all__ = [
    "SingleBoxCNN",
    "SingleBoxResNet",
    "DigitClassifierCNN",
    "Stage4SlotDetectorFC",
    "Stage4BoxDetector",
    "Stage5UnifiedDetector",
    "Grid14Detector",
]
