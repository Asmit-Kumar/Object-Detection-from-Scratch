"""
Strict-load the trained checkpoints that match the current architectures.

Weights are not in git, so each case is skipped when its file is missing (e.g. in CI).
"""
import unittest
from pathlib import Path

import models

ROOT = Path(__file__).resolve().parent.parent

CASES = [
    ("fcos", "weights/fcos_n_28x14_best.pth", {"size": "n"}),
    ("fcos", "weights/fcos_s_28x14_best.pth", {"size": "s"}),
    ("fcos", "weights/fcos_m_28x14_best.pth", {"size": "m"}),
    ("multiscale", "checkpoint/two_scale_28x14_3a_detector_n_best.pth", {"size": "n"}),
    ("multiscale", "checkpoint/two_scale_28x14_3a_detector_s_best.pth", {"size": "s"}),
    ("classifier", "weights/classifier_resent_bymerge_s_best.pth", {"num_classes": 47}),
]


class TestTrainedCheckpoints(unittest.TestCase):

    def test_strict_load(self):
        for name, rel_path, kwargs in CASES:
            path = ROOT / rel_path
            with self.subTest(checkpoint=rel_path):
                if not path.exists():
                    self.skipTest(f"{rel_path} not present")
                model = models.load_model(name, str(path), "cpu", **kwargs)
                self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
