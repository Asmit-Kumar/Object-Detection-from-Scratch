"""
The archived architectures (models/archive/) must keep matching the code they were restored from,
so their old checkpoints keep loading.

The signatures pin every state_dict key and shape; they were recorded after checking each class
against its original source in git (same seeded weights, same outputs). Changing one means the old
checkpoints of that architecture no longer load.
"""
import hashlib
import unittest
import warnings
from pathlib import Path

import models

ROOT = Path(__file__).resolve().parent.parent

# (name, size, num_anchors) -> signature. 'archive.grid14' size 'l' shares its layout with the current
# 28x28 'grid' size 'l' (only layer3's stride differs, which a state_dict does not record).
ARCHIVE_SIGNATURES = {
    ("archive.single_box_cnn", None, None): "b285cb2c852d01c4",
    ("archive.single_box_resnet", None, None): "52ff9faa2ad5d446",
    ("archive.digit_cnn", None, None): "fff9c1307a1e2c47",
    ("archive.stage4_fc", "n", None): "d41978252b2388aa",
    ("archive.stage4_fc", "s", None): "40fd147c39cea7d8",
    ("archive.stage4", "n", None): "029cd7d0fb986739",
    ("archive.stage4", "s", None): "17430e7b42b880af",
    ("archive.stage4", "m", None): "71dd713b936af562",
    ("archive.stage5", "n", None): "1d63c72d23dbc8b1",
    ("archive.stage5", "s", None): "525599c5159cbb10",
    ("archive.stage5", "m", None): "ad998b8849f38c67",
    ("archive.stage5", "l", None): "6006a799cd3456bb",
    ("archive.grid14", "n", 1): "cb131e477c1dfbc8",
    ("archive.grid14", "n", 3): "e7fe5d9dbcf7e200",
    ("archive.grid14", "s", 1): "c266a953529136b3",
    ("archive.grid14", "s", 3): "c80ed1434d8a946b",
    ("archive.grid14", "m", 1): "1389a57c91c90d0d",
    ("archive.grid14", "m", 3): "34c1931e14af40f5",
    ("archive.grid14", "l", 1): "4f4c2abe63ce098d",
    ("archive.grid14", "l", 3): "a11259d9116e183c",
}

# One representative checkpoint per architecture (skipped when the weights are not present).
CHECKPOINTS = [
    ("archive.single_box_cnn", None, {}, "weights/bbox_model_torch.pth"),
    ("archive.single_box_resnet", None, {}, "weights/plain_res_bbox_model_torch_light.pth"),
    ("archive.stage4_fc", "n", {}, "weights/detector_n_best.pth"),
    ("archive.stage4_fc", "s", {}, "weights/detector_s_best.pth"),
    ("archive.stage4", "n", {}, "weights/detector_n_new_best.pth"),
    ("archive.stage4", "s", {}, "weights/detector_s_new_best.pth"),
    ("archive.stage4", "m", {}, "weights/detector_m_new_best.pth"),
    ("archive.stage5", "s", {}, "weights/s_detector_s_best.pth"),
    ("archive.stage5", "l", {}, "weights/s_detector_l_best.pth"),
    ("archive.grid14", "s", {"num_anchors": 1}, "weights/1_grid_detector_s_best.pth"),
    ("archive.grid14", "m", {"num_anchors": 3}, "weights/3_grid_detector_m_best.pth"),
]


def state_dict_signature(model) -> str:
    lines = (f"{key}:{tuple(value.shape)}" for key, value in model.state_dict().items())
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()[:16]


class TestArchivedArchitectures(unittest.TestCase):

    def test_state_dict_signatures(self):
        for (name, size, anchors), expected in ARCHIVE_SIGNATURES.items():
            with self.subTest(name=name, size=size, anchors=anchors):
                kwargs = {"num_anchors": anchors} if anchors else {}
                model = models.build_model(name, size=size or "s", **kwargs)
                self.assertEqual(state_dict_signature(model), expected)

    def test_every_archive_name_is_pinned(self):
        registered = {name for name in models.MODEL_NAMES if name.startswith("archive.")}
        self.assertEqual(registered, {name for name, _, _ in ARCHIVE_SIGNATURES})

    def test_legacy_grid14_import(self):
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            from legacy_models.grid_detector_14x14 import ObjectDetectorResNet
        finally:
            sys.path.remove(str(ROOT / "scripts"))
        self.assertIs(ObjectDetectorResNet, models.Grid14Detector)
        self.assertEqual(ObjectDetectorResNet.CONFIGS["s"], (64, [64, 128, 128, 256], [1, 1, 1, 1], 14))

    def test_trained_checkpoints_load_strictly(self):
        for name, size, kwargs, rel_path in CHECKPOINTS:
            path = ROOT / rel_path
            with self.subTest(checkpoint=rel_path):
                if not path.exists():
                    self.skipTest(f"{rel_path} not present")
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    models.load_model(name, str(path), "cpu", size=size or "s", **kwargs)


if __name__ == "__main__":
    unittest.main()
