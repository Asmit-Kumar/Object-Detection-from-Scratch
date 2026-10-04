"""
The modules moved out of utils/ must stay importable under their old names, as the same module
objects, so older scripts and pickles that reference utils.<module> keep working.
"""
import importlib
import unittest

MOVED = {
    "utils.reader": "dataio.reader",
    "utils.dataset": "dataio.dataset",
    "utils.fcos_targets": "dataio.fcos_targets",
    "utils.trainer": "training.trainer",
    "utils.losses": "training.losses",
    "utils.callbacks": "training.callbacks",
    "utils.logger": "training.logger",
    "utils.pipeline": "inference.pipeline",
}


class TestMovedModules(unittest.TestCase):

    def test_old_names_alias_new_modules(self):
        for old, new in MOVED.items():
            with self.subTest(module=old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))

    def test_old_from_imports(self):
        from utils.dataset import K, S, get_detection_loaders  # noqa: F401
        from utils.pipeline import DetectionPipeline, _decode_fcos_candidates  # noqa: F401
        from utils.trainer import fit  # noqa: F401
        from dataio.dataset import get_detection_loaders as moved
        self.assertIs(get_detection_loaders, moved)


if __name__ == "__main__":
    unittest.main()
