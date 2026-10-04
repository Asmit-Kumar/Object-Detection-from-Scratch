"""
Architecture and loading tests for the models package.

The signatures pin every state_dict key and shape. If one changes, existing checkpoints no longer
load into that architecture: either the change is a mistake, or it is a new architecture that needs
new checkpoints (then update the signature on purpose).
"""
import hashlib
import os
import tempfile
import unittest
import warnings

import torch

import models

STATE_DICT_SIGNATURES = {
    ("grid", "n"): "3c7d6cae1c029fc3",
    ("grid", "s"): "44983b84527b27fb",
    ("grid", "m"): "c354dbf97aad44c0",
    ("grid", "l"): "4f4c2abe63ce098d",
    ("multiscale", "n"): "2dd489220cfce71a",
    ("multiscale", "s"): "2ace66867a0c8b25",
    ("multiscale", "m"): "a07245e246f364d2",
    ("multiscale", "l"): "8bc4b84d5b3b1485",
    ("fcos", "n"): "710d021395ec1833",
    ("fcos", "s"): "6c85a23cb4ef228f",
    ("fcos", "m"): "b603841a1aa0a2c9",
    ("fcos", "l"): "15137d25bac94879",
    ("classifier", 62): "9e4a91167494659c",
    ("classifier", 47): "2be9c324c6cebec0",
}


def state_dict_signature(model: torch.nn.Module) -> str:
    lines = (f"{key}:{tuple(value.shape)}" for key, value in model.state_dict().items())
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()[:16]


def same_weights(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


class TestArchitectures(unittest.TestCase):

    def test_state_dict_signatures(self):
        for (name, variant), expected in STATE_DICT_SIGNATURES.items():
            with self.subTest(name=name, variant=variant):
                if name == "classifier":
                    model = models.build_model(name, num_classes=variant)
                else:
                    model = models.build_model(name, size=variant)
                self.assertEqual(state_dict_signature(model), expected)

    def test_output_shapes(self):
        x = torch.randn(2, 1, 224, 224)
        with torch.no_grad():
            grid = models.build_model("grid", size="n").eval()(x)
            multi = models.build_model("multiscale", size="n").eval()(x)
            fcos = models.build_model("fcos", size="n").eval()(x)
            cls = models.build_model("classifier", num_classes=47).eval()(torch.randn(2, 1, 28, 28))

        self.assertEqual(tuple(grid.shape), (2, 28, 28, 1, 5 + 47))
        self.assertEqual({k: tuple(v.shape) for k, v in multi.items()},
                         {28: (2, 28, 28, 3, 52), 14: (2, 14, 14, 3, 52)})
        self.assertEqual(tuple(fcos[28]["cls_logits"].shape), (2, 28, 28, 47))
        self.assertEqual(tuple(fcos[14]["reg_ltrb"].shape), (2, 14, 14, 4))
        self.assertEqual(tuple(fcos[14]["centerness_logits"].shape), (2, 14, 14))
        self.assertEqual(tuple(cls.shape), (2, 47))

    def test_legacy_configs_format(self):
        # scripts unpack these as (stem, channels, blocks, last)
        self.assertEqual(models.ObjectDetectorResNet.CONFIGS["s"], (64, [64, 128, 128, 256], [1, 1, 1, 1], 28))
        self.assertIs(models.MultiScaleObjectDetectorResNet.CONFIGS, models.ObjectDetectorResNet.CONFIGS)
        self.assertEqual(models.FCOSObjectDetectorResNet.CONFIGS["n"], (32, [32, 64, 64, 128], [1, 1, 1, 1], 48))

    def test_getters_match_build_model(self):
        pairs = [
            (lambda: models.get_detector(size="m"), lambda: models.build_model("grid", size="m")),
            (lambda: models.get_multiscale_detector(size="s"), lambda: models.build_model("multiscale", size="s")),
            (lambda: models.get_fcos_detector(size="n"), lambda: models.build_model("fcos", size="n")),
            (lambda: models.get_classifier(), lambda: models.build_model("classifier")),
        ]
        for old, new in pairs:
            torch.manual_seed(0)
            a = old()
            torch.manual_seed(0)
            b = new()
            self.assertTrue(same_weights(a, b))

    def test_overrides_and_errors(self):
        model = models.build_model("fcos", size="n", fpn_channels=32, num_classes=62)
        self.assertEqual((model.fpn_channels, model.num_classes), (32, 62))
        with self.assertRaises(ValueError):
            models.build_model("yolo")
        with self.assertRaises(ValueError):
            models.build_model("fcos", size="xl")


class TestLoading(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        torch.manual_seed(0)
        self.source = models.build_model("fcos", size="n")

    def _save(self, payload) -> str:
        path = os.path.join(self.tmp.name, "ckpt.pth")
        torch.save(payload, path)
        return path

    def test_checkpoint_formats(self):
        sd = self.source.state_dict()
        for payload in (sd, {"model_state_dict": sd, "best_score": 1.0}, {"model_state": sd, "epoch": 3}):
            with self.subTest(keys=list(payload)[:2]):
                loaded = models.load_model("fcos", self._save(payload), "cpu", size="n")
                self.assertTrue(same_weights(self.source, loaded))
                self.assertFalse(loaded.training)

    def test_anchors_are_restored(self):
        path = self._save({"model_state_dict": self.source.state_dict(), "anchors_wh": torch.ones(3, 2)})
        self.assertEqual(tuple(models.load_model("fcos", path, "cpu", size="n").anchors_wh.shape), (3, 2))

    def test_mismatch_raises_in_load_model_and_warns_in_legacy_loaders(self):
        partial = {k: v for k, v in self.source.state_dict().items() if not k.startswith("cls_head")}
        path = self._save(partial)
        with self.assertRaises(RuntimeError):
            models.load_model("fcos", path, "cpu", size="n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            models.load_fcos_detector(path, "cpu", size="n")
        self.assertTrue(any("does not match fcos/n" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
