"""
The ltrb pipeline (targets, loss, decode, dataset flag) is shared by FCOS and the anchor-free YOLOv8
detector. The FCOS names it replaced must stay importable as the same objects.
"""
import types
import unittest

import torch

import models
from dataio.dataset import DetectionDataset
from dataio.ltrb_targets import LTRBTargetGenerator, LTRBTargets
from inference.pipeline import _decode_ltrb_candidates
from training.losses import LTRBLoss


def sample_annotations():
    boxes = [
        torch.tensor([[20.0, 30.0, 40.0, 36.0], [120.0, 100.0, 50.0, 44.0]]),
        torch.zeros(0, 4),
        torch.tensor([[60.0, 60.0, 30.0, 30.0]]),
    ]
    labels = [torch.tensor([3, 17]), torch.zeros(0, dtype=torch.long), torch.tensor([46])]
    return boxes, labels


def random_preds(num_classes=47, with_centerness=True, batch=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    preds = {}
    for grid, stride in ((28, 8.0), (14, 16.0)):
        scale = {
            "cls_logits": torch.randn(batch, grid, grid, num_classes, generator=g),
            "reg_ltrb": torch.rand(batch, grid, grid, 4, generator=g) * 40 * stride / 8,
        }
        if with_centerness:
            scale["centerness_logits"] = torch.randn(batch, grid, grid, generator=g)
        preds[grid] = scale
    return preds


class TestFormerNames(unittest.TestCase):

    def test_fcos_names_are_the_ltrb_objects(self):
        from dataio import fcos_targets, ltrb_targets
        from inference import pipeline
        from training import losses, trainer
        self.assertIs(fcos_targets, ltrb_targets)
        self.assertIs(ltrb_targets.FCOSTargets, LTRBTargets)
        self.assertIs(ltrb_targets.FCOSTargetGenerator, LTRBTargetGenerator)
        self.assertIs(losses.FCOSLoss, LTRBLoss)
        self.assertIs(pipeline._decode_fcos_candidates, _decode_ltrb_candidates)
        self.assertIs(trainer.evaluate_fcos_detection, trainer.evaluate_ltrb_detection)
        self.assertIs(trainer.FCOSLoss, LTRBLoss)


class TestLTRBLoss(unittest.TestCase):

    def test_centerness_is_optional(self):
        boxes, labels = sample_annotations()
        with_cent = random_preds(with_centerness=True)
        without = {grid: {k: v for k, v in scale.items() if k != "centerness_logits"}
                   for grid, scale in with_cent.items()}
        criterion = LTRBLoss()

        _, parts_with = criterion(with_cent, (boxes, labels))
        _, parts_without = criterion(without, (boxes, labels))

        self.assertGreater(parts_with["cent_loss"], 0.0)
        self.assertEqual(parts_without["cent_loss"], 0.0)
        # the classification and regression terms never depend on a predicted centerness
        self.assertEqual(parts_with["cls_loss"], parts_without["cls_loss"])
        self.assertEqual(parts_with["reg_loss"], parts_without["reg_loss"])
        self.assertEqual(parts_with["num_pos"], parts_without["num_pos"])
        self.assertGreater(parts_with["num_pos"], 0)

    def test_accepts_precomputed_targets(self):
        boxes, labels = sample_annotations()
        preds = random_preds(with_centerness=False)
        criterion = LTRBLoss()
        targets = criterion.target_generator.generate_targets(boxes, labels, torch.device("cpu"))
        self.assertIsInstance(targets, LTRBTargets)
        self.assertEqual(criterion(preds, targets)[1], criterion(preds, (boxes, labels))[1])


class TestDecode(unittest.TestCase):
    STRIDES = {28: 8.0, 14: 16.0}

    def setUp(self):
        self.detector = types.SimpleNamespace(STRIDES=self.STRIDES)

    def test_without_centerness_the_score_is_the_class_probability(self):
        preds = random_preds(with_centerness=False, batch=2)
        for scale in preds.values():  # one clearly confident location, everything else suppressed
            scale["cls_logits"] -= 20.0
        preds[28]["cls_logits"][0, 5, 6, 9] = 8.0
        preds[28]["reg_ltrb"][0, 5, 6] = torch.tensor([4.0, 5.0, 6.0, 7.0])

        (boxes, scores, classes), (empty_boxes, _, _) = _decode_ltrb_candidates(
            self.detector, preds, score_threshold=0.5
        )

        self.assertEqual(boxes.shape, (1, 4))
        self.assertEqual(int(classes[0]), 9)
        self.assertAlmostEqual(float(scores[0]), float(torch.sigmoid(torch.tensor(8.0))), places=5)
        xc, yc = (6 + 0.5) * 8.0, (5 + 0.5) * 8.0  # grid cell centre, as used by the target generator
        self.assertTrue(torch.allclose(boxes[0], torch.tensor([xc - 4, yc - 5, 10.0, 12.0])))
        self.assertEqual(empty_boxes.shape, (0, 4))

    def test_centerness_still_gates_the_score(self):
        preds = random_preds(with_centerness=True, batch=1)
        for scale in preds.values():
            scale["cls_logits"] -= 20.0
        preds[14]["cls_logits"][0, 3, 3, 2] = 6.0
        preds[14]["centerness_logits"][0, 3, 3] = 2.0

        ((_, scores, classes),) = _decode_ltrb_candidates(self.detector, preds, score_threshold=0.1)

        expected = torch.sqrt(torch.sigmoid(torch.tensor(6.0)) * torch.sigmoid(torch.tensor(2.0)))
        self.assertEqual(int(classes[0]), 2)
        self.assertAlmostEqual(float(scores[0]), float(expected), places=5)


class TestDatasetFlag(unittest.TestCase):

    @staticmethod
    def records():
        boxes, labels = sample_annotations()
        return [types.SimpleNamespace(image=torch.zeros(1, 224, 224), boxes=b, labels=l)
                for b, l in zip(boxes, labels)]

    def test_collate_builds_ltrb_targets(self):
        dataset = DetectionDataset(reader=[], grid_sizes=(28, 14), ltrb=True)
        images, targets, (boxes, labels) = dataset.collate_fn(self.records())
        self.assertEqual(tuple(images.shape), (3, 1, 224, 224))
        self.assertIsInstance(targets, LTRBTargets)
        self.assertEqual(tuple(targets.reg_targets[14].shape), (3, 14, 14, 4))
        self.assertEqual(len(boxes), 3)

    def test_deferred_targets_are_left_to_the_training_device(self):
        dataset = DetectionDataset(reader=[], grid_sizes=(28, 14), ltrb=True, defer_ltrb_targets=True)
        _, targets, annotations = dataset.collate_fn(self.records())
        self.assertIsNone(targets)
        self.assertEqual(len(annotations), 2)

    def test_anchor_mode_still_needs_anchors(self):
        with self.assertRaises(ValueError):
            DetectionDataset(reader=[], grid_sizes=(28, 14), ltrb=False)


class TestEveryHeadTypeAndSize(unittest.TestCase):
    """The shared targets / loss / decode only rely on the output keys, so every ltrb model fits."""

    def test_loss_backward_and_decode_in_fp32_and_bf16(self):
        boxes, labels = sample_annotations()
        images = torch.randn(3, 1, 224, 224)
        criterion = LTRBLoss()
        for name in ("fcos", "yolov8"):
            for size in "nsml":
                with self.subTest(model=name, size=size):
                    model = models.build_model(name, size=size).train()
                    outputs = model(images)
                    self.assertEqual(
                        "centerness_logits" in outputs[28], name == "fcos",
                        "only FCOS predicts a centerness",
                    )
                    loss, parts = criterion(outputs, (boxes, labels))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.parameters()))

                    model.eval()
                    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                        half = model(images)
                        half_loss, _ = criterion(half, (boxes, labels))
                        candidates = _decode_ltrb_candidates(model, half, score_threshold=0.0)
                    self.assertTrue(torch.isfinite(half_loss))
                    self.assertEqual(len(candidates), 3)
                    self.assertTrue(all(torch.isfinite(c[0]).all() and c[0].shape[1:] == (4,) for c in candidates))


class TestYOLOv8TrainsOnLTRBPipeline(unittest.TestCase):

    def test_loss_backward_and_decode(self):
        torch.manual_seed(0)
        model = models.build_model("yolov8", size="n")
        boxes, labels = sample_annotations()
        images = torch.randn(3, 1, 224, 224)

        model.train()
        loss, parts = LTRBLoss()(model(images), (boxes, labels))
        loss.backward()

        # a freshly built model starts near the focal prior; without the class-bias init this is ~250
        self.assertLess(parts["loss"], 10.0)
        self.assertEqual(parts["cent_loss"], 0.0)
        self.assertGreater(parts["num_pos"], 0)
        missing = [name for name, p in model.named_parameters()
                   if p.grad is None or not torch.isfinite(p.grad).all()]
        self.assertEqual(missing, [])

        model.eval()
        with torch.no_grad():
            candidates = _decode_ltrb_candidates(model, model(images), score_threshold=0.0)
        self.assertEqual(len(candidates), 3)
        for pred_boxes, scores, classes in candidates:
            self.assertEqual(pred_boxes.shape[0], scores.shape[0])
            self.assertEqual(pred_boxes.shape[0], classes.shape[0])
            self.assertEqual(pred_boxes.shape[1:], (4,))


if __name__ == "__main__":
    unittest.main()
