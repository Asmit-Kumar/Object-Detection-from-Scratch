import io
import unittest
from unittest import mock

import torch

from models import build_model
from dataio.task_aligned import TaskAlignedAssigner, flatten_ltrb_predictions
from inference.pipeline import _decode_ltrb_candidates, DetectionPipeline
from training.losses import YOLOv10Loss
from tests.test_ltrb import sample_annotations, random_preds


class TestTaskAlignedAssignment(unittest.TestCase):
    def test_top_one_is_global_across_scales_and_agrees_with_top_many(self):
        model = build_model("yolov10", size="n")
        preds = random_preds(with_centerness=False, batch=1)
        logits, _, points, boxes = flatten_ltrb_predictions(preds, model.STRIDES)
        annotations = ([torch.tensor([[0., 0., 224., 224.]])], [torch.tensor([0])])
        one = TaskAlignedAssigner(1)(logits, boxes, points, *annotations)
        many = TaskAlignedAssigner(10)(logits, boxes, points, *annotations)
        self.assertEqual((one.matched_gt >= 0).sum().item(), 1)
        self.assertEqual((many.matched_gt >= 0).sum().item(), 10)
        self.assertTrue((many.matched_gt[one.matched_gt >= 0] == 0).all())
        self.assertFalse(one.scores.requires_grad)

    def test_prediction_quality_changes_selection_and_collision_has_one_owner(self):
        points = torch.tensor([[4., 4.], [12., 4.]])
        boxes = torch.tensor([[[0., 0., 16., 8.], [0., 0., 16., 8.]]])
        annotations = ([torch.tensor([[0., 0., 16., 8.], [0., 0., 16., 8.]])], [torch.tensor([0, 1])])
        logits = torch.tensor([[[5., 5.], [-5., -5.]]])
        first = TaskAlignedAssigner(1)(logits, boxes, points, *annotations)
        second = TaskAlignedAssigner(1)(logits.flip(1), boxes, points, *annotations)
        self.assertEqual(first.matched_gt.tolist(), [[0, -1]])
        self.assertEqual(second.matched_gt.tolist(), [[-1, 0]])
        self.assertTrue((first.scores > 0).sum(-1).le(1).all())

    def test_spatial_prior_and_empty_image(self):
        logits = torch.zeros(2, 2, 2)
        points = torch.tensor([[4., 4.], [100., 100.]])
        boxes = torch.tensor([[[0., 0., 8., 8.], [0., 0., 8., 8.]]] * 2)
        annotations = ([torch.tensor([[0., 0., 8., 8.]]), torch.empty(0, 4)],
                       [torch.tensor([1]), torch.empty(0, dtype=torch.long)])
        result = TaskAlignedAssigner(10)(logits, boxes, points, *annotations)
        self.assertEqual(result.matched_gt.tolist(), [[0, -1], [-1, -1]])
        self.assertEqual(result.scores[0, 0, 1].item(), 1.)


class TestYOLOv10(unittest.TestCase):
    def test_dual_loss_and_branch_gradient_ownership(self):
        model = build_model("yolov10", size="n").train()
        outputs = model(torch.randn(3, 1, 224, 224))
        self.assertEqual(set(outputs), {"main", "aux"})
        criterion = YOLOv10Loss()
        annotations = sample_annotations()
        main_loss, _ = criterion(outputs["main"], annotations)
        main_loss.backward(retain_graph=True)
        self.assertIsNone(model.stem.conv.weight.grad)
        self.assertGreater(model.head_p3.classes[-1].weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        loss, parts = criterion(outputs, annotations)
        aux_loss, _ = criterion._branch_loss(outputs["aux"], annotations, criterion.one_to_many)
        torch.testing.assert_close(loss, main_loss + 0.5 * aux_loss)
        self.assertLessEqual(parts["num_pos"], sum(len(b) for b in annotations[0]))
        self.assertGreater(parts["aux_num_pos"], parts["num_pos"])
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.stem.conv.weight.grad.abs().sum().item(), 0)

    def test_empty_and_precomputed_annotations(self):
        criterion = YOLOv10Loss()
        preds = random_preds(with_centerness=False, batch=1)
        annotations = ([torch.empty(0, 4)], [torch.empty(0, dtype=torch.long)])
        loss, parts = criterion(preds, annotations)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["num_pos"], 0)
        self.assertEqual(parts["reg_loss"], 0)
        dense = criterion.target_generator.generate_targets(*annotations, torch.device("cpu"))
        torch.testing.assert_close(loss, criterion(preds, dense, annotations)[0])
        torch.testing.assert_close(loss, criterion(preds, *annotations)[0])
        with self.assertRaisesRegex(ValueError, "raw"):
            criterion(preds, dense)

    def test_export_reload_skips_auxiliary_and_preserves_nms_free_predictions(self):
        model = build_model("yolov10", size="n").eval()
        images = torch.randn(2, 1, 224, 224)
        auxiliary_calls = []
        handle = model.auxiliary_network.register_forward_hook(lambda *args: auxiliary_calls.append(1))
        with torch.no_grad():
            expected = model(images)
        handle.remove()
        self.assertEqual(auxiliary_calls, [])
        exported = model.to_inference()
        self.assertTrue(exported.end_to_end)
        self.assertFalse(hasattr(exported, "auxiliary_network"))
        self.assertFalse(any("auxiliary" in key for key in exported.state_dict()))
        self.assertLess(sum(p.numel() for p in exported.parameters()), sum(p.numel() for p in model.parameters()))
        buffer = io.BytesIO()
        torch.save(exported.state_dict(), buffer)
        buffer.seek(0)
        restored = build_model("yolov10", size="n", inference_only=True).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        with torch.no_grad():
            actual = restored(images)
        for grid in (28, 14):
            for key in expected[grid]:
                self.assertTrue(torch.equal(actual[grid][key], expected[grid][key]))
        pipeline = DetectionPipeline.__new__(DetectionPipeline)
        pipeline.ltrb_detector, pipeline.ltrb_model = True, "yolov10"
        pipeline.detector_size, pipeline.device = "n", "cpu"
        with mock.patch("models.load_model", return_value=model):
            deployed = pipeline._load_detector()
        self.assertFalse(hasattr(deployed, "auxiliary_network"))
        self.assertTrue(deployed.end_to_end)

    def test_decoder_keeps_overlaps_without_nms_and_caps_each_image(self):
        model = build_model("yolov10", size="n", max_detections=2)
        points = torch.tensor([[[4., 4.], [12., 4.], [20., 4.]]])
        same_box = torch.cat((points, torch.tensor([[[40., 40.]]]) - points), -1).expand(2, -1, -1).unsqueeze(1)
        outputs = {28: {"cls_logits": torch.tensor([[[[4., -4.], [3., -3.], [2., -2.]]]]).expand(2, -1, -1, -1),
                        "reg_ltrb": same_box}}
        with mock.patch("torchvision.ops.batched_nms", side_effect=AssertionError("NMS called")), \
             mock.patch("torchvision.ops.nms", side_effect=AssertionError("NMS called")):
            result = _decode_ltrb_candidates(model, outputs, 0.5)
            for boxes, scores, classes in result:
                self.assertEqual(len(boxes), 2)
                torch.testing.assert_close(boxes, torch.tensor([[0., 0., 40., 40.]] * 2))
                self.assertTrue((scores[:-1] >= scores[1:]).all())
                self.assertEqual(classes.tolist(), [0, 0])
            self.assertTrue(all(len(boxes) == 0 for boxes, _, _ in _decode_ltrb_candidates(model, outputs, 1.0)))
            pipeline = DetectionPipeline.__new__(DetectionPipeline)
            pipeline.ltrb_detector, pipeline.detector, pipeline.iou_threshold = True, model, 0.45
            with mock.patch.object(model, "forward", return_value=outputs):
                self.assertEqual(len(pipeline._detector_candidates(torch.zeros(2, 1, 224, 224), 0.5)), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_training_and_final_e2e_evaluation_without_nms(self):
        from training.trainer import train_one_epoch_detection, evaluate_detection

        model = build_model("yolov10", size="n").cuda()
        criterion = YOLOv10Loss()
        loader = [(torch.rand(3, 1, 224, 224), None, sample_annotations())]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.auxiliary_network.head_p3.classes[-1].weight.detach().clone()
        loss = train_one_epoch_detection(model, loader, criterion, optimizer, torch.device("cuda"))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.auxiliary_network.head_p3.classes[-1].weight))
        with mock.patch("torchvision.ops.batched_nms", side_effect=AssertionError("NMS called")):
            metrics = evaluate_detection(model, loader, criterion, torch.device("cuda"), conf_threshold=0.)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertIn("f1", metrics["end_to_end"])


if __name__ == "__main__":
    unittest.main()
