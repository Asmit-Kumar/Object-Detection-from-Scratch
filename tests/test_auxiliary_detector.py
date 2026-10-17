import unittest
from unittest import mock

import torch

import models
from models.auxiliary_network_detector import make_aux_preset
from models.configs import DetectorPreset, YOLOV8_PRESETS
from inference.pipeline import _decode_ltrb_candidates
from training.losses import LTRBLoss
from tests.test_ltrb import random_preds, sample_annotations


class TestAuxiliaryArchitecture(unittest.TestCase):
    def test_neck_scaling_and_independent_parameters_for_every_preset(self):
        for size, preset in YOLOV8_PRESETS.items():
            with self.subTest(size=size):
                model = models.build_model("auxiliary", size=size)
                aux = model.auxiliary_network
                self.assertEqual(aux.project_p3.conv.in_channels, preset.channels[2])
                self.assertEqual(aux.project_p4.conv.in_channels, preset.channels[3])
                for name in ("reduce_p4", "top_down", "down_p3", "bottom_up"):
                    main_layer, aux_layer = getattr(model, name), getattr(aux, name)
                    main_conv = main_layer.conv2 if name in ("top_down", "bottom_up") else main_layer
                    aux_conv = aux_layer.conv2 if name in ("top_down", "bottom_up") else aux_layer
                    self.assertEqual(aux_conv.conv.out_channels * 2, main_conv.conv.out_channels)
                for name in ("top_down", "bottom_up"):
                    self.assertEqual(len(getattr(aux, name).blocks),
                                     max(1, (len(getattr(model, name).blocks) + 1) // 2))
                main_parameters = {p.data_ptr() for name, p in model.named_parameters()
                                   if not name.startswith("auxiliary_network.")}
                self.assertFalse(main_parameters & {p.data_ptr() for p in aux.parameters()})
                self.assertTrue(all(module.eps == 1e-3 and module.momentum == 0.03
                                    for module in model.modules()
                                    if isinstance(module, torch.nn.BatchNorm2d)))

    def test_custom_configuration_is_scaled_from_the_actual_main_neck(self):
        preset = DetectorPreset(10, (10, 20, 30, 50), (1, 2, 3, 5))
        auxiliary = make_aux_preset(preset)
        self.assertEqual(auxiliary.channels, (5, 10, 15, 25))
        self.assertEqual(auxiliary.blocks, (1, 1, 2, 3))
        model = models.build_model("auxiliary", channels=list(preset.channels), blocks=list(preset.blocks))
        self.assertEqual(model.auxiliary_network.top_down.conv2.conv.out_channels, 15)
        self.assertEqual(model.auxiliary_network.bottom_up.conv2.conv.out_channels, 25)
        self.assertEqual(len(model.auxiliary_network.top_down.blocks), 2)
        self.assertEqual(len(model.auxiliary_network.bottom_up.blocks), 3)

    def test_both_branches_train_and_share_one_backbone_execution(self):
        model = models.build_model("auxiliary", size="n").train()
        images = torch.randn(3, 1, 224, 224)
        calls = []
        handle = model.stem.register_forward_hook(lambda *args: calls.append("stem"))
        outputs = model(images)
        handle.remove()
        self.assertEqual(calls, ["stem"])
        self.assertEqual(set(outputs), {"main", "aux"})
        for branch in outputs.values():
            self.assertEqual(set(branch), {28, 14})
            for grid, predictions in branch.items():
                self.assertEqual(set(predictions), {"cls_logits", "reg_ltrb"})
                self.assertEqual(predictions["cls_logits"].shape, (3, grid, grid, 47))
                self.assertEqual(predictions["reg_ltrb"].shape, (3, grid, grid, 4))
                self.assertTrue((predictions["reg_ltrb"] > 0).all())
        loss, _ = LTRBLoss()(outputs, sample_annotations())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual([name for name, p in model.named_parameters()
                          if p.grad is None or not torch.isfinite(p.grad).all()], [])

    def test_auxiliary_loss_reaches_backbone_without_using_main_neck_or_heads(self):
        model = models.build_model("auxiliary", size="n").train()
        outputs = model(torch.randn(3, 1, 224, 224))
        loss, _ = LTRBLoss()(outputs["aux"], sample_annotations())
        loss.backward()
        self.assertGreater(model.stem.conv.weight.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.auxiliary_network.parameters()))
        for name in ("reduce_p4", "top_down", "down_p3", "bottom_up", "head_p3", "head_p4",
                     "scale_p3", "scale_p4"):
            self.assertTrue(all(p.grad is None for p in getattr(model, name).parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_training_loop_with_deferred_targets_and_cuda_autocast(self):
        from training.trainer import train_one_epoch_detection, evaluate_detection

        device = torch.device("cuda")
        model = models.build_model("auxiliary", size="n").to(device)
        criterion = LTRBLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        loader = [(torch.randn(3, 1, 224, 224), None, sample_annotations())]
        before = model.auxiliary_network.head_p3.classes[-1].weight.detach().clone()
        loss = train_one_epoch_detection(model, loader, criterion, optimizer, device)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.auxiliary_network.head_p3.classes[-1].weight))
        metrics = evaluate_detection(model, loader, criterion, device)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_eval_skips_auxiliary_and_export_removes_it_with_identical_predictions(self):
        model = models.build_model("auxiliary", size="n").eval()
        images = torch.randn(2, 1, 224, 224)
        auxiliary_calls = []
        handle = model.auxiliary_network.register_forward_hook(lambda *args: auxiliary_calls.append("aux"))
        with torch.no_grad():
            expected = model(images)
        handle.remove()
        self.assertEqual(auxiliary_calls, [])
        inference_model = model.to_inference()
        self.assertIsInstance(inference_model, models.YOLOv8ObjectDetector)
        self.assertFalse(inference_model.training)
        self.assertFalse(any("auxiliary" in key for key in inference_model.state_dict()))
        self.assertEqual(sum(p.numel() for p in inference_model.parameters()),
                         sum(p.numel() for p in models.build_model("yolov8", size="n").parameters()))
        with torch.no_grad():
            actual = inference_model(images)
        for grid in (28, 14):
            for key in ("cls_logits", "reg_ltrb"):
                self.assertTrue(torch.equal(actual[grid][key], expected[grid][key]))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = inference_model(images)
            decoded = _decode_ltrb_candidates(inference_model, outputs, score_threshold=0.0)
        self.assertEqual(len(decoded), 2)
        self.assertTrue(all(torch.isfinite(boxes).all() for boxes, _, _ in decoded))


class TestAuxiliaryLoss(unittest.TestCase):
    def test_identical_formulation_weighting_and_target_reuse(self):
        criterion = LTRBLoss(lambda_reg=2.0, focal_alpha=0.3, focal_gamma=1.5)
        main = random_preds(with_centerness=False, seed=0)
        auxiliary = random_preds(with_centerness=False, seed=1)
        annotations = sample_annotations()
        main_loss, main_parts = criterion(main, annotations)
        aux_loss, aux_parts = criterion(auxiliary, annotations)
        generate = mock.Mock(wraps=criterion.target_generator.generate_targets)
        criterion.target_generator.generate_targets = generate
        total, parts = criterion({"main": main, "aux": auxiliary}, annotations)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(criterion.lambda_aux, 0.5)
        torch.testing.assert_close(total, main_loss + 0.5 * aux_loss)
        self.assertEqual(parts["main_loss"], main_loss.item())
        self.assertEqual(parts["aux_loss"], aux_loss.item())
        self.assertEqual(parts["loss"], total.item())
        self.assertEqual(parts["num_pos"], main_parts["num_pos"])
        for key in ("cls_loss", "reg_loss", "cent_loss"):
            self.assertEqual(parts[key], main_parts[key] + 0.5 * aux_parts[key])

    def test_precomputed_and_separate_annotations_match(self):
        criterion = LTRBLoss()
        outputs = {"main": random_preds(with_centerness=False),
                   "aux": random_preds(with_centerness=False, seed=1)}
        boxes, labels = sample_annotations()
        targets = criterion.target_generator.generate_targets(boxes, labels, torch.device("cpu"))
        expected = criterion(outputs, (boxes, labels))[1]
        self.assertEqual(criterion(outputs, targets)[1], expected)
        self.assertEqual(criterion(outputs, boxes, labels)[1], expected)

    def test_empty_annotations_are_finite_for_both_branches(self):
        outputs = {"main": random_preds(with_centerness=False, batch=1),
                   "aux": random_preds(with_centerness=False, batch=1, seed=1)}
        loss, parts = LTRBLoss()(outputs, ([torch.zeros(0, 4)], [torch.zeros(0, dtype=torch.long)]))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["num_pos"], 0)
        self.assertEqual(parts["reg_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
