"""Output-contract tests for the YOLOv8-inspired anchor-free detector."""
import unittest

import torch
from models.yolov8_detector import YOLOv8ObjectDetector


class TestYOLOv8Detector(unittest.TestCase):
    def test_output_contract(self):
        model = YOLOv8ObjectDetector(num_classes=47).eval()
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 224, 224)

        with torch.no_grad():
            outputs = model(input_tensor)

        self.assertEqual(set(outputs), {28, 14})
        for grid_size in (28, 14):
            with self.subTest(grid_size=grid_size):
                scale = outputs[grid_size]
                self.assertEqual(set(scale), {"cls_logits", "reg_ltrb"})
                self.assertEqual(scale["cls_logits"].shape, (batch_size, grid_size, grid_size, 47))
                self.assertEqual(scale["reg_ltrb"].shape, (batch_size, grid_size, grid_size, 4))
                self.assertTrue(torch.all(scale["reg_ltrb"] > 0))


if __name__ == "__main__":
    unittest.main()
