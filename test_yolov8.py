"""Test script for YOLOv8 anchor-free detector."""
import torch
from models.yolov8_detector import YOLOv8ObjectDetector


def test_yolov8_detector():
    # Create model instance with default parameters (num_classes=47)
    model = YOLOv8ObjectDetector(num_classes=47)
    model.eval()

    # Create random input tensor of shape (2, 1, 224, 224)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 224, 224)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)

    # Check output structure
    print("Output keys:", list(outputs.keys()))
    assert set(outputs) == {28, 14}

    # Expected per scale: {'cls_logits': (B, H, W, C), 'reg_ltrb': (B, H, W, 4)} where C=47.
    # There is no objectness / centerness branch.
    C = 47
    for grid_size in (28, 14):
        scale = outputs[grid_size]
        print(f"\nScale {grid_size} outputs: { {k: tuple(v.shape) for k, v in scale.items()} }")
        assert set(scale) == {"cls_logits", "reg_ltrb"}, f"Scale {grid_size} keys: {sorted(scale)}"
        assert scale["cls_logits"].shape == (batch_size, grid_size, grid_size, C)
        assert scale["reg_ltrb"].shape == (batch_size, grid_size, grid_size, 4)
        # ltrb distances are exp-activated, so they are strictly positive
        assert torch.all(scale["reg_ltrb"] > 0)
        print(f"Scale {grid_size} shapes match expected")

    print("All output components verified correctly!")
    print("\n=== TEST PASSED ===")


if __name__ == "__main__":
    test_yolov8_detector()
