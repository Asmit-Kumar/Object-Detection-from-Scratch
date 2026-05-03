"""
Visual evaluation script: runs the PyTorch pipeline on the first 12 test images
and saves a 3x4 detection grid to pipeline_test_results.png.
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.reader import FileReader as FR
from models import load_bbox_model, load_digit_model
from digit_detector import DigitDetectionPipeline

BBOX_WEIGHTS  = "checkpoint/best_od_checkpoint_light.pth"
DIGIT_WEIGHTS = "Models/digit_best_model_ultralight.pth"
TEST_DIR      = "TestImages"
OUT_FILE      = "pipeline_test_results.png"
N_SHOW        = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load models
bbox_model  = load_bbox_model(BBOX_WEIGHTS,  device)
digit_model = load_digit_model(DIGIT_WEIGHTS, device)

pipeline = DigitDetectionPipeline(
    bbox_model=bbox_model,
    classifier_model=digit_model,
    use_classical_fallback=True,
    normalize_bbox=True,
    device=device,
)

# Check directory
if not os.path.exists(TEST_DIR):
    print(f"Directory not found: {TEST_DIR}")
    raise SystemExit(1)

image_files = FR.read_files(TEST_DIR, "png")[:N_SHOW]
print(f"Running pipeline on {len(image_files)} test images...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, file_name in enumerate(image_files):
    img = FR.read_image(os.path.join(TEST_DIR, file_name))  # (1, 128, 128)
    result = pipeline.predict(img, image_id=file_name)

    ax = axes[i]
    ax.imshow(img.squeeze().numpy(), cmap="gray")

    if result:
        x_min, y_min, x_max, y_max = result["bbox"]
        ax.add_patch(patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor="lime", facecolor="none"
        ))
        ax.set_title(
            f"Pred: {result['digit']} ({result['confidence']*100:.1f}%)\n"
            f"[BBox: {result.get('bbox_source', 'cnn')}]",
            color="lime", fontsize=10
        )
    else:
        ax.set_title("No Detection", color="red")

    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150)
print(f"Saved visualization grid -> {OUT_FILE}")
