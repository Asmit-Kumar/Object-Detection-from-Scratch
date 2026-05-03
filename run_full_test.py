"""
Full-test runner: evaluates the PyTorch pipeline on the entire TestImages dataset
and saves per-image prediction logs to logs/run_torch.json.
"""

import os
import time
import torch
from utils.reader import FileReader as FR
from models import load_bbox_model, load_digit_model
from digit_detector import DigitDetectionPipeline

BBOX_WEIGHTS   = "checkpoint/best_od_checkpoint_light.pth"
DIGIT_WEIGHTS  = "Models/digit_best_model_ultralight.pth"
TEST_DIR       = "TestImages"
LOG_PATH       = "logs/run_torch.json"
BATCH_SIZE     = 256

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
pipeline.init_logger(LOG_PATH)

# Load all test images
image_files = FR.read_files(TEST_DIR, "png")
print(f"Loading {len(image_files)} images...")

t0 = time.perf_counter()
images_tensor = torch.stack([FR.read_image(os.path.join(TEST_DIR, f)) for f in image_files])
print(f"Loaded {tuple(images_tensor.shape)} in {time.perf_counter()-t0:.2f}s")

# Run batched inference (pass filenames so logs get correct IDs directly)
print("Running batched inference...")
t0 = time.perf_counter()
pipeline.predict_batch(images_tensor, batch_size=BATCH_SIZE, image_ids=image_files)
print(f"Inference completed in {time.perf_counter()-t0:.2f}s")

# Save
pipeline.save_logs()
print(f"Saved {len(pipeline.logs)} results -> {LOG_PATH}")
