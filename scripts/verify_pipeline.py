"""Verification script for the OD-From-Scratch pipeline.
Checks everything from data loading to trainer compatibility.
No model is actually trained - just the data pipeline and interface.
"""
import sys
sys.path.insert(0, r'c:\OD-From-Scratch')

import torch
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── 1. DataReader (PNG) ──────────────────────────────────────────────────────
print('\n[1] DataReader (PNG) ...')
from utils.reader import DataReader, Record

data_root = Path(r'c:\OD-From-Scratch\data\OD')
train_reader = DataReader(root=data_root / 'train')
test_reader  = DataReader(root=data_root / 'test')

rec = train_reader.get_record(0)
assert isinstance(rec, Record), 'FAIL: get_record did not return Record'
assert rec.image.shape[0] == 1, 'FAIL: image should be (1, H, W)'
assert rec.boxes.ndim == 2 and rec.boxes.shape[1] == 4, 'FAIL: boxes shape wrong'
print('  OK  train + test readers loaded, get_record returns Record')

# ── 2. DetectionDataset.collate_fn returns (images, records) ─────────────────
print('\n[2] DetectionDataset.collate_fn returns (images, records) ...')
from utils.dataset import DetectionDataset
from utils.reader import DataReader

batch_records = [train_reader.get_record(i) for i in range(4)]
result = DetectionDataset.collate_fn(batch_records)

assert len(result) == 2, f'FAIL: collate_fn returned {len(result)} items, expected 2'
imgs, recs = result
assert imgs.shape == (4, 1, 224, 224), f'FAIL: images shape {imgs.shape}'
assert len(recs) == 4, 'FAIL: records list should have 4 items'
assert isinstance(recs[0], Record), 'FAIL: records should contain Record objects'
print('  OK  collate_fn returns (images, list[Record])')

# ── 3. get_detection_loaders ─────────────────────────────────────────────────
print('\n[3] get_detection_loaders ...')
from utils.dataset import get_detection_loaders

train_loader, val_loader = get_detection_loaders(
    data_root=data_root,
    batch_size=32,
    val_size=1000,
    num_workers=0,
    pin_memory=False,
    transform=None,
)
imgs_b, recs_b = next(iter(train_loader))
assert imgs_b.ndim == 4, 'FAIL: images should be 4D'
assert len(recs_b) == 32, 'FAIL: records batch should be 32'
print(f'  OK  train_loader batch: images={imgs_b.shape}, records={len(recs_b)}')

train_loader_t, val_loader_t, test_loader_t = get_detection_loaders(
    data_root=data_root,
    batch_size=32,
    val_size=1000,
    test=True,
    num_workers=0,
    pin_memory=False,
    transform=None,
)
imgs_test, recs_test = next(iter(test_loader_t))
assert imgs_test.ndim == 4, 'FAIL: test images should be 4D'
print(f'  OK  test_loader batch: images={imgs_test.shape}, records={len(recs_test)}')

# ── 4. records_to_padded_tensors ─────────────────────────────────────────────
print('\n[4] records_to_padded_tensors ...')
from utils.trainer import records_to_padded_tensors

boxes, labels, mask, classes_list = records_to_padded_tensors(recs_b, DEVICE)
assert boxes.shape[0] == 32, 'FAIL: boxes batch dim wrong'
assert boxes.shape[2] == 4, 'FAIL: boxes last dim should be 4'
assert mask.dtype == torch.bool, 'FAIL: mask should be bool'
assert boxes.device.type == DEVICE.type, f'FAIL: boxes not on {DEVICE}'
print(f'  OK  boxes={boxes.shape}, mask={mask.shape}, all on {DEVICE}')

# ── 5. DetectionLoss with padded tensors ─────────────────────────────────────
print('\n[5] DetectionLoss ...')
from utils.losses import DetectionLoss
from models.object_detector_res import ObjectDetectorResNet

model = ObjectDetectorResNet(max_objects=24).to(DEVICE)
criterion = DetectionLoss(lambda_conf=1.0)

imgs_gpu = imgs_b.to(DEVICE)
with torch.no_grad():
    outputs = model(imgs_gpu)  # (B, 24, 5)

loss = criterion(outputs, boxes, mask)
assert loss.item() > 0, 'FAIL: loss should be positive'
print(f'  OK  DetectionLoss = {loss.item():.4f}')

# ── 6. Visualizer.visualize_multi_detection_batch signature ──────────────────
print('\n[6] Visualizer.visualize_multi_detection_batch signature ...')
import inspect
from utils.visualizer import Visualizer

sig = inspect.signature(Visualizer.visualize_multi_detection_batch)
params = list(sig.parameters.keys())
assert 'padded_boxes_v' not in params, f'FAIL: old param name found: {params}'
print(f'  OK  signature params: {params}')

# ── 7. Docstrings stale check ────────────────────────────────────────────────
print('\n[7] Stale docstring check in trainer.py ...')
from utils import trainer as _trainer
import inspect

train_doc  = inspect.getdoc(_trainer.train_one_epoch_detection)
eval_doc   = inspect.getdoc(_trainer.evaluate_detection)

stale_phrase = 'Expects loader to yield (images, boxes, labels, mask, classes_list)'
print(f'  train_one_epoch_detection docstring stale: {stale_phrase in (train_doc or "")}')
print(f'  evaluate_detection docstring stale:        {stale_phrase in (eval_doc or "")}')
# Only warn, not fail - docstrings don't affect runtime

print('\n\nAll critical checks PASSED!')
