"""
dataset_setup.py
----------------
One-time setup and validation script for the EMNIST ByClass dataset.

Run this once to download the dataset and verify the class distribution:
    python dataset_setup.py

EMNIST ByClass — 62 classes:
    0–9   → digits
    10–35 → uppercase A–Z
    36–61 → lowercase a–z

Training set: ~697,932 samples
Test set:     ~116,323 samples
"""

from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
from collections import Counter

EMNIST_ROOT = "./data"
EMNIST_SPLIT = "byclass"  # 62 classes


def download_emnist():
    """Download EMNIST ByClass train and test splits."""
    print("Downloading EMNIST ByClass train split...")
    train_dataset = EMNIST(root=EMNIST_ROOT, split=EMNIST_SPLIT, train=True, download=True)
    print(f"  Train samples : {len(train_dataset):,}")

    print("Downloading EMNIST ByClass test split...")
    test_dataset = EMNIST(root=EMNIST_ROOT, split=EMNIST_SPLIT, train=False, download=True)
    print(f"  Test samples  : {len(test_dataset):,}")

    return train_dataset, test_dataset


def print_class_info(dataset, name="Dataset"):
    """Print class count and per-class sample distribution."""
    num_classes = len(dataset.classes)
    print(f"\n{name} — {num_classes} classes, {len(dataset):,} total samples")
    print(f"  Classes: {dataset.classes}")

    # Count samples per class (limit to first 50k for speed)
    limit = min(len(dataset), 50_000)
    targets = [dataset.targets[i].item() for i in range(limit)]
    counts = Counter(targets)

    print(f"\n  Per-class sample counts (first {limit:,} samples):")
    for cls_id in sorted(counts):
        label = dataset.classes[cls_id] if cls_id < num_classes else str(cls_id)
        print(f"    [{cls_id:2d}] {label!r:4s} → {counts[cls_id]:6,}")


if __name__ == "__main__":
    train_ds, test_ds = download_emnist()
    print_class_info(train_ds, "Train")
    print_class_info(test_ds, "Test")

    print("\n✓ EMNIST ByClass dataset ready.")
    print("  Use DatasetBuilder.load_emnist_dataset() to get a DataLoader-ready dataset.")
