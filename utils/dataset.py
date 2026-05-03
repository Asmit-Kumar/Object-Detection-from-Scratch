import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from utils.reader import FileReader as FR
from torchvision import transforms


class AugmentedTensorDataset(Dataset):
    """
    Custom Dataset that wraps tensors and applies dynamic torchvision transforms
    on the fly during __getitem__.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

    def __len__(self):
        return self.images.size(0)


class DatasetBuilder:
    """
    Utility class for building PyTorch datasets
    from image and label directories.
    """

    @staticmethod
    def load_bbox_dataset(
        images_dir,
        labels_dir,
        image_ext="png",
        image_shape=(128, 128),
        normalize_bbox=False,
        limit=None
    ):
        """
        Load images and bounding box labels as PyTorch tensors on CPU.
        """
        image_files = FR.read_files(images_dir, image_ext, limit=limit)
        if image_files is None:
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        images = []
        bboxes = []

        for img_file in image_files:
            name = os.path.splitext(img_file)[0]
            label_path = os.path.join(labels_dir, f"{name}.txt")

            if not os.path.exists(label_path):
                continue

            image = FR.read_image(os.path.join(images_dir, img_file))
            bbox = FR.read_label(label_path)

            images.append(image)
            bboxes.append(bbox)

        if len(images) == 0:
            raise ValueError("No matching image-label pairs found.")

        images_tensor = torch.stack(images)            # (N, 1, H, W)
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)  # (N, 4)

        if normalize_bbox:
            h, w = image_shape
            scale = torch.tensor([w, h, w, h], dtype=torch.float32)
            bboxes_tensor = bboxes_tensor / scale

        return images_tensor, bboxes_tensor

    @staticmethod
    def load_classification_dataset(
        images_dir,
        class_dir,
        image_ext="png",
        num_classes=10,
        one_hot=False,
        limit=None
    ):
        """
        Load images and class labels as PyTorch tensors on CPU.
        """
        image_files = FR.read_files(images_dir, image_ext, limit=limit)
        if image_files is None:
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        images = []
        labels = []

        for img_file in image_files:
            name = os.path.splitext(img_file)[0]
            class_path = os.path.join(class_dir, f"{name}.txt")

            if not os.path.exists(class_path):
                continue

            image = FR.read_image(os.path.join(images_dir, img_file))
            with open(class_path, "r") as f:
                label = int(f.read().strip())

            images.append(image)
            labels.append(label)

        if len(images) == 0:
            raise ValueError("No matching image-label pairs found.")

        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        if one_hot:
            labels_tensor = F.one_hot(labels_tensor, num_classes=num_classes).to(torch.float32)

        return images_tensor, labels_tensor

    @staticmethod
    def as_dataloader(images, labels, batch_size=32, shuffle=True, drop_last=False, transform=None):
        """
        Wrap tensor arrays into a PyTorch DataLoader, optionally with transforms.
        """
        dataset = AugmentedTensorDataset(images, labels, transform=transform)
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last
        )

        return loader

    @staticmethod
    def create_dataloaders(images, labels, batch_size=32, val_split=0.15, test_split=0.15, seed=42, drop_last=False, train_transform=None, eval_transform=None):
        """
        Wrap tensor arrays into PyTorch DataLoaders with automatic train/val/test splits and augmentations.
        If val_split and test_split are 0, returns a single DataLoader.
        If one of them is 0, returns a tuple of two DataLoaders.

        Args:
            images (torch.Tensor): Images tensor.
            labels (torch.Tensor): Labels tensor.
            batch_size (int): Batch size. Defaults to 32.
            val_split (float): Fraction of data to use for validation. Defaults to 0.15.
            test_split (float): Fraction of data to use for testing. Defaults to 0.15.
            seed (int): Random seed for reproducibility. Defaults to 42.
            drop_last (bool): Whether to drop the last incomplete batch. Defaults to False.
            train_transform (callable): Transforms applied only to the training set.
            eval_transform (callable): Transforms applied to the validation and testing sets.

        Returns:
            DataLoader | tuple: 1 to 3 DataLoaders depending on split fractions.
        """
        num_samples = len(images)
        test_size = int(test_split * num_samples)
        val_size = int(val_split * num_samples)
        train_size = num_samples - val_size - test_size
        
        indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed))
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        loaders = []
        
        if train_size > 0:
            train_ds = AugmentedTensorDataset(images[train_idx], labels[train_idx], transform=train_transform)
            loaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last))
            
        if val_size > 0:
            val_ds = AugmentedTensorDataset(images[val_idx], labels[val_idx], transform=eval_transform)
            loaders.append(DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=drop_last))
            
        if test_size > 0:
            test_ds = AugmentedTensorDataset(images[test_idx], labels[test_idx], transform=eval_transform)
            loaders.append(DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=drop_last))
            
        if len(loaders) == 1:
            return loaders[0]
        return tuple(loaders)
