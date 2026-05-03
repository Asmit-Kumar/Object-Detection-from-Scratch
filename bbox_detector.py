from utils.reader import FileReader as FR
import os
import torch
import numpy as np
from scipy.ndimage import label

class BboxDetector:
    """
    Detector class for bounding box prediction validation and data loading.
    
    This class handles loading of image and label data, and provides
    methods for calculating bounding boxes using classical computer vision techniques
    (connected components).
    """

    def __init__(self, images_path, txt_path):
        """
        Initialize the BboxDetector.

        Args:
            images_path (str): Path to the directory containing images.
            txt_path (str): Path to the directory containing label text files.
        """
        self.images_path = images_path
        self.images_files = FR.read_files(images_path, "png")
        self.txt_path = txt_path
        self.txt_files = FR.read_files(txt_path, "txt")
        self.digits = []
        self.actual_bboxes = []
        self.pred_bboxes = []

    def get_images(self):
        """
        Load all images from the configured path.

        Returns:
            list: List of image tensors.
        """
        images = []
        for file_name in self.images_files:
            images.append(FR.read_image(os.path.join(self.images_path, file_name)))

        return images

    def get_labels(self):
        """
        Load all ground truth bounding boxes from labels.
        
        Populates self.actual_bboxes.
        """
        for file in self.txt_files:
            bbox = FR.read_label(os.path.join(self.txt_path, file))
            self.actual_bboxes.append(bbox)

    def _read_image(self, file_name):
        """
        Helper to read a single image by filename.

        Args:
            file_name (str): Name of the file in the images directory.

        Returns:
            torch.Tensor: Processed image.
        """
        image = FR.read_image(os.path.join(self.images_path, file_name))
        return image

    @staticmethod
    def find_digit_bbox(img, threshold=0.15, margin=2):
        """
        Calculate bounding box using connected components (classical approach).

        Args:
            img (torch.Tensor | np.ndarray): Input image.
            threshold (float, optional): Threshold for binarization. Defaults to 0.15.
            margin (int, optional): Margin to add to the bbox. Defaults to 2.

        Returns:
            list: [x_min, y_min, x_max, y_max] or None if no object found.
        """
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            
        img = img.squeeze()

        binary = img > threshold

        labels_map, num = label(binary)

        if num == 0:
            return None

        largest_label = max(
            range(1, num + 1),
            key=lambda i: np.sum(labels_map == i)
        )

        ys, xs = np.where(labels_map == largest_label)

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(img.shape[0] - 1, y_max + margin)
        x_max = min(img.shape[1] - 1, x_max + margin)

        return [x_min, y_min, x_max, y_max]

    def get_pred_bboxes(self):
        """
        Generate predicted bounding boxes for all loaded images using the classical method.
        
        Populates self.pred_bboxes.
        """
        for image_file in self.images_files:
            image = self._read_image(image_file)
            self.pred_bboxes.append(self.find_digit_bbox(image))
