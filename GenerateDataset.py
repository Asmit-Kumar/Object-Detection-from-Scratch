import os
import random
import numpy as np
import pandas as pd
from PIL import Image

IMAGES_DIR = "TestImages"
CLASS_DIR = "Testclass"
LABELS_DIR = "Testlabel"

def generateData(images, labels=None):
    """
    Generate synthetic dataset by placing 28x28 digits onto 128x128 canvases.
    Saves generated images, bounding box coordinates, and class labels to disk.

    Args:
        images (np.ndarray): Array of flattened digit images.
        labels (np.ndarray, optional): Array of corresponding digit classes.
    """

    def createCanvas(digit):
        """
        Place a digit randomly on a 128x128 canvas.

        Args:
            digit (np.ndarray): 28x28 digit image.

        Returns:
            tuple: (128x128 canvas image, [x_min, y_min, x_max, y_max] bbox)
        """
        canvas = np.zeros((128, 128), dtype=np.uint8)

        # row = y, col = x
        y_min = random.randint(0, 100)
        x_min = random.randint(0, 100)
        y_max = y_min + 28
        x_max = x_min + 28

        canvas[y_min:y_max, x_min:x_max] = digit

        bbox = [x_min, y_min, x_max, y_max]
        return canvas, bbox

    def saveImage(img, name):
        """Save the generated 128x128 image to the images directory."""
        os.makedirs(IMAGES_DIR, exist_ok=True)
        Image.fromarray(img).save(f"{IMAGES_DIR}/{name}.png")

    def saveBbox(bbox, name):
        """Save the bounding box coordinates to a text file."""
        os.makedirs(LABELS_DIR, exist_ok=True)
        with open(f"{LABELS_DIR}/{name}.txt", "w") as f:
            f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

    def saveClassLabel(label, name):
        """Save the digit class label to a text file."""
        os.makedirs(CLASS_DIR, exist_ok=True)
        with open(f"{CLASS_DIR}/{name}.txt", "w") as f:
            f.write(str(label))

    for i in range(len(images)):
        digit = images[i].reshape(28, 28).astype(np.uint8)

        canvas, bbox = createCanvas(digit)

        saveImage(canvas, i)
        saveBbox(bbox, i)

        if labels:
            class_label = labels[i]
            saveClassLabel(class_label, i)

    print("Successfully generated dataset, Count:", len(images))


if __name__ == "__main__":
    # df = pd.read_csv("csvs/train.csv")
    #
    # images = df.drop("label", axis=1).values
    # labels = df["label"].values

    df = pd.read_csv("csvs/test.csv")

    generateData(df.values)

