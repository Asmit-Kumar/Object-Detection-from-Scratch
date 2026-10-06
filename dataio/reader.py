from pathlib import Path
import json
import torch
import cv2
from dataclasses import dataclass
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'OD'


@dataclass
class Record:
    image: torch.Tensor
    boxes: torch.Tensor
    labels: torch.Tensor
    classes: list[str]

@dataclass(slots=True)
class Metadata:
    image: str
    boxes: np.ndarray
    labels: np.ndarray
    classes: list[str]


class DataReader:
    """Reads a generated dataset directory (images/ + metadata.jsonl)."""

    def __init__(
            self,
            root: str | Path,
            image_size: tuple = (224, 224),
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.images_dir = self.root / 'images'
        self.records: list[Metadata] = []       # parsed JSONL rows
        self.metadata_path = self.root / 'metadata.jsonl'
        self._load_metadata()

    def _load_metadata(self):
        """Parse metadata.jsonl → self.records list."""
        self.records = []

        with open(self.metadata_path) as f:
            for line in f:
                if not line.strip():
                    continue

                rec = json.loads(line)
                objects = rec["objects"]

                boxes = []
                labels = []
                classes = []

                for obj in objects:
                    boxes.append(obj["bbox"])
                    labels.append(obj["label"])
                    classes.append(obj["char"])

                self.records.append(
                    Metadata(
                        image=rec["image"],
                        boxes=np.asarray(boxes, dtype=np.float32),
                        labels=np.asarray(labels, dtype=np.int64),
                        classes=classes,
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def read_image(self, filename: str) -> torch.Tensor:
        """Read image → float32 (1, H, W) normalised to [0, 1]."""
        path = str(self.images_dir / filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # uint8 (H, W)
        if img is None:
            raise FileNotFoundError(path)
        return torch.from_numpy(img).unsqueeze(0).float().div_(255.0)

    def get_record(self, idx: int) -> Record:
        """Return Record with image Tensor(1, H, W), boxes Tensor(M, 4), labels Tensor(M,)."""
        rec = self.records[idx]
        image = self.read_image(rec.image)
        boxes = torch.from_numpy(rec.boxes)
        labels = torch.from_numpy(rec.labels)
        classes = rec.classes

        return Record(
            image=image,
            boxes=boxes,
            labels=labels,
            classes=classes
        )


if __name__ == '__main__':
    reader = DataReader(ROOT_DIR / 'train', image_size=(224, 224))
    record = reader.get_record(np.random.randint(0, len(reader.records)))
    for cls, lbl, bbox in zip(record.classes, record.labels, record.boxes):
        print(f" - Char '{cls}' (label {lbl}) at bbox {bbox}")

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots()
    ax.imshow(record.image.detach().cpu().squeeze().numpy(), cmap='gray')

    for cls, bbox in zip(record.classes, record.boxes):
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 2, cls, color='red', fontsize=12, fontweight='bold')

    plt.show()
