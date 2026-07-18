import json
import threading
from itertools import count
from pathlib import Path
import cv2
import numpy as np
from canvas import CanvasGenerator, Canvas

ROOT_DIR = Path(__file__).parent.parent / 'data' / 'OD'

PLACEMENT_PROBS = {
    "random": 0.50,
    "grid":   0.35,
    "words":  0.10,
    "line":   0.05,
}

_DEFAULT_KEYS = tuple(PLACEMENT_PROBS)
_DEFAULT_PROBS = np.array(tuple(PLACEMENT_PROBS.values()), dtype=float)


class DatasetGenerator:
    def __init__(
        self,
        split: str = 'bymerge',
        dest_dir: Path = ROOT_DIR,
        train_len: int = 255_000,
        test_len: int = 45_000,
        placement=None,
        subset=None,
        num_workers: int = 4,
        write_buffer_size: int = 128,
    ) -> None:
        self.split = split
        self.dest_dir = Path(dest_dir)
        self.train_len = train_len
        self.test_len = test_len
        self.placement = placement
        self.subset = subset
        self.num_workers = num_workers
        self.write_buffer_size = write_buffer_size
        self.canvas_generator = CanvasGenerator(split=split)

    def generate(self):
        if self.subset is not None:
            self._generate_split(self.subset)
        else:
            t1 = threading.Thread(target=self._generate_split, args=('train',))
            t2 = threading.Thread(target=self._generate_split, args=('test',))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

    def _image_dir(self, subset: str) -> Path:
        if isinstance(self.placement, str):
            return self.dest_dir / self.placement / subset / 'images'
        return self.dest_dir / subset / 'images'

    def _metadata_path(self, subset: str) -> Path:
        return self._image_dir(subset).parent / 'metadata.jsonl'

    def _get_placement(self):
        if isinstance(self.placement, str):
            return self.placement
        elif isinstance(self.placement, dict):
            keys = list(self.placement.keys())
            probs = np.array(list(self.placement.values()), dtype=float)
            probs /= probs.sum()
            return np.random.choice(keys, p=probs)
        else:
            return np.random.choice(_DEFAULT_KEYS, p=_DEFAULT_PROBS)

    def _generate_split(self, subset: str):
        total = getattr(self, f"{subset}_len")
        image_dir = self._image_dir(subset)
        metadata_path = self._metadata_path(subset)

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        counter_gen = count()
        lock = threading.Lock()
        buffer: list[str] = []

        def worker(f):
            while True:
                with lock:
                    idx = next(counter_gen)
                if idx >= total:
                    return
                placement = self._get_placement()
                canvas: Canvas = self.canvas_generator.generate_canvas(subset=subset, placement=placement)
                
                filename = f"{idx:06d}.png"
                img_path = image_dir / filename
                cv2.imwrite(
                    str(img_path),
                    canvas.image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )
                    
                record = {"image_idx": idx, "placement": canvas.placement, "objects": [
                    {
                        "bbox": obj.bbox,
                        "label": obj.label,
                        "char": obj.char,
                    }
                    for obj in canvas.objects
                ], "image": filename}

                line = json.dumps(record, separators=(',', ':'))
                with lock:
                    buffer.append(line)
                    if len(buffer) >= self.write_buffer_size:
                        f.write('\n'.join(buffer) + '\n')
                        buffer.clear()

                if idx % 1000 == 0:
                    print(f"[{subset}] {idx}/{total}")

        with open(metadata_path, 'w') as f:
            threads = [threading.Thread(target=worker, args=(f,)) for _ in range(self.num_workers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            if buffer:
                f.write('\n'.join(buffer) + '\n')
                buffer.clear()
        
        print(f"[{subset}] Done - {total} canvases saved to {image_dir.parent}")


if __name__ == '__main__':
    import time

    dest_path = ROOT_DIR / 'gen_test'

    start = time.time()
    gen = DatasetGenerator()
    gen.generate()
    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f}s")
