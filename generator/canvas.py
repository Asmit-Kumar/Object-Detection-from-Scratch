from dataclasses import dataclass
from dataset import EmnistDataset
import numpy as np
import cv2
from scipy.stats import poisson


@dataclass
class CanvasObject:
    bbox: list[int]
    label: int
    char: str

@dataclass
class Canvas:
    placement: str
    image: np.ndarray
    objects: list[CanvasObject]

PLACEMENT_DISTRIBUTIONS = {
    'random': 'uniform',
    'grid': 'uniform',
    'line': {'digits': 0.05, 'uppercase': 0.10, 'lowercase': 0.85},
    'words': {'digits': 0.05, 'uppercase': 0.10, 'lowercase': 0.85},
}

class CanvasGenerator:
    def __init__(self, canvas_size=224, mean_objs=None, split='bymerge'):
        self.canvas_size = canvas_size
        self.dataset = EmnistDataset(split=split)
        self.min_count = 2
        self.mean_objs = mean_objs if mean_objs is not None else canvas_size // 28
        self.rotation_range = [i for i in range(-15, 16, 3)]

    def _get_placement_max_count(self, placement):
        n = self.canvas_size // 28    
        abs_cap = n * (n - 2)
        if placement in ('line', 'words'):
            return min(3 * n, abs_cap)
        else:
            return min(n * 2, abs_cap * 2)

    def get_count(self, placement):
        max_count = self._get_placement_max_count(placement)
        counts = range(self.min_count, max_count + 1)
        probabilities = poisson.pmf(counts, mu=self.mean_objs)
        probabilities /= probabilities.sum()
        return np.random.choice(counts, p=probabilities)

    def sample_chars(self, count, subset="train", distribution='uniform'):
        dataset = getattr(self.dataset, subset)
        patches = dataset.multi_random_samples(count, distribution=distribution)
        return patches

    def _get_bboxes(self, count, placement):
        placements = {
            'random': self._bboxes_random,
            'grid': self._bboxes_grid,
            'line': self._bboxes_line,
            'words': self._bboxes_words,
        }
        fn = placements.get(placement)
        if fn is None:
            print(f"Unknown placement: {placement}. \nUsing random placement.")
            fn = self._bboxes_random

        return fn(count)

    def _bboxes_random(self, count):
        # stride = 28 ensures no nominal cell overlap (previously 28-2 = 26, causing up to 2px overlap)
        stride = 28
        candidates = [
            [x, y, 28, 28]
            for y in range(0, self.canvas_size - 28, stride)
            for x in range(0, self.canvas_size - 28, stride)
        ]
        np.random.shuffle(candidates)
        return candidates[:count]

    def _bboxes_grid(self, count):
        stride = 28 + np.random.randint(2, 8)
        cols = (self.canvas_size - 28) // stride + 1
        rows = (self.canvas_size - 28) // stride + 1

        max_ox = max(0, self.canvas_size - 28 - (cols - 1) * stride)
        max_oy = max(0, self.canvas_size - 28 - (rows - 1) * stride)
        ox = np.random.randint(0, max_ox + 1)
        oy = np.random.randint(0, max_oy + 1)

        bboxes = [
            [ox + c * stride, oy + r * stride, 28, 28]
            for r in range(rows)
            for c in range(cols)
        ]
        np.random.shuffle(bboxes)
        return bboxes[:count]

    def _bboxes_line(self, count):
        # spacing minimum bumped from 1 to 5 to absorb worst-case rotation drift (~3.6px at ±15°)
        spacing = np.random.randint(5, 10)
        line_height = 28 + np.random.randint(8, 20)
        chars_per_row = (self.canvas_size + spacing) // (28 + spacing)

        bboxes = []
        remaining = count
        y = 0
        max_x = 0

        while remaining > 0 and y + 28 <= self.canvas_size:
            chars_this_line = min(remaining, np.random.randint(3, max(4, chars_per_row + 1)))
            
            line_width = chars_this_line * 28 + (chars_this_line - 1) * spacing
            if line_width > self.canvas_size:
                chars_this_line = (self.canvas_size + spacing) // (28 + spacing)
                line_width = chars_this_line * 28 + (chars_this_line - 1) * spacing
                if chars_this_line == 0:
                    break

            for i in range(chars_this_line):
                bboxes.append([i * (28 + spacing), y, 28, 28])

            max_x = max(max_x, line_width)
            remaining -= chars_this_line
            y += line_height

        if not bboxes:
            return []

        total_height = bboxes[-1][1] + 28
        x_offset = np.random.randint(0, max(1, self.canvas_size - max_x))
        y_offset = np.random.randint(0, max(1, self.canvas_size - total_height))

        for bbox in bboxes:
            bbox[0] += x_offset
            bbox[1] += y_offset

        return bboxes

    def _bboxes_words(self, count):
        # char_spacing minimum bumped from 1 to 5 to absorb worst-case rotation drift (~3.6px)
        # word_gap minimum bumped from 16 to 20 similarly
        char_spacing = np.random.randint(5, 8)
        word_gap = np.random.randint(20, 32)
        line_height = 28 + np.random.randint(12, 28)

        bboxes = []
        remaining = count
        y = 0
        max_x = 0

        while remaining > 0 and y + 28 <= self.canvas_size:
            cursor_x = 0
            line_bboxes = []

            while remaining > 0 and cursor_x < self.canvas_size:
                word_len = min(remaining, np.random.randint(2, 6))
                word_width = word_len * 28 + (word_len - 1) * char_spacing
                if cursor_x + word_width > self.canvas_size:
                    break
                for _ in range(word_len):
                    line_bboxes.append([cursor_x, y, 28, 28])
                    cursor_x += 28 + char_spacing
                cursor_x += word_gap - char_spacing
                remaining -= word_len

            if not line_bboxes:
                break

            for bbox in line_bboxes:
                bboxes.append(bbox)

            line_width = line_bboxes[-1][0] + 28
            max_x = max(max_x, line_width)
            y += line_height

        if not bboxes:
            return []

        total_height = bboxes[-1][1] + 28
        x_offset = np.random.randint(0, max(1, self.canvas_size - max_x))
        y_offset = np.random.randint(0, max(1, self.canvas_size - total_height))

        for bbox in bboxes:
            bbox[0] += x_offset
            bbox[1] += y_offset

        return bboxes


    def _get_patched_canvas(self, patches, bboxes):
        background = np.random.randint(8, 48)
        canvas = np.full((self.canvas_size, self.canvas_size), background, dtype=np.uint8)
        for patch, bbox in zip(patches, bboxes):
            x, y, w, h = bbox
            img = patch.image
            assert img.shape == (h, w)
            canvas[y:y+h, x:x+w] = np.maximum(canvas[y:y+h, x:x+w], img)

        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-10, 10)
        canvas = np.clip(alpha * canvas + beta, 0, 255).astype(np.uint8)

        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

        sigma = np.random.uniform(1, 5)
        noise = np.random.normal(0, sigma, canvas.shape)
        canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)

        return canvas

    @staticmethod
    def _boxes_overlap(b1, b2):
        """Return True if two [x, y, w, h] boxes overlap (share any pixel area)."""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    def generate_canvas(self, subset="train", placement="random"):
        count = self.get_count(placement)
        distribution = PLACEMENT_DISTRIBUTIONS.get(placement, 'uniform')
        patches = self.sample_chars(count, subset=subset, distribution=distribution)
        bboxes = self._get_bboxes(count, placement)

        patches = patches[:len(bboxes)]

        # Apply rotation and tight-bbox crop per character, then check for overlap.
        # On overlap: revert the offending character to its original (unrotated) bbox.
        # If it still overlaps, drop it entirely.
        original_bboxes = [list(b) for b in bboxes]
        keep = [True] * len(patches)

        for i in range(len(patches)):
            orig_bbox = list(bboxes[i])
            orig_image = patches[i].image.copy()

            angle = self._get_random_rotation()
            if angle is not None:
                M = cv2.getRotationMatrix2D((14.0, 14.0), angle, 1.0)
                patches[i].image = cv2.warpAffine(
                    patches[i].image, M, (28, 28), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )

            non_zero = cv2.findNonZero(patches[i].image)
            if non_zero is not None:
                tx, ty, tw, th = cv2.boundingRect(non_zero)
                bboxes[i] = [orig_bbox[0] + tx, orig_bbox[1] + ty, tw, th]
                patches[i].image = patches[i].image[ty:ty+th, tx:tx+tw]
            else:
                bboxes[i] = orig_bbox

            # Post-hoc overlap check against all previously accepted characters
            overlapping = any(
                keep[j] and self._boxes_overlap(bboxes[i], bboxes[j])
                for j in range(i)
            )
            if overlapping:
                # Strategy 1: revert to the original unrotated 28x28 box
                patches[i].image = orig_image
                bboxes[i] = orig_bbox
                # Re-check with unrotated box
                still_overlapping = any(
                    keep[j] and self._boxes_overlap(bboxes[i], bboxes[j])
                    for j in range(i)
                )
                if still_overlapping:
                    # Strategy 2: drop the character entirely
                    keep[i] = False

        # Filter out dropped characters
        patches = [p for p, k in zip(patches, keep) if k]
        bboxes  = [b for b, k in zip(bboxes,  keep) if k]

        canvas_img = self._get_patched_canvas(patches, bboxes)
        
        objects = [
            CanvasObject(bbox=bbox, label=patch.label, char=patch.char)
            for patch, bbox in zip(patches, bboxes)
        ]
        
        return Canvas(
            placement=placement,
            image=canvas_img,
            objects=objects
        )

    def _get_random_rotation(self):
        if np.random.random() < 0.25:
            return np.random.choice(self.rotation_range)
        return None


if __name__ == "__main__":
    generator = CanvasGenerator()
    canvas = generator.generate_canvas(placement='line')
    print(f"Generated canvas of size {canvas.image.shape} with {len(canvas.objects)} objects.")
    for obj in canvas.objects:
        print(f" - Char '{obj.char}' (label {obj.label}) at bbox {obj.bbox}")

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots()
    ax.imshow(canvas.image, cmap='gray')

    for obj in canvas.objects:
        x, y, w, h = obj.bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 2, obj.char, color='red', fontsize=12, fontweight='bold')

    plt.show()
    