from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Union
import numpy as np

DATA_ROOT = Path(__file__).parent.parent / "data" / "EMNIST"
RAW_IMAGES_PATH = 'raw/emnist-{}-{}-images-idx3-ubyte'
RAW_LABEL_PATH = 'raw/emnist-{}-{}-labels-idx1-ubyte'

_DIGITS = list("0123456789")
_UPPERCASE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_LOWERCASE = list("abcdefghijklmnopqrstuvwxyz")
_BYMERGE_LOWERCASE = list("abdefghnqrt")

EMNIST_CLASS_NAMES = {
    "byclass": _DIGITS + _UPPERCASE + _LOWERCASE,
    "bymerge": _DIGITS + _UPPERCASE + _BYMERGE_LOWERCASE,
    "balanced": _DIGITS + _UPPERCASE + _BYMERGE_LOWERCASE,
    "letters": ["N/A"] + _LOWERCASE,
    "digits": _DIGITS,
    "mnist": _DIGITS,
}


def get_emnist_class_names(split: str) -> list[str]:
    try:
        return list(EMNIST_CLASS_NAMES[split])
    except KeyError as exc:
        available = ", ".join(sorted(EMNIST_CLASS_NAMES))
        raise ValueError(f"Unknown EMNIST split '{split}'. Available splits: {available}") from exc


@dataclass
class CharacterSample:
    image: np.ndarray
    label: int
    char: str


class BaseCharacterDataset:
    classes: np.ndarray | None
    class_ids: np.ndarray | None
    label_to_char: dict[int, str]
    char_to_label: dict[str, int]
    images: np.ndarray | None
    labels: np.ndarray | None
    class_to_idxs: dict[int, list[int]]

    def __init__(
            self,
            images: np.ndarray | None = None,
            labels: np.ndarray | None = None,
            class_names: Sequence[str] | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.class_to_idxs = defaultdict(list)

        self.class_ids = None
        self.label_to_char = {}
        self.char_to_label = {}
        self.classes = np.array(class_names) if class_names is not None else None
        self.distributions = {}

        self.label_to_char = {
            class_id: class_name
            for class_id, class_name in enumerate(self.classes)
        } if self.classes is not None else {}
        self.char_to_label = {
            class_name: class_id
            for class_id, class_name in enumerate(self.classes)
        } if self.classes is not None else {}


    def generate_indices(self):
        if self.labels is None:
            return

        self.class_to_idxs.clear()
        for idx, cls in enumerate(self.labels):
            self.class_to_idxs[int(cls)].append(idx)
        self.class_ids = np.array(sorted(self.class_to_idxs.keys()))
        self._set_default_distribution()


    def _resolve_class_id(self, cls: Any) -> int:
        if hasattr(cls, 'item'):
            cls = cls.item()
        if isinstance(cls, str):
            if cls not in self.char_to_label:
                raise KeyError(f"Unknown class name: {cls}")
            return self.char_to_label[cls]
        return int(cls)

    def sample(self, cls: Any, count: int = 1) -> Union[CharacterSample, list[CharacterSample]]:
        cls_id = self._resolve_class_id(cls)
        if cls_id not in self.class_to_idxs:
            raise KeyError(f"Class id {cls_id} is not present in this dataset")

        indices = np.random.choice(
            self.class_to_idxs[cls_id],
            size=count,
            replace=False
        )

        char = self.label_to_char.get(cls_id, str(cls_id))

        if count == 1:
            return CharacterSample(image=self.images[indices[0]], label=cls_id, char=char)

        return [CharacterSample(image=self.images[idx], label=cls_id, char=char) for idx in indices]

    def _get_distribution(self, distribution: Union[str, dict, np.ndarray] = 'uniform') -> np.ndarray:
        if isinstance(distribution, np.ndarray):
            assert len(distribution) == len(self.class_ids), "Custom distribution array must match the number of classes."
            return distribution / distribution.sum()

        if isinstance(distribution, dict):
            group_counts = {'digits': 0, 'uppercase': 0, 'lowercase': 0}
            for cls_id in self.class_ids:
                char = self.label_to_char.get(cls_id, str(cls_id))
                if char not in distribution:
                    if char.isdigit(): group_counts['digits'] += 1
                    elif char.isupper(): group_counts['uppercase'] += 1
                    elif char.islower(): group_counts['lowercase'] += 1

            weights = np.zeros(len(self.class_ids))
            for i, cls_id in enumerate(self.class_ids):
                char = self.label_to_char.get(cls_id, str(cls_id))
                if char in distribution:
                    weights[i] = distribution[char]
                elif char.isdigit() and "digits" in distribution:
                    weights[i] = distribution["digits"] / max(1, group_counts['digits'])
                elif char.isupper() and "uppercase" in distribution:
                    weights[i] = distribution["uppercase"] / max(1, group_counts['uppercase'])
                elif char.islower() and "lowercase" in distribution:
                    weights[i] = distribution["lowercase"] / max(1, group_counts['lowercase'])
                else:
                    weights[i] = 0.0

            if weights.sum() == 0:
                raise ValueError("Custom distribution has zero sum for available classes.")
            return weights / weights.sum()

        return self.distributions[distribution]

    def _set_default_distribution(self):
        def get_digit_heavy_distribution():
            weights = np.zeros(len(self.class_ids))
            for i, cls_id in enumerate(self.class_ids):
                char = self.label_to_char.get(cls_id, str(cls_id))
                weights[i] = 9.0 if char.isdigit() else 1.0
            return weights / weights.sum()

        def get_english_distribution():
            english_freq = {
                'a': 8.167, 'b': 1.492, 'c': 2.782, 'd': 4.253, 'e': 12.702,
                'f': 2.228, 'g': 2.015, 'h': 6.094, 'i': 6.966, 'j': 0.153,
                'k': 0.772, 'l': 4.025, 'm': 2.406, 'n': 6.749, 'o': 7.507,
                'p': 1.929, 'q': 0.095, 'r': 5.987, 's': 6.327, 't': 9.056,
                'u': 2.758, 'v': 0.978, 'w': 2.360, 'x': 0.150, 'y': 1.974, 'z': 0.074,
            }
            weights = np.zeros(len(self.class_ids))
            for i, cls_id in enumerate(self.class_ids):
                char = self.label_to_char.get(cls_id, str(cls_id))
                lower_char = char.lower()

                if char.isalpha():
                    base_freq = english_freq.get(lower_char, 0.1)
                    if char.isupper():
                        weights[i] = base_freq * 0.05
                    else:
                        weights[i] = base_freq
                elif char.isdigit():
                    weights[i] = 1.0
                else:
                    weights[i] = 0.1
            return weights / weights.sum()


        self.distributions['uniform'] = np.ones(len(self.class_ids)) / len(self.class_ids)
        self.distributions['digit-heavy'] = get_digit_heavy_distribution()
        self.distributions['english'] = get_english_distribution()

    def random_sample(self, distribution: Union[str, dict, np.ndarray] = 'uniform'):
        if self.class_ids is None or len(self.class_ids) == 0:
            raise ValueError("No classes available in this dataset")
        p = self._get_distribution(distribution)
        cls = np.random.choice(self.class_ids, p=p)
        cls_id = self._resolve_class_id(cls)
        return self.sample(cls_id)

    def multi_random_samples(self, count: int = 1, distribution: Union[str, dict, np.ndarray] = 'uniform', replace: bool = True):
        assert count >= 1
        if not replace:
            assert count <= len(self.class_ids)
        p = self._get_distribution(distribution)
        classes = np.random.choice(self.class_ids, size=count, replace=replace, p=p)
        return [self.sample(cls) for cls in classes]

class EmnistDataset:

    def __init__(self, root: Union[str, Path] = DATA_ROOT, split: str = 'bymerge') -> None:
        self.root = Path(root) if isinstance(root, str) else root
        self.split = split
        self.class_names = get_emnist_class_names(split)

        self.train = BaseCharacterDataset(class_names=self.class_names)
        self.test = BaseCharacterDataset(class_names=self.class_names)

        self._set_base_dataset()

    def _set_base_dataset(self) -> None:
        for subset in ['train', 'test']:
            images_path_str = self.root / RAW_IMAGES_PATH.format(self.split, subset)
            labels_path_str = self.root / RAW_LABEL_PATH.format(self.split, subset)
            dataset = getattr(self, subset)
            dataset.images = read_idx(images_path_str)
            dataset.labels = read_idx(labels_path_str)
            dataset.generate_indices()


def read_idx(path):

    if path.exists():
        with open(path, "rb") as f:
            data = f.read()

        magic = get_int(data[:4])
        ty = magic // 256
        nd = magic % 256

        assert ty == 8
        shape = [
            get_int(data[4 * (i + 1): 4 * (i + 2)])
            for i in range(nd)
        ]
        parsed = np.frombuffer(
            data,
            dtype=np.uint8,
            offset=4*(nd+1)
        ).reshape(shape)
        
        if nd == 3:
            parsed = parsed.transpose((0, 2, 1))
            
        return parsed

    return None


def get_int(b: bytes):
    return int.from_bytes(b, byteorder="big")


if __name__ == "__main__":
    print("Initializing EmnistDataset...")
    ed = EmnistDataset()
    
    # 1. Verify classes property is a numpy array
    print("\n--- Testing 'classes' property ---")
    print(f"Classes type: {type(ed.train.classes)}")
    print(f"First 15 classes: {ed.train.classes[:15]}")
    assert isinstance(ed.train.classes, np.ndarray), "classes should be a numpy array"
    
    # 2. Verify label_to_char and char_to_label mappings
    print("\n--- Testing mapping dictionaries ---")
    sample_label = ed.train.class_ids[0]
    sample_char = ed.train.label_to_char[sample_label]
    print(f"Label {sample_label} maps to Character: '{sample_char}'")
    print(f"Character '{sample_char}' maps back to Label: {ed.train.char_to_label[sample_char]}")
    assert ed.train.char_to_label[sample_char] == sample_label, "Reverse mapping mismatch"

    # 3. Verify sampling by character and by label ID
    print("\n--- Testing sample(cls) by both character and label ---")
    char_to_test = sample_char
    label_to_test = sample_label
    
    sample_by_char = ed.train.sample(char_to_test)
    sample_by_label = ed.train.sample(label_to_test)
    print(f"Sampled by char '{char_to_test}' shape: {sample_by_char.image.shape}, label: {sample_by_char.label}, class: {sample_by_char.char}")
    print(f"Sampled by label ID '{label_to_test}' shape: {sample_by_label.image.shape}, label: {sample_by_label.label}, class: {sample_by_label.char}")
    assert sample_by_char.image.shape == (28, 28) or sample_by_char.image.shape == (28, 28, 1), "Incorrect image shape"

    # 4. Verify random_sample returns CharacterSample
    print("\n--- Testing random_sample() ---")
    random_samp = ed.train.random_sample()
    print(f"random_sample returned: Image shape = {random_samp.image.shape}, Label = {random_samp.label}, Character = '{random_samp.char}'")
    assert isinstance(random_samp.char, str), "Returned character must be a string"
    assert isinstance(random_samp.image, np.ndarray), "Returned image must be a numpy array"

    # 5. Verify error handling for invalid character
    print("\n--- Testing invalid sample lookup ---")
    try:
        ed.train.sample('invalid_char')
    except KeyError as e:
        print(f"Caught expected KeyError for invalid char: {e}")

    print("\n--- Testing distributions ---")
    print(ed.train._get_distribution())

    print("\nAll sanity checks completed successfully!")
