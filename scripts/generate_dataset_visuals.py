"""
Saves clean, plain generated dataset images (without detection overlays)
for all 4 placement strategies (random, grid, words, line) to assets/dataset/
for use in README.md.
"""
from pathlib import Path
import cv2

BENCHMARK_DIR = Path('data/OD_benchmark')
OUT_DIR = Path('assets/dataset')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLACEMENTS = ['random', 'grid', 'words', 'line']
N_SAMPLES = 2

for p in PLACEMENTS:
    img_dir = BENCHMARK_DIR / p / 'test' / 'images'
    samples = sorted(img_dir.glob('*.png'))[:N_SAMPLES]
    for idx, s in enumerate(samples):
        img = cv2.imread(str(s), cv2.IMREAD_GRAYSCALE)
        out_path = OUT_DIR / f"{p}_{idx + 1}.png"
        cv2.imwrite(str(out_path), img)
        print(f"Saved plain dataset image: {out_path}")

print("Done generating plain dataset visuals.")
