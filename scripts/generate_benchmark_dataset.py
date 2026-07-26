import sys, time
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))

from generator.generator import DatasetGenerator

BENCHMARK_DIR = Path('data/OD_benchmark')
PLACEMENTS = ['random', 'grid', 'words', 'line']
SAMPLES_PER_TYPE = 2500

print(f"Starting benchmark dataset generation in {BENCHMARK_DIR}...")
t0 = time.time()

for p in PLACEMENTS:
    print(f"\n--- Generating {SAMPLES_PER_TYPE:,} samples for placement='{p}' -> {BENCHMARK_DIR / p} ---")
    gen = DatasetGenerator(
        dest_dir=BENCHMARK_DIR,
        test_len=SAMPLES_PER_TYPE,
        placement=p,
        subset='test',
        num_workers=4
    )
    gen.generate()

elapsed = time.time() - t0
total = SAMPLES_PER_TYPE * len(PLACEMENTS)
print(f"\nSuccessfully generated {total:,} benchmark canvases across all 4 placements in {elapsed:.2f} seconds!")
