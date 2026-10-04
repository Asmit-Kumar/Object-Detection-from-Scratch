"""
Benchmark Dataset Generation Script.

Generates a stratified 10,000-image evaluation dataset (2,500 images per layout placement:
'random', 'grid', 'words', 'line') saved under data/OD_benchmark/.
"""
import argparse, sys, time
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))

from generator.generator import DatasetGenerator

BENCHMARK_DIR = Path('data/OD_benchmark')
PLACEMENTS = ['random', 'grid', 'words', 'line']
SAMPLES_PER_TYPE = 2500

# Held-out threshold-tuning set (disjoint images from the 10k benchmark).
TUNE_DIR = BENCHMARK_DIR / 'tune'
TUNE_SAMPLES_PER_TYPE = 500


def generate_split(dest_dir: Path, samples_per_type: int) -> None:
    for p in PLACEMENTS:
        print(f"\n--- Generating {samples_per_type:,} samples for placement='{p}' -> {dest_dir / p} ---")
        gen = DatasetGenerator(
            dest_dir=dest_dir,
            test_len=samples_per_type,
            placement=p,
            subset='test',
            num_workers=4
        )
        gen.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the benchmark and threshold-tuning datasets.')
    parser.add_argument('--tune-only', action='store_true',
                        help=f'Only generate the tuning set in {TUNE_DIR}; never touches the 10k benchmark.')
    args = parser.parse_args()

    t0 = time.time()
    if not args.tune_only:
        print(f"Starting benchmark dataset generation in {BENCHMARK_DIR}...")
        generate_split(BENCHMARK_DIR, SAMPLES_PER_TYPE)
        total = SAMPLES_PER_TYPE * len(PLACEMENTS)
        print(f"\nSuccessfully generated {total:,} benchmark canvases across all 4 placements.")

    print(f"Starting tuning dataset generation in {TUNE_DIR}...")
    generate_split(TUNE_DIR, TUNE_SAMPLES_PER_TYPE)
    total = TUNE_SAMPLES_PER_TYPE * len(PLACEMENTS)
    print(f"\nSuccessfully generated {total:,} tuning canvases across all 4 placements.")

    elapsed = time.time() - t0
    print(f"Total generation time: {elapsed:.2f} seconds")
