"""
Runs the full benchmark across all 3 detector sizes (n, s, m) and all 4 placement layouts.
Prints a combined results table for use in BENCHMARK.md.
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))

from utils.pipeline import DetectionPipeline

SIZES = ['n', 's', 'm']
SIZE_LABELS = {'n': 'Nano (0.57M)', 's': 'Small (2.23M)', 'm': 'Medium (8.84M)'}


if __name__ == '__main__':
    all_results = {}

    for size in SIZES:
        print(f'\n{"="*55}')
        print(f'  Detector: {SIZE_LABELS[size]}')
        print(f'{"="*55}')
        pipeline = DetectionPipeline(detector_size=size, conf_threshold=0.70)
        metrics = pipeline.evaluate_benchmark_parallel(benchmark_dir='data/OD_benchmark')
        all_results[size] = metrics
        del pipeline   # free GPU memory before loading next model

    print('\n\n' + '='*80)
    print('  FULL RESULTS — All Detectors x All Layouts')
    print('='*80)
    header = f"{'Detector':<16} {'Layout':<10} {'Det P':>8} {'Det R':>8} {'Cls Acc':>9} {'E2E F1':>8} {'FPS':>7}"
    print(header)
    print('-'*80)
    for size, m_by_layout in all_results.items():
        for layout in ['random', 'grid', 'words', 'line']:
            m = m_by_layout.get(layout, {})
            if 'error' in m:
                print(f'{SIZE_LABELS[size]:<16} {layout:<10}  ERROR')
            else:
                p   = m['det_precision']
                r   = m['det_recall']
                ca  = m['classifier_acc']
                f1  = m['e2e_f1']
                fps = m['fps']
                print(f'{SIZE_LABELS[size]:<16} {layout:<10} {p:>8.4f} {r:>8.4f} {ca:>9.4f} {f1:>8.4f} {fps:>7.1f}')
    print('='*80)
