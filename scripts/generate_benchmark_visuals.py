"""
Generates benchmark visualization images for all 3 detector sizes × 4 placements × 2 samples.
Saves annotated detection outputs to result/benchmark/<model>/<placement>_<idx>.png
"""
import sys
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.pipeline import DetectionPipeline, PLACEMENTS, BENCHMARK_PATH

SIZES = ['n', 's', 'm']
SIZE_LABELS = {'n': 'Nano', 's': 'Small', 'm': 'Medium'}

# Representative clean benchmark sample images per layout
SAMPLE_MAP = {
    'random': ['000002.png', '000003.png'],
    'grid':   ['000001.png', '000003.png'],
    'words':  ['000001.png', '000003.png'],
    'line':   ['000002.png', '000003.png'],
}

if __name__ == '__main__':
    for size in SIZES:
        out_dir = Path('result/benchmark') / size
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n[{SIZE_LABELS[size]}] Loading pipeline...')
        pipeline = DetectionPipeline(detector_size=size, conf_threshold=0.70)

        for placement in PLACEMENTS:
            img_dir = Path(BENCHMARK_PATH) / placement / 'test' / 'images'
            sample_files = SAMPLE_MAP[placement]

            for idx, file_name in enumerate(sample_files):
                sample = img_dir / file_name
                detections = pipeline.predict_image(sample)
                img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)

                fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)

                for det in detections:
                    x, y, w, h = det.bbox
                    rect = patches.Rectangle(
                        (x, y), w, h,
                        linewidth=1.5, edgecolor='#00FFCC', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x, max(0, y - 3),
                        f"{det.char_label} {det.joint_conf:.2f}",
                        color='#FFE03A', fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.55, ec='none')
                    )

                n_det = len(detections)
                ax.set_title(
                    f'{SIZE_LABELS[size]} | {placement} | {n_det} detections',
                    fontsize=9, pad=5
                )
                ax.axis('off')
                plt.tight_layout(pad=0.4)

                out_path = out_dir / f'{placement}_{idx + 1}.png'
                plt.savefig(out_path, bbox_inches='tight', dpi=130)
                plt.close()
                print(f'  Saved: {out_path}')

        del pipeline  # free GPU memory before next model
        print(f'[{SIZE_LABELS[size]}] Done.')

    print('\nAll benchmark visuals generated successfully in result/benchmark/')
