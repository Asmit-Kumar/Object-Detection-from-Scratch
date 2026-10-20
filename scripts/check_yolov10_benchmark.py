"""Prepare/check the exact YOLOv10-style benchmark artifacts without changing training."""
import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from models import load_model
from inference.pipeline import _decode_fcos_candidates

PROVENANCE = ROOT / 'benchmark/yolov10_provenance.json'
SIZES = ('n', 's', 'm')
PLACEMENTS = ('random', 'grid', 'words', 'line')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare():
    assert torch.cuda.is_available(), 'CUDA is required.'
    print('PyTorch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))
    if PROVENANCE.exists():
        manifest = load(PROVENANCE)
    else:
        protected = [ROOT / 'notebooks/detection.ipynb']
        protected += [p for p in (ROOT / 'benchmark').glob('*')
                      if p.suffix in ('.json', '.md') and not p.name.startswith(('yolov10', '08_'))]
        manifest = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'preserved_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in protected},
            'baseline_master_rows': [line for line in (ROOT / 'BENCHMARK.md').read_text(encoding='utf-8').splitlines()
                                     if line.startswith('| **') and 'objects**' not in line],
            'models': {},
        }
    for size in SIZES:
        source = ROOT / f'checkpoint/yolov10_{size}_28x14_best.pth'
        destination = ROOT / f'weights/{source.name}'
        if destination.exists():
            assert digest(source) == digest(destination), f'Existing weights differ: {destination}'
        else:
            shutil.copy2(source, destination)
        logs = list((ROOT / 'logs/YOLOv10StyleObjectDetector').glob(f'yolov10_{size}_*.json'))
        assert len(logs) == 1, 'Choose the matching training log explicitly when multiple runs exist.'
        log = load(logs[0])
        best = max(log['epochs'], key=lambda e: e['val_metric'])
        model = load_model('yolov10', str(destination), device='cuda', size=size, strict=True)
        exported = model.to_inference()
        assert exported.end_to_end and exported.inference_only
        assert not any(k.startswith('auxiliary_network.') for k in exported.state_dict())
        images = torch.rand(2, 1, 224, 224, device='cuda')
        with (torch.no_grad(),
              patch('torchvision.ops.batched_nms', side_effect=AssertionError('NMS called')),
              patch('torchvision.ops.nms', side_effect=AssertionError('NMS called'))):
            expected, actual = model(images), exported(images)
            for scale in (28, 14):
                for key in expected[scale]:
                    torch.testing.assert_close(expected[scale][key], actual[scale][key], rtol=0, atol=0)
            _decode_fcos_candidates(exported, actual, score_threshold=0.01)
        info = {
            'checkpoint': destination.relative_to(ROOT).as_posix(),
            'checkpoint_sha256': digest(destination),
            'training_params': sum(p.numel() for p in model.parameters()),
            'inference_params': sum(p.numel() for p in exported.parameters()),
            'log': logs[0].relative_to(ROOT).as_posix(), 'log_sha256': digest(logs[0]),
            'completed_epochs': len(log['epochs']), 'best_validation_epoch': best['epoch'],
            'best_validation_metric': log['best_val_metric'],
            'training_time_min': log['total_time_min'],
            'export_exact': True, 'nms_guard_passed': True,
        }
        assert info['completed_epochs'] == 60
        manifest['models'][size] = info
        print(size, json.dumps(info))
        del model, exported, expected, actual
        torch.cuda.empty_cache()
    PROVENANCE.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('PASS: strict checkpoint loading, exact main-only export, NMS guards, completed training.')


def check_artifacts():
    for name in ('check_yolov10_benchmark.py', 'run_single_stage_benchmark.py',
                 'run_density_sweep.py', 'generate_single_stage_visuals.py', 'render_benchmark_reports.py'):
        path = ROOT / 'scripts' / name
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    manifest = load(PROVENANCE)
    results = load(ROOT / 'benchmark/yolov10_results.json')
    density = load(ROOT / 'benchmark/yolov10_density_sweep_results.json')
    assert set(results) == set(density) == set(SIZES)
    for size, run in results.items():
        info = manifest['models'][size]
        assert run['device'] == 'cuda' and run['nms_free'] and run['nms_iou'] is None
        assert run['head'] == 'one-to-one' and run['max_detections'] == 300
        assert run['precision'] == 'float32' and run['match_iou'] == 0.50
        assert run['threshold_source'] == 'held-out tune set'
        assert len(run['threshold_sweep']) == 13
        assert max(run['threshold_sweep'], key=lambda r: r['e2e_f1'])['conf'] == run['opt_conf']
        assert density[size]['conf'] == run['opt_conf']
        assert digest(ROOT / run['checkpoint']) == run['checkpoint_sha256'] == info['checkpoint_sha256']
        assert digest(ROOT / f'checkpoint/yolov10_{size}_28x14_best.pth') == info['checkpoint_sha256']
        assert run['training_params'] == info['training_params']
        assert run['inference_params'] == info['inference_params']
        assert set(run['placements']) == set(PLACEMENTS)
        for placement, metrics in run['placements'].items():
            assert metrics['total_images'] == 2500
            tp, gt, pred = metrics['e2e_tp'], metrics['total_gt'], metrics['total_preds']
            assert round(2 * tp / (gt + pred), 4) == metrics['e2e_f1']
            assert tp <= metrics['det_tp'] <= min(gt, pred)
            for bucket in density[size]['layouts'][placement]:
                assert 0 <= bucket['recall'] <= 1
                assert round(bucket['tp'] / (bucket['tp'] + bucket['fn']), 4) == bucket['recall']
            for sample in (1, 2):
                path = ROOT / f'result/benchmark/yolov10/{size}/{placement}_{sample}.png'
                assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert digest(ROOT / info['log']) == info['log_sha256']
    for name, sha in manifest['preserved_sha256'].items():
        assert digest(ROOT / name) == sha, f'Unrelated artifact changed: {name}'
    master = (ROOT / 'BENCHMARK.md').read_text(encoding='utf-8')
    for row in manifest['baseline_master_rows']:
        assert row in master, f'Baseline row changed: {row}'
    report_path = ROOT / 'benchmark/08_yolov10_nms_free.md'
    report = report_path.read_text(encoding='utf-8')
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', report):
        assert (report_path.parent / target).exists(), target
    print('PASS: CUDA metrics, threshold selection, checkpoint identity, 30,000 images, density, '
          '24 visuals, report links, baseline and notebook preservation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-only', action='store_true')
    if parser.parse_args().artifacts_only:
        check_artifacts()
    else:
        prepare()
