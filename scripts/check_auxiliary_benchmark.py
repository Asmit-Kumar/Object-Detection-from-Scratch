"""Check trained auxiliary checkpoints and CUDA before benchmarking."""
import json
import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import load_model


def check_artifacts():
    results = json.loads(Path('benchmark/auxiliary_results.json').read_text())
    density = json.loads(Path('benchmark/auxiliary_density_sweep_results.json').read_text())
    assert set(results) == set(density) == {'n', 's'}
    for size, run in results.items():
        assert run['device'] == 'cuda'
        assert run['threshold_source'] == 'held-out tune set'
        assert len(run['threshold_sweep']) == 13
        assert max(run['threshold_sweep'], key=lambda r: r['e2e_f1'])['conf'] == run['opt_conf']
        assert density[size]['conf'] == run['opt_conf']
        assert hashlib.sha256(Path(run['checkpoint']).read_bytes()).hexdigest() == run['checkpoint_sha256']
        assert Path(run['checkpoint']).read_bytes() == Path(f'checkpoint/auxiliary_{size}_28x14_best.pth').read_bytes()
        for placement, metrics in run['placements'].items():
            assert metrics['total_images'] == 2500
            tp, gt, pred = metrics['e2e_tp'], metrics['total_gt'], metrics['total_preds']
            assert round(2 * tp / (gt + pred), 4) == metrics['e2e_f1']
            assert metrics['e2e_tp'] <= metrics['det_tp'] <= min(gt, pred)
            assert all(0 <= b['recall'] <= 1 for b in density[size]['layouts'][placement])
            for sample in (1, 2):
                image = Path(f'result/benchmark/auxiliary/{size}/{placement}_{sample}.png')
                assert image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    master = Path('BENCHMARK.md').read_text(encoding='utf-8')
    old_master = subprocess.check_output(['git', 'show', 'HEAD:BENCHMARK.md']).decode('utf-8')
    for row in old_master.splitlines():
        if row.startswith('| **') and 'objects**' not in row:
            assert row in master, f'Baseline row changed: {row}'
    report_path = Path('benchmark/07_auxiliary_yolov8_resnet.md')
    report = report_path.read_text(encoding='utf-8')
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', report):
        assert (report_path.parent / target).exists(), target
    for script in ('check_auxiliary_benchmark.py', 'run_single_stage_benchmark.py',
                   'run_density_sweep.py', 'generate_single_stage_visuals.py', 'render_benchmark_reports.py'):
        path = Path('scripts') / script
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    print('PASS: CUDA results, sweep selection, checkpoint hashes, 10,000 images/size, E2E counts, '
          'density thresholds, 16 visuals, report links, baseline preservation, and script syntax.')


def main():
    print('PyTorch:', torch.__version__)
    assert torch.cuda.is_available(), 'CUDA is required for this benchmark.'
    print('GPU:', torch.cuda.get_device_name(0))
    for size in ('n', 's'):
        path = Path(f'checkpoint/auxiliary_{size}_28x14_best.pth')
        model = load_model('auxiliary', str(path), device='cuda', size=size, strict=True)
        inference = model.to_inference()
        images = torch.rand(2, 1, 224, 224, device='cuda')
        with torch.no_grad():
            expected, actual = model(images), inference(images)
        for scale in (28, 14):
            for key in expected[scale]:
                torch.testing.assert_close(expected[scale][key], actual[scale][key], rtol=0, atol=0)
        assert not any(k.startswith('auxiliary_network.') for k in inference.state_dict())
        print(size, 'train params:', sum(p.numel() for p in model.parameters()),
              'inference params:', sum(p.numel() for p in inference.parameters()), 'export identical')
        for log_path in Path('logs/AuxiliaryObjectDetector').glob(f'auxiliary_{size}_*.json'):
            log = json.loads(log_path.read_text())
            print(log_path, 'epochs:', len(log['epochs']), 'summary:', log.get('summary'))
        del model, inference
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-only', action='store_true')
    if parser.parse_args().artifacts_only:
        check_artifacts()
    else:
        main()
