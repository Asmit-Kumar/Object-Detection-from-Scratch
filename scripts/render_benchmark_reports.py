"""
Render the generated sections of the benchmark reports from their raw results.

Every number inside a ``<!-- BEGIN GENERATED: name -->`` / ``<!-- END GENERATED: name -->``
block is read from its source here, never hand-copied. Everything outside the markers
(architecture write-ups, analysis prose) is left untouched.

Sources:
  - benchmark/fcos_results.json, fcos_cls_diagnostics.json, fcos_density_sweep_results.json
  - benchmark/yolov8_results.json, yolov8_density_sweep_results.json
  - benchmark/multi_anchor_results.json (+ .pre_tune_fix.json for the before/after table),
    grid_density_sweep_results.json                     (Grid K=1 Focal/BCE, Multi-Anchor K=3)
  - benchmark/single_stage_results.json, density_sweep_results.json   (Single-Stage, Two-Stage density)
  - benchmark/01_two_stage_resnet.md per-layout tables  (Two-Stage; no JSON is saved for it)
  - Params: FCOS by instantiation; other architectures from their checkpoint state_dicts.

Rendered files: BENCHMARK.md, benchmark/03_*.md through benchmark/06_*.md
"""
import json
import re
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))

import torch

from models import get_fcos_detector

B = Path('benchmark')
FCOS_RESULTS = B / 'fcos_results.json'
FCOS_DIAG = B / 'fcos_cls_diagnostics.json'
FCOS_DENSITY = B / 'fcos_density_sweep_results.json'
YOLOV8_RESULTS = B / 'yolov8_results.json'
YOLOV8_DENSITY = B / 'yolov8_density_sweep_results.json'
GRID_RESULTS = B / 'multi_anchor_results.json'
GRID_RESULTS_BEFORE = B / 'multi_anchor_results.pre_tune_fix.json'
GRID_DENSITY = B / 'grid_density_sweep_results.json'
SINGLE_STAGE_RESULTS = B / 'single_stage_results.json'
LEGACY_DENSITY = B / 'density_sweep_results.json'
TWO_STAGE_REPORT = B / '01_two_stage_resnet.md'
SINGLE_STAGE_REPORT = B / '02_single_stage_unified_resnet.md'
GRID_REPORT = B / '03_grid_based_spatial_resnet.md'
MULTI_ANCHOR_REPORT = B / '04_multi_anchor_spatial_resnet.md'
FCOS_REPORT = B / '05_fcos_anchor_free_resnet.md'
YOLOV8_REPORT = B / '06_yolov8_anchor_free_resnet.md'
MASTER = Path('BENCHMARK.md')
TWO_STAGE_SCRIPT = Path('scripts/run_full_benchmark.py')

PLACEMENTS = ['random', 'grid', 'words', 'line']
SIZES = ['n', 's', 'm']
SIZE_NAMES = {'n': 'Nano', 's': 'Small', 'm': 'Medium', 'l': 'Large'}
SIZE_ICONS = {'n': '🔬', 's': '⚡', 'm': '🎯', 'l': '🏆'}
LAYOUT_LABELS = {'random': '🎲 Random', 'grid': '📐 Grid', 'words': '📝 Words', 'line': '📏 Line'}
DENSITY_LABELS = {'random': '🎲 Random Layout', 'grid': '📐 Grid Layout', 'words': '📝 Words Layout', 'line': '📏 Line Layout'}

GRID_VARIANTS = {
    # key: (results-JSON name, checkpoint prefix, density-JSON name, visuals dir, label)
    'focal': ('Grid (K=1, Focal)', '1_grid_detector_', 'grid_focal_stage', 'grid_focal_stage', 'Grid Spatial ($K=1$, Focal)'),
    'bce': ('Grid (K=1, Original BCE)', 'grid_detector_', 'grid_bce_stage', 'grid_stage', 'Grid Spatial ($K=1$, BCE)'),
    'k3': ('Multi-Anchor (K=3)', '3_grid_detector_', 'multi_anchor_stage', '3_grid_stage', 'Multi-Anchor ($K=3$)'),
}

CONF_SRC_HELDOUT = 'held-out tune set'
CONF_SRC_FIXED = 'fixed'
CONF_SRC_HARDCODED = 'hard-coded, source unrecorded'
CONF_SRC_TEST_SWEEP = 'test-sweep override'


# ── Formatting ────────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def pct(x: float) -> str:
    return f'{x * 100:.2f}%'


def pp(x: float) -> str:
    return f'{x * 100:+.2f} pp'


def millions(n: int) -> str:
    return f'{n / 1e6:.2f}M'


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def first_existing(*paths: str) -> str:
    for p in paths:
        if Path(p).exists():
            return p
    raise FileNotFoundError(paths)


def checkpoint_params(path: str) -> int:
    """Trainable parameter count stored in a checkpoint (BatchNorm buffers excluded)."""
    raw = torch.load(path, map_location='cpu', weights_only=False)
    state = raw.get('model_state_dict', raw) if isinstance(raw, dict) else raw
    skip = ('running_mean', 'running_var', 'num_batches_tracked')
    return sum(v.numel() for k, v in state.items() if torch.is_tensor(v) and not k.endswith(skip))


# ── Data layer: one normalized row per (architecture, size) ───────────────────
# row = {label, size, params, conf, conf_src, placements{p: metrics}, avg{...}, loop_fps, pure_fps}

def _row(label, size, params, conf, conf_src, placements, loop_key, pure=None, extra=None):
    keys = ('det_precision', 'det_recall', 'classifier_acc', 'e2e_f1')
    row = {
        'label': label, 'size': size, 'params': params, 'conf': conf, 'conf_src': conf_src,
        'placements': {p: dict(placements[p], loop_fps=placements[p][loop_key]) for p in PLACEMENTS},
        'avg': {k: round(mean(placements[p][k] for p in PLACEMENTS), 4) for k in keys},
        'loop_fps': mean(placements[p][loop_key] for p in PLACEMENTS),
        'pure_fps': pure,
    }
    row.update(extra or {})
    return row


def two_stage_row(size: str) -> dict:
    text = TWO_STAGE_REPORT.read_text(encoding='utf-8')
    section = text.split(f'### {SIZE_ICONS[size]} {SIZE_NAMES[size]} —', 1)[1].split('###', 1)[0]
    placements = {}
    for p, label in LAYOUT_LABELS.items():
        m = re.search(rf'\| {re.escape(label)}\s*\|(.+)\|', section)
        cells = [c.strip().strip('*').replace('%', '').replace('FPS', '').strip() for c in m.group(1).split('|')]
        placements[p] = {'det_precision': float(cells[0]), 'det_recall': float(cells[1]),
                         'classifier_acc': float(cells[2]) / 100, 'e2e_f1': float(cells[3]), 'fps': float(cells[4])}
    det = first_existing(f'weights/detector_{size}_new_best.pth', f'checkpoint/detector_{size}_new_best.pth')
    cls = first_existing('weights/classifier_resent_bymerge_s_best.pth', 'checkpoint/classifier_resent_bymerge_s_best.pth')
    conf = float(re.search(r'DetectionPipeline\([^)]*conf_threshold=([\d.]+)', TWO_STAGE_SCRIPT.read_text()).group(1))
    return _row('Two-Stage Pipeline', size, checkpoint_params(det) + checkpoint_params(cls), conf, CONF_SRC_FIXED,
                placements, 'fps', extra={'report': TWO_STAGE_REPORT.name})


def single_stage_row(size: str) -> dict:
    placements = load_json(SINGLE_STAGE_RESULTS)[size]
    text = SINGLE_STAGE_REPORT.read_text(encoding='utf-8')
    conf = float(re.search(rf'\| \*\*{SIZE_NAMES[size]} \(`{size}`\)\*\* \|.*?\| `([\d.]+)`', text).group(1))
    ckpt = first_existing(f'weights/s_detector_{size}_best.pth', f'checkpoint/s_detector_{size}_best.pth')
    return _row('Single-Stage Unified', size, checkpoint_params(ckpt), conf, CONF_SRC_HARDCODED,
                placements, 'fps', extra={'report': SINGLE_STAGE_REPORT.name})


def grid_row(variant: str, size: str, before: bool = False) -> dict:
    name, prefix, _, _, label = GRID_VARIANTS[variant]
    data = load_json(GRID_RESULTS_BEFORE if before else GRID_RESULTS)[name][size]
    ckpt = first_existing(f'weights/{prefix}{size}_best.pth', f'checkpoint/{prefix}{size}_best.pth')
    pure = None if before else data['pure_inference']['median_img_s']
    src = '**tuned on test set**' if before else CONF_SRC_HELDOUT
    report = MULTI_ANCHOR_REPORT.name if variant == 'k3' else GRID_REPORT.name
    return _row(label, size, checkpoint_params(ckpt), data['opt_conf'], src, data['placements'],
                'fps' if before else 'eval_loop_fps', pure,
                extra={'report': report, 'architecture': data.get('architecture'), 'prefix': prefix})


def fcos_row(size: str, res: dict, params: dict) -> dict:
    d = res[size]
    return _row('FCOS Anchor-Free', size, params[size], d['opt_conf'], CONF_SRC_HELDOUT, d['placements'],
                'eval_loop_fps', d['pure_inference']['median_img_s'], extra={'report': FCOS_REPORT.name})


def yolov8_row(size: str, res: dict) -> dict:
    d = res[size]
    checkpoint = first_existing(
        f'weights/yolov8_{size}_28x14_best.pth',
        f'checkpoint/yolov8_{size}_28x14_best.pth',
    )
    return _row('YOLOv8 Anchor-Free', size, checkpoint_params(checkpoint), d['opt_conf'],
                d.get('threshold_source', CONF_SRC_HELDOUT),
                d['placements'], 'eval_loop_fps', d['pure_inference']['median_img_s'],
                extra={'report': YOLOV8_REPORT.name, 'checkpoint': checkpoint})


def fcos_params() -> dict:
    return {sz: sum(p.numel() for p in get_fcos_detector(size=sz).parameters()) for sz in SIZES}


# ── Shared blocks ─────────────────────────────────────────────────────────────

def layout_table(row: dict, heading: str) -> str:
    out = [heading, '',
           '| Layout | Det Precision | Det Recall | Classifier Acc | End-to-End F1 | Eval-Loop Throughput |',
           '|:---|:---:|:---:|:---:|:---:|:---:|']
    for p in PLACEMENTS:
        r = row['placements'][p]
        out.append(f'| {LAYOUT_LABELS[p]} | {r["det_precision"]:.4f} | {r["det_recall"]:.4f} | '
                   f'{pct(r["classifier_acc"])} | {r["e2e_f1"]:.4f} | {r["loop_fps"]:.1f} img/s |')
    a = row['avg']
    out.append(f'| **Average** | **{a["det_precision"]:.4f}** | **{a["det_recall"]:.4f}** | '
               f'**{pct(a["classifier_acc"])}** | **{a["e2e_f1"]:.4f}** | **{row["loop_fps"]:.1f} img/s** |')
    if row['pure_fps'] is not None:
        out += ['', f'Pure inference (`model(images)` only, batch 128): **{row["pure_fps"]:,.1f} img/s**.']
    return '\n'.join(out)


def summary_table(rows: list, first_col: str, with_src: bool = True, with_report: bool = False) -> str:
    head = f'| {first_col} | Size | Params | `conf` |' + (' `conf` Source |' if with_src else '') + \
           ' Avg Det Precision | Avg Det Recall | Avg Classifier Acc | Avg End-to-End F1 | Eval-Loop Throughput | Pure Inference |' + \
           (' Full Report |' if with_report else '')
    ncols = head.count('|') - 1
    out = [head, '|:---|' + ':---:|' * (ncols - 1)]
    prev = None
    for r in rows:
        if prev is not None and r['label'] != prev:
            out.append('|' + ' ─── |' * ncols)
        prev = r['label']
        a = r['avg']
        pure = f'{r["pure_fps"]:,.0f} img/s' if r['pure_fps'] is not None else '—'
        cells = [f'**{r["label"]}**', f'{SIZE_NAMES[r["size"]]} (`{r["size"]}`)', millions(r['params']), f'{r["conf"]:.2f}']
        if with_src:
            cells.append(r['conf_src'])
        cells += [f'{a["det_precision"]:.4f}', f'{a["det_recall"]:.4f}', pct(a['classifier_acc']), f'{a["e2e_f1"]:.4f}',
                  f'{r["loop_fps"]:.0f} img/s', pure]
        if with_report:
            cells.append(f'[`benchmark/{r["report"]}`](./benchmark/{r["report"]})')
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def density_table(columns: list, layout: str, title: str = None) -> str:
    """columns: [(header, buckets_list)] with identical bucket order."""
    first = columns[0][1]
    out = ([title] if title else []) + [
        '| Density Bucket ($n_{gt}$) | Test Images | ' + ' | '.join(h for h, _ in columns) + ' |',
        '|:---:|:---:|' + ':---:|' * len(columns),
    ]
    for i, b in enumerate(first):
        recalls = []
        for _, buckets in columns:
            assert buckets[i]['bucket'] == b['bucket'] and buckets[i]['n_images'] == b['n_images']
            recalls.append(buckets[i]['recall'])
        best = max(recalls)
        cells = [f'**{r * 100:.1f}%**' if r == best else f'{r * 100:.1f}%' for r in recalls]
        lo, hi = b['bucket'].split('-')
        out.append(f'| **{lo} – {hi} objects** | {b["n_images"]:,} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def eval_loop_note() -> str:
    return ('_Eval-loop throughput is wall-clock over the whole evaluation loop (PNG decoding with `num_workers=0`, '
            'target encoding in the collate function, host↔device copies, post-processing and scipy Hungarian '
            'matching); it is dominated by that overhead and varies run to run, so it is not a model-speed '
            'comparison. Pure inference times only `model(images)` on GPU-resident batches of 128 (median of '
            'repeated, synchronized runs after a ≥2 s warm-up)._')


def heldout_note(scope: str) -> str:
    return '\n'.join([
        f'Confidence thresholds for {scope} are tuned to maximize end-to-end F1 on a **held-out tuning set** '
        '(`data/OD_benchmark/tune/random`, 500 images from `scripts/generate_benchmark_dataset.py --tune-only`) that '
        'shares no images with the 10,000-image benchmark.',
    ])


def grid_tuning_update(variants: list) -> str:
    """Before/after table for the grid variants (test-set tuning → held-out tuning)."""
    out = [
        '> [!NOTE]',
        '> **Re-evaluated with held-out threshold tuning.** These models were originally tuned on '
        '`data/OD_benchmark/random`, i.e. on the test set they report. ' + heldout_note('these models')
        + ' All tables in this report are generated from `benchmark/multi_anchor_results.json` by '
        '`scripts/render_benchmark_reports.py`. The checkpoints use the original $14 \\times 14$ architecture, '
        'loaded from the frozen snapshot `scripts/legacy_models/grid_detector_14x14.py`; re-scoring at the old '
        'thresholds reproduces the original numbers exactly.',
        '',
    ]
    if not GRID_RESULTS_BEFORE.exists():
        return '\n'.join(out).rstrip()
    out += ['| Variant | Size | `conf` (test-tuned → held-out) | Avg Det P | Avg Det R | Avg Cls Acc | Avg E2E F1 |',
            '|:---|:---:|:---:|:---:|:---:|:---:|:---:|']
    for v in variants:
        for sz in SIZES:
            o, n = grid_row(v, sz, before=True), grid_row(v, sz)
            arrow = lambda k, f: f'{f(o["avg"][k])} → {f(n["avg"][k])}' if o['avg'][k] != n['avg'][k] else f(n['avg'][k])
            conf = f'{o["conf"]:.2f} → **{n["conf"]:.2f}**' if o['conf'] != n['conf'] else f'{n["conf"]:.2f}'
            out.append(f'| {n["label"]} | {SIZE_NAMES[sz]} | {conf} | {arrow("det_precision", lambda x: f"{x:.4f}")} | '
                       f'{arrow("det_recall", lambda x: f"{x:.4f}")} | {arrow("classifier_acc", pct)} | '
                       f'{arrow("e2e_f1", lambda x: f"{x:.4f}")} |')
    return '\n'.join(out)


def grid_params_table(variants: list) -> str:
    out = ['| Size Preset | Variant | Anchors ($K$) | Total Parameters | `conf` | Checkpoint Weight File | Architecture |',
           '|:---|:---|:---:|:---:|:---:|:---|:---|']
    for sz in SIZES:
        for v in variants:
            r = grid_row(v, sz)
            data = load_json(GRID_RESULTS)[GRID_VARIANTS[v][0]][sz]
            k = 3 if v == 'k3' else 1
            out.append(f'| **{SIZE_NAMES[sz]} (`{sz}`)** | {r["label"]} | {k} | **{r["params"]:,}** ({millions(r["params"])}) | '
                       f'`{r["conf"]:.2f}` | `weights/{r["prefix"]}{sz}_best.pth` | {data["architecture"]} |')
    out += ['', '_Parameter counts are read from the checkpoint state_dicts (the architecture these models were trained '
                'with). The grid benchmark has no NMS: every cell above `conf` is a prediction, matched to ground truth '
                'with Hungarian matching at IoU ≥ 0.50._']
    return '\n'.join(out)


def grid_visuals(variant: str) -> str:
    name, prefix, _, vis_dir, label = GRID_VARIANTS[variant]
    out = ['Sample detections are rendered by `scripts/generate_single_stage_visuals.py --detector grid` at each '
           'model\'s held-out-tuned `conf` (with NMS IoU 0.35 and a 0.30 class-confidence gate for display).', '']
    for sz in SIZES:
        r = grid_row(variant, sz)
        base = f'../result/benchmark/{vis_dir}/{sz}'
        out += [f'### {SIZE_ICONS[sz]} {SIZE_NAMES[sz]} ({millions(r["params"])}) — `conf = {r["conf"]:.2f}`', '',
                '| random | random | grid | grid |', '|:---:|:---:|:---:|:---:|',
                f'| ![]({base}/random_1.png) | ![]({base}/random_2.png) | ![]({base}/grid_1.png) | ![]({base}/grid_2.png) |',
                '| **words** | **words** | **line** | **line** |',
                f'| ![]({base}/words_1.png) | ![]({base}/words_2.png) | ![]({base}/line_1.png) | ![]({base}/line_2.png) |', '']
    return '\n'.join(out).rstrip()


def grid_density(variant: str, size: str) -> dict:
    node = load_json(GRID_DENSITY)[GRID_VARIANTS[variant][2]][size]
    return node


# ── Report 03: Grid K=1 (Focal + BCE) ─────────────────────────────────────────

def render_report_03() -> dict:
    blocks = {
        'update': grid_tuning_update(['focal', 'bce']),
        'params': grid_params_table(['focal', 'bce']),
    }
    for v, title in (('focal', 'Focal'), ('bce', 'BCE')):
        blocks[f'results_{v}'] = '\n\n'.join(
            layout_table(r, f'### {SIZE_ICONS[sz]} {title} {SIZE_NAMES[sz]} — {millions(r["params"])} Params (`conf = {r["conf"]:.2f}`)')
            for sz in SIZES for r in [grid_row(v, sz)])
    blocks['summary'] = summary_table([grid_row(v, sz) for v in ('focal', 'bce') for sz in SIZES], 'Model Variant') \
        + '\n\n' + eval_loop_note()
    dens = []
    for v in ('focal', 'bce'):
        cols = [(f'{"Focal" if v == "focal" else "BCE"} {SIZE_NAMES[sz]} (`conf` {grid_density(v, sz)["conf"]:.2f})',
                 grid_density(v, sz)['layouts']['random']) for sz in SIZES]
        dens.append(density_table(cols, 'random', f'#### {"Focal Loss" if v == "focal" else "Original BCE"}'))
    blocks['density'] = ('Recall per density bucket on the `random` layout (`benchmark/grid_density_sweep_results.json`, '
                         'evaluated with `utils.trainer.evaluate_density_sweep` at each model\'s held-out-tuned `conf`):\n\n'
                         + '\n\n'.join(dens))
    blocks['visuals'] = grid_visuals('bce')
    return blocks


# ── Report 04: Multi-Anchor K=3 ───────────────────────────────────────────────

def render_report_04() -> dict:
    blocks = {
        'update': grid_tuning_update(['k3']),
        'params': grid_params_table(['k3']),
        'results': '\n\n'.join(
            layout_table(r, f'### {SIZE_ICONS[sz]} Multi-Anchor {SIZE_NAMES[sz]} — {millions(r["params"])} Params (`conf = {r["conf"]:.2f}`)')
            for sz in SIZES for r in [grid_row('k3', sz)]),
        'comparison': summary_table([r for sz in SIZES for r in (grid_row('bce', sz), grid_row('k3', sz))],
                                    'Model Variant') + '\n\n' + eval_loop_note(),
    }
    k1, k3 = grid_density('bce', 'm'), grid_density('k3', 'm')
    blocks['density'] = (f'Recall per density bucket for the **Medium** models across all 4 layouts '
                         f'(`benchmark/grid_density_sweep_results.json`; $K=1$ BCE at `conf` {k1["conf"]:.2f}, '
                         f'$K=3$ at `conf` {k3["conf"]:.2f}, IoU 0.50):\n\n' + '\n\n'.join(
        density_table([(f'Grid Medium ($K=1$) — {millions(grid_row("bce", "m")["params"])}', k1['layouts'][p]),
                       (f'Multi-Anchor Medium ($K=3$) — {millions(grid_row("k3", "m")["params"])}', k3['layouts'][p])],
                      p, f'### {DENSITY_LABELS[p]}') for p in PLACEMENTS))
    blocks['visuals'] = grid_visuals('k3')
    return blocks


# ── Report 05: FCOS ───────────────────────────────────────────────────────────

def render_report_05(res: dict, params: dict) -> dict:
    diag = load_json(FCOS_DIAG)
    rows = {sz: fcos_row(sz, res, params) for sz in SIZES}

    def per_size(key, fmt):
        return ', '.join(f'**{fmt(rows[sz]["avg"][key])}** ({SIZE_NAMES[sz]})' for sz in SIZES)

    recalls = {sz: [res[sz]['placements'][p]['det_recall'] for p in PLACEMENTS] for sz in SIZES}
    lo_sz, lo_p = min(((sz, p) for sz in SIZES for p in PLACEMENTS), key=lambda k: res[k[0]]['placements'][k[1]]['det_recall'])
    summary = [
        f'- **Detection Precision**: {per_size("det_precision", pct)}',
        f'- **Detection Recall**: {per_size("det_recall", pct)}',
        f'- **Classifier Accuracy** (on detected objects): {per_size("classifier_acc", pct)}',
        f'- **End-to-End F1**: {per_size("e2e_f1", lambda v: f"{v:.4f}")}',
        '- **Pure Inference Throughput** (`model(images)` only, batch 128, CUDA): '
        + ', '.join(f'**{rows[sz]["pure_fps"]:,.0f} img/s** ({SIZE_NAMES[sz]})' for sz in SIZES),
        '',
        'Per-layout recall ranges across `random`, `grid`, `words`, `line`: '
        + '; '.join(f'{SIZE_NAMES[sz]} {pct(min(recalls[sz]))}–{pct(max(recalls[sz]))}' for sz in SIZES)
        + f'. The lowest cell is {SIZE_NAMES[lo_sz]} on `{lo_p}` ({pct(res[lo_sz]["placements"][lo_p]["det_recall"])}).',
    ]
    blocks = {'summary': '\n'.join('> ' + l if l else '>' for l in summary)}

    blocks['tuning'] = '\n'.join([
        heldout_note('FCOS') + ' Selected thresholds: '
        + ', '.join(f'{SIZE_NAMES[sz]} `{res[sz]["opt_conf"]:.2f}`' for sz in SIZES) + '.',
        '',
        '> [!WARNING]',
        '> Grid ($K=1$) and Multi-Anchor ($K=3$) were originally tuned on `data/OD_benchmark/random` (the test set) and '
        'have since been re-evaluated with the same held-out protocol (Reports 03–04). Single-Stage (Report 02) '
        'thresholds are hard-coded values with no recorded tuning source, and Two-Stage (Report 01) uses a fixed '
        '`0.70`; those two rows may be optimistically biased relative to the others.',
    ])

    out = ['| Size Preset | Model Name | Total Parameters | FPN Strides | Optimal `conf` | Checkpoint Weight File |',
           '|:---|:---|:---:|:---:|:---:|:---|']
    for sz in SIZES:
        out.append(f'| **{SIZE_NAMES[sz]} (`{sz}`)** | `FCOSObjectDetectorResNet` (v4 {SIZE_NAMES[sz]}) | '
                   f'**{params[sz]:,}** ({millions(params[sz])}) | 8, 16 | `{res[sz]["opt_conf"]:.2f}` | '
                   f'`weights/fcos_{sz}_28x14_best.pth` |')
    out += ['', '_Parameter counts are computed by instantiating `FCOSObjectDetectorResNet` for each size._']
    blocks['params'] = '\n'.join(out)

    blocks['results'] = '\n\n'.join(
        layout_table(rows[sz], f'### {SIZE_ICONS[sz]} FCOS {SIZE_NAMES[sz]} — {millions(params[sz])} Params (`conf = {res[sz]["opt_conf"]:.2f}`)')
        for sz in SIZES) + ('\n\n_Classifier Acc = correct class / matched detections, so it is graded only on objects '
                            'each model detected; see the diagnostics section for a like-for-like comparison._')

    out = ['| Size | Pure Inference (median) | Pure Inference (min–max over repeats) | Eval-Loop Throughput (avg of 4 layouts) | Eval-Loop range across layouts |',
           '|:---|:---:|:---:|:---:|:---:|']
    for sz in SIZES:
        reps = res[sz]['pure_inference']['repeat_img_s']
        loop = [res[sz]['placements'][p]['eval_loop_fps'] for p in PLACEMENTS]
        out.append(f'| **{SIZE_NAMES[sz]}** | **{rows[sz]["pure_fps"]:,.1f} img/s** | {min(reps):,.1f}–{max(reps):,.1f} | '
                   f'{rows[sz]["loop_fps"]:.1f} img/s | {min(loop):.1f}–{max(loop):.1f} |')
    pi = res[SIZES[0]]['pure_inference']
    out += ['', f'- **Pure inference** times only `model(images)` on GPU-resident batches of {pi["batch_size"]} (real images, '
                f'fp32): ≥{pi["warmup_sec"]:.0f} s of warm-up, then {pi["repeats"]} repeats × {pi["iters_per_repeat"]} forward '
                'passes with `torch.cuda.synchronize()` around each repeat; the median is reported.',
            '- **Eval-loop throughput** is wall-clock over the whole evaluation loop: PNG decoding (`num_workers=0`), FCOS '
            'target generation in the collate function, host↔device copies, decoding + NMS, and scipy Hungarian matching. '
            'It is dominated by that overhead, not by the model, and varies run to run — do not use it to compare model speed.']
    blocks['throughput'] = '\n'.join(out)

    comp = [two_stage_row('m'), single_stage_row('l'), grid_row('bce', 'm'), grid_row('k3', 'm')] + [rows[sz] for sz in SIZES]
    blocks['comparison'] = summary_table(comp, 'Model Architecture') + '\n\n' + (
        '_Sources: Two-Stage — `benchmark/01_two_stage_resnet.md` (per-layout table; no JSON is saved), conf from '
        '`scripts/run_full_benchmark.py`; Single-Stage — `benchmark/single_stage_results.json`, conf from Report 02; '
        'Grid/Multi-Anchor — `benchmark/multi_anchor_results.json`; FCOS — `benchmark/fcos_results.json`. Averages are '
        'unweighted means over the 4 layouts. Non-FCOS params are counted from checkpoint state_dicts. Eval-loop '
        'throughput comes from a different evaluation loop per architecture._')

    dens = load_json(FCOS_DENSITY)
    stale = [sz for sz in SIZES if abs(dens[sz]['conf'] - res[sz]['opt_conf']) > 1e-9]
    out = ['Thresholds used for this sweep (`benchmark/fcos_density_sweep_results.json`): '
           + ', '.join(f'{SIZE_NAMES[sz]} `{dens[sz]["conf"]:.2f}`' for sz in SIZES) + '.']
    if stale:
        out += ['', '> [!CAUTION]', '> The sweep was run at thresholds that differ from the current `opt_conf` for: '
                + ', '.join(f'{SIZE_NAMES[sz]} (sweep `{dens[sz]["conf"]:.2f}` vs. `{res[sz]["opt_conf"]:.2f}`)' for sz in stale)
                + '. Rerun `scripts/run_density_sweep.py --detector fcos`.']
    for p in PLACEMENTS:
        out += ['', density_table([(f'FCOS {SIZE_NAMES[sz]} ({millions(params[sz])})', dens[sz]['layouts'][p]) for sz in SIZES],
                                  p, f'### {DENSITY_LABELS[p]}')]
    blocks['density'] = '\n'.join(out)
    blocks['cls_diagnostics'] = render_cls_diagnostics(diag)
    return blocks


def render_cls_diagnostics(diag: dict) -> str:
    opt, val = diag['opt_conf'], diag['val_conf']
    out = [
        'Classifier accuracy above is `correct / detected`, so a model with higher recall is graded on more hard objects. '
        '`scripts/fcos_cls_diagnostics.py` records the outcome for every ground-truth object (key: placement, image, GT '
        'index) and re-scores each model on the **intersection** of objects that all three sizes detect. Output: '
        '`benchmark/fcos_cls_diagnostics.json`.',
        '',
        '### Selection effect: own detections vs. common intersection',
        '',
        '| Setting | Objects | ' + ' | '.join(SIZE_NAMES[sz] for sz in SIZES) + ' |',
        '|:---|:---:|' + ':---:|' * len(SIZES),
    ]
    for label, d in (('opt_conf', opt), ('val_conf', val)):
        conf_desc = ', '.join(f'{d["per_size"][sz]["conf"]:.2f}' for sz in SIZES)
        det_n = ' / '.join(f'{sum(pl["det_tp"] for pl in d["per_size"][sz]["placements"].values()):,}' for sz in SIZES)
        out.append(f'| Own detections, mean of layouts (`{label}` = {conf_desc}) | {det_n} | '
                   + ' | '.join(pct(d['per_size'][sz]['cls_acc_detected_mean']) for sz in SIZES) + ' |')
        i = d['intersection']
        out.append(f'| Intersection, pooled (`{label}`) | {i["n_objects"]:,} of {i["n_total_gt"]:,} | '
                   + ' | '.join(pct(i['cls_acc'][sz]['pooled']) for sz in SIZES) + ' |')
    out += ['']

    i = opt['intersection']
    extra = opt['detected_by_size_missed_by_nano']
    own_spread = opt['per_size']['n']['cls_acc_detected_mean'] - opt['per_size']['m']['cls_acc_detected_mean']
    inter_spread = i['cls_acc']['n']['pooled'] - i['cls_acc']['m']['pooled']
    mc = i['pairwise_mcnemar']
    out += ['| Paired test on intersection (`opt_conf`) | A right, B wrong | B right, A wrong | McNemar exact p |',
            '|:---|:---:|:---:|:---:|']
    for pair, v in mc.items():
        a, b = pair.split('_vs_')
        out.append(f'| {SIZE_NAMES[a]} (A) vs. {SIZE_NAMES[b]} (B) | {v[f"{a}_correct_{b}_wrong"]:,} | '
                   f'{v[f"{b}_correct_{a}_wrong"]:,} | {v["p_value"]:.3g} |')
    out += [
        '',
        f'Objects that the larger models detect but Nano misses are hard: Small classifies {pct(extra["s"]["cls_acc"])} of '
        f'its {extra["s"]["n_objects"]:,} such objects correctly and Medium {pct(extra["m"]["cls_acc"])} of '
        f'{extra["m"]["n_objects"]:,}. Removing them shrinks the Nano–Medium accuracy spread from {pp(own_spread)} '
        f'(own detections) to {pp(inter_spread)} (intersection).',
        '',
        '### Benchmark vs. validation accuracy',
        '',
        'Validation `val_cls_acc` is taken at the best-checkpoint epoch (last `is_best: true`) of each training log. The '
        'validation split comes from `data/OD/train` and so reuses the EMNIST-train glyph pool seen in training; the '
        'benchmark uses EMNIST-test glyphs. Benchmark accuracy is weighted by the training placement mix ('
        + ', '.join(f'{p} {w:.2f}' for p, w in diag['protocol']['train_mix_weights'].items()) + ').',
        '',
        '| Size | Best Epoch | Val Cls Acc | Benchmark (train-mix), `conf` 0.50 | Gap | Benchmark (train-mix), `opt_conf` | Gap | Intersection (train-mix), `conf` 0.50 | Gap |',
        '|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|',
    ]
    for sz in SIZES:
        v = diag['val'][sz]
        out.append(f'| **{SIZE_NAMES[sz]}** | {v["best_epoch"]} | {pct(v["val_cls_acc"])} | '
                   f'{pct(val["per_size"][sz]["cls_acc_detected_train_mix"])} | {pp(val["gap_train_mix_vs_val"][sz])} | '
                   f'{pct(opt["per_size"][sz]["cls_acc_detected_train_mix"])} | {pp(opt["gap_train_mix_vs_val"][sz])} | '
                   f'{pct(val["intersection"]["cls_acc"][sz]["train_mix"])} | {pp(val["gap_intersection_train_mix_vs_val"][sz])} |')
    gaps, inter_gaps = val['gap_train_mix_vs_val'], val['gap_intersection_train_mix_vs_val']
    grows = gaps['n'] > gaps['s'] > gaps['m'] and inter_gaps['n'] > inter_gaps['s'] > inter_gaps['m']
    out += [
        '', '### Conclusion', '',
        f'- **Mostly a selection effect.** About {1 - inter_spread / own_spread:.0%} of the Nano–Medium classifier-accuracy '
        'spread on own detections disappears on the common intersection. Larger models are graded on extra hard objects '
        'that Nano never detects.',
        f'- **A small real difference survives.** On identical objects Nano is still {pp(inter_spread)} ahead of Medium '
        f'(McNemar p = {mc["n_vs_m"]["p_value"]:.2g}) and ahead of Small (p = {mc["n_vs_s"]["p_value"]:.2g}); Small vs. '
        f'Medium is not significant (p = {mc["s_vs_m"]["p_value"]:.2g}).',
        (f'- **The val→benchmark gap grows with model size** ({", ".join(f"{SIZE_NAMES[sz]} {pp(gaps[sz])}" for sz in SIZES)} '
         f'at `conf` 0.50; {", ".join(pp(inter_gaps[sz]) for sz in SIZES)} on the intersection), consistent with mild '
         'memorization of EMNIST-train glyphs by the larger models. Val accuracy is itself scored on each model\'s own '
         f'detections (val recall {", ".join(f"{SIZE_NAMES[sz]} {pct(diag["val"][sz]["val_recall"])}" for sz in SIZES)}), '
         'so the larger models\' val numbers already include more hard objects, which if anything understates this trend. '
         if grows else
         f'- **The val→benchmark gap does not grow monotonically with size** '
         f'({", ".join(f"{SIZE_NAMES[sz]} {pp(gaps[sz])}" for sz in SIZES)} at `conf` 0.50). ')
        + 'This is suggestive, not conclusive: validation also differs in matching (greedy vs. Hungarian) and precision '
          '(bf16 vs. fp32), and a direct check would require scoring on EMNIST-test glyphs in the training placement mix.',
    ]
    return '\n'.join(out)


# ── Report 06: YOLOv8 ─────────────────────────────────────────────────────────

def render_yolov8_report(res: dict) -> dict:
    rows = {sz: yolov8_row(sz, res) for sz in SIZES}
    device = str(res[SIZES[0]].get('device', 'unknown')).upper()

    def per_size(key, fmt):
        return ', '.join(f'**{fmt(rows[sz]["avg"][key])}** ({SIZE_NAMES[sz]})' for sz in SIZES)

    recalls = {sz: [res[sz]['placements'][p]['det_recall'] for p in PLACEMENTS] for sz in SIZES}
    summary = [
        f'- **Detection Precision**: {per_size("det_precision", pct)}',
        f'- **Detection Recall**: {per_size("det_recall", pct)}',
        f'- **Classifier Accuracy**: {per_size("classifier_acc", pct)}',
        f'- **End-to-End F1**: {per_size("e2e_f1", lambda v: f"{v:.4f}")}',
        f'- **Pure Inference Throughput** (`model(images)` only, batch 128, {device}): '
        + ', '.join(f'**{rows[sz]["pure_fps"]:,.0f} img/s** ({SIZE_NAMES[sz]})' for sz in SIZES),
        '',
        'Per-layout detection recall ranges across `random`, `grid`, `words`, and `line`: '
        + '; '.join(f'{SIZE_NAMES[sz]} {pct(min(recalls[sz]))}–{pct(max(recalls[sz]))}' for sz in SIZES) + '.',
    ]
    blocks = {'summary': '\n'.join(summary)}
    threshold_bits = ', '.join(
        f'{SIZE_NAMES[sz]} `{res[sz]["opt_conf"]:.2f}` ({res[sz].get("threshold_source", CONF_SRC_HELDOUT)})'
        for sz in SIZES)
    sources = {res[sz].get('threshold_source', CONF_SRC_HELDOUT) for sz in SIZES}
    if sources == {CONF_SRC_HELDOUT}:
        blocks['tuning'] = heldout_note('YOLOv8') + ' Selected thresholds: ' + threshold_bits + '.'
    else:
        by_source = {}
        for sz in SIZES:
            by_source.setdefault(res[sz].get('threshold_source', CONF_SRC_HELDOUT), []).append(SIZE_NAMES[sz])
        provenance = []
        for source, names in by_source.items():
            label = ' and '.join(names)
            verb = 'use' if len(names) > 1 else 'uses'
            if source == CONF_SRC_HELDOUT:
                provenance.append(f'{label} {verb} the held-out tuning set (`data/OD_benchmark/tune/random`)')
            elif source == CONF_SRC_TEST_SWEEP:
                provenance.append(f'{label} {verb} an explicit benchmark test-sweep override')
            else:
                provenance.append(f'{label} {verb} {source}')
        blocks['tuning'] = (
            'YOLOv8 confidence thresholds are recorded per size. ' + '; '.join(provenance) + '. '
            f'Selected thresholds: {threshold_bits}.'
        )

    out = ['| Size Preset | Model Name | Total Parameters | Prediction Scales | Optimal `conf` | Checkpoint File |',
           '|:---|:---|:---:|:---:|:---:|:---|']
    for sz in SIZES:
        row = rows[sz]
        out.append(f'| **{SIZE_NAMES[sz]} (`{sz}`)** | `YOLOv8ObjectDetector` | **{row["params"]:,}** '
                   f'({millions(row["params"])}) | 28 × 28, 14 × 14 | `{row["conf"]:.2f}` | `{row["checkpoint"]}` |')
    out += ['', '_Parameter counts are read from each trained checkpoint state_dict, excluding BatchNorm buffers._']
    blocks['params'] = '\n'.join(out)

    blocks['results'] = '\n\n'.join(
        layout_table(rows[sz], f'### {SIZE_ICONS[sz]} YOLOv8 {SIZE_NAMES[sz]} — '
                     f'{millions(rows[sz]["params"])} Params (`conf = {res[sz]["opt_conf"]:.2f}`)')
        for sz in SIZES)

    out = ['| Size | Pure Inference (median) | Pure Inference (min–max over repeats) | '
           'Eval-Loop Throughput (avg of 4 layouts) | Eval-Loop range across layouts |',
           '|:---|:---:|:---:|:---:|:---:|']
    for sz in SIZES:
        repeats = res[sz]['pure_inference']['repeat_img_s']
        loop = [res[sz]['placements'][p]['eval_loop_fps'] for p in PLACEMENTS]
        out.append(f'| **{SIZE_NAMES[sz]}** | **{rows[sz]["pure_fps"]:,.1f} img/s** | '
                   f'{min(repeats):,.1f}–{max(repeats):,.1f} | {rows[sz]["loop_fps"]:.1f} img/s | '
                   f'{min(loop):.1f}–{max(loop):.1f} |')
    pure = res[SIZES[0]]['pure_inference']
    out += ['', f'- Pure inference times only `model(images)` on {device}-resident batches of {pure["batch_size"]}; '
                f'it uses at least {pure["warmup_sec"]:.0f} seconds of warm-up and {pure["repeats"]} repeats × '
                f'{pure["iters_per_repeat"]} synchronized forward passes. The median is reported.',
            '- Eval-loop throughput includes image loading, target collation, decoding, NMS, device transfers, and '
            'Hungarian matching; it is context for the full evaluation path rather than a model-only speed measure.']
    blocks['throughput'] = '\n'.join(out)

    density = load_json(YOLOV8_DENSITY)
    stale = [sz for sz in SIZES if abs(density[sz]['conf'] - res[sz]['opt_conf']) > 1e-9]
    out = ['Density recall uses the thresholds recorded in `benchmark/yolov8_results.json` and IoU 0.50.',
           'Threshold provenance: ' + ', '.join(
               f'{SIZE_NAMES[sz]} {res[sz].get("threshold_source", CONF_SRC_HELDOUT)}' for sz in SIZES) + '.']
    if stale:
        out += ['', '> [!CAUTION]', '> The density sweep thresholds differ from the latest tuned thresholds for: '
                + ', '.join(f'{SIZE_NAMES[sz]} ({density[sz]["conf"]:.2f} vs. {res[sz]["opt_conf"]:.2f})' for sz in stale)
                + '. Rerun `scripts/run_density_sweep.py --detector yolov8`.']
    for placement in PLACEMENTS:
        cols = [(f'YOLOv8 {SIZE_NAMES[sz]} ({millions(rows[sz]["params"])})', density[sz]['layouts'][placement])
                for sz in SIZES]
        out += ['', density_table(cols, placement, f'### {DENSITY_LABELS[placement]}')]
    blocks['density'] = '\n'.join(out)

    out = ['Visuals use each model\'s recorded threshold and the shared ltrb decoder with class-wise NMS.', '']
    for sz in SIZES:
        base = f'../result/benchmark/yolov8/{sz}'
        out += [f'### {SIZE_ICONS[sz]} {SIZE_NAMES[sz]} ({millions(rows[sz]["params"])}) — `conf = {rows[sz]["conf"]:.2f}`', '',
                '| random | random | grid | grid |', '|:---:|:---:|:---:|:---:|',
                f'| ![]({base}/random_1.png) | ![]({base}/random_2.png) | ![]({base}/grid_1.png) | ![]({base}/grid_2.png) |',
                '| **words** | **words** | **line** | **line** |',
                f'| ![]({base}/words_1.png) | ![]({base}/words_2.png) | ![]({base}/line_1.png) | ![]({base}/line_2.png) |', '']
    blocks['visuals'] = '\n'.join(out).rstrip()
    return blocks


# ── BENCHMARK.md (master index) ───────────────────────────────────────────────

def render_master(res: dict, params: dict, yolov8_res: dict) -> dict:
    rows = ([two_stage_row(sz) for sz in SIZES] + [single_stage_row(sz) for sz in ('n', 's', 'm', 'l')]
            + [grid_row(v, sz) for v in ('focal', 'bce', 'k3') for sz in SIZES]
            + [fcos_row(sz, res, params) for sz in SIZES]
            + [yolov8_row(sz, yolov8_res) for sz in SIZES])
    blocks = {'master_table': summary_table(rows, 'Architecture Paradigm', with_report=True) + '\n\n' + (
        '_`conf` source: **held-out tune set** = tuned on `data/OD_benchmark/tune/random`, disjoint from the benchmark; '
        '**test-sweep override** = selected directly from the benchmark sweep; **fixed** / **hard-coded** = not tuned '
        'on held-out data (provenance of the Single-Stage values is not recorded). '
        'Averages are unweighted means over the 4 layouts. FCOS params are counted from model instantiation; other '
        'parameter counts come from checkpoint state_dicts._')}

    out = ['| Paradigm | Size | Params | Pure Inference (`model(images)`, batch 128) | Eval-Loop Throughput |',
           '|:---|:---:|:---:|:---:|:---:|']
    prev = None
    for r in rows:
        if prev is not None and r['label'] != prev:
            out.append('|' + ' ─── |' * 5)
        prev = r['label']
        pure = f'**{r["pure_fps"]:,.0f} img/s**' if r['pure_fps'] is not None else 'not measured'
        out.append(f'| **{r["label"]}** | {SIZE_NAMES[r["size"]]} (`{r["size"]}`) | {millions(r["params"])} | {pure} | {r["loop_fps"]:.0f} img/s |')
    blocks['throughput'] = '\n'.join(out) + '\n\n' + eval_loop_note() + (
        ' Two-Stage and Single-Stage checkpoints were not re-timed with the pure-inference protocol.')

    legacy = load_json(LEGACY_DENSITY)
    fd = load_json(FCOS_DENSITY)
    yd = load_json(YOLOV8_DENSITY)
    cols = [('Two-Stage Medium', legacy['two_stage']['m']), ('Single-Stage Medium', legacy['single_stage']['m']),
            ('Grid Medium ($K=1$, BCE)', grid_density('bce', 'm')['layouts']['random']),
            ('Multi-Anchor Medium ($K=3$)', grid_density('k3', 'm')['layouts']['random'])] + \
           [(f'FCOS {SIZE_NAMES[sz]}', fd[sz]['layouts']['random']) for sz in SIZES] + \
           [(f'YOLOv8 {SIZE_NAMES[sz]}', yd[sz]['layouts']['random']) for sz in SIZES]
    struct = []
    for p in ('grid', 'words', 'line'):
        struct.append(f'| `{p}` | ' + ' | '.join(
            pct(mean(b['recall'] for b in grid_density(v, 'm')['layouts'][p])) for v in ('bce', 'k3'))
            + ' | ' + ' | '.join(pct(res[sz]['placements'][p]['det_recall']) for sz in SIZES)
            + ' | ' + ' | '.join(pct(yolov8_res[sz]['placements'][p]['det_recall']) for sz in SIZES) + ' |')
    blocks['density'] = '\n'.join([
        'Recall per ground-truth density bucket on the **`random`** layout (Two-Stage/Single-Stage from '
        '`benchmark/density_sweep_results.json`, Grid/Multi-Anchor from `grid_density_sweep_results.json`, FCOS from '
        '`fcos_density_sweep_results.json`, and YOLOv8 from `yolov8_density_sweep_results.json`):', '',
        density_table(cols, 'random'), '',
        'The `random` layout hides the single-scale grid failure. Recall on the structured layouts '
        '(Grid/Multi-Anchor: mean over density buckets; FCOS and YOLOv8: overall layout recall):', '',
        '| Layout | Grid Medium ($K=1$, BCE) | Multi-Anchor Medium ($K=3$) | '
        + ' | '.join(f'FCOS {SIZE_NAMES[sz]}' for sz in SIZES) + ' | '
        + ' | '.join(f'YOLOv8 {SIZE_NAMES[sz]}' for sz in SIZES) + ' |',
        '|:---|' + ':---:|' * 8,
    ] + struct)
    return blocks


# ── Driver ────────────────────────────────────────────────────────────────────

def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf'((?:> )?<!-- BEGIN GENERATED: {name} [^>]*-->\n)(.*?)((?:> )?<!-- END GENERATED: {name} -->)', re.S)
    assert pattern.search(text), f'markers for block {name!r} not found'
    return pattern.sub(lambda m: m.group(1) + body + '\n' + m.group(3), text)


def render_file(path: Path, blocks: dict) -> None:
    text = path.read_text(encoding='utf-8')
    for name, body in blocks.items():
        text = replace_block(text, name, body)
    leftover = set(re.findall(r'<!-- BEGIN GENERATED: (\w+)', text)) - set(blocks)
    assert not leftover, f'{path}: no renderer for blocks {leftover}'
    path.write_text(text, encoding='utf-8')
    print(f"Rendered {len(blocks)} generated sections of '{path}'")


def main():
    res = load_json(FCOS_RESULTS)
    params = fcos_params()
    yolov8_res = load_json(YOLOV8_RESULTS)
    render_file(FCOS_REPORT, render_report_05(res, params))
    render_file(YOLOV8_REPORT, render_yolov8_report(yolov8_res))
    render_file(GRID_REPORT, render_report_03())
    render_file(MULTI_ANCHOR_REPORT, render_report_04())
    render_file(MASTER, render_master(res, params, yolov8_res))


if __name__ == '__main__':
    main()
