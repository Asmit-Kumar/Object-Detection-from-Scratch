"""
Moved to models/archive/grid14_detector.py (class Grid14Detector).

This module keeps the old import working for scripts/run_single_stage_benchmark.py:
    from legacy_models.grid_detector_14x14 import ObjectDetectorResNet
The architecture is unchanged: same layers, CONFIGS, constructor and outputs as the frozen
snapshot of commit 2be0b81 that used to live here (pinned by tests/test_archive.py).
"""
from models.archive.grid14_detector import Grid14Detector as ObjectDetectorResNet  # noqa: F401
from models.common import SimpleResBlock  # noqa: F401
