"""Moved to inference/pipeline.py. This alias keeps `utils.pipeline` imports (and pickles) working."""
import sys

from inference import pipeline as _module

sys.modules[__name__] = _module
