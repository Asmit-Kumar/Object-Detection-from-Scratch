"""Moved to training/trainer.py. This alias keeps `utils.trainer` imports (and pickles) working."""
import sys

from training import trainer as _module

sys.modules[__name__] = _module
