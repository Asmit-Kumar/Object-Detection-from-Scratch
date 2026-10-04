"""Moved to training/callbacks.py. This alias keeps `utils.callbacks` imports (and pickles) working."""
import sys

from training import callbacks as _module

sys.modules[__name__] = _module
