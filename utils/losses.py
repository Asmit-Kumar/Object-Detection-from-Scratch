"""Moved to training/losses.py. This alias keeps `utils.losses` imports (and pickles) working."""
import sys

from training import losses as _module

sys.modules[__name__] = _module
