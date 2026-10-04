"""Moved to dataio/fcos_targets.py. This alias keeps `utils.fcos_targets` imports (and pickles) working."""
import sys

from dataio import fcos_targets as _module

sys.modules[__name__] = _module
