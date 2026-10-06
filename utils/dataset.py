"""Moved to dataio/dataset.py. This alias keeps `utils.dataset` imports (and pickles) working."""
import sys

from dataio import dataset as _module

sys.modules[__name__] = _module
