"""Moved to dataio/reader.py. This alias keeps `utils.reader` imports (and pickles) working."""
import sys

from dataio import reader as _module

sys.modules[__name__] = _module
