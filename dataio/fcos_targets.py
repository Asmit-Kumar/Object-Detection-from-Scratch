"""Renamed to dataio/ltrb_targets.py. This alias keeps `dataio.fcos_targets` imports (and pickles) working."""
import sys

from dataio import ltrb_targets as _module

sys.modules[__name__] = _module
