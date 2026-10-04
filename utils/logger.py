"""Moved to training/logger.py. This alias keeps `utils.logger` imports (and pickles) working."""
import sys

from training import logger as _module

sys.modules[__name__] = _module
