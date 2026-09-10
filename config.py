"""Backward-compatibility shim.

The implementation now lives in :mod:`rlmalloc.config`.  Importing this flat
module keeps the historical ``import config`` working.
"""

from rlmalloc.config import *  # noqa: F401,F403
