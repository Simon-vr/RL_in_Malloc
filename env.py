"""Backward-compatibility shim for the legacy ``env`` module.

The implementation now lives in :mod:`rlmalloc.env`.
"""

from rlmalloc.env import MemoryEnv  # noqa: F401

__all__ = ["MemoryEnv"]
