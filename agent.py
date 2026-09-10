"""Backward-compatibility shim for the legacy ``agent`` module.

The implementation now lives in :mod:`rlmalloc.agent`.
"""

from rlmalloc.agent import (  # noqa: F401
    DQNAgent,
    QNetwork,
    Transition,
    save_checkpoint_meta,
)

__all__ = ["DQNAgent", "QNetwork", "Transition", "save_checkpoint_meta"]
