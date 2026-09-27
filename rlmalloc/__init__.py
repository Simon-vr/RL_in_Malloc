"""RLmalloc: a DQN-based dynamic memory allocator.

The agent observes a small set of candidate free blocks (pre-filtered by a
First-Fit-style rule) and learns to pick one, guided by a free-space
concentration reward.  See ``docs/METHOD.md`` for the design and
``docs/DESIGN.md`` for the documented assumptions and limitations.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
