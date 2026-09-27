"""Evaluation metrics.

* Occupancy     = 1 - S_total / M
* Duration      = number of *successful* allocation requests (tracked by env)
* Fragmentation = 1 - max(l_i) / S_total          (lower is better)
* HHI           = sum((l_i / S_total) ** 2)       (higher = more concentrated)

HHI (Herfindahl-Hirschman index of the free-block size shares) is defined once,
as the sum of squared free-block size shares::

    HHI = sum_i (l_i / S_total) ** 2 ,   S_total = sum_i l_i

where ``l_i`` are the sizes of the free blocks at the end of a round.

Properties:

* range ``1/N <= HHI <= 1`` for ``N`` free blocks (``HHI = 1`` for a single
  free block, ``HHI = 1/N`` for ``N`` equal blocks, ``HHI = 0`` when no memory
  is free);
* higher = free space concentrated in fewer / larger blocks = healthy, i.e.
  low external fragmentation;
* bounded by the largest free share, which equals ``1 - Fragmentation``:
  ``HHI <= 1 - Fragmentation``.
"""

from __future__ import annotations

from typing import Iterable, Tuple

Block = Tuple[int, int]


def compute_metrics(
    free_blocks: Iterable[Block],
    memory_size: int,
) -> Tuple[float, float, float]:
    """Return ``(occupancy, fragmentation, hhi)`` under the unified HHI."""
    blocks = list(free_blocks)
    total_free = sum(sz for _, sz in blocks)

    occupancy = (memory_size - total_free) / memory_size

    if total_free == 0:
        return occupancy, 0.0, 0.0

    fragmentation = 1.0 - max(sz for _, sz in blocks) / total_free
    hhi = sum((sz / total_free) ** 2 for _, sz in blocks)
    return occupancy, fragmentation, hhi
