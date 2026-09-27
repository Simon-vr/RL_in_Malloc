"""Evaluation metrics defined in the paper (Section II.A).

* Occupancy     = 1 - S_total / M
* Duration      = number of *successful* allocation requests (tracked by env)
* Fragmentation = 1 - max(l_i) / S_total          (lower is better)
* HHI           = sum((l_i / S_total) ** 2)       (higher = more concentrated)

Unified HHI definition
----------------------
This project uses exactly **one** HHI definition everywhere: the paper's own
Eq.(1) / Section II.A formula

    HHI = sum_i (l_i / S_total) ** 2 ,   S_total = sum_i l_i

where ``l_i`` are the sizes of the free blocks at the end of a round.

Properties:

* range ``1/N <= HHI <= 1`` for ``N`` free blocks (``HHI = 1`` for a single
  free block, ``HHI = 1/N`` for ``N`` equal blocks, ``HHI = 0`` when no memory
  is free);
* higher = free space concentrated in fewer / larger blocks = healthy, i.e.
  low external fragmentation;
* it is bounded above by the largest free share, which equals
  ``1 - Fragmentation``:  ``HHI <= 1 - Fragmentation``.

The paper's *printed* experimental numbers (e.g. DQN 0.8165, and Worst-Fit
highest overall) are **not** consistent with this formula -- they behave like
``1 - HHI``. That complement has the opposite direction (higher = more
fragmented) and is not the HHI, so it is deliberately **not** reported as a
metric here. To compare against the paper's printed table, compute
``1 - HHI``. See ``docs/DEVIATIONS.md`` for the proof.
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
