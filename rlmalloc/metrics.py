"""Evaluation metrics defined in the paper (Section II.A).

* Occupancy    = 1 - S_total / M
* Duration     = number of *successful* allocation requests (tracked by env)
* Fragmentation= 1 - max(l_i) / S_total          (lower is better)
* HHI          = sum((l_i / S_total) ** 2)       (higher is more concentrated)

HHI convention
--------------
The paper defines HHI in Eq.(1) and Section II.A as the *raw* sum
``sum((l_i/S_total)^2)``.  The legacy code reported ``1 - sum(...)``.  We
report the paper-faithful ``hhi`` and additionally expose
``hhi_complement = 1 - hhi`` for readers used to the other convention.
See ``docs/DEVIATIONS.md``.
"""

from __future__ import annotations

from typing import Iterable, Tuple

Block = Tuple[int, int]


def compute_metrics(
    free_blocks: Iterable[Block],
    memory_size: int,
) -> Tuple[float, float, float, float]:
    """Return ``(occupancy, fragmentation, hhi, hhi_complement)``."""
    blocks = list(free_blocks)
    total_free = sum(sz for _, sz in blocks)

    occupancy = (memory_size - total_free) / memory_size

    if total_free == 0:
        return occupancy, 0.0, 0.0, 1.0

    fragmentation = 1.0 - max(sz for _, sz in blocks) / total_free
    hhi = sum((sz / total_free) ** 2 for _, sz in blocks)
    return occupancy, fragmentation, hhi, 1.0 - hhi
