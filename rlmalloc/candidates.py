"""Canonical candidate-block construction.

This is the *single source of truth* for the candidate set used by both the
state builder (:mod:`rlmalloc.env`) and the action decoder (also
:mod:`rlmalloc.env`).  Keeping one function is what prevents the historical
state/action mismatch bug (state listed blocks sorted by size/start in one
order, while the action was decoded in another).

Paper reference: p.2 III.B.1 -- "sort by start address in ascending order and
take the first k free blocks whose sizes satisfy the current request".
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

Block = Tuple[int, int]  # (start, size)


def build_candidates(
    free_blocks: Iterable[Block],
    request_size: int,
    k: int,
) -> List[Block]:
    """Return up to ``k`` blocks that can satisfy ``request_size``.

    Blocks are sorted by start address ascending and filtered by
    ``size >= request_size``.  The returned list has length ``<= k``.
    """
    if request_size is None:
        return []
    return [(s, sz) for (s, sz) in sorted(free_blocks) if sz >= request_size][:k]


def candidate_starts(candidates: Sequence[Block]) -> List[int]:
    """Convenience accessor for tests / debugging."""
    return [s for s, _ in candidates]
