"""Canonical candidate-block construction.

This is the *single source of truth* for the candidate set used by both the
state builder (:mod:`rlmalloc.env`) and the action decoder (also
:mod:`rlmalloc.env`).  Keeping one function guarantees that the block shown in
the state is exactly the block an action selects.

Candidate rule: sort free blocks by start address ascending and keep the first
``k`` whose size satisfies the current request.
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
