"""Allocation policies.

Heuristic baselines operate over *all* free blocks, exactly as the paper
defines them (p.4 IV.A).  They may therefore pick a block outside the DQN's
first-k candidate set; this asymmetry is intentional and documented in
``docs/DEVIATIONS.md``.

The DQN "policy" is not a block-picking function: the agent outputs an index
into the canonical candidate set and :meth:`rlmalloc.env.MemoryEnv.step`
decodes it.  ``dqn_action`` is the thin adapter used by the evaluation loop.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

Block = Tuple[int, int]

HEURISTIC_NAMES = ["best_fit", "first_fit", "worst_fit"]


def first_fit(free_blocks, request_size: int) -> Optional[Block]:
    for start, size in sorted(free_blocks):
        if size >= request_size:
            return start, size
    return None


def best_fit(free_blocks, request_size: int) -> Optional[Block]:
    best: Optional[Block] = None
    min_waste = float("inf")
    for start, size in free_blocks:
        if size >= request_size:
            waste = size - request_size
            if waste < min_waste:
                min_waste = waste
                best = (start, size)
    return best


def worst_fit(free_blocks, request_size: int) -> Optional[Block]:
    best: Optional[Block] = None
    max_size = -1
    for start, size in free_blocks:
        if size >= request_size and size > max_size:
            max_size = size
            best = (start, size)
    return best


_HEURISTICS: dict[str, Callable] = {
    "first_fit": first_fit,
    "best_fit": best_fit,
    "worst_fit": worst_fit,
}

# Display names used in the report / figures.
DISPLAY_NAMES = {
    "dqn": "DQN Agent",
    "first_fit": "First-Fit",
    "best_fit": "Best-Fit",
    "worst_fit": "Worst-Fit",
}


def get_heuristic(name: str) -> Callable:
    if name not in _HEURISTICS:
        raise KeyError(f"unknown heuristic {name!r}; "
                       f"available: {sorted(_HEURISTICS)}")
    return _HEURISTICS[name]


def dqn_action(agent, state) -> int:
    """Return the greedy action index from a trained agent."""
    return int(agent.select_action(state, is_test=True).item())
