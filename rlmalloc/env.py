"""Memory-allocation environment (the MDP).

Two modes:

* ``mode="train"``  requests are sampled i.i.d. and random frees occur after
  each allocation with probability ``RELEASE_RATE``.
* ``mode="eval"``   the environment replays an explicit event list generated
  by :func:`rlmalloc.workloads.generate_workload`.  Free events resolve an
  allocation *ordinal* (the n-th successful allocation) to its block.

Key behaviour (paper Section III.B):

* candidate set = :func:`rlmalloc.candidates.build_candidates` (sorted by
  start address, first k that fit) -- used by both state and action decoding;
* step(action) with an out-of-range/invalid action returns reward ``-1.0`` and
  terminates the episode;
* reward is the HHI ``sum((l_i / S_total)^2)`` of the free-block set, bounded
  in [0, 1] (0 when memory is empty);
* duration counts only *successful* allocations.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import config
from .candidates import build_candidates
from .workloads import Event, sample_request

Block = Tuple[int, int]


class MemoryEnv:
    """Dynamic memory allocator environment."""

    def __init__(
        self,
        dist: str = config.DEFAULT_TRAIN_DIST,
        request_sequence: Optional[List[Event]] = None,
        rng: Optional[np.random.Generator] = None,
        memory_size: int = config.MEMORY_SIZE,
    ) -> None:
        self.memory_size = memory_size
        self.dist = dist
        self.request_sequence = request_sequence
        self.mode = "eval" if request_sequence is not None else "train"
        self.rng = rng if rng is not None else np.random.default_rng()
        self._py_rng = random.Random()

        # runtime state (populated by reset)
        self.free_blocks: List[Block] = []
        self.allocated: Dict[int, Block] = {}
        self.next_alloc_ordinal = 0
        self.duration = 0
        self.request_idx = 0
        self.current_request_size: Optional[int] = None
        self.current_ordinal: Optional[int] = None
        self.reset()

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def set_python_seed(self, seed: int) -> None:
        self._py_rng = random.Random(seed)

    def reset(self) -> torch.Tensor:
        self.free_blocks = [(0, self.memory_size)]
        self.allocated = {}
        self.next_alloc_ordinal = 0
        self.duration = 0
        self.request_idx = 0

        if self.mode == "eval":
            self.current_request_size = None
            self.current_ordinal = None
            self._advance_eval_request()
        else:
            self.current_request_size = self._sample_train_request()
            self.current_ordinal = None
        return self._get_state()

    # ------------------------------------------------------------------ #
    # free-list maintenance
    # ------------------------------------------------------------------ #
    def _merge_free_blocks(self) -> None:
        self.free_blocks.sort()
        if len(self.free_blocks) < 2:
            return
        merged: List[Block] = []
        cur_start, cur_size = self.free_blocks[0]
        for start, size in self.free_blocks[1:]:
            if cur_start + cur_size == start:
                cur_size += size
            else:
                merged.append((cur_start, cur_size))
                cur_start, cur_size = start, size
        merged.append((cur_start, cur_size))
        self.free_blocks = merged

    def _release(self, ordinal: int) -> bool:
        """Release an allocation by ordinal. Returns True if it existed."""
        if ordinal not in self.allocated:
            return False
        start, size = self.allocated.pop(ordinal)
        self.free_blocks.append((start, size))
        self._merge_free_blocks()
        return True

    # ------------------------------------------------------------------ #
    # request advancement
    # ------------------------------------------------------------------ #
    def _sample_train_request(self) -> int:
        return sample_request(config.DISTRIBUTIONS[self.dist], self.rng)

    def _advance_eval_request(self) -> None:
        """Consume free events, then arm the next allocation request.

        Sets ``current_request_size``/``current_ordinal`` to ``None`` once the
        event list is exhausted.  Non-recursive (fixes legacy trailing-free
        fallthrough).
        """
        self.current_request_size = None
        self.current_ordinal = None
        while self.request_idx < len(self.request_sequence):
            event = self.request_sequence[self.request_idx]
            self.request_idx += 1
            if event[0] == "free":
                self._release(int(event[1]))
            elif event[0] == "alloc":
                self.current_ordinal = int(event[1])
                self.current_request_size = int(event[2])
                return
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown event type: {event!r}")

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def _get_state(self) -> torch.Tensor:
        state = np.zeros(config.STATE_SIZE, dtype=np.float32)
        if self.current_request_size is not None:
            cands = build_candidates(
                self.free_blocks, self.current_request_size,
                config.N_CANDIDATE_BLOCKS,
            )
            for i, (start, size) in enumerate(cands):
                state[i * 2] = size / self.memory_size
                state[i * 2 + 1] = start / self.memory_size
            state[-1] = self.current_request_size / self.memory_size
        return torch.from_numpy(state).unsqueeze(0)

    # ------------------------------------------------------------------ #
    # reward / termination
    # ------------------------------------------------------------------ #
    def _reward(self) -> float:
        total = sum(sz for _, sz in self.free_blocks)
        if total == 0:
            return 0.0
        return float(sum((sz / total) ** 2 for _, sz in self.free_blocks))

    def is_done(self) -> bool:
        if self.current_request_size is None:
            return True
        return not any(sz >= self.current_request_size
                       for _, sz in self.free_blocks)

    # ------------------------------------------------------------------ #
    # stepping
    # ------------------------------------------------------------------ #
    def _commit_allocation(self, start: int, size: int):
        # split block
        self.free_blocks.remove((start, size))
        req = int(self.current_request_size)
        ordinal = self.next_alloc_ordinal
        self.allocated[ordinal] = (start, req)
        self.next_alloc_ordinal += 1
        self.duration += 1

        if size > req:
            self.free_blocks.append((start + req, size - req))
            self.free_blocks.sort()

        if self.mode == "train":
            if self.allocated and self._py_rng.random() < config.RELEASE_RATE:
                victim = self._py_rng.choice(list(self.allocated.keys()))
                self._release(victim)
        else:
            self._advance_eval_request()

        reward = self._reward()

        if self.mode == "train":
            self.current_request_size = self._sample_train_request()

        done = self.is_done()
        return (
            self._get_state(),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done], dtype=torch.bool),
        )

    def step(self, action: int):
        """DQN action: index into the canonical candidate list."""
        cands = build_candidates(
            self.free_blocks, self.current_request_size,
            config.N_CANDIDATE_BLOCKS,
        )
        if action < 0 or action >= len(cands):
            return (
                self._get_state(),
                torch.tensor([config.INVALID_ACTION_REWARD], dtype=torch.float32),
                torch.tensor([True], dtype=torch.bool),
            )
        start, size = cands[action]
        return self._commit_allocation(start, size)

    def allocate_block(self, start: int, size: int):
        """Explicit block choice (used by heuristic baselines)."""
        if self.current_request_size is None:
            return (
                self._get_state(),
                torch.tensor([config.INVALID_ACTION_REWARD], dtype=torch.float32),
                torch.tensor([True], dtype=torch.bool),
            )
        if size < self.current_request_size or (start, size) not in self.free_blocks:
            return (
                self._get_state(),
                torch.tensor([config.INVALID_ACTION_REWARD], dtype=torch.float32),
                torch.tensor([True], dtype=torch.bool),
            )
        return self._commit_allocation(start, size)
