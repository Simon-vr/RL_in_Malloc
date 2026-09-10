"""Regression tests for the candidate-set / state / action consistency.

This is the guard against the historical bug where the state vector listed
blocks in one order (e.g. by size) while ``step`` decoded the action in
another order (by start address).
"""

import numpy as np
import torch

from rlmalloc import config
from rlmalloc.candidates import build_candidates, candidate_starts
from rlmalloc.env import MemoryEnv


def test_sorted_by_start_and_filtered():
    free = [(4096, 100), (0, 50), (100, 20), (200, 30)]
    cands = build_candidates(free, 25, 3)
    assert cands == [(0, 50), (200, 30), (4096, 100)]


def test_respects_k_and_filter():
    free = [(0, 100), (100, 100), (200, 100), (300, 10)]
    assert build_candidates(free, 50, 5) == [(0, 100), (100, 100), (200, 100)]
    assert len(build_candidates(free, 50, 2)) == 2
    assert build_candidates(free, 500, 5) == []


def test_candidate_starts_helper():
    assert candidate_starts([(5, 1), (2, 3)]) == [5, 2]


def test_state_encodes_exact_candidate_order():
    env = MemoryEnv(request_sequence=[("alloc", 0, 100)])
    cands = build_candidates(env.free_blocks, env.current_request_size,
                             config.N_CANDIDATE_BLOCKS)
    state = env._get_state()
    assert state.shape == (1, config.STATE_SIZE)

    for i, (start, size) in enumerate(cands):
        assert state[0, 2 * i].item() == size / env.memory_size
        assert state[0, 2 * i + 1].item() == start / env.memory_size

    # padding past the number of candidates must be zero
    for i in range(len(cands), config.N_CANDIDATE_BLOCKS):
        assert state[0, 2 * i].item() == 0.0
        assert state[0, 2 * i + 1].item() == 0.0
    assert state[0, -1].item() == env.current_request_size / env.memory_size


def test_step_decodes_same_order_as_state():
    """Take action i and verify the i-th candidate from the state is used."""
    events = [
        ("alloc", 0, 50),
        ("alloc", 1, 50),
        ("free", 0, 0),
        ("alloc", 2, 50),
    ]
    env = MemoryEnv(request_sequence=events)
    env.step(0)  # ordinal 0
    env.step(0)  # ordinal 1; advance processes free(0) and arms alloc 2

    cands = build_candidates(env.free_blocks, env.current_request_size,
                             config.N_CANDIDATE_BLOCKS)
    assert len(cands) >= 2
    target = cands[1]
    ordinal = env.next_alloc_ordinal
    env.step(1)
    assert env.allocated[ordinal][0] == target[0]
