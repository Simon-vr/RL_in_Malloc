"""Tests for the MDP environment semantics."""

from rlmalloc import config
from rlmalloc.candidates import build_candidates
from rlmalloc.env import MemoryEnv


def test_invalid_action_is_terminal_with_negative_reward():
    # request larger than total memory -> no candidate -> any action invalid
    env = MemoryEnv(request_sequence=[("alloc", 0, 10_000)])
    state, reward, done = env.step(0)
    assert reward.item() == config.INVALID_ACTION_REWARD
    assert bool(done.item()) is True
    assert env.duration == 0


def test_duration_counts_only_successful_allocations():
    events = [("alloc", 0, 100), ("alloc", 1, 100), ("alloc", 2, 100)]
    env = MemoryEnv(request_sequence=events)
    done = env.is_done()
    while not done:
        cands = build_candidates(env.free_blocks, env.current_request_size, 5)
        assert cands
        _, _, done_t = env.step(0)
        done = bool(done_t.item())
    assert env.duration == 3


def test_free_by_ordinal_releases_the_right_block():
    events = [
        ("alloc", 0, 100),
        ("alloc", 1, 100),
        ("free", 1, 0),
        ("alloc", 2, 100),
    ]
    env = MemoryEnv(request_sequence=events)
    done = env.is_done()
    while not done:
        _, _, done_t = env.step(0)
        done = bool(done_t.item())
    assert env.duration == 3
    # ordinal 1 was freed, so only 0 and 2 remain allocated
    assert set(env.allocated.keys()) == {0, 2}


def test_advance_is_non_recursive_and_handles_trailing_frees():
    events = [("alloc", 0, 100), ("free", 0, 0)]
    env = MemoryEnv(request_sequence=events)
    # after the single allocation, the trailing free must be processed and
    # the episode must terminate (not recurse forever).
    _, _, done = env.step(0)
    assert bool(done.item()) is True
    assert env.duration == 1
    assert env.allocated == {}
    assert env.free_blocks == [(0, config.MEMORY_SIZE)]


def test_reward_is_hhi_and_bounded():
    env = MemoryEnv(request_sequence=[("alloc", 0, 100)])
    _, reward, _ = env.step(0)
    r = float(reward.item())
    assert 0.0 <= r <= 1.0


def test_training_mode_runs_and_respects_release():
    env = MemoryEnv(dist="lognormal_train")
    env.set_python_seed(123)
    state = env.reset()
    assert state.shape == (1, config.STATE_SIZE)
    for _ in range(100):
        _, _, done = env.step(0)
        if bool(done.item()):
            break
    assert env.duration >= 1
