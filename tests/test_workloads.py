"""Tests for the workload generator and distributions."""

import numpy as np

from rlmalloc import config
from rlmalloc.workloads import (
    DISTRIBUTIONS,
    generate_workload,
    sample_request,
    sample_request_by_name,
)


def test_sample_in_bounds_for_all_distributions():
    rng = np.random.default_rng(0)
    for name in config.DISTRIBUTIONS:
        for _ in range(2000):
            s = sample_request_by_name(name, rng)
            assert config.MIN_REQUEST_SIZE <= s <= config.MAX_REQUEST_SIZE


def test_workload_free_targets_are_live_on_reference_path():
    rng = np.random.default_rng(1)
    events = generate_workload("lognormal_train", n_allocs=200,
                               release_rate=0.5, rng=rng)
    live = set()
    for event in events:
        if event[0] == "alloc":
            _, _ordinal, _size = event
            live.add(_ordinal)
        else:
            _, ordinal, _ = event
            assert ordinal in live, "free targets a non-live ordinal"
            live.discard(ordinal)


def test_workload_is_deterministic_for_fixed_seed():
    e1 = generate_workload("uniform", 50, 0.3, np.random.default_rng(7))
    e2 = generate_workload("uniform", 50, 0.3, np.random.default_rng(7))
    assert e1 == e2


def test_workload_has_exactly_n_allocs():
    events = generate_workload("bimodal", n_allocs=37, release_rate=0.4,
                               rng=np.random.default_rng(3))
    n_alloc = sum(1 for e in events if e[0] == "alloc")
    assert n_alloc == 37


def test_bimodal_uses_two_centres():
    rng = np.random.default_rng(0)
    cfg = DISTRIBUTIONS["bimodal"]
    samples = np.array([sample_request(cfg, rng) for _ in range(20000)])
    assert (samples < 64).mean() > 0.4
    assert (samples > 128).mean() > 0.1
