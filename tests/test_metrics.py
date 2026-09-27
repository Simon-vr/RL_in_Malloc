"""Tests for the metric definitions (paper Section II.A / Eq.1).

The single unified HHI is ``sum((l_i/S_total)^2)``.
"""

import pytest

from rlmalloc.metrics import compute_metrics


def test_single_contiguous_free_block():
    occ, frag, hhi = compute_metrics([(0, 100)], memory_size=100)
    assert occ == pytest.approx(0.0)
    assert frag == pytest.approx(0.0)
    assert hhi == pytest.approx(1.0)


def test_memory_exhausted():
    occ, frag, hhi = compute_metrics([], memory_size=100)
    assert occ == pytest.approx(1.0)
    assert frag == pytest.approx(0.0)
    assert hhi == pytest.approx(0.0)


def test_two_equal_blocks():
    occ, frag, hhi = compute_metrics([(0, 50), (50, 50)], memory_size=100)
    assert occ == pytest.approx(0.0)
    assert frag == pytest.approx(0.5)
    assert hhi == pytest.approx(0.5)  # 1/N for N equal blocks


def test_hhi_is_raw_sum():
    # total = 30 -> (10/30)^2 + (20/30)^2 = 5/9
    _, _, hhi = compute_metrics([(0, 10), (10, 20)], memory_size=100)
    assert hhi == pytest.approx(5.0 / 9.0)


def test_hhi_bounded_by_one_minus_fragmentation():
    # For free-block shares s_i, sum(s_i^2) <= max(s_i) = 1 - fragmentation.
    _, frag, hhi = compute_metrics(
        [(0, 10), (10, 20), (30, 5)], memory_size=100
    )
    assert hhi <= (1.0 - frag) + 1e-12


def test_fragmentation_uses_largest_block():
    _, frag, _ = compute_metrics([(0, 10), (10, 120)], memory_size=200)
    assert frag == pytest.approx(1.0 - 120.0 / 130.0)
