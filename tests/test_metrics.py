"""Tests for the metric definitions (paper Section II.A / Eq.1)."""

import pytest

from rlmalloc.metrics import compute_metrics


def test_single_contiguous_free_block():
    occ, frag, hhi, comp = compute_metrics([(0, 100)], memory_size=100)
    assert occ == pytest.approx(0.0)
    assert frag == pytest.approx(0.0)
    assert hhi == pytest.approx(1.0)
    assert comp == pytest.approx(0.0)


def test_memory_exhausted():
    occ, frag, hhi, comp = compute_metrics([], memory_size=100)
    assert occ == pytest.approx(1.0)
    assert frag == pytest.approx(0.0)
    assert hhi == pytest.approx(0.0)
    assert comp == pytest.approx(1.0)


def test_two_equal_blocks():
    occ, frag, hhi, comp = compute_metrics([(0, 50), (50, 50)], memory_size=100)
    assert occ == pytest.approx(0.0)
    assert frag == pytest.approx(0.5)
    assert hhi == pytest.approx(0.5)
    assert comp == pytest.approx(0.5)


def test_hhi_is_raw_sum_not_one_minus_sum():
    # total = 30 -> (10/30)^2 + (20/30)^2 = 5/9
    _, _, hhi, comp = compute_metrics([(0, 10), (10, 20)], memory_size=100)
    assert hhi == pytest.approx(5.0 / 9.0)
    assert comp == pytest.approx(1.0 - 5.0 / 9.0)


def test_fragmentation_uses_largest_block():
    _, frag, _, _ = compute_metrics([(0, 10), (10, 120)], memory_size=200)
    assert frag == pytest.approx(1.0 - 120.0 / 130.0)
